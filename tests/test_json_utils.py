"""``utils/json_utils`` — two parsers with deliberately different strictness.

``parse_json_response`` is what the LLM-judge metrics call. Its contract is that
the whole response, once the ```json fences are stripped, must *be* JSON: a
judge prompt pins its output format tightly, and a judge that starts
editorialising has not answered, so the right outcome is a raised
``JSONDecodeError`` the metric turns into a ``None`` score. Guessing at the JSON
buried in the prose would silently score a run off a response the prompt did not
ask for.

``extract_json_object`` is the looser reading, added for structured *generation*,
where a ``Here is the JSON:`` preamble is a real answer and burning the one
allowed reask on it costs a real API call. The two are separate functions on
purpose — these tests pin that separation, so a change made for the generation
layer cannot quietly loosen the metrics' contract.
"""

import json

import pytest

from llminspector.utils.json_utils import extract_json_object, parse_json_response

# One payload, rendered every way a model plausibly returns it. Shared between
# both parsers so the "extract handles everything parse does" claim is literal
# rather than two independently drifting lists.
PAYLOAD = {"text": "yes", "score": 0.5}
BARE = '{"text": "yes", "score": 0.5}'

BOTH_PARSE = [
    ("bare", BARE),
    ("json_fence", f"```json\n{BARE}\n```"),
    ("plain_fence", f"```\n{BARE}\n```"),
    ("leading_whitespace", f"\n\n  {BARE}  \n"),
]

ONLY_EXTRACT_PARSES = [
    ("prose_before", f"Here is the JSON:\n{BARE}"),
    ("prose_after", f"{BARE}\n\nHope that helps."),
    ("prose_both_sides", f"Sure! Here you go:\n{BARE}\nLet me know if you need more."),
    ("prose_around_a_fence", f"Sure:\n```json\n{BARE}\n```\nThat's the answer."),
]


# --------------------------------------------------------------------------- #
# parse_json_response — the strict one the judge metrics rely on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text", [t for _, t in BOTH_PARSE], ids=[i for i, _ in BOTH_PARSE]
)
def test_parse_json_response_handles_bare_and_fenced_json(text):
    """The legacy strip-fences-then-loads sequence, centralised."""
    assert parse_json_response(text) == PAYLOAD


@pytest.mark.parametrize(
    "text", [t for _, t in ONLY_EXTRACT_PARSES], ids=[i for i, _ in ONLY_EXTRACT_PARSES]
)
def test_parse_json_response_rejects_prose_wrapped_json(text):
    """Strictness is the feature, not an oversight.

    A judge metric that accepted prose-wrapped JSON would score off a response
    that ignored the output format its prompt pinned. Raising here is what makes
    the metric record an error and a ``None`` score instead. The tolerant
    reading lives in :func:`extract_json_object`; keep it there.
    """
    with pytest.raises(json.JSONDecodeError):
        parse_json_response(text)


def test_parse_json_response_honours_a_custom_fence_marker():
    """``json_block`` is a parameter, and the metrics' JSON_BLOCK is its default."""
    assert parse_json_response("```JSON\n" + BARE + "\n```", json_block="```JSON")


# --------------------------------------------------------------------------- #
# extract_json_object — everything above, plus prose
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [t for _, t in BOTH_PARSE + ONLY_EXTRACT_PARSES],
    ids=[i for i, _ in BOTH_PARSE + ONLY_EXTRACT_PARSES],
)
def test_extract_json_object_is_a_superset_of_parse_json_response(text):
    """Same value for every shape the strict parser accepts, plus the prose ones."""
    assert extract_json_object(text) == PAYLOAD


def test_extract_json_object_slices_to_the_outermost_braces():
    """Nested objects must not be truncated at the first closing brace.

    The fallback is ``find("{")``/``rfind("}")`` precisely so a nested payload
    survives; a ``find("}")`` would slice ``{"a": {"b": 1}`` and fail.
    """
    text = 'Here is what I found:\n{"a": {"b": 1}, "c": [2, 3]}\nDone.'
    assert extract_json_object(text) == {"a": {"b": 1}, "c": [2, 3]}


def test_extract_json_object_reraises_when_there_is_no_object_at_all():
    """No braces means nothing to slice — the original decode error propagates.

    Returning ``None`` instead would hand the caller a value indistinguishable
    from a model that legitimately answered ``null``, and would deprive
    ``generate_structured`` of the error text it puts in the reask prompt.
    """
    with pytest.raises(json.JSONDecodeError):
        extract_json_object("I'm sorry, I cannot help with that request.")


def test_extract_json_object_reraises_on_a_mismatched_brace_order():
    """``end <= start`` is not a sliceable object either."""
    with pytest.raises(json.JSONDecodeError):
        extract_json_object("closing } then opening {")


def test_extract_json_object_returns_a_truncated_object_unparsed():
    """An unterminated object still raises rather than being repaired."""
    with pytest.raises(json.JSONDecodeError):
        extract_json_object('Here you go: {"text": "yes", "score":')


# --------------------------------------------------------------------------- #
# non-object JSON — pinned as implemented, see the docstring
# --------------------------------------------------------------------------- #


def test_a_bare_json_list_parses_on_the_strict_path():
    """Neither function is object-only when the *whole* response is JSON.

    ``parse_json_response`` is a plain ``json.loads``, so a top-level array (or
    string, or number) comes back as-is, and ``extract_json_object`` inherits
    that because it tries the strict path first. The name says "object" but the
    happy path is "any JSON value" — worth knowing before relying on the return
    being a dict.
    """
    assert parse_json_response("[1, 2]") == [1, 2]
    assert extract_json_object("[1, 2]") == [1, 2]
    assert extract_json_object("```json\n[1, 2]\n```") == [1, 2]


def test_a_prose_wrapped_json_list_is_not_recovered():
    """The fallback only knows about braces, so an array in prose is lost.

    Asymmetric with the object case above, and deliberate to pin rather than
    endorse: the fallback exists for ``generate_structured``, whose schemas are
    pydantic models and therefore always JSON *objects*. If a caller ever needs
    a top-level array from a chatty model, this is the line to revisit.
    """
    with pytest.raises(json.JSONDecodeError):
        extract_json_object("Here is the list: [1, 2]")
