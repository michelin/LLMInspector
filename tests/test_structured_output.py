"""Structured output and async embeddings are *capabilities with defaults*.

The point of this module is what it does **not** need. ``models/CLAUDE.md`` says
"the contract is deliberately small... a new provider is a class with three
methods", and the whole design of ``generate_structured`` / ``a_generate_structured``
and ``a_embed_text`` / ``a_embed_texts`` is subordinate to that: they are
concrete methods on the ABCs, never ``@abstractmethod``, so adding them did not
turn the three-method contract into a four- or five-method one.

So the central tests here are driven by stubs that implement *only* the three
required methods — ``ScriptedLLM`` from ``tests/conftest.py`` on the LLM side, a
minimal embedding stub on the other — and assert the inherited defaults work.
``tests/test_provider_contract.py`` makes the same guarantee for metrics; this is
its counterpart for the generation layer.

Everything else follows from that: the reask costs a real API call, so call
counts are asserted exactly rather than loosely, and every scenario is run
through both the sync and the async entry point and compared, because divergent
sync/async behaviour is the single worst defect this layer can carry.
"""

import asyncio
import json
import threading
from typing import Any, List, NamedTuple, Optional

import pytest
from pydantic import BaseModel, ValidationError

from llminspector.models.base_model import (
    STRUCTURED_OUTPUT_INSTRUCTION,
    STRUCTURED_OUTPUT_REASK,
    BaseEmbeddingModel,
    BaseLLM,
)
from llminspector.models.errors import StructuredOutputError
from tests.conftest import FakeEmbedding, ScriptedLLM

# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #


class Answer(BaseModel):
    """The minimal two-required-field schema most assertions use."""

    text: str
    score: float


class Citation(BaseModel):
    """Nested model, with an optional field so validation has something to bite."""

    source: str
    page: Optional[int] = None


class Report(BaseModel):
    """Nesting plus optionality — the shape a real synthesizer schema has."""

    answer: Answer
    citations: List[Citation] = []
    note: Optional[str] = None


PROMPT = "How many requests per minute does the API allow?"
VALID = '{"text": "500 per minute", "score": 0.9}'
MALFORMED = "I'm not going to answer that."
SCHEMA_INVALID = '{"text": "500 per minute"}'  # valid JSON, missing `score`


def _assert_answer(result: Any) -> None:
    """The one place the expected parse of ``VALID`` is spelled out."""
    assert isinstance(result, Answer)
    assert result.text == "500 per minute"
    assert result.score == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# sync/async parity harness
#
# Every scenario below runs through both entry points. The two implementations
# are near-duplicates by necessity (there is no way to write one async body that
# a sync caller can drive without an event loop), which is exactly the situation
# where they drift. `asyncio.run` inside a plain sync test keeps the project free
# of pytest-asyncio.
# --------------------------------------------------------------------------- #


def _sync(model: BaseLLM, prompt: str, schema: Any, **kwargs: Any) -> Any:
    return model.generate_structured(prompt, schema, **kwargs)


def _async(model: BaseLLM, prompt: str, schema: Any, **kwargs: Any) -> Any:
    return asyncio.run(model.a_generate_structured(prompt, schema, **kwargs))


both_paths = pytest.mark.parametrize("run", [_sync, _async], ids=["sync", "async"])


# --------------------------------------------------------------------------- #
# the central assertion — the provider contract did not grow
# --------------------------------------------------------------------------- #


@both_paths
def test_a_three_method_provider_gets_structured_output_for_free(run):
    """``ScriptedLLM`` implements only get_model_name / generate / a_generate.

    That it can none the less return a validated pydantic model is the proof
    that structured output landed as an inherited default and *not* as a fourth
    required method. If this ever needs a fourth method on the stub, the change
    that caused it broke the documented provider contract.
    """
    model = ScriptedLLM([VALID])
    _assert_answer(run(model, PROMPT, Answer))
    assert model.calls == 1


def test_the_stub_does_not_define_structured_output_itself():
    """Guards the test above against accidentally testing an override.

    If ``ScriptedLLM`` ever grew its own ``generate_structured``, every
    assertion in this module would be exercising the stub rather than the ABC
    default, and would keep passing after the default was deleted.
    """
    assert "generate_structured" not in vars(ScriptedLLM)
    assert "a_generate_structured" not in vars(ScriptedLLM)
    assert isinstance(ScriptedLLM([VALID]), BaseLLM)


@both_paths
def test_a_nested_schema_with_optional_fields_round_trips(run):
    """Defaults and nesting are pydantic's job — this pins that we let it do it."""
    model = ScriptedLLM(
        [
            '{"answer": {"text": "500 per minute", "score": 0.9}, '
            '"citations": [{"source": "docs", "page": 4}, {"source": "faq"}]}'
        ]
    )
    report = run(model, PROMPT, Report)
    assert isinstance(report, Report)
    assert report.answer.score == pytest.approx(0.9)
    assert [c.source for c in report.citations] == ["docs", "faq"]
    assert report.citations[1].page is None  # optional default, not an error
    assert report.note is None
    assert model.calls == 1


# --------------------------------------------------------------------------- #
# response shapes that must parse on the FIRST call
#
# The call count is part of every assertion, not decoration: a reask is a real,
# billed API call and a whole extra round trip. A shape that "works" but costs
# two calls is a regression even though the return value is right.
# --------------------------------------------------------------------------- #

FIRST_CALL_SHAPES = [
    ("bare", VALID),
    ("json_fence", f"```json\n{VALID}\n```"),
    ("plain_fence", f"```\n{VALID}\n```"),
    ("prose_before", f"Here is the JSON:\n{VALID}"),
    ("prose_after", f"{VALID}\n\nLet me know if you need anything else."),
    ("prose_both_sides", f"Sure, here you go:\n{VALID}\nHope that helps."),
    ("fence_and_prose", f"Certainly.\n```json\n{VALID}\n```\nThat's the answer."),
]


@both_paths
@pytest.mark.parametrize(
    "response",
    [r for _, r in FIRST_CALL_SHAPES],
    ids=[i for i, _ in FIRST_CALL_SHAPES],
)
def test_common_response_shapes_parse_without_a_reask(run, response):
    """Every shape a cooperative model actually emits costs exactly one call."""
    model = ScriptedLLM([response])
    _assert_answer(run(model, PROMPT, Answer))
    assert model.calls == 1, "wasted a reask on a parseable response"


# --------------------------------------------------------------------------- #
# the reask path
# --------------------------------------------------------------------------- #


@both_paths
def test_malformed_json_triggers_exactly_one_reask(run):
    model = ScriptedLLM([MALFORMED, VALID])
    _assert_answer(run(model, PROMPT, Answer))
    assert model.calls == 2


@both_paths
def test_schema_invalid_json_also_triggers_the_reask(run):
    """The reask fires on ``ValidationError``, not only ``JSONDecodeError``.

    ``{"text": "..."}`` is perfectly good JSON — it just isn't an ``Answer``.
    Catching only the decode error would hand the caller a bare ValidationError
    from inside the model layer on the single most common LLM failure mode:
    right format, missing field.
    """
    model = ScriptedLLM([SCHEMA_INVALID, VALID])
    _assert_answer(run(model, PROMPT, Answer))
    assert model.calls == 2


@both_paths
def test_a_wrongly_typed_field_triggers_the_reask(run):
    """`score` as unparseable text is a ValidationError, not a decode error."""
    model = ScriptedLLM(['{"text": "ok", "score": "very high"}', VALID])
    _assert_answer(run(model, PROMPT, Answer))
    assert model.calls == 2


@both_paths
def test_the_reask_prompt_carries_everything_the_model_needs(run):
    """Original prompt + schema + the exact bad response + the error text.

    A reask missing any of the four is a wasted call: the model cannot tell what
    it did wrong, so it repeats it and the run fails after two calls instead of
    succeeding on the second.
    """
    model = ScriptedLLM([SCHEMA_INVALID, VALID])
    run(model, PROMPT, Answer)

    reask = model.prompts[1]
    assert PROMPT in reask, "reask dropped the user's original prompt"
    assert "score" in reask and "Answer" in reask, "reask dropped the schema"
    assert SCHEMA_INVALID in reask, "reask dropped the response that failed"
    assert "ValidationError" in reask, "reask dropped the error type"
    # The reask is an *extension* of the first prompt, not a replacement, so the
    # schema block does not have to be re-rendered.
    assert reask.startswith(model.prompts[0])


@both_paths
def test_the_reask_prompt_names_a_decode_error_too(run):
    model = ScriptedLLM([MALFORMED, VALID])
    run(model, PROMPT, Answer)
    reask = model.prompts[1]
    assert MALFORMED in reask
    assert "JSONDecodeError" in reask


# --------------------------------------------------------------------------- #
# giving up — exactly two calls, never three
# --------------------------------------------------------------------------- #


@both_paths
def test_two_failures_raise_after_exactly_two_calls(run):
    """One reask, then stop.

    A malformed response is not a transient condition, so this axis deliberately
    does *not* share the exponential backoff in ``models/retry.py``: retrying a
    confused model five times buys five identical paragraphs of prose. The
    ``calls == 2`` assertion is what stops someone "improving" it into a loop.
    """
    model = ScriptedLLM([MALFORMED, "still not JSON"])
    with pytest.raises(StructuredOutputError) as err:
        run(model, PROMPT, Answer)

    assert model.calls == 2
    assert err.value.schema_name == "Answer"
    assert err.value.response == "still not JSON"  # the *final* attempt
    assert isinstance(err.value.__cause__, (ValidationError, json.JSONDecodeError))


@both_paths
def test_two_validation_failures_chain_the_validation_error(run):
    model = ScriptedLLM([SCHEMA_INVALID, SCHEMA_INVALID])
    with pytest.raises(StructuredOutputError) as err:
        run(model, PROMPT, Answer)

    assert model.calls == 2
    assert isinstance(err.value.__cause__, ValidationError)
    assert err.value.response == SCHEMA_INVALID


@both_paths
def test_structured_output_error_is_a_value_error(run):
    """Callers already writing ``except ValueError`` around a parse keep working.

    Both underlying failures (``json.JSONDecodeError``, pydantic's
    ``ValidationError``) are ``ValueError`` subclasses, so narrowing the raised
    type would silently change which handler catches a two-strike failure.
    """
    assert issubclass(StructuredOutputError, ValueError)
    model = ScriptedLLM([MALFORMED, MALFORMED])
    with pytest.raises(ValueError):
        run(model, PROMPT, Answer)


def test_the_error_message_quotes_the_final_response():
    """The message is the only artefact that reaches a log, so it must be usable."""
    model = ScriptedLLM([MALFORMED, "nope"])
    with pytest.raises(StructuredOutputError) as err:
        model.generate_structured(PROMPT, Answer)
    message = str(err.value)
    assert "Answer" in message
    assert "nope" in message
    assert "reask" in message


# --------------------------------------------------------------------------- #
# sync/async parity, asserted directly
# --------------------------------------------------------------------------- #


class Outcome(NamedTuple):
    """Everything one entry point observably did, for a field-by-field compare."""

    result: Any
    failure: Any
    calls: int
    prompts: List[str]


def _outcome(run, script) -> Outcome:
    """Run ``script`` through one entry point on its own fresh ``ScriptedLLM``.

    Captures the result *or* the failure, plus the number of calls and the exact
    prompt text of every one of them — the prompts are the part that catches a
    copy-paste divergence in the two near-identical method bodies.
    """
    model = ScriptedLLM(list(script))
    try:
        return Outcome(
            run(model, PROMPT, Answer), None, model.calls, list(model.prompts)
        )
    except StructuredOutputError as exc:  # compared structurally below
        return Outcome(None, exc, model.calls, list(model.prompts))


@pytest.mark.parametrize(
    "script",
    [
        [VALID],
        [MALFORMED, VALID],
        [SCHEMA_INVALID, VALID],
        [MALFORMED, "still not JSON"],
        [SCHEMA_INVALID, SCHEMA_INVALID],
    ],
    ids=[
        "happy",
        "reask_decode",
        "reask_validation",
        "raise_decode",
        "raise_validation",
    ],
)
def test_sync_and_async_are_indistinguishable(script):
    """Same script in, same everything out — including the prompts sent.

    The two bodies in ``base_model.py`` are necessarily near-duplicates, which is
    precisely the shape that rots: a fix applied to one and not the other makes
    ``evaluate()`` (async) behave differently from a notebook (sync) on the same
    data. This is the test that fails when that happens.
    """
    sync = _outcome(_sync, script)
    other = _outcome(_async, script)

    assert sync.calls == other.calls
    assert sync.prompts == other.prompts
    assert (sync.failure is None) == (other.failure is None)
    if sync.failure is None:
        assert sync.result == other.result
    else:
        assert sync.failure.schema_name == other.failure.schema_name
        assert sync.failure.response == other.failure.response
        # Exception instances never compare equal, so compare the chained cause
        # by type — a decode failure on one path and a validation failure on the
        # other would mean the two bodies took different branches.
        assert type(sync.failure.__cause__) is type(other.failure.__cause__)


# --------------------------------------------------------------------------- #
# the prompt itself
# --------------------------------------------------------------------------- #


def test_structured_prompt_contains_the_user_prompt_and_the_schema():
    """The rendered JSON Schema is what the model is actually being shown."""
    prompt = ScriptedLLM([VALID])._structured_prompt(PROMPT, Answer)

    assert prompt.startswith(PROMPT)
    schema = Answer.model_json_schema()
    for field in schema["properties"]:
        assert field in prompt, f"schema field {field!r} missing from the prompt"
    assert schema["title"] in prompt
    # The schema is embedded as JSON, so it must itself round-trip.
    assert json.loads(json.dumps(schema))


def test_structured_prompt_forbids_fences_and_prose():
    """``extract_json_object`` tolerates both; the prompt still asks for neither.

    Tolerance is the safety net, not the request. Dropping the directive would
    push every response onto the fallback slicing path, which cannot recover a
    top-level array and mis-slices any response containing a stray brace.
    """
    prompt = ScriptedLLM([VALID])._structured_prompt(PROMPT, Answer)
    lowered = prompt.lower()
    assert "code fences" in lowered
    assert "explanation" in lowered and "preamble" in lowered


def test_the_word_json_appears_verbatim_in_the_instruction():
    """Azure's ``json_object`` response format *requires* it in the prompt.

    ``AzureOpenAIModel.generate_structured`` sets
    ``response_format={"type": "json_object"}`` and relies on this shared
    instruction to satisfy the API's precondition — the request 400s otherwise.
    So the literal token "JSON" in ``STRUCTURED_OUTPUT_INSTRUCTION`` is
    load-bearing across a provider boundary: reflow the constant freely, but do
    not paraphrase "JSON" away.
    """
    assert "JSON" in STRUCTURED_OUTPUT_INSTRUCTION
    assert "JSON" in ScriptedLLM([VALID])._structured_prompt(PROMPT, Answer)
    # The reask replaces the tail of the prompt on the second call, so it has to
    # carry the token too, or the retry is the request that 400s.
    assert "JSON" in STRUCTURED_OUTPUT_REASK


def test_reask_prompt_renders_the_error_with_its_type():
    model = ScriptedLLM([VALID])
    structured = model._structured_prompt(PROMPT, Answer)
    error = ValueError("boom")
    reask = model._reask_prompt(structured, "garbage", error)

    assert reask.startswith(structured)
    assert "garbage" in reask
    assert "ValueError: boom" in reask


def test_parse_structured_is_a_static_pure_function():
    """No provider state involved, so any provider's parse behaves identically."""
    assert isinstance(BaseLLM.__dict__["_parse_structured"], staticmethod)
    _assert_answer(BaseLLM._parse_structured(f"```json\n{VALID}\n```", Answer))
    with pytest.raises(ValidationError):
        BaseLLM._parse_structured(SCHEMA_INVALID, Answer)


# --------------------------------------------------------------------------- #
# kwargs forwarding
#
# ``ScriptedLLM`` ignores **kwargs, so it cannot show this. A local recorder
# rather than a change to conftest: kwarg capture is only interesting here.
# --------------------------------------------------------------------------- #


class KwargRecordingLLM(BaseLLM):
    """Three-method provider that also records the kwargs of each call."""

    def __init__(self, responses: List[str]) -> None:
        self._queue = list(responses)
        self.prompts: List[str] = []
        self.kwargs: List[dict] = []

    def get_model_name(self) -> str:
        return "kwarg-recorder"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self._next(prompt, kwargs)

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        return self._next(prompt, kwargs)

    def _next(self, prompt: str, kwargs: dict) -> str:
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        return self._queue.pop(0)


@both_paths
def test_kwargs_reach_the_provider_on_the_first_call(run):
    """This forwarding is how the Azure override injects ``response_format``.

    ``AzureOpenAIModel`` implements native JSON mode purely by seeding a kwarg
    and delegating up; if the base stopped passing ``**kwargs`` through, that
    override would silently become a no-op and Azure would fall back to
    prompt-only structured output.
    """
    model = KwargRecordingLLM([VALID])
    run(model, PROMPT, Answer, response_format={"type": "json_object"}, temperature=0)

    assert model.kwargs == [
        {"response_format": {"type": "json_object"}, "temperature": 0}
    ]


@both_paths
def test_kwargs_reach_the_provider_on_the_reask_too(run):
    """A reask that dropped ``response_format`` would lose Azure's JSON mode
    exactly when the model has already demonstrated it needs the help."""
    model = KwargRecordingLLM([MALFORMED, VALID])
    _assert_answer(run(model, PROMPT, Answer, response_format={"type": "json_object"}))

    assert len(model.kwargs) == 2
    assert (
        model.kwargs[0]
        == model.kwargs[1]
        == {"response_format": {"type": "json_object"}}
    )


# --------------------------------------------------------------------------- #
# async embeddings — the other capability-with-a-default
# --------------------------------------------------------------------------- #


class ThreadRecordingEmbedding(BaseEmbeddingModel):
    """Embedding provider with only the three required methods.

    ``FakeEmbedding`` in ``conftest.py`` defines its own async wrappers (it
    predates the ABC defaults), so it cannot demonstrate that a bare provider
    inherits working async methods. This one can, and it records the thread each
    sync call ran on so the off-the-event-loop guarantee is testable.
    """

    def __init__(self) -> None:
        self.threads: List[int] = []

    def get_model_name(self) -> str:
        return "thread-recorder"

    def embed_text(self, text: str) -> List[float]:
        self.threads.append(threading.get_ident())
        return [float(len(text))]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self.threads.append(threading.get_ident())
        return [[float(len(t))] for t in texts]


def test_a_three_method_embedding_provider_gets_async_for_free():
    """Same guarantee as ``generate_structured``, on the embedding ABC.

    The async methods are concrete defaults, so ``BaseEmbeddingModel`` stayed at
    get_model_name / embed_text / embed_texts.
    """
    stub = ThreadRecordingEmbedding()
    assert "a_embed_texts" not in vars(ThreadRecordingEmbedding)

    texts = ["alpha", "bravo charlie"]
    assert asyncio.run(stub.a_embed_texts(texts)) == stub.embed_texts(texts)
    assert asyncio.run(stub.a_embed_text("alpha")) == stub.embed_text("alpha")


def test_the_async_default_runs_off_the_event_loop_thread():
    """``asyncio.to_thread``, not a coroutine that merely *looks* async.

    Embedding is a blocking HTTP call. Wrapping it in an ``async def`` that
    calls it directly would satisfy every type checker and still stall the event
    loop for the whole corpus — which is the exact workload the generation
    pipeline runs concurrently with everything else, and the entire reason these
    two methods exist. Comparing thread identities is the only assertion that
    actually distinguishes the two implementations.
    """
    stub = ThreadRecordingEmbedding()

    async def drive():
        loop_thread = threading.get_ident()
        await stub.a_embed_texts(["alpha", "bravo"])
        await stub.a_embed_text("charlie")
        return loop_thread

    loop_thread = asyncio.run(drive())

    assert len(stub.threads) == 2
    assert loop_thread not in stub.threads, "the blocking call ran on the event loop"


def test_conftest_fake_embedding_keeps_its_own_overrides():
    """A provider may still override; the default is a floor, not a ceiling.

    ``FakeEmbedding`` runs the sync call inline, which is why it is *not* the
    vehicle for the thread assertion above — pinning that here keeps the two
    tests from being read as contradictory.
    """
    assert "a_embed_texts" in vars(FakeEmbedding)
    fake = FakeEmbedding(dim=4)
    assert asyncio.run(fake.a_embed_text("alpha")) == fake.embed_text("alpha")
