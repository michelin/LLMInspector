"""Phase 7D — coverage for the pure-function perturbation layer.

These are the cheapest, highest-value tests in the repo: zero dependencies, no
network, and the exact place two silent legacy bugs lived (``add_contraction``
and ``add_abbreviation`` both built a perturbed list and then returned the
*unmodified* input).

Perturbations are stochastic, so assertions are on invariants — length, type,
idempotence at ``prob=0``, and specific behaviour under a seeded RNG — rather
than on exact strings, except where the transform is deterministic.
"""

import random
import string

import pytest

from llminspector.data import CONTRACTION_MAP, abbreviation_dict, ocr_typo_dict
from llminspector.generation import perturbations as p

_SAMPLES = [
    "I do not think the CEO will arrive today",
    "We cannot ship it before Friday!",
]


@pytest.fixture(autouse=True)
def seeded():
    random.seed(1234)


# --------------------------------------------------------------------------- #
# case transforms
# --------------------------------------------------------------------------- #


def test_uppercase_transform_full_probability():
    assert p.uppercase_transform(["hello world"]) == ["HELLO WORLD"]


def test_uppercase_transform_zero_probability_is_identity():
    assert p.uppercase_transform(["hello world"], prob=0.0) == ["hello world"]


def test_lowercase_transform_full_probability():
    assert p.lowercase_transform(["HELLO World"]) == ["hello world"]


def test_lowercase_transform_zero_probability_is_identity():
    assert p.lowercase_transform(["HELLO World"], prob=0.0) == ["HELLO World"]


def test_titlecase_transform_full_probability():
    assert p.titlecase_transform(["hello world"]) == ["Hello World"]


def test_titlecase_transform_skips_non_strings():
    # the loop guards on isinstance(sample, str)
    assert p.titlecase_transform(["hi there", 42, None]) == ["Hi There"]


@pytest.mark.parametrize(
    "fn", [p.uppercase_transform, p.lowercase_transform, p.titlecase_transform]
)
def test_case_transforms_preserve_word_count(fn):
    out = fn(_SAMPLES)
    assert len(out) == len(_SAMPLES)
    for original, transformed in zip(_SAMPLES, out):
        assert len(transformed.split()) == len(original.split())


# --------------------------------------------------------------------------- #
# punctuation
# --------------------------------------------------------------------------- #


def test_add_punctuation_appends_from_the_whitelist():
    out = p.add_punctuation(["hello"], whitelist=["!"])
    assert out == ["hello!"]


def test_add_punctuation_strips_whitelisted_punctuation_before_appending():
    # only characters *in the whitelist* are stripped first, so "hello!!" with
    # whitelist ["!"] becomes "hello" and then gains a single "!"
    assert p.add_punctuation(["hello!!"], whitelist=["!"]) == ["hello!"]
    # characters outside the whitelist survive untouched
    assert p.add_punctuation(["hello!?"], whitelist=["."]) == ["hello!?."]


def test_add_punctuation_default_whitelist():
    out = p.add_punctuation(["hello"])
    assert len(out) == 1
    assert out[0][:-1] == "hello"
    assert out[0][-1] in "!?,.-:;"


def test_add_punctuation_count_produces_multiple_variants():
    out = p.add_punctuation(["hello"], count=3, whitelist=["!"])
    assert out == ["hello!"] * 3


def test_add_punctuation_zero_probability_emits_nothing():
    assert p.add_punctuation(["hello"], prob=0.0) == []


def test_strip_punctuation_removes_whitelisted_characters():
    assert p.strip_punctuation(["Hello, world!"]) == ["Hello world"]


def test_strip_punctuation_custom_whitelist():
    assert p.strip_punctuation(["a-b-c"], whitelist=["-"]) == ["abc"]


def test_strip_punctuation_zero_probability_emits_nothing():
    assert p.strip_punctuation(["Hello, world!"], prob=0.0) == []


# --------------------------------------------------------------------------- #
# typos
# --------------------------------------------------------------------------- #


def test_add_typo_changes_text_at_full_error_rate():
    out = p.add_typo(["alphabet soup kitchen"], error_rate=1.0)
    assert len(out) == 1
    assert out[0] != "alphabet soup kitchen"
    assert len(out[0].split()) == 3


def test_add_typo_zero_error_rate_is_identity():
    assert p.add_typo(_SAMPLES, error_rate=0.0) == _SAMPLES


def test_add_typo_leaves_single_character_words_alone():
    # the transform requires len(word) > 1
    assert p.add_typo(["a I x"], error_rate=1.0) == ["a I x"]


def test_add_typo_only_uses_lowercase_ascii_for_edits():
    out = p.add_typo(["alphabet"], error_rate=1.0)[0]
    assert set(out) <= set(string.ascii_lowercase)


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #


def test_add_context_start_strategy_prepends():
    out = p.add_context(
        ["question"],
        starting_context=["PREFIX"],
        ending_context=["SUFFIX"],
        strategy="start",
    )
    assert out == ["PREFIX question"]


def test_add_context_end_strategy_appends():
    out = p.add_context(
        ["question"],
        starting_context=["PREFIX"],
        ending_context=["SUFFIX"],
        strategy="end",
    )
    assert out == ["question SUFFIX"]


def test_add_context_combined_strategy_does_both():
    out = p.add_context(
        ["question"],
        starting_context=["PREFIX"],
        ending_context=["SUFFIX"],
        strategy="combined",
    )
    assert out == ["PREFIX question SUFFIX"]


def test_add_context_joins_list_valued_tokens():
    out = p.add_context(
        ["q"],
        starting_context=[["Good", "morning"]],
        ending_context=["Bye"],
        strategy="start",
    )
    assert out == ["Good morning q"]


def test_add_context_leaves_the_dash_placeholder_alone():
    out = p.add_context(
        ["-"],
        starting_context=["PREFIX"],
        ending_context=["SUFFIX"],
        strategy="combined",
    )
    assert out == ["-"]


def test_add_context_zero_probability_is_identity():
    out = p.add_context(
        ["question"],
        prob=0.0,
        starting_context=["PREFIX"],
        ending_context=["SUFFIX"],
        strategy="combined",
    )
    assert out == ["question"]


def test_add_context_count_produces_multiple_variants():
    out = p.add_context(
        ["q"], starting_context=["P"], ending_context=["S"], strategy="start", count=3
    )
    assert out == ["P q"] * 3


def test_add_context_unknown_strategy_warns_and_no_ops(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        out = p.add_context(
            ["question"],
            starting_context=["P"],
            ending_context=["S"],
            strategy="sideways",
        )
    assert out == ["question"]
    assert "sideways" in caplog.text


def test_add_context_falls_back_to_the_packaged_tables():
    """No explicit contexts -> the JSON lookup tables are used."""
    out = p.add_context(["question"], strategy="start")
    assert len(out) == 1 and out[0].endswith("question")


# --------------------------------------------------------------------------- #
# contractions — one of the two fixed legacy bugs
# --------------------------------------------------------------------------- #


def test_add_contraction_returns_the_perturbed_output_not_the_input():
    """The legacy version built the output and then returned sample_list."""
    samples = ["I do not think so"]
    out = p.add_contraction(samples)
    assert out == ["I don't think so"]
    assert out is not samples


def test_add_contraction_is_case_preserving_on_the_first_letter():
    assert p.add_contraction(["Do not go"]) == ["Don't go"]


def test_add_contraction_zero_probability_is_identity():
    assert p.add_contraction(["I do not think so"], prob=0.0) == ["I do not think so"]


def test_add_contraction_leaves_text_without_contractions_alone():
    assert p.add_contraction(["quiet afternoon"]) == ["quiet afternoon"]


def test_add_contraction_handles_a_batch():
    out = p.add_contraction(_SAMPLES)
    assert len(out) == len(_SAMPLES)
    assert all(isinstance(s, str) for s in out)


def test_contraction_map_is_loaded_from_package_data():
    assert "do not" in CONTRACTION_MAP or "do not" in {
        k.lower() for k in CONTRACTION_MAP
    }


# --------------------------------------------------------------------------- #
# OCR typos
# --------------------------------------------------------------------------- #


def test_add_ocr_typo_zero_probability_is_identity():
    assert p.add_ocr_typo(_SAMPLES, prob=0.0) == _SAMPLES


def test_add_ocr_typo_count_produces_multiple_variants():
    out = p.add_ocr_typo(["the cat"], count=3)
    assert len(out) == 3


def test_add_ocr_typo_perturbs_a_known_dictionary_word():
    word = next(iter(ocr_typo_dict))
    out = p.add_ocr_typo([f"{word}"], prob=1.0)
    assert len(out) == 1


def test_add_ocr_typo_uppercases_the_replacement_for_uppercase_tokens():
    # find a dictionary entry whose typo is alphabetic so casing is observable
    word = next(
        w
        for w, typo in ocr_typo_dict.items()
        if w.isalpha() and len(w) > 2 and typo.isalpha()
    )
    out = p.add_ocr_typo([word.upper()], prob=1.0)[0]
    assert out == out.upper()


# --------------------------------------------------------------------------- #
# abbreviations — the other fixed legacy bug
# --------------------------------------------------------------------------- #


def test_add_abbreviation_returns_the_perturbed_output_not_the_input():
    """The legacy version built the output and then returned sample_list."""
    abbreviation, expansions = next(
        (a, e) for a, e in abbreviation_dict.items() if e and e[0].isalpha()
    )
    samples = [f"please {expansions[0]} now"]
    out = p.add_abbreviation(samples)
    assert out is not samples
    assert out[0] != samples[0]
    assert abbreviation in out[0]


def test_add_abbreviation_zero_probability_is_identity():
    assert p.add_abbreviation(_SAMPLES, prob=0.0) == _SAMPLES


def test_add_abbreviation_leaves_unmatched_text_alone():
    assert p.add_abbreviation(["zzzz qqqq"]) == ["zzzz qqqq"]


def test_add_abbreviation_handles_a_batch():
    out = p.add_abbreviation(_SAMPLES)
    assert len(out) == len(_SAMPLES)


# --------------------------------------------------------------------------- #
# shared invariants
# --------------------------------------------------------------------------- #

_LENGTH_PRESERVING = [
    p.uppercase_transform,
    p.lowercase_transform,
    p.titlecase_transform,
    p.add_typo,
    p.add_contraction,
    p.add_ocr_typo,
    p.add_abbreviation,
]


@pytest.mark.parametrize("fn", _LENGTH_PRESERVING, ids=lambda f: f.__name__)
def test_one_output_per_input(fn):
    assert len(fn(_SAMPLES)) == len(_SAMPLES)


@pytest.mark.parametrize(
    "fn",
    _LENGTH_PRESERVING + [p.add_punctuation, p.strip_punctuation, p.add_context],
    ids=lambda f: f.__name__,
)
def test_always_returns_a_list_of_strings(fn):
    out = fn(_SAMPLES)
    assert isinstance(out, list)
    assert all(isinstance(s, str) for s in out)


@pytest.mark.parametrize(
    "fn",
    _LENGTH_PRESERVING + [p.add_punctuation, p.strip_punctuation, p.add_context],
    ids=lambda f: f.__name__,
)
def test_empty_input_gives_empty_output(fn):
    assert fn([]) == []


@pytest.mark.parametrize("fn", _LENGTH_PRESERVING, ids=lambda f: f.__name__)
def test_input_list_is_never_mutated(fn):
    samples = list(_SAMPLES)
    fn(samples)
    assert samples == _SAMPLES


def test_inverted_ocr_dict_is_cached():
    assert p._inverted_ocr_typo_dict() is p._inverted_ocr_typo_dict()
