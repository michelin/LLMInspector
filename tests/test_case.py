"""Phase 1 exit-criteria tests for ``LLMTestCase`` construction + validation."""

import pytest
from pydantic import ValidationError

from llminspector.test_case import LLMTestCase


def test_minimal_construction():
    tc = LLMTestCase(input="What is the capital of France?")
    assert tc.input == "What is the capital of France?"
    assert tc.actual_output is None
    assert tc.expected_output is None
    assert tc.retrieval_context is None
    assert tc.policy is None


def test_full_construction():
    tc = LLMTestCase(
        input="q",
        actual_output="a",
        expected_output="gt",
        retrieval_context=["c1", "c2"],
        policy="be nice",
    )
    assert tc.actual_output == "a"
    assert tc.expected_output == "gt"
    assert tc.retrieval_context == ["c1", "c2"]
    assert tc.policy == "be nice"


def test_input_required():
    with pytest.raises(ValidationError):
        LLMTestCase()  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_input_rejects_blank(bad):
    with pytest.raises(ValidationError):
        LLMTestCase(input=bad)


def test_retrieval_context_coerces_single_string():
    tc = LLMTestCase(input="q", retrieval_context="just one")
    assert tc.retrieval_context == ["just one"]


def test_retrieval_context_drops_blanks():
    tc = LLMTestCase(input="q", retrieval_context=["a", "", "  ", "b"])
    assert tc.retrieval_context == ["a", "b"]


def test_retrieval_context_all_blank_becomes_none():
    tc = LLMTestCase(input="q", retrieval_context=["", "   "])
    assert tc.retrieval_context is None
