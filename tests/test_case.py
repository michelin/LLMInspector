"""Phase 1 exit-criteria tests for ``LLMTestCase`` construction + validation."""

import pytest
from pydantic import ValidationError

from llminspector.dataset import EvaluationDataset
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


# -- golden_id / metadata (Phase 1) -------------------------------------------


def test_golden_id_and_metadata_default_empty():
    """Both are absent for a case read straight from a spreadsheet."""
    tc = LLMTestCase(input="q")
    assert tc.golden_id is None
    assert tc.metadata == {}


def test_golden_id_and_metadata_are_settable():
    tc = LLMTestCase(input="q", golden_id="seed-1", metadata={"lineage": "evolved"})
    assert tc.golden_id == "seed-1"
    assert tc.metadata == {"lineage": "evolved"}


def test_metadata_default_is_per_instance():
    """Each case gets its own dict rather than one shared class-level default.

    A shared mutable default would let lineage written onto one scored row leak
    into every other row. ``default_factory`` prevents it; this test is what
    says we depend on that.
    """
    a = LLMTestCase(input="q")
    b = LLMTestCase(input="q")
    a.metadata["source"] = "a"
    assert b.metadata == {}
    assert a.metadata is not b.metadata


def test_model_dump_key_set():
    """Pin the serialized shape so a new field is a decision, not an accident."""
    dumped = LLMTestCase(input="q").model_dump()
    assert set(dumped) == {
        "input",
        "actual_output",
        "expected_output",
        "retrieval_context",
        "policy",
        "golden_id",
        "metadata",
    }


def test_new_fields_do_not_widen_the_exported_table():
    """Adding ``golden_id`` / ``metadata`` must not widen any exported table.

    ``EvaluationDataset.to_pandas`` and ``EvaluationResult.to_pandas`` both name
    their columns explicitly rather than dumping the model, which is *why* two
    new fields on ``LLMTestCase`` leave the workbook shape untouched. That is a
    property of the export code, not of this schema — so this test is what would
    catch someone "simplifying" either exporter into ``model_dump()`` and
    silently pushing lineage columns into every user's spreadsheet.
    """
    dataset = EvaluationDataset(
        test_cases=[
            LLMTestCase(
                input="q",
                actual_output="a",
                golden_id="seed-1",
                metadata={"lineage": "evolved"},
            )
        ]
    )
    df = dataset.to_pandas()
    assert list(df.columns) == [
        "question",
        "answer",
        "ground_truth",
        "contexts",
        "policy",
    ]
