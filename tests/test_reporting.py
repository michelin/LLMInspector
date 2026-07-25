"""Phase 6 — reporting (revised in Phase 8.6).

``reporting.to_dataframe`` / ``to_excel`` were removed: they forwarded to
identically-named ``EvaluationResult`` methods and added nothing. Serialization
is the result object's job; ``reporting`` holds the analysis on top.
"""

import pytest

from llminspector import evaluate, reporting
from llminspector.dataset import EvaluationDataset
from llminspector.evaluate import ResultColumns
from llminspector.metrics.base_metric import BaseMetric
from llminspector.test_case import LLMTestCase


class _Const(BaseMetric):
    def __init__(self, name, required, value, exc=None):
        super().__init__()
        self.metric_name = name
        self.required_inputs = set(required)
        self._value = value
        self._exc = exc

    async def a_measure(self, tc):
        if self._exc is not None:
            self.record_failure(self._exc)
            self.score = None
        else:
            self.score = self._value
        return self.score


def _result():
    ds = EvaluationDataset(
        test_cases=[
            LLMTestCase(input="q1", actual_output="a1"),
            LLMTestCase(input="q2", actual_output="a2"),
        ]
    )
    metrics = [
        _Const("bert_score", {"actual_output"}, 0.8),
        _Const("answer_sentiment", {"actual_output"}, "Positive"),
    ]
    return evaluate(ds, metrics, show_progress=False)


def test_result_serializes_itself():
    df = _result().to_pandas()
    assert "bert_score" in df.columns
    assert df["bert_score"].tolist() == [0.8, 0.8]


def test_result_to_excel(tmp_path):
    path = tmp_path / "out.xlsx"
    _result().to_excel(str(path))
    assert path.exists()


def test_forwarders_are_gone():
    """One door per export — the result object's."""
    assert not hasattr(reporting, "to_dataframe")
    assert not hasattr(reporting, "to_excel")
    assert set(reporting.__all__) == {"summary", "errors"}


def test_summary_only_numeric():
    stats = reporting.summary(_result())
    assert "bert_score" in stats
    assert stats["bert_score"]["mean"] == 0.8
    # non-numeric column skipped
    assert "answer_sentiment" not in stats


def test_errors_frame_is_empty_for_a_clean_run():
    df = reporting.errors(_result())
    assert list(df.columns) == ["row", "metric", "error"]
    assert df.empty


def test_errors_frame_lists_failures():
    ds = EvaluationDataset(
        test_cases=[
            LLMTestCase(input="q1", actual_output="a1"),
            LLMTestCase(input="q2", actual_output="a2"),
        ]
    )
    m = _Const("bert_score", {"actual_output"}, None, exc=RuntimeError("boom"))
    df = reporting.errors(evaluate(ds, [m], show_progress=False))
    assert df["row"].tolist() == [0, 1]
    assert df["metric"].tolist() == ["bert_score", "bert_score"]
    assert all("boom" in e for e in df["error"])


# -- write schema is separate from the read schema (8.6) --------------------


def test_output_column_names_default_to_the_familiar_ones():
    df = _result().to_pandas()
    assert list(df.columns)[:5] == [
        "question",
        "answer",
        "ground_truth",
        "contexts",
        "policy",
    ]


def test_output_column_names_are_configurable_independently():
    df = _result().to_pandas(
        columns=ResultColumns(input_col="prompt", actual_output_col="response")
    )
    assert list(df.columns)[:2] == ["prompt", "response"]


def test_result_columns_can_mirror_a_dataset_read_mapping():
    from llminspector.dataset.dataset import ColumnMapping

    columns = ResultColumns.from_column_mapping(
        ColumnMapping(input_col="prompt", policy_col="rules")
    )
    df = _result().to_pandas(columns=columns)
    assert "prompt" in df.columns and "rules" in df.columns


def test_renaming_an_input_column_no_longer_renames_an_output_column():
    """The two schemas used to be the same type, so this leaked."""
    from llminspector.dataset.dataset import ColumnMapping

    ds = EvaluationDataset.from_pandas(
        __import__("pandas").DataFrame({"prompt": ["q"], "answer": ["a"]}),
        input_col="prompt",
    )
    result = evaluate(
        ds, [_Const("bert_score", {"actual_output"}, 0.8)], show_progress=False
    )
    # read as "prompt", still exported as "question" unless asked otherwise
    assert "question" in result.to_pandas().columns
    assert ColumnMapping is not ResultColumns
