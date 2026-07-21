"""Phase 6 — reporting exporters."""

from llminspector import EvaluationDataset, LLMTestCase, evaluate, reporting
from llminspector.metrics.base_metric import BaseMetric


class _Const(BaseMetric):
    def __init__(self, name, required, value):
        super().__init__()
        self.metric_name = name
        self.required_inputs = set(required)
        self._value = value

    async def a_measure(self, tc):
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


def test_to_dataframe():
    df = reporting.to_dataframe(_result())
    assert "bert_score" in df.columns
    assert df["bert_score"].tolist() == [0.8, 0.8]


def test_to_excel(tmp_path):
    path = tmp_path / "out.xlsx"
    reporting.to_excel(_result(), str(path))
    assert path.exists()


def test_summary_only_numeric():
    stats = reporting.summary(_result())
    assert "bert_score" in stats
    assert stats["bert_score"]["mean"] == 0.8
    # non-numeric column skipped
    assert "answer_sentiment" not in stats
