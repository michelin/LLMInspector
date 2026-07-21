"""Phase 4 — the evaluate engine.

Driven with lightweight fake metrics (no langchain/ragas) plus, for the
dependency path, the real FaithfulnessMetric which fails gracefully to ``None``
when given no model. Covers availability filtering, input-order preservation,
the answer_correctness/overall_accuracy overwrite + dependency auto-add/null,
column expansion, and DataFrame export.
"""

import asyncio

import pytest

from llminspector import EvaluationDataset, LLMTestCase, evaluate
from llminspector.evaluate import EvaluationResult
from llminspector.metrics.base_metric import BaseMetric


class FakeMetric(BaseMetric):
    def __init__(
        self, name, required_inputs, value,
        produces_reasoning=False, reason=None, delay=0.0,
    ):
        super().__init__(model=None)
        self.metric_name = name
        self.required_inputs = set(required_inputs)
        self.produces_reasoning = produces_reasoning
        self._value = value
        self._reason = reason
        self._delay = delay

    async def a_measure(self, test_case):
        if self._delay:
            await asyncio.sleep(self._delay)
        self.score = self._value(test_case) if callable(self._value) else self._value
        self.reason = self._reason
        self.is_successful()
        return self.score


def _dataset(*test_cases):
    return EvaluationDataset(test_cases=list(test_cases))


# --------------------------------------------------------------------------- #

def test_requires_metrics():
    with pytest.raises(ValueError):
        evaluate(_dataset(LLMTestCase(input="q")), metrics=[])


def test_basic_scores_and_result_shape():
    ds = _dataset(
        LLMTestCase(input="hello", actual_output="hi"),
        LLMTestCase(input="world", actual_output="yo"),
    )
    m = FakeMetric("answer_sentiment", {"actual_output"}, "Positive")
    result = evaluate(ds, metrics=[m], show_progress=False)
    assert isinstance(result, EvaluationResult)
    assert len(result) == 2
    assert [r["answer_sentiment"] for r in result.rows] == ["Positive", "Positive"]


def test_availability_filtering_skips_unavailable_metric():
    ds = _dataset(
        LLMTestCase(input="q1", actual_output="a1", retrieval_context=["c"]),
        LLMTestCase(input="q2", actual_output="a2"),  # no context
    )
    faith = FakeMetric(
        "faithfulness", {"input", "actual_output", "retrieval_context"}, 0.9
    )
    result = evaluate(ds, metrics=[faith], show_progress=False)
    assert result.rows[0]["faithfulness"] == 0.9
    assert result.rows[1]["faithfulness"] is None  # skipped, stays None


def test_input_order_preserved_despite_completion_order():
    # Earlier rows sleep longer, so they complete last; output must stay aligned.
    tcs = [LLMTestCase(input=f"q{i}", actual_output=f"a{i}") for i in range(3)]
    ds = _dataset(*tcs)
    m = FakeMetric(
        "answer_sentiment", {"actual_output"},
        value=lambda tc: tc.input,
        delay=None,
    )
    # per-row delay: q0 -> 0.03, q1 -> 0.02, q2 -> 0.01
    class _Ordered(FakeMetric):
        async def a_measure(self, test_case):
            idx = int(test_case.input[1:])
            await asyncio.sleep((3 - idx) * 0.02)
            self.score = test_case.input
            self.is_successful()
            return self.score

    om = _Ordered("answer_sentiment", {"actual_output"}, None)
    result = evaluate(ds, metrics=[om], batch_size=3, show_progress=False)
    assert [r["answer_sentiment"] for r in result.rows] == ["q0", "q1", "q2"]


def test_reasoning_column_added():
    ds = _dataset(LLMTestCase(input="q", actual_output="a"))
    m = FakeMetric(
        "conciseness", {"input", "actual_output"}, 0.8,
        produces_reasoning=True, reason="tight",
    )
    result = evaluate(ds, metrics=[m], show_progress=False)
    assert result.rows[0]["conciseness"] == 0.8
    assert result.rows[0]["conciseness_reasoning"] == "tight"


def test_overall_accuracy_overwrites_answer_correctness():
    ds = _dataset(
        LLMTestCase(input="q", actual_output="a", expected_output="gt",
                    retrieval_context=["c"])
    )
    ac = FakeMetric("answer_correctness", {"input", "actual_output", "expected_output"}, 1.0)
    fa = FakeMetric("faithfulness", {"input", "actual_output", "retrieval_context"}, 0.5)
    ar = FakeMetric("answer_relevancy", {"input", "actual_output"}, 0.5)
    result = evaluate(ds, metrics=[ac, fa, ar], show_progress=False)
    # 0.5*1.0 + 0.3*0.5 + 0.2*0.5 = 0.75
    assert result.rows[0]["answer_correctness"] == 0.75
    assert result.rows[0]["faithfulness"] == 0.5  # requested -> kept


def test_dependency_auto_add_and_null():
    ds = _dataset(
        LLMTestCase(input="q", actual_output="a", expected_output="gt")
    )
    ac = FakeMetric("answer_correctness", {"input", "actual_output", "expected_output"}, 1.0)
    ar = FakeMetric("answer_relevancy", {"input", "actual_output"}, 0.0)
    # faithfulness auto-added (real metric, no model -> fails to None), then nulled.
    result = evaluate(ds, metrics=[ac, ar], show_progress=False)
    # 2-metric blend: 0.75*1.0 + 0.25*0.0 = 0.75
    assert result.rows[0]["answer_correctness"] == 0.75
    assert result.rows[0]["faithfulness"] is None  # dependency-only -> nulled


def test_code_detect_expansion():
    ds = _dataset(LLMTestCase(input="q", actual_output="print(1)"))
    m = FakeMetric(
        "answer_code_detected", {"actual_output"},
        {"code_detected": True, "code_language": "python"},
    )
    row = evaluate(ds, metrics=[m], show_progress=False).rows[0]
    assert row["answer_code_detected"] is True
    assert row["answer_code_language"] == "python"


def test_content_moderation_expansion():
    ds = _dataset(LLMTestCase(input="q", actual_output="a"))
    flags = {
        "hate_speech": 0, "fairness": 1, "sexually_explicit_information": 0,
        "violence": 0, "self_harm": 0, "dangerous_content": 0,
        "harassment": 0, "profanity": 1, "toxicity_risk": 0,
    }
    m = FakeMetric("answer_content_moderation", {"actual_output"}, flags)
    row = evaluate(ds, metrics=[m], show_progress=False).rows[0]
    assert row["answer_fairness"] == 1 and row["answer_profanity"] == 1
    assert "answer_content_moderation" not in row  # original key removed


def test_policy_expansion():
    ds = _dataset(LLMTestCase(input="q", actual_output="a", policy="no pii"))
    m = FakeMetric(
        "policy_check", {"input", "actual_output", "policy"},
        {"is_policy_violated": True, "policy_violation_reason": "leak"},
    )
    row = evaluate(ds, metrics=[m], show_progress=False).rows[0]
    assert row["is_policy_violated"] is True
    assert row["policy_violation_reason"] == "leak"
    assert "policy_check" not in row


def test_total_tokens_aggregation():
    ds = _dataset(LLMTestCase(input="q", actual_output="a"))
    qt = FakeMetric("question_tokens", {"input"}, 3)
    at = FakeMetric("answer_tokens", {"actual_output"}, 5)
    row = evaluate(ds, metrics=[qt, at], show_progress=False).rows[0]
    assert row["total_tokens"] == 8


def test_to_pandas_has_source_and_metric_columns():
    ds = _dataset(
        LLMTestCase(input="q1", actual_output="a1"),
        LLMTestCase(input="q2", actual_output="a2"),
    )
    m = FakeMetric("answer_sentiment", {"actual_output"}, "Positive")
    df = evaluate(ds, metrics=[m], show_progress=False).to_pandas()
    assert list(df.columns)[:5] == ["question", "answer", "ground_truth", "contexts", "policy"]
    assert "answer_sentiment" in df.columns
    assert df["question"].tolist() == ["q1", "q2"]
    assert df["answer_sentiment"].tolist() == ["Positive", "Positive"]
