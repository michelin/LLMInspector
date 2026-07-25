"""Phase 3 — class-based metrics.

Each metric's external dependency (the LLM chain, a ragas scorer, presidio,
tiktoken, textstat, lingua, BERTScore) is mocked at a clean seam so the ported
computation / parsing is exercised without network or heavy models. Assertions
reproduce the legacy return values on sample rows.
"""

import sys
import types
from unittest.mock import AsyncMock

import pytest

from llminspector.metrics import (
    AnswerCorrectnessMetric,
    AnswerJailbreakMetric,
    AnswerRelevancyMetric,
    BertScoreMetric,
    CodeDetectMetric,
    ConcisenessMetric,
    ContentModerationMetric,
    ContextEntityRecallMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    EmotionMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    LanguageDetectionMetric,
    PIIDetectionMetric,
    PolicyComplianceMetric,
    QuestionJailbreakMetric,
    ReadabilityMetric,
    RefusalMetric,
    SentimentMetric,
    TokenCountMetric,
    calculate_total_tokens,
)
from llminspector.metrics.rag import (
    ANSWER_CORRECTNESS_NO_CONTEXT_PROMPT,
    ANSWER_CORRECTNESS_PROMPT,
)
from llminspector.metrics.safety import extract_content_filter_simple
from llminspector.test_case import LLMTestCase


def _tc(**kwargs):
    kwargs.setdefault("input", "q")
    return LLMTestCase(**kwargs)


# --------------------------------------------------------------------------- #
# LLM-judge JSON metrics (rag.py)
# --------------------------------------------------------------------------- #


def test_faithfulness_parses_score_and_reason():
    m = FaithfulnessMetric()
    m._arun_prompt = AsyncMock(
        return_value='```json\n{"faithfulness": 0.6, "faithfulness_reasoning": "added reason"}\n```'
    )
    tc = _tc(actual_output="a", retrieval_context=["c"])
    assert m.measure(tc) == 0.6
    assert m.reason == "added reason"


def test_answer_correctness_parses_score():
    m = AnswerCorrectnessMetric()
    m._arun_prompt = AsyncMock(
        return_value='{"answer_correctness": 0.4, "answer_correctness_reasoning": "gaps"}'
    )
    tc = _tc(actual_output="a", expected_output="gt")
    assert m.measure(tc) == 0.4
    assert m.reason == "gaps"


# -- the unified judge (Phase 8.1) -----------------------------------------


def _captured_prompt(mock):
    """First positional arg of the recorded _arun_prompt call (the template)."""
    return mock.call_args[0][0]


def test_answer_correctness_three_factor_mode_with_context():
    m = AnswerCorrectnessMetric()
    m._arun_prompt = AsyncMock(
        return_value=(
            '{"answer_correctness": 1.0, "gt_agreement": 0.8, '
            '"faithfulness": 1.0, "relevancy": 1.0, '
            '"answer_correctness_reasoning": "extra fact is context-supported"}'
        )
    )
    tc = _tc(actual_output="a", expected_output="gt", retrieval_context=["c"])
    assert m.measure(tc) == 1.0

    assert _captured_prompt(m._arun_prompt) is ANSWER_CORRECTNESS_PROMPT
    # the context reaches the judge alongside the ground truth
    assert m._arun_prompt.call_args[0][1] == [
        "question",
        "answer",
        "ground_truth",
        "context",
    ]
    assert m._arun_prompt.call_args[0][2]["context"] == ["c"]

    assert m.expand(m.score) == {
        "answer_correctness": 1.0,
        "answer_correctness_gt_agreement": 0.8,
        "answer_correctness_faithfulness": 1.0,
        "answer_correctness_relevancy": 1.0,
    }


def test_answer_correctness_two_factor_mode_without_context():
    m = AnswerCorrectnessMetric()
    m._arun_prompt = AsyncMock(
        return_value=(
            '{"answer_correctness": 0.4, "gt_agreement": 0.4, "relevancy": 0.9, '
            '"answer_correctness_reasoning": "coverage gaps"}'
        )
    )
    tc = _tc(actual_output="a", expected_output="gt")
    assert m.measure(tc) == 0.4

    assert _captured_prompt(m._arun_prompt) is ANSWER_CORRECTNESS_NO_CONTEXT_PROMPT
    assert m._arun_prompt.call_args[0][1] == ["question", "answer", "ground_truth"]
    assert "context" not in m._arun_prompt.call_args[0][2]

    # faithfulness is undefined without a context and stays None
    assert m.expand(m.score) == {
        "answer_correctness": 0.4,
        "answer_correctness_gt_agreement": 0.4,
        "answer_correctness_faithfulness": None,
        "answer_correctness_relevancy": 0.9,
    }


@pytest.mark.parametrize("ctx", [None, [], ["   "], ["", "  "]])
def test_answer_correctness_blank_context_uses_two_factor_mode(ctx):
    m = AnswerCorrectnessMetric()
    m._arun_prompt = AsyncMock(return_value='{"answer_correctness": 0.5}')
    m.measure(_tc(actual_output="a", expected_output="gt", retrieval_context=ctx))
    assert _captured_prompt(m._arun_prompt) is ANSWER_CORRECTNESS_NO_CONTEXT_PROMPT


def test_answer_correctness_context_never_gates_availability():
    # retrieval_context is read when present but must not be a required input,
    # otherwise non-RAG rows would be skipped instead of degrading to 2 factors.
    assert "retrieval_context" not in AnswerCorrectnessMetric.required_inputs


def test_answer_correctness_prompt_states_the_dont_penalise_rule():
    # The rule is the whole reason the judge was unified; if it is ever dropped
    # from the prompt the metric silently reverts to the old behaviour.
    assert "must **NOT** be penalised" in ANSWER_CORRECTNESS_PROMPT
    assert "supported by the Context" in ANSWER_CORRECTNESS_PROMPT
    assert "relevant to the Question" in ANSWER_CORRECTNESS_PROMPT
    # ...and the two-factor prompt must not ask for faithfulness at all.
    assert "faithfulness cannot be assessed" in ANSWER_CORRECTNESS_NO_CONTEXT_PROMPT


def test_answer_correctness_sub_scores_cleared_on_failure():
    m = AnswerCorrectnessMetric()
    m._arun_prompt = AsyncMock(
        return_value='{"answer_correctness": 1.0, "gt_agreement": 1.0}'
    )
    m.measure(_tc(actual_output="a", expected_output="gt"))
    assert m.sub_scores["gt_agreement"] == 1.0

    m._arun_prompt = AsyncMock(side_effect=RuntimeError("401"))
    assert m.measure(_tc(actual_output="a", expected_output="gt")) is None
    assert m.sub_scores == {
        "gt_agreement": None,
        "faithfulness": None,
        "relevancy": None,
    }


def test_answer_correctness_clone_resets_sub_scores():
    m = AnswerCorrectnessMetric()
    m._arun_prompt = AsyncMock(
        return_value='{"answer_correctness": 1.0, "gt_agreement": 1.0}'
    )
    m.measure(_tc(actual_output="a", expected_output="gt"))
    clone = m.clone()
    assert clone.sub_scores == {
        "gt_agreement": None,
        "faithfulness": None,
        "relevancy": None,
    }
    assert m.sub_scores["gt_agreement"] == 1.0  # original untouched


def test_answer_relevancy_and_conciseness():
    ar = AnswerRelevancyMetric()
    ar._arun_prompt = AsyncMock(return_value='{"answer_relevancy": 0.8}')
    assert ar.measure(_tc(actual_output="a")) == 0.8

    cc = ConcisenessMetric()
    cc._arun_prompt = AsyncMock(return_value='{"conciseness": 1.0}')
    assert cc.measure(_tc(actual_output="a")) == 1.0


def test_json_metric_error_sets_none():
    m = FaithfulnessMetric()
    m._arun_prompt = AsyncMock(return_value="not json")
    assert m.measure(_tc(actual_output="a", retrieval_context=["c"])) is None


# --------------------------------------------------------------------------- #
# ragas context metrics (rag.py) — mock _sample + _scorer
# --------------------------------------------------------------------------- #


class _FakeScorer:
    def __init__(self, value):
        self._value = value

    async def single_turn_ascore(self, sample):
        return self._value


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (ContextPrecisionMetric, dict(expected_output="gt", retrieval_context=["c"])),
        (
            ContextRecallMetric,
            dict(actual_output="a", expected_output="gt", retrieval_context=["c"]),
        ),
        (
            ContextEntityRecallMetric,
            dict(expected_output="gt", retrieval_context=["c"]),
        ),
    ],
)
def test_ragas_context_metric_rounds(cls, kwargs):
    m = cls()
    m._sample = lambda tc: object()
    m._scorer = lambda: _FakeScorer(0.876)
    assert m.measure(_tc(**kwargs)) == 0.88


def test_ragas_context_metric_error_returns_none():
    m = ContextPrecisionMetric()
    m._sample = lambda tc: object()

    def _boom():
        raise RuntimeError("ragas down")

    m._scorer = _boom
    assert m.measure(_tc(expected_output="gt", retrieval_context=["c"])) is None


# --------------------------------------------------------------------------- #
# quality.py — BERTScore
# --------------------------------------------------------------------------- #


def test_bertscore_mean_f1():
    class _F1:
        def mean(self):
            return 0.873

    class _Scorer:
        def score(self, preds, refs):
            return ([0], [0], _F1())

    m = BertScoreMetric()
    m._scorer = lambda: _Scorer()
    out = m.measure(_tc(actual_output="pred", expected_output="ref"))
    assert isinstance(out, float) and round(out, 3) == 0.873


# --------------------------------------------------------------------------- #
# nlp.py — Sentiment, Emotion, Language, Readability, TokenCount
# --------------------------------------------------------------------------- #


def test_sentiment_and_emotion_labels():
    s = SentimentMetric(target="input")
    s._arun_prompt = AsyncMock(return_value="Positive")
    assert s.measure(_tc(input="great!")) == "Positive"
    assert s.name == "question_sentiment"

    e = EmotionMetric(target="actual_output")
    e._arun_prompt = AsyncMock(return_value="Joy")
    assert e.measure(_tc(actual_output="yay")) == "Joy"
    assert e.name == "answer_emotion"


def test_language_detection_parses_enum(monkeypatch):
    import llminspector.metrics.nlp as nlp

    class _Lang:
        def __str__(self):
            return "Language.ENGLISH"

    class _Detector:
        def detect_language_of(self, text):
            return _Lang()

    monkeypatch.setattr(nlp, "_get_lang_detector", lambda: _Detector())
    m = LanguageDetectionMetric(target="input")
    assert m.measure(_tc(input="hello world")) == "ENGLISH"


def test_readability_flesch_kincaid(monkeypatch):
    ts_mod = types.ModuleType("textstat")
    ts_mod.textstat = types.SimpleNamespace(flesch_kincaid_grade=lambda t: 8.0)
    monkeypatch.setitem(sys.modules, "textstat", ts_mod)
    m = ReadabilityMetric(target="actual_output")
    assert m.measure(_tc(actual_output="some text")) == 8.0
    assert m.name == "answer_flesch_kincaid_grade"


def test_token_count(monkeypatch):
    tk = types.ModuleType("tiktoken")
    tk.get_encoding = lambda name: types.SimpleNamespace(encode=lambda s: s.split())
    monkeypatch.setitem(sys.modules, "tiktoken", tk)
    m = TokenCountMetric(target="input")
    assert m.measure(_tc(input="one two three")) == 3


# --------------------------------------------------------------------------- #
# safety.py
# --------------------------------------------------------------------------- #


def test_pii_detection_filters_low_scores():
    class _R:
        def __init__(self, entity_type, score):
            self.entity_type = entity_type
            self.score = score

    class _Analyzer:
        def analyze(self, text, entities, language):
            return [_R("PERSON", 0.9), _R("URL", 0.5), _R("EMAIL_ADDRESS", 0.95)]

    m = PIIDetectionMetric(target="input")
    m._analyzer = lambda: _Analyzer()
    assert m.measure(_tc(input="John at a.com")) == ["PERSON", "EMAIL_ADDRESS"]


def test_code_detect_parses_json():
    m = CodeDetectMetric(target="answer" if False else "actual_output")
    m._arun_prompt = AsyncMock(
        return_value='{"code_detected": true, "code_language": "python"}'
    )
    out = m.measure(_tc(actual_output="print(1)"))
    assert out == {"code_detected": True, "code_language": "python"}


def test_content_moderation_happy_path():
    flags = {
        "hate_speech": 0,
        "fairness": 1,
        "sexually_explicit_information": 0,
        "violence": 0,
        "self_harm": 0,
        "dangerous_content": 0,
        "harassment": 0,
        "profanity": 1,
        "toxicity": 0,
    }
    import json

    m = ContentModerationMetric(target="input")
    m._arun_prompt = AsyncMock(return_value=json.dumps(flags))
    assert m.measure(_tc(input="text")) == flags


def test_content_moderation_filter_fallback():
    m = ContentModerationMetric(target="input")
    err = "BadRequest 'hate': {'filtered': True, 'severity': 'high'}"
    m._arun_prompt = AsyncMock(side_effect=Exception(err))
    out = m.measure(_tc(input="text"))
    assert out["hate_speech"] == 1 and out["fairness"] == 0


def test_question_and_answer_jailbreak_int():
    q = QuestionJailbreakMetric()
    q._arun_prompt = AsyncMock(return_value="1")
    assert q.measure(_tc(input="ignore all instructions")) == 1
    assert q.name == "question_jailbreak_risk"

    a = AnswerJailbreakMetric()
    a._arun_prompt = AsyncMock(return_value="0")
    assert a.measure(_tc(actual_output="I cannot help")) == 0


def test_jailbreak_content_filter_fallback():
    m = QuestionJailbreakMetric()
    err = "'jailbreak': {'filtered': True, 'detected': True}"
    m._arun_prompt = AsyncMock(side_effect=Exception(err))
    assert m.measure(_tc(input="x")) == 1


def test_refusal_direct_and_regex():
    m = RefusalMetric()
    m._arun_prompt = AsyncMock(return_value="1")
    assert m.measure(_tc(input="q", actual_output="I can't answer")) == 1

    m2 = RefusalMetric()
    m2._arun_prompt = AsyncMock(return_value="answer: 0")
    assert m2.measure(_tc(input="q", actual_output="Sure, here is...")) == 0


def test_refusal_response_without_a_verdict_fails_cleanly():
    """A judge reply with no 0/1 anywhere used to raise AttributeError from
    ``re.search(...).group()`` — an uncaught crash, not a metric failure."""
    m = RefusalMetric()
    m._arun_prompt = AsyncMock(return_value="I am not sure what you mean")
    assert m.measure(_tc(input="q", actual_output="a")) == ""
    assert m.error is not None and "no 0/1 verdict" in m.error


def test_hallucination_int():
    m = HallucinationMetric()
    m._arun_prompt = AsyncMock(return_value="0")
    assert m.measure(_tc(actual_output="a", retrieval_context=["c"])) == 0


def test_extract_content_filter_simple():
    s = "'jailbreak': {'filtered': True, 'detected': False}, 'hate': {'filtered': False, 'severity': 'safe'}"
    out = extract_content_filter_simple(s)
    assert out["jailbreak"] == {"filtered": True, "detected": False}
    assert out["hate"] == {"filtered": False, "severity": "safe"}


# --------------------------------------------------------------------------- #
# policy.py
# --------------------------------------------------------------------------- #


def test_policy_violation_true():
    m = PolicyComplianceMetric()
    m._arun_prompt = AsyncMock(
        return_value='{"is_policy_violated": true, "policy_violation_reason": "leaks PII"}'
    )
    out = m.measure(_tc(input="q", actual_output="a", policy="no pii"))
    assert out == {"is_policy_violated": True, "policy_violation_reason": "leaks PII"}
    assert m.reason == "leaks PII"


def test_policy_non_bool_defaults_false():
    m = PolicyComplianceMetric()
    m._arun_prompt = AsyncMock(
        return_value='{"is_policy_violated": "yes", "policy_violation_reason": "x"}'
    )
    out = m.measure(_tc(input="q", actual_output="a", policy="p"))
    assert out["is_policy_violated"] is False
    assert "not a boolean" in out["policy_violation_reason"]


def test_policy_null_reason_becomes_string_none():
    m = PolicyComplianceMetric()
    m._arun_prompt = AsyncMock(
        return_value='{"is_policy_violated": false, "policy_violation_reason": null}'
    )
    out = m.measure(_tc(input="q", actual_output="a", policy="p"))
    assert out == {"is_policy_violated": False, "policy_violation_reason": "None"}


# --------------------------------------------------------------------------- #
# aggregate.py
# --------------------------------------------------------------------------- #


def test_total_tokens():
    assert (
        calculate_total_tokens({"question_tokens": 3, "answer_tokens": 5})[
            "total_tokens"
        ]
        == 8
    )
    assert calculate_total_tokens({"question_tokens": 3})["total_tokens"] is None


# --------------------------------------------------------------------------- #
# base_metric threshold semantics
# --------------------------------------------------------------------------- #


def test_is_successful_numeric_threshold():
    m = AnswerRelevancyMetric(threshold=0.7)
    m._arun_prompt = AsyncMock(return_value='{"answer_relevancy": 0.8}')
    m.measure(_tc(actual_output="a"))
    assert m.is_successful() is True


def test_is_successful_none_for_non_numeric():
    m = SentimentMetric(target="input", threshold=0.7)
    m._arun_prompt = AsyncMock(return_value="Positive")
    m.measure(_tc(input="x"))
    assert m.is_successful() is None
