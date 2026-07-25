"""Phase 8.2 — metrics own their output contract.

The evaluate engine used to carry a hand-maintained 50-entry ``_REORDER_KEYS``
list plus three bespoke ``_expand_*`` functions that hardcoded metric names and
``question_`` / ``answer_`` prefixes. Each metric now declares the columns it
owns (``expand`` / ``output_columns``) and where they sit (``sort_key``).

``GOLDEN_HEADER`` below is the exact column order that list produced. It is a
frozen artefact — the engine must reproduce it from the metrics alone.
"""

import ast
import inspect
from importlib import import_module

import pytest

from llminspector import metrics as metrics_pkg

# llminspector.evaluate.__init__ rebinds the name `evaluate` to the function, so
# the submodule has to be fetched from sys.modules rather than by attribute.
evaluate_mod = import_module("llminspector.evaluate.evaluate")
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
    ContextRelevanceMetric,
    ContextUtilisationMetric,
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
)
from llminspector.metrics.base_metric import (
    BaseMetric,
    DualTargetMetric,
    RagasBackedMetric,
)

GOLDEN_HEADER = [
    "question_emotion",
    "question_sentiment",
    "question_language",
    "question_pii_detected",
    "question_flesch_kincaid_grade",
    "question_tokens",
    "question_code_detected",
    "question_code_language",
    "answer_emotion",
    "answer_sentiment",
    "answer_language",
    "answer_pii_detected",
    "answer_flesch_kincaid_grade",
    "answer_tokens",
    "answer_no_refusal",
    "answer_code_detected",
    "answer_code_language",
    "bert_score",
    "faithfulness",
    "faithfulness_reasoning",
    "answer_correctness",
    "answer_correctness_reasoning",
    "answer_correctness_gt_agreement",
    "answer_correctness_faithfulness",
    "answer_correctness_relevancy",
    "answer_relevancy",
    "answer_relevancy_reasoning",
    "conciseness",
    "conciseness_reasoning",
    "context_relevance",
    "context_utilisation",
    "context_entity_recall",
    "context_precision",
    "context_recall",
    "question_jailbreak_risk",
    "answer_jailbreak_risk",
    "answer_hallucination_risk",
    "question_hate_speech",
    "question_fairness",
    "question_sexually_explicit_information",
    "question_violence",
    "question_self_harm",
    "question_dangerous_content",
    "question_harassment",
    "question_profanity",
    "question_toxicity_risk",
    "answer_hate_speech",
    "answer_fairness",
    "answer_sexually_explicit_information",
    "answer_violence",
    "answer_self_harm",
    "answer_dangerous_content",
    "answer_harassment",
    "answer_profanity",
    "answer_toxicity_risk",
    "is_policy_violated",
    "policy_violation_reason",
]


def _every_metric():
    """One instance of every metric, both targets for the dual ones."""
    dual = [
        EmotionMetric,
        SentimentMetric,
        LanguageDetectionMetric,
        PIIDetectionMetric,
        ReadabilityMetric,
        TokenCountMetric,
        CodeDetectMetric,
        ContentModerationMetric,
    ]
    single = [
        RefusalMetric,
        BertScoreMetric,
        FaithfulnessMetric,
        AnswerCorrectnessMetric,
        AnswerRelevancyMetric,
        ConcisenessMetric,
        ContextRelevanceMetric,
        ContextUtilisationMetric,
        ContextEntityRecallMetric,
        ContextPrecisionMetric,
        ContextRecallMetric,
        QuestionJailbreakMetric,
        AnswerJailbreakMetric,
        HallucinationMetric,
        PolicyComplianceMetric,
    ]
    instances = [cls() for cls in single]
    for cls in dual:
        instances += [cls(target="input"), cls(target="actual_output")]
    return instances


# --------------------------------------------------------------------------- #


def test_column_order_reproduces_the_golden_header():
    assert evaluate_mod._column_order(_every_metric()) == GOLDEN_HEADER


def test_column_order_is_independent_of_input_order():
    metrics = _every_metric()
    assert evaluate_mod._column_order(metrics) == evaluate_mod._column_order(
        list(reversed(metrics))
    )


def _executable_source(module) -> str:
    """Module source with comments and docstrings stripped.

    The engine's docstring legitimately names metrics when explaining what the
    retired ``overall_accuracy`` blend used to do; only executable code counts.
    """
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


def test_engine_module_contains_no_metric_names():
    """The whole point of 8.2 — no metric name may appear in the engine's code."""
    source = _executable_source(evaluate_mod)
    names = {m.name for m in _every_metric()} | set(GOLDEN_HEADER)
    leaked = sorted(n for n in names if n in source)
    assert leaked == [], f"evaluate.py still hardcodes metric names: {leaked}"


def test_every_column_is_owned_by_exactly_one_metric():
    owners: dict = {}
    for metric in _every_metric():
        for column in metric.output_columns:
            owners.setdefault(column, []).append(metric.name)
    clashes = {col: who for col, who in owners.items() if len(who) > 1}
    assert clashes == {}


@pytest.mark.parametrize("metric", _every_metric(), ids=lambda m: m.name)
def test_expand_key_set_is_stable_across_scores(metric):
    """expand(None) must reserve exactly the columns a real score produces."""
    assert set(metric.expand(None)) == set(metric.output_columns) - (
        {f"{metric.name}_reasoning"} if metric.produces_reasoning else set()
    )


@pytest.mark.parametrize("metric", _every_metric(), ids=lambda m: m.name)
def test_headline_column_comes_first(metric):
    assert metric.output_columns[0] == next(iter(metric.expand(None)))


def test_all_metrics_declare_a_sort_key():
    """A metric left on the default would silently land at the far right."""
    defaulted = [m.name for m in _every_metric() if m.sort_key == BaseMetric.sort_key]
    assert defaulted == []


def test_public_metric_classes_are_all_covered():
    """Guards against a new metric being added without a place in this file."""
    exported = {
        obj
        for obj in vars(metrics_pkg).values()
        if inspect.isclass(obj)
        and issubclass(obj, BaseMetric)
        and obj not in (BaseMetric, DualTargetMetric, RagasBackedMetric)
    }
    covered = {type(m) for m in _every_metric()}
    assert exported - covered == set()


# -- the structured-score expanders ----------------------------------------


def test_code_detect_expand():
    m = CodeDetectMetric(target="actual_output")
    assert m.expand({"code_detected": True, "code_language": "python"}) == {
        "answer_code_detected": True,
        "answer_code_language": "python",
    }
    # a_measure's failure path sets "" — reported as-is
    assert m.expand("") == {"answer_code_detected": "", "answer_code_language": None}
    assert m.expand(None) == {
        "answer_code_detected": None,
        "answer_code_language": None,
    }


def test_content_moderation_expand_pins_the_column_set():
    m = ContentModerationMetric(target="input")
    # judge answered a subset and invented a key: columns stay fixed
    out = m.expand({"fairness": 1, "profanity": 1, "not_a_category": 9})
    assert "question_not_a_category" not in out
    assert out["question_fairness"] == 1
    assert out["question_profanity"] == 1
    assert out["question_hate_speech"] is None
    assert set(out) == set(m.output_columns)
    # skipped row: same header, all None
    assert set(m.expand(None)) == set(out)


def test_content_moderation_toxicity_column_matches_the_prompt():
    """The content-filter fallback used to emit `toxicity`, which no declared
    column matched, so the value was dropped and `*_toxicity_risk` stayed None."""
    m = ContentModerationMetric(target="actual_output")
    assert "answer_toxicity_risk" in m.output_columns
    assert m.expand({"toxicity_risk": 1})["answer_toxicity_risk"] == 1


def test_policy_expand():
    m = PolicyComplianceMetric()
    assert m.expand(
        {"is_policy_violated": True, "policy_violation_reason": "leak"}
    ) == {
        "is_policy_violated": True,
        "policy_violation_reason": "leak",
    }
    assert m.expand(None) == {
        "is_policy_violated": None,
        "policy_violation_reason": None,
    }
    # the metric is named policy_check but owns no column under that name
    assert "policy_check" not in m.output_columns
