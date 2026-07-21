"""The evaluate engine — ``evaluate(dataset, metrics=[...])``.

Ports ``helper.py``'s async pipeline onto the Phase 3 metric objects:

* per-row ``asyncio.gather`` over that row's available metrics,
* batches with a tqdm progress bar,
* availability filtering (a metric runs only when its ``required_inputs`` are
  present on the test case),
* dependency auto-add (``answer_correctness`` pulls in ``faithfulness`` /
  ``answer_relevancy`` for the ``overall_accuracy`` blend, then drops them from
  the output if the caller did not request them),
* result column-expansion (code detect / content moderation / policy),
* ``overall_accuracy`` overwriting ``answer_correctness`` and ``total_tokens``,
* ``reorder_results`` column ordering.

Two intentional divergences from the legacy code, both noted inline:
1. Batch results are reassembled **in input order** (the legacy
   ``as_completed`` + positional ``pd.concat`` could misalign metrics to the
   wrong row).
2. Result columns reflect only the requested metrics (+ their expansions),
   rather than always emitting every possible metric column.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

from ..dataset.dataset import EvaluationDataset
from ..metrics.aggregate import calculate_overall_accuracy, calculate_total_tokens
from ..metrics.base_metric import BaseMetric
from ..test_case.test_case import LLMTestCase
from .result import EvaluationResult

# answer_correctness is reported as the overall_accuracy blend, which needs
# these two computed even when the caller didn't ask for them.
_ACCURACY_DEPENDENCIES = ("faithfulness", "answer_relevancy")

_REORDER_KEYS = [
    "question_emotion", "question_sentiment", "question_language",
    "question_pii_detected", "question_flesch_kincaid_grade", "question_tokens",
    "question_code_detected", "question_code_language",
    "answer_emotion", "answer_sentiment", "answer_language",
    "answer_pii_detected", "answer_flesch_kincaid_grade", "answer_tokens",
    "answer_no_refusal",
    "answer_code_detected", "answer_code_language",
    "bert_score",
    "faithfulness", "faithfulness_reasoning",
    "answer_correctness", "answer_correctness_reasoning",
    "answer_relevancy", "answer_relevancy_reasoning",
    "conciseness", "conciseness_reasoning",
    "context_relevance", "context_utilisation", "context_entity_recall",
    "context_precision", "context_recall",
    "question_jailbreak_risk", "answer_jailbreak_risk", "answer_hallucination_risk",
    "question_hate_speech", "question_fairness", "question_sexually_explicit_information",
    "question_violence", "question_self_harm", "question_dangerous_content",
    "question_harassment", "question_profanity", "question_toxicity_risk",
    "answer_hate_speech", "answer_fairness", "answer_sexually_explicit_information",
    "answer_violence", "answer_self_harm", "answer_dangerous_content",
    "answer_harassment", "answer_profanity", "answer_toxicity_risk",
    "is_policy_violated", "policy_violation_reason",
    "total_tokens",
]

_MODERATION_SUBKEYS = [
    "hate_speech", "fairness", "sexually_explicit_information", "violence",
    "self_harm", "dangerous_content", "harassment", "profanity", "toxicity_risk",
]


class _NullProgressBar:
    """Fallback when tqdm is unavailable; matches the update/close interface."""

    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


def _make_progress_bar(total: int, show_progress: bool):
    if not show_progress:
        return _NullProgressBar()
    try:
        from tqdm import tqdm
    except ImportError:
        return _NullProgressBar()
    return tqdm(total=total, desc="Evaluating")


# --------------------------------------------------------------------------- #
# availability
# --------------------------------------------------------------------------- #

def _availability(test_case: LLMTestCase) -> Dict[str, bool]:
    """Which LLMTestCase attributes are populated (mirrors legacy check)."""
    ctx = test_case.retrieval_context

    def _str_ok(value: Any) -> bool:
        return value is not None and str(value).strip() != ""

    return {
        "input": _str_ok(test_case.input),
        "actual_output": _str_ok(test_case.actual_output),
        "expected_output": _str_ok(test_case.expected_output),
        "retrieval_context": ctx is not None
        and len(ctx) > 0
        and all(str(c).strip() != "" for c in ctx),
        "policy": _str_ok(test_case.policy),
    }


def _metric_available(metric: BaseMetric, avail: Dict[str, bool]) -> bool:
    return all(avail.get(req, False) for req in metric.required_inputs)


# --------------------------------------------------------------------------- #
# result column-expansion (ported from helper.py)
# --------------------------------------------------------------------------- #

def _expand_code_detect(results: Dict[str, Any]) -> None:
    for prefix in ("question", "answer"):
        key = f"{prefix}_code_detected"
        if key not in results:
            continue
        value = results[key]
        if isinstance(value, dict):
            code_info = value
            results[f"{prefix}_code_detected"] = code_info.get("code_detected")
            results[f"{prefix}_code_language"] = code_info.get("code_language")
        elif value is not None:
            results[f"{prefix}_code_detected"] = value
            results[f"{prefix}_code_language"] = None
        else:
            results[f"{prefix}_code_detected"] = None
            results[f"{prefix}_code_language"] = None


def _expand_content_moderation(results: Dict[str, Any]) -> None:
    for prefix in ("question", "answer"):
        key = f"{prefix}_content_moderation"
        if key not in results:
            continue
        value = results[key]
        if isinstance(value, dict):
            for subkey, subval in value.items():
                results[f"{prefix}_{subkey}"] = subval
        else:
            for subkey in _MODERATION_SUBKEYS:
                results[f"{prefix}_{subkey}"] = None
        del results[key]


def _expand_policy_check(results: Dict[str, Any]) -> None:
    if "policy_check" not in results:
        return
    value = results["policy_check"]
    if value is not None and isinstance(value, dict):
        results["is_policy_violated"] = value.get("is_policy_violated")
        results["policy_violation_reason"] = value.get("policy_violation_reason")
    else:
        results["is_policy_violated"] = None
        results["policy_violation_reason"] = None
    del results["policy_check"]


def _reorder(results: Dict[str, Any]) -> Dict[str, Any]:
    ordered: Dict[str, Any] = {}
    for key in _REORDER_KEYS:
        if key in results:
            ordered[key] = results[key]
    for key in results:
        if key not in ordered:
            ordered[key] = results[key]
    return ordered


# --------------------------------------------------------------------------- #
# metric-set assembly (dependency auto-add)
# --------------------------------------------------------------------------- #

def _build_metric_set(metrics: Sequence[BaseMetric]):
    """Return (prototypes, dependency_only_names).

    Adds faithfulness / answer_relevancy prototypes when answer_correctness is
    present but they are not, so the accuracy blend can be computed. Their names
    are returned so the caller can null them from the output afterwards.
    """
    prototypes: List[BaseMetric] = list(metrics)
    present = {m.name for m in prototypes}
    dependency_only: List[str] = []

    if "answer_correctness" in present:
        model = next(
            (m.model for m in prototypes if m.name == "answer_correctness"), None
        )
        from ..metrics.rag import AnswerRelevancyMetric, FaithfulnessMetric

        dep_classes = {
            "faithfulness": FaithfulnessMetric,
            "answer_relevancy": AnswerRelevancyMetric,
        }
        for dep_name in _ACCURACY_DEPENDENCIES:
            if dep_name not in present:
                prototypes.append(dep_classes[dep_name](model=model))
                dependency_only.append(dep_name)

    return prototypes, dependency_only


async def _run_metric(metric: BaseMetric, test_case: LLMTestCase):
    """Run one metric, mirroring legacy per-metric error isolation."""
    try:
        await metric.a_measure(test_case)
        return metric
    except Exception as e:  # noqa: BLE001 - isolate per-metric failures
        print(f"Error calculating {metric.name}: {str(e)}")
        metric.score = None
        return metric


# --------------------------------------------------------------------------- #
# per-row + batch orchestration
# --------------------------------------------------------------------------- #

async def _evaluate_row(
    test_case: LLMTestCase,
    prototypes: Sequence[BaseMetric],
    dependency_only: Sequence[str],
) -> Dict[str, Any]:
    avail = _availability(test_case)

    # Fresh clones per row so concurrent rows never share mutable score state.
    row_metrics = [p.clone() for p in prototypes]

    # Stable base columns for every requested metric (+ reasoning slots).
    results: Dict[str, Any] = {}
    for m in row_metrics:
        results[m.name] = None
        if m.produces_reasoning:
            results[f"{m.name}_reasoning"] = None

    runnable = [m for m in row_metrics if _metric_available(m, avail)]
    if runnable:
        completed = await asyncio.gather(*(_run_metric(m, test_case) for m in runnable))
        for m in completed:
            results[m.name] = m.score
            if m.produces_reasoning:
                results[f"{m.name}_reasoning"] = m.reason

    # Column expansion (order matches legacy).
    _expand_code_detect(results)
    _expand_content_moderation(results)
    _expand_policy_check(results)
    calculate_total_tokens(results)

    # overall_accuracy is reported in place of answer_correctness.
    if "answer_correctness" in results:
        overall_acc = calculate_overall_accuracy(results)
        if overall_acc is not None:
            results["answer_correctness"] = overall_acc

    # Drop dependency-only metrics the caller didn't request.
    for dep_name in dependency_only:
        if dep_name in results:
            results[dep_name] = None
        for suffix in ("_reasoning", "_key_findings"):
            key = f"{dep_name}{suffix}"
            if key in results:
                results[key] = None

    return _reorder(results)


async def _aevaluate(
    test_cases: Sequence[LLMTestCase],
    metrics: Sequence[BaseMetric],
    batch_size: int,
    show_progress: bool,
) -> List[Dict[str, Any]]:
    prototypes, dependency_only = _build_metric_set(metrics)
    rows: List[Optional[Dict[str, Any]]] = [None] * len(test_cases)

    pbar = _make_progress_bar(len(test_cases), show_progress)
    try:
        for start in range(0, len(test_cases), batch_size):
            batch = list(enumerate(test_cases))[start : start + batch_size]

            async def _row(idx: int, tc: LLMTestCase):
                return idx, await _evaluate_row(tc, prototypes, dependency_only)

            tasks = [_row(idx, tc) for idx, tc in batch]
            # as_completed for responsive progress; idx keeps input order intact.
            for future in asyncio.as_completed(tasks):
                idx, row = await future
                rows[idx] = row
                pbar.update(1)
    finally:
        pbar.close()

    return [r if r is not None else {} for r in rows]


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #

def evaluate(
    dataset: EvaluationDataset,
    metrics: Sequence[BaseMetric],
    *,
    batch_size: int = 5,
    show_progress: bool = True,
) -> EvaluationResult:
    """Evaluate ``dataset``'s test cases against instantiated ``metrics``.

    Parameters
    ----------
    dataset:
        An :class:`~llminspector.dataset.dataset.EvaluationDataset` with
        ``test_cases``.
    metrics:
        Instantiated metric objects (each built with its model). Metrics whose
        ``required_inputs`` are missing on a given row are skipped for that row.
    batch_size:
        Number of rows evaluated concurrently per batch.
    show_progress:
        Show a tqdm progress bar.
    """
    if not metrics:
        raise ValueError("evaluate() requires at least one metric.")

    test_cases = list(dataset.test_cases)
    rows = asyncio.run(
        _aevaluate(test_cases, list(metrics), batch_size, show_progress)
    )
    return EvaluationResult(rows=rows, test_cases=test_cases)
