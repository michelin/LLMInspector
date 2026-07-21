"""Aggregate post-metrics — ``overall_accuracy`` and ``total_tokens``.

Ported verbatim from ``helper.calculate_overall_accuracy`` /
``calculate_total_tokens``. These operate on a per-row results dict (metric
name -> value), not on a single test case, so they are functions rather than
:class:`~llminspector.metrics.base_metric.BaseMetric` subclasses.

NOTE: ``overall_accuracy`` is a stopgap. A later phase will fold this weighted
blend directly into the answer-correctness judge prompt (see REFACTOR_PHASES.md
"Future work"), at which point this function is expected to be retired.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def calculate_total_tokens(results: Dict[str, Any]) -> Dict[str, Any]:
    """Set ``results["total_tokens"] = answer_tokens + question_tokens`` if both
    are present, else ``None``. Mutates and returns ``results``.
    """
    if results.get("answer_tokens") is not None and results.get("question_tokens") is not None:
        results["total_tokens"] = results["answer_tokens"] + results["question_tokens"]
    else:
        results["total_tokens"] = None
    return results


def calculate_overall_accuracy(results: Dict[str, Any]) -> Optional[float]:
    """Weighted blend of answer_correctness / faithfulness / answer_relevancy.

    * 3 metrics present: 0.5*correctness + 0.3*faithfulness + 0.2*relevancy
    * correctness + relevancy only: 0.75*correctness + 0.25*relevancy
    * otherwise: ``None``. Result rounded to 2 dp.
    """
    try:
        if (
            results.get("answer_correctness") is not None
            and results.get("faithfulness") is not None
            and results.get("answer_relevancy") is not None
        ):
            overall_accuracy = (
                0.5 * results["answer_correctness"]
                + 0.3 * results["faithfulness"]
                + 0.2 * results["answer_relevancy"]
            )
        elif (
            results.get("answer_correctness") is not None
            and results.get("answer_relevancy") is not None
        ):
            overall_accuracy = (
                0.75 * results["answer_correctness"]
                + 0.25 * results["answer_relevancy"]
            )
        else:
            return None
        return round(overall_accuracy, 2)
    except Exception:  # noqa: BLE001 - mirror legacy behavior
        return None
