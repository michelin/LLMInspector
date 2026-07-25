"""Aggregate post-metrics — ``total_tokens``.

Operates on a per-row results dict (metric name -> value), not on a single test
case, so it is a function rather than a
:class:`~llminspector.metrics.base_metric.BaseMetric` subclass.

``calculate_overall_accuracy`` used to live here: a 0.5/0.3/0.2 weighted blend
of correctness / faithfulness / relevancy that overwrote the
``answer_correctness`` column. Phase 8.1 retired it — all three factors are now
judged together inside
:class:`~llminspector.metrics.rag.AnswerCorrectnessMetric`'s prompt, which is
what lets context-supported content missing from the ground truth escape the
penalty an external blend could not avoid.
"""

from __future__ import annotations

from typing import Any, Dict


def calculate_total_tokens(results: Dict[str, Any]) -> Dict[str, Any]:
    """Set ``results["total_tokens"] = answer_tokens + question_tokens`` if both
    are present, else ``None``. Mutates and returns ``results``.
    """
    if (
        results.get("answer_tokens") is not None
        and results.get("question_tokens") is not None
    ):
        results["total_tokens"] = results["answer_tokens"] + results["question_tokens"]
    else:
        results["total_tokens"] = None
    return results
