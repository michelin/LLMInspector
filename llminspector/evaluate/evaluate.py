"""The evaluate engine — ``evaluate(dataset, metrics=[...])``.

Ports ``helper.py``'s async pipeline onto the Phase 3 metric objects:

* per-row ``asyncio.gather`` over that row's available metrics,
* batches with a tqdm progress bar,
* availability filtering (a metric runs only when its ``required_inputs`` are
  present on the test case),
* ``total_tokens``,
* column ordering.

The engine runs exactly the metrics it is handed. It used to silently add
``faithfulness`` / ``answer_relevancy`` behind the caller's back to feed the
``overall_accuracy`` blend and then null them out again; Phase 8.1 folded all
three factors into the ``answer_correctness`` judge itself, so the auto-add, the
blend, and the column overwrite are all gone.

**This module knows no metric names.** Phase 8.2 moved the output contract onto
the metrics: each one declares the columns it owns (``expand`` /
``output_columns``) and where they sit in the table (``sort_key``). Adding a
metric touches only that metric's file.

Two intentional divergences from the legacy code, both noted inline:

1. Batch results are reassembled **in input order** (the legacy
   ``as_completed`` + positional ``pd.concat`` could misalign metrics to the
   wrong row).
2. Result columns reflect only the requested metrics (+ their expansions),
   rather than always emitting every possible metric column.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..dataset.dataset import EvaluationDataset
from ..metrics.aggregate import calculate_total_tokens
from ..metrics.base_metric import BaseMetric
from ..test_case.test_case import LLMTestCase
from .result import EvaluationResult

logger = logging.getLogger(__name__)


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
# column ordering
# --------------------------------------------------------------------------- #


def _column_order(prototypes: Sequence[BaseMetric]) -> List[str]:
    """The export order of the metric columns, from the metrics themselves.

    Metrics sort by ``sort_key`` (ties broken by name for determinism), and each
    contributes its ``output_columns`` as a contiguous block.
    """
    order: List[str] = []
    for metric in sorted(prototypes, key=lambda m: (m.sort_key, m.name)):
        order.extend(metric.output_columns)
    return order


def _reorder(results: Dict[str, Any], order: Sequence[str]) -> Dict[str, Any]:
    """Apply ``order``, then append anything left (aggregates like total_tokens)."""
    ordered: Dict[str, Any] = {}
    for key in order:
        if key in results:
            ordered[key] = results[key]
    for key in results:
        if key not in ordered:
            ordered[key] = results[key]
    return ordered


async def _run_metric(metric: BaseMetric, test_case: LLMTestCase):
    """Run one metric, isolating its failure from the rest of the row.

    Metrics normally record their own failures (``BaseMetric.record_failure``);
    this catches anything that escapes that, so one broken metric cannot abort
    the run.
    """
    try:
        await metric.a_measure(test_case)
    except Exception as e:  # noqa: BLE001 - isolate per-metric failures
        metric.record_failure(e)
        metric.score = None
        # A metric that raised past its own error path may never have reached
        # is_successful(); a stale verdict must not outlive the score.
        metric.success = None
    return metric


# --------------------------------------------------------------------------- #
# per-row + batch orchestration
# --------------------------------------------------------------------------- #


async def _evaluate_row(
    test_case: LLMTestCase,
    prototypes: Sequence[BaseMetric],
    order: Sequence[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """Return ``(ordered results, metric failures)`` for one test case."""
    avail = _availability(test_case)

    # Fresh clones per row so concurrent rows never share mutable score state.
    row_metrics = [p.clone() for p in prototypes]

    # Stable base columns for every requested metric (+ reasoning slots), so a
    # metric that is skipped on this row still emits its columns as None.
    results: Dict[str, Any] = {}
    for m in row_metrics:
        results.update(m.expand(None))
        if m.produces_reasoning:
            results[f"{m.name}_reasoning"] = None
        if m.success_column is not None:
            results[m.success_column] = None

    failures: List[str] = []
    runnable = [m for m in row_metrics if _metric_available(m, avail)]
    if runnable:
        completed = await asyncio.gather(*(_run_metric(m, test_case) for m in runnable))
        for m in completed:
            results.update(m.expand(m.score))
            if m.produces_reasoning:
                results[f"{m.name}_reasoning"] = m.reason
            if m.success_column is not None:
                results[m.success_column] = m.success
            if m.error is not None:
                failures.append(f"{m.name}: {m.error}")

    calculate_total_tokens(results)

    return _reorder(results, order), failures


async def _aevaluate(
    test_cases: Sequence[LLMTestCase],
    metrics: Sequence[BaseMetric],
    batch_size: int,
    show_progress: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prototypes: List[BaseMetric] = list(metrics)
    order = _column_order(prototypes)
    rows: List[Optional[Dict[str, Any]]] = [None] * len(test_cases)
    errors: List[Dict[str, Any]] = []

    pbar = _make_progress_bar(len(test_cases), show_progress)
    try:
        for start in range(0, len(test_cases), batch_size):
            batch = list(enumerate(test_cases))[start : start + batch_size]

            async def _row(idx: int, tc: LLMTestCase):
                return idx, await _evaluate_row(tc, prototypes, order)

            tasks = [_row(idx, tc) for idx, tc in batch]
            # as_completed for responsive progress; idx keeps input order intact.
            for future in asyncio.as_completed(tasks):
                idx, (row, failures) = await future
                rows[idx] = row
                errors += [{"row": idx, "error": f} for f in failures]
                pbar.update(1)
    finally:
        pbar.close()

    errors.sort(key=lambda e: (e["row"], e["error"]))
    if errors:
        logger.warning(
            "%d metric failure(s) across %d row(s); see EvaluationResult.errors",
            len(errors),
            len({e["row"] for e in errors}),
        )
    return [r if r is not None else {} for r in rows], errors


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #

_DEFAULT_BATCH_SIZE = 5


def _resolve_batch_size(
    metrics: Sequence[BaseMetric], batch_size: Optional[int]
) -> int:
    """Reconcile the engine's row concurrency with the providers' own ceiling.

    ``batch_size`` and the provider's ``max_workers`` (which also drives the
    ragas ``RunConfig``) used to be set independently, with nothing keeping them
    consistent. When ``batch_size`` is not given it is taken from the strictest
    provider limit in the metric set, so one number governs both.
    """
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        return batch_size
    limits = [
        int(getattr(m.model, "max_workers"))
        for m in metrics
        if m.model is not None and getattr(m.model, "max_workers", None) is not None
    ]
    return min(limits) if limits else _DEFAULT_BATCH_SIZE


async def a_evaluate(
    dataset: EvaluationDataset,
    metrics: Sequence[BaseMetric],
    *,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> EvaluationResult:
    """Async form of :func:`evaluate` — await this inside a running event loop.

    The engine is async all the way down, so notebooks, FastAPI handlers, and
    anything else already running a loop should use this. :func:`evaluate` calls
    ``asyncio.run``, which raises ``RuntimeError: asyncio.run() cannot be called
    from a running event loop`` in exactly those places.

    Parameters
    ----------
    dataset:
        An :class:`~llminspector.dataset.dataset.EvaluationDataset` with
        ``test_cases``.
    metrics:
        Instantiated metric objects (each built with its model). Metrics whose
        ``required_inputs`` are missing on a given row are skipped for that row.
    batch_size:
        Rows evaluated concurrently per batch. Defaults to the strictest
        ``max_workers`` among the metrics' providers, else 5.
    show_progress:
        Show a tqdm progress bar.
    """
    if not metrics:
        raise ValueError("evaluate() requires at least one metric.")

    metrics = list(metrics)
    test_cases = list(dataset.test_cases)
    rows, errors = await _aevaluate(
        test_cases, metrics, _resolve_batch_size(metrics, batch_size), show_progress
    )
    return EvaluationResult(rows=rows, test_cases=test_cases, errors=errors)


def evaluate(
    dataset: EvaluationDataset,
    metrics: Sequence[BaseMetric],
    *,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> EvaluationResult:
    """Evaluate ``dataset``'s test cases against instantiated ``metrics``.

    Thin synchronous wrapper around :func:`a_evaluate`. Inside a running event
    loop (Jupyter, FastAPI) use ``await a_evaluate(...)`` instead — see there for
    the full parameter documentation.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "evaluate() cannot be called from a running event loop "
            "(Jupyter, FastAPI, ...). Use `await a_evaluate(...)` instead."
        )

    return asyncio.run(
        a_evaluate(
            dataset,
            metrics,
            batch_size=batch_size,
            show_progress=show_progress,
        )
    )
