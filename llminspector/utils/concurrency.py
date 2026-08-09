"""Bounded concurrent ``map`` over an async function.

``a_map`` is the generalised form of the pattern in
``evaluate/evaluate.py::_aevaluate``: fan out over a sequence, keep results in
input order, collect per-item failures instead of raising, drive a tqdm bar off
completion. It differs from that engine in one respect that matters:

**A semaphore bounds concurrency; there are no batches.** The evaluate engine
slices its rows into fixed windows of ``batch_size`` and awaits the whole window
before starting the next, so a single slow row blocks every free worker until it
finishes (head-of-line blocking). ``a_map`` starts every item at once and lets a
semaphore admit at most ``limit`` of them, so the moment one item finishes the
next one starts.

``evaluate.py`` is deliberately **not** refactored onto this. Its batching
behaviour is pinned by ``tests/test_evaluate_engine.py`` and
``tests/test_column_contract.py``, and converging the two is a separate decision
with its own test churn. Until then the small progress-bar helpers below are
knowingly duplicated there; do not edit ``evaluate.py`` to import them without
making that call explicitly.

This module sits at the very bottom of the layering: it imports nothing from
``llminspector`` and nothing heavy at module level.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class _NullProgressBar:
    """Fallback when tqdm is unavailable; matches the update/close interface."""

    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


def _make_progress_bar(total: int, desc: Optional[str]) -> Any:
    """A tqdm bar labelled ``desc``, or a no-op bar.

    ``desc is None`` means the caller wants no visible progress at all, so the
    null bar stands in and tqdm is never imported. tqdm is a declared dependency
    but the import stays lazy and absence-tolerant: a bar is never worth failing
    a run over.
    """
    if desc is None:
        return _NullProgressBar()
    try:
        from tqdm import tqdm
    except ImportError:
        return _NullProgressBar()
    return tqdm(total=total, desc=desc)


async def a_map(
    items: Sequence[T],
    fn: Callable[[T], Awaitable[R]],
    *,
    limit: int = 5,
    desc: Optional[str] = None,
) -> tuple[List[Optional[R]], List[Dict[str, Any]]]:
    """Apply ``fn`` to every item in ``items``, at most ``limit`` at a time.

    Parameters
    ----------
    items:
        The inputs. Consumed once, positionally.
    fn:
        An async callable taking one item. It is called once per item.
    limit:
        Maximum number of ``fn`` calls in flight. Must be >= 1.
    desc:
        Progress bar label. ``None`` (the default) shows no bar.

    Returns
    -------
    ``(results, errors)``. ``results`` has the same length and order as
    ``items`` — ``results[i]`` is the value ``fn(items[i])`` returned, or
    ``None`` if that call raised, so it can always be zipped against ``items``.
    ``errors`` is a list of ``{"index": i, "error": "ExcType: message"}``,
    sorted by index.

    A failing item never aborts the rest of the run; it is recorded and the
    remaining items keep going.
    """
    # Validated ahead of the empty-input shortcut: a bad ``limit`` is a caller
    # bug, and an empty batch is no reason to let it through unnoticed.
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")

    # Nothing to schedule — no bar, no semaphore, no tasks.
    if not items:
        return [], []

    results: List[Optional[R]] = [None] * len(items)
    errors: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(limit)

    async def _run(index: int, item: T) -> tuple[int, Optional[R], Optional[str]]:
        # The semaphore is taken inside the task rather than around task
        # creation, so all items are scheduled immediately and each one starts
        # the instant a slot frees rather than waiting on a batch boundary.
        async with semaphore:
            try:
                return index, await fn(item), None
            except Exception as exc:  # noqa: BLE001 - isolate per-item failures
                # Exception, never BaseException: asyncio.CancelledError derives
                # from BaseException and must propagate. Swallowing it would
                # turn one cancelled run into N ordinary-looking item failures
                # and leave the caller with a full result set it never got.
                return index, None, f"{type(exc).__name__}: {exc}"

    tasks = [asyncio.ensure_future(_run(i, item)) for i, item in enumerate(items)]
    pbar = _make_progress_bar(len(items), desc)
    try:
        # as_completed only drives the bar; the index carried out of _run is
        # what puts each result back in its input slot.
        for future in asyncio.as_completed(tasks):
            index, value, error = await future
            results[index] = value
            if error is not None:
                errors.append({"index": index, "error": error})
            pbar.update(1)
    finally:
        # On a propagating cancellation the remaining tasks are still pending;
        # cancel them rather than leaving the loop to complain about tasks
        # destroyed mid-flight.
        for task in tasks:
            if not task.done():
                task.cancel()
        pbar.close()

    # Completion order is nondeterministic, so sort before returning — callers
    # (and their tests) get a stable errors list.
    errors.sort(key=lambda e: e["index"])
    if errors:
        logger.warning(
            "%d of %d item(s) failed during %s; see the returned errors list",
            len(errors),
            len(items),
            desc or "a_map",
        )
    return results, errors
