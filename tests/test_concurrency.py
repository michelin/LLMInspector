"""Phase 0 — ``llminspector.utils.concurrency.a_map``.

The bounded, order-preserving, failure-collecting fan-out that later phases
build on. The behaviours pinned here are the ones the evaluate engine learned
the hard way (input-order reassembly, errors collected rather than raised) plus
the one it does *not* have: a semaphore instead of fixed batches, so a slow item
never blocks a free worker.

Self-contained by design — no conftest fixtures, no network, and every sleep is
a few hundredths of a second. Async tests are driven with ``asyncio.run`` inside
ordinary sync test functions; the project does not use pytest-asyncio.
"""

import asyncio
import builtins
import sys
import types

import pytest

from llminspector.utils.concurrency import _make_progress_bar, _NullProgressBar, a_map

# --------------------------------------------------------------------------- #
# ordering
# --------------------------------------------------------------------------- #


def test_results_come_back_in_input_order():
    """Deliberately inverted durations: item 0 finishes last, still lands first."""
    items = [0, 1, 2, 3, 4]

    async def fn(i):
        # Item 0 sleeps longest, item 4 not at all — completion order is the
        # exact reverse of input order.
        await asyncio.sleep((len(items) - i) * 0.01)
        return f"r{i}"

    results, errors = asyncio.run(a_map(items, fn, limit=5))

    assert results == ["r0", "r1", "r2", "r3", "r4"]
    assert errors == []
    # The contract callers rely on: results zip against their inputs.
    assert list(zip(items, results)) == [(i, f"r{i}") for i in items]


def test_result_list_length_matches_input_even_with_failures():
    items = list(range(6))

    async def fn(i):
        if i % 2:
            raise RuntimeError("odd")
        return i

    results, errors = asyncio.run(a_map(items, fn, limit=3))

    assert len(results) == len(items)
    assert results == [0, None, 2, None, 4, None]
    assert [e["index"] for e in errors] == [1, 3, 5]


# --------------------------------------------------------------------------- #
# the semaphore
# --------------------------------------------------------------------------- #


def _instrumented():
    """An async fn that records how many calls are in flight, and the peak."""
    state = {"live": 0, "peak": 0}

    async def fn(item):
        state["live"] += 1
        state["peak"] = max(state["peak"], state["live"])
        await asyncio.sleep(0.01)
        state["live"] -= 1
        return item

    return state, fn


def test_concurrency_never_exceeds_limit():
    state, fn = _instrumented()

    results, errors = asyncio.run(a_map(list(range(12)), fn, limit=3))

    assert results == list(range(12))
    assert errors == []
    assert state["peak"] <= 3
    # With four times as many items as slots the pool must actually saturate;
    # `<= limit` alone would also pass for a serial implementation.
    assert state["peak"] == 3


def test_limit_above_item_count_runs_everything_at_once():
    state, fn = _instrumented()

    asyncio.run(a_map(list(range(4)), fn, limit=10))

    assert state["peak"] == 4


def test_no_head_of_line_blocking():
    """One slow item must not hold up the items behind it.

    The batching engine in evaluate.py would block here: with a window of 2,
    item 1 would wait for item 0 to finish. a_map admits the next item as soon
    as a slot frees, so the slow item completes last. Asserted on a completion
    log rather than wall-clock time, which would be flaky.
    """
    completed = []

    async def fn(i):
        if i == 0:
            await asyncio.sleep(0.05)
        else:
            await asyncio.sleep(0)
        completed.append(i)
        return i

    results, _ = asyncio.run(a_map(list(range(6)), fn, limit=2))

    assert results == list(range(6))
    assert completed[-1] == 0, f"slow item did not finish last: {completed}"
    assert sorted(completed) == list(range(6))


# --------------------------------------------------------------------------- #
# failures
# --------------------------------------------------------------------------- #


def test_one_failure_does_not_abort_the_batch():
    async def fn(i):
        if i == 1:
            raise ValueError("boom")
        await asyncio.sleep(0)
        return i * 10

    results, errors = asyncio.run(a_map([0, 1, 2], fn, limit=2))

    assert results == [0, None, 20]
    assert errors == [{"index": 1, "error": "ValueError: boom"}]


def test_errors_are_sorted_by_index_regardless_of_completion_order():
    """Item 2 fails immediately, item 1 fails later — errors still read 1, 2."""

    async def fn(i):
        if i == 1:
            await asyncio.sleep(0.03)
            raise KeyError("late")
        if i == 2:
            raise ValueError("early")
        return i

    _, errors = asyncio.run(a_map([0, 1, 2, 3], fn, limit=4))

    assert [e["index"] for e in errors] == [1, 2]
    assert errors[0]["error"] == "KeyError: 'late'"
    assert errors[1]["error"] == "ValueError: early"


def test_all_items_failing_yields_all_none():
    async def fn(_):
        raise TypeError("nope")

    results, errors = asyncio.run(a_map([1, 2, 3], fn, limit=2))

    assert results == [None, None, None]
    assert len(errors) == 3
    assert {e["error"] for e in errors} == {"TypeError: nope"}


def test_failures_log_a_warning(caplog):
    async def fn(i):
        raise ValueError("boom")

    with caplog.at_level("WARNING", logger="llminspector.utils.concurrency"):
        asyncio.run(a_map([0, 1], fn, limit=2))

    messages = [record.getMessage() for record in caplog.records]
    assert any("2 of 2 item(s) failed" in message for message in messages)


def test_cancelled_error_propagates_rather_than_being_collected():
    """A cancelled run must not look like N ordinary item failures."""

    async def fn(i):
        if i == 1:
            raise asyncio.CancelledError()
        await asyncio.sleep(0.01)
        return i

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(a_map([0, 1, 2], fn, limit=3))


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #


def test_empty_input_returns_empty_pair():
    calls = []

    async def fn(item):  # pragma: no cover - must never run
        calls.append(item)
        return item

    assert asyncio.run(a_map([], fn, limit=3, desc="unused")) == ([], [])
    assert calls == []


def test_limit_below_one_raises_value_error():
    async def fn(item):  # pragma: no cover - never reached
        return item

    for bad in (0, -1):
        with pytest.raises(ValueError) as excinfo:
            asyncio.run(a_map([1, 2], fn, limit=bad))
        assert str(bad) in str(excinfo.value)


def test_limit_is_validated_before_the_empty_shortcut():
    """A bad limit is a caller bug; an empty batch must not hide it."""

    async def fn(item):  # pragma: no cover - never reached
        return item

    with pytest.raises(ValueError):
        asyncio.run(a_map([], fn, limit=0))


# --------------------------------------------------------------------------- #
# progress bar
# --------------------------------------------------------------------------- #


class _FakeBar:
    instances = []

    def __init__(self, total=None, desc=None):
        self.total = total
        self.desc = desc
        self.updates = 0
        self.closed = False
        _FakeBar.instances.append(self)

    def update(self, n=1):
        self.updates += n

    def close(self):
        self.closed = True


@pytest.fixture
def fake_tqdm(monkeypatch):
    """Stand in for the real tqdm module and record every bar built."""
    _FakeBar.instances = []
    module = types.ModuleType("tqdm")
    module.tqdm = _FakeBar
    monkeypatch.setitem(sys.modules, "tqdm", module)
    return _FakeBar


@pytest.fixture
def no_tqdm(monkeypatch):
    """Make every `import tqdm` raise, as it would if tqdm were absent."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "tqdm" or name.startswith("tqdm."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.delitem(sys.modules, "tqdm", raising=False)


def test_desc_builds_a_bar_and_closes_it(fake_tqdm):
    async def fn(i):
        return i

    results, _ = asyncio.run(a_map([0, 1, 2], fn, limit=2, desc="working"))

    assert results == [0, 1, 2]
    assert len(fake_tqdm.instances) == 1
    bar = fake_tqdm.instances[0]
    assert (bar.total, bar.desc) == (3, "working")
    assert bar.updates == 3
    assert bar.closed


def test_desc_none_builds_no_bar_at_all(fake_tqdm):
    async def fn(i):
        return i

    asyncio.run(a_map([0, 1, 2], fn, limit=2))

    assert fake_tqdm.instances == []
    assert isinstance(_make_progress_bar(3, None), _NullProgressBar)


def test_bar_is_closed_even_when_items_fail(fake_tqdm):
    async def fn(i):
        raise ValueError("boom")

    asyncio.run(a_map([0, 1], fn, limit=2, desc="working"))

    assert fake_tqdm.instances[0].closed


def test_missing_tqdm_falls_back_to_the_null_bar(no_tqdm):
    assert isinstance(_make_progress_bar(3, "working"), _NullProgressBar)

    async def fn(i):
        return i

    results, errors = asyncio.run(a_map([0, 1, 2], fn, limit=2, desc="working"))

    assert results == [0, 1, 2]
    assert errors == []


def test_null_progress_bar_is_a_no_op():
    bar = _NullProgressBar()
    assert bar.update(3) is None
    assert bar.close() is None
