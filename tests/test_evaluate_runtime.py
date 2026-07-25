"""Phase 8.4 — async entry point, visible failures, rate-limit backoff.

Three defects this pins:

1. ``evaluate()`` called ``asyncio.run``, which raises inside any already-running
   loop — Jupyter, FastAPI — including the ``getting_started.ipynb`` the package
   ships. ``a_evaluate`` is now the real entry point.
2. Metric failures were ``print``-ed and scored ``None``, so a run where every
   call returned 401 produced a clean table of ``None``s indistinguishable from
   *skipped for missing input*.
3. There was no 429 handling anywhere.
"""

import asyncio
import logging

import pytest

from llminspector import a_evaluate, evaluate
from llminspector.dataset import EvaluationDataset
from llminspector.evaluate.evaluate import _resolve_batch_size
from llminspector.metrics.base_metric import BaseMetric
from llminspector.models import retry
from llminspector.test_case import LLMTestCase


class FakeMetric(BaseMetric):
    def __init__(self, name, required_inputs, value=None, exc=None):
        super().__init__(model=None)
        self.metric_name = name
        self.sort_key = 1
        self.required_inputs = set(required_inputs)
        self._value = value
        self._exc = exc

    async def a_measure(self, test_case):
        if self._exc is not None:
            self.record_failure(self._exc)
            self.score = None
        else:
            self.score = self._value
        self.is_successful()
        return self.score


def _dataset(*test_cases):
    return EvaluationDataset(test_cases=list(test_cases))


# --------------------------------------------------------------------------- #
# 1. async entry point
# --------------------------------------------------------------------------- #


def test_a_evaluate_runs_inside_a_running_loop():
    """The exact call that used to raise RuntimeError in a notebook."""
    ds = _dataset(LLMTestCase(input="q", actual_output="a"))

    async def main():
        return await a_evaluate(
            ds,
            [FakeMetric("answer_sentiment", {"actual_output"}, "Positive")],
            show_progress=False,
        )

    result = asyncio.run(main())
    assert result.rows[0]["answer_sentiment"] == "Positive"


def test_sync_evaluate_inside_a_loop_says_what_to_use_instead():
    ds = _dataset(LLMTestCase(input="q", actual_output="a"))
    metric = FakeMetric("answer_sentiment", {"actual_output"}, "Positive")

    async def main():
        with pytest.raises(RuntimeError, match="a_evaluate"):
            evaluate(ds, [metric], show_progress=False)

    asyncio.run(main())


def test_sync_and_async_entry_points_agree():
    ds = _dataset(
        LLMTestCase(input="q1", actual_output="a1"),
        LLMTestCase(input="q2", actual_output="a2"),
    )

    def metrics():
        return [FakeMetric("answer_sentiment", {"actual_output"}, "Positive")]

    sync_rows = evaluate(ds, metrics(), show_progress=False).rows
    async_rows = asyncio.run(a_evaluate(ds, metrics(), show_progress=False)).rows
    assert sync_rows == async_rows


def test_a_evaluate_requires_metrics():
    with pytest.raises(ValueError):
        asyncio.run(a_evaluate(_dataset(LLMTestCase(input="q")), []))


# --------------------------------------------------------------------------- #
# 2. failures are visible in the result, not only on stdout
# --------------------------------------------------------------------------- #


def test_failed_metric_is_recorded_on_the_result():
    ds = _dataset(
        LLMTestCase(input="q1", actual_output="a1"),
        LLMTestCase(input="q2", actual_output="a2"),
    )
    m = FakeMetric(
        "answer_sentiment",
        {"actual_output"},
        exc=PermissionError("401 Unauthorized"),
    )
    result = evaluate(ds, [m], show_progress=False)

    # the table alone cannot tell you anything went wrong...
    assert [r["answer_sentiment"] for r in result.rows] == [None, None]
    # ...but the result object can
    assert len(result.errors) == 2
    assert result.errors[0]["row"] == 0
    assert "401 Unauthorized" in result.errors[0]["error"]
    assert result.error_summary() == {"answer_sentiment": 2}
    assert "errors=2" in repr(result)


def test_skipped_metric_is_not_reported_as_an_error():
    """A metric skipped for missing inputs is not a failure."""
    ds = _dataset(LLMTestCase(input="q"))  # no actual_output
    m = FakeMetric("answer_sentiment", {"actual_output"}, "Positive")
    result = evaluate(ds, [m], show_progress=False)
    assert result.rows[0]["answer_sentiment"] is None
    assert result.errors == []


def test_a_successful_run_has_no_errors():
    ds = _dataset(LLMTestCase(input="q", actual_output="a"))
    m = FakeMetric("answer_sentiment", {"actual_output"}, "Positive")
    result = evaluate(ds, [m], show_progress=False)
    assert result.errors == []
    assert result.error_summary() == {}
    assert "errors" not in repr(result)


def test_failures_go_to_logging_not_stdout(caplog, capsys):
    ds = _dataset(LLMTestCase(input="q", actual_output="a"))
    m = FakeMetric("answer_sentiment", {"actual_output"}, exc=RuntimeError("boom"))
    with caplog.at_level(logging.WARNING):
        evaluate(ds, [m], show_progress=False)
    assert "boom" in caplog.text
    assert capsys.readouterr().out == ""


def test_one_failing_metric_does_not_stop_the_others():
    ds = _dataset(LLMTestCase(input="q", actual_output="a"))
    bad = FakeMetric("answer_sentiment", {"actual_output"}, exc=RuntimeError("boom"))
    good = FakeMetric("answer_emotion", {"actual_output"}, "Joy")
    result = evaluate(ds, [bad, good], show_progress=False)
    assert result.rows[0]["answer_emotion"] == "Joy"
    assert len(result.errors) == 1


def test_clone_resets_the_error_state():
    m = FakeMetric("answer_sentiment", {"actual_output"}, exc=RuntimeError("boom"))
    m.measure(LLMTestCase(input="q", actual_output="a"))
    assert m.error is not None
    assert m.clone().error is None


# --------------------------------------------------------------------------- #
# 3. the two concurrency knobs are reconciled
# --------------------------------------------------------------------------- #


class _ModelWithLimit:
    def __init__(self, max_workers):
        self.max_workers = max_workers


def test_batch_size_defaults_to_the_strictest_provider_limit():
    metrics = [
        FakeMetric("a", {"input"}),
        FakeMetric("b", {"input"}),
        FakeMetric("c", {"input"}),
    ]
    metrics[0].model = _ModelWithLimit(6)
    metrics[1].model = _ModelWithLimit(2)
    assert _resolve_batch_size(metrics, None) == 2


def test_batch_size_falls_back_when_no_provider_declares_a_limit():
    assert _resolve_batch_size([FakeMetric("a", {"input"})], None) == 5


def test_explicit_batch_size_wins():
    m = FakeMetric("a", {"input"})
    m.model = _ModelWithLimit(2)
    assert _resolve_batch_size([m], 9) == 9


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        _resolve_batch_size([FakeMetric("a", {"input"})], 0)


# --------------------------------------------------------------------------- #
# 4. rate-limit backoff
# --------------------------------------------------------------------------- #


class _RateLimited(Exception):
    status_code = 429


@pytest.mark.parametrize(
    "exc",
    [
        _RateLimited("slow down"),
        type("RateLimitError", (Exception,), {})("nope"),
        Exception("Error code: 429 - Too Many Requests"),
        Exception("You exceeded your rate limit"),
    ],
)
def test_rate_limit_errors_are_detected(exc):
    assert retry.is_rate_limit_error(exc)


@pytest.mark.parametrize(
    "exc", [ValueError("bad input"), PermissionError("401 Unauthorized")]
)
def test_other_errors_are_not_treated_as_rate_limits(exc):
    assert not retry.is_rate_limit_error(exc)


def test_retry_succeeds_after_a_rate_limit(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _RateLimited("429")
        return "ok"

    assert retry.with_rate_limit_retry(flaky, initial_delay=0) == "ok"
    assert len(calls) == 3


def test_retry_gives_up_and_reraises(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)

    def always():
        raise _RateLimited("429")

    with pytest.raises(_RateLimited):
        retry.with_rate_limit_retry(always, max_retries=2, initial_delay=0)


def test_non_rate_limit_errors_are_not_retried():
    calls = []

    def boom():
        calls.append(1)
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        retry.with_rate_limit_retry(boom)
    assert len(calls) == 1, "a non-429 must not be retried"


def test_async_retry_succeeds_after_a_rate_limit(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(retry.asyncio, "sleep", no_sleep)
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise _RateLimited("429")
        return "ok"

    result = asyncio.run(retry.a_with_rate_limit_retry(flaky, initial_delay=0))
    assert result == "ok" and len(calls) == 2


def test_retry_after_header_is_honoured():
    class _WithHeader(Exception):
        status_code = 429
        headers = {"retry-after": "7"}

    assert retry.retry_after_seconds(_WithHeader()) == 7.0
    assert retry.retry_after_seconds(ValueError("no headers")) is None


def test_retry_after_is_capped_at_max_delay():
    class _Huge(Exception):
        status_code = 429
        headers = {"retry-after": "99999"}

    assert retry._next_delay(_Huge(), 0, 1.0, 2.0, 60.0) == 60.0
