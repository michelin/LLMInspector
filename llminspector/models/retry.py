"""Rate-limit backoff for provider calls.

A tool whose job is hammering an LLM endpoint in a loop will hit HTTP 429. There
was no retry anywhere in the package: a rate-limited call surfaced as a metric
exception, was swallowed by the metric's error path, and became a ``None`` score
indistinguishable from *skipped for missing input*.

Provider-agnostic on purpose — the detection below works off status codes and
message text rather than importing any vendor's exception classes, so a new
provider gets backoff without touching this module.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_BACKOFF_FACTOR = 2.0

_RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "too many requests", "429")


def is_rate_limit_error(exc: BaseException) -> bool:
    """Best-effort detection of a rate-limit / 429 response.

    Checks, in order: an explicit ``429`` on the usual status attributes, a
    class name containing ``RateLimit`` (openai, anthropic, httpx wrappers all
    follow this), then the message text.
    """
    for attribute in ("status_code", "status", "http_status", "code"):
        value = getattr(exc, attribute, None)
        if value == 429 or str(value) == "429":
            return True
    if "ratelimit" in type(exc).__name__.replace("_", "").lower():
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _RATE_LIMIT_MARKERS)


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """The server's ``Retry-After`` hint, when the exception carries one."""
    headers = getattr(exc, "headers", None) or getattr(
        getattr(exc, "response", None), "headers", None
    )
    if not headers:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
        return float(value) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _next_delay(
    exc: BaseException, attempt: int, initial: float, factor: float, maximum: float
) -> float:
    """Server hint if given, else exponential backoff with full jitter."""
    hint = retry_after_seconds(exc)
    if hint is not None:
        return min(hint, maximum)
    ceiling = min(initial * (factor**attempt), maximum)
    return random.uniform(0, ceiling)


def with_rate_limit_retry(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    **kwargs: Any,
) -> T:
    """Call ``func``, retrying rate-limit failures with backoff.

    Anything that is not a rate-limit error propagates immediately — this is
    not a general-purpose retry.
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised unless rate-limited
            if attempt == max_retries or not is_rate_limit_error(exc):
                raise
            delay = _next_delay(exc, attempt, initial_delay, backoff_factor, max_delay)
            logger.warning(
                "Rate limited (attempt %d/%d); retrying in %.1fs: %s",
                attempt + 1,
                max_retries,
                delay,
                exc,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


async def a_with_rate_limit_retry(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    **kwargs: Any,
) -> T:
    """Async form of :func:`with_rate_limit_retry`."""
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised unless rate-limited
            if attempt == max_retries or not is_rate_limit_error(exc):
                raise
            delay = _next_delay(exc, attempt, initial_delay, backoff_factor, max_delay)
            logger.warning(
                "Rate limited (attempt %d/%d); retrying in %.1fs: %s",
                attempt + 1,
                max_retries,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
