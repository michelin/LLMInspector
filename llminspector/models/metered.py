"""``MeteredModel`` — token accounting for any provider.

``BaseLLM.generate`` returns ``str``, which throws away the ``usage_metadata``
the provider sent back. That is fine for a metric making one call per row; it is
not fine for a generation pipeline making four to six calls per record, where a
run can cost real money with no visibility at all.

This is a decorator, not a contract change. It wraps any :class:`BaseLLM`,
counts prompt and completion tokens with ``tiktoken`` (already core), and exposes
the totals. Provider-agnostic, accurate to roughly ±10%, and zero change to the
three-method contract.

Two alternatives were considered and rejected:

* **Changing ``generate``'s return type** to carry usage — breaks every metric
  in the package, and every provider anyone has written.
* **An out-of-band usage callback** — racy under concurrency without threading a
  context object through every call site, which is most of the cost of doing it
  properly with none of the benefit.

Revisit if the estimate proves too coarse; the honest fix then is to read the
provider's own usage numbers, which means a capability method on the ABC.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Type, TypeVar

from pydantic import BaseModel

from .base_model import BaseLLM

__all__ = ["MeteredModel"]

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)

#: Matches ``TokenCountMetric`` and ``TokenChunker`` so the package only ever
#: downloads one BPE table.
DEFAULT_ENCODING = "o200k_base"


@lru_cache(maxsize=4)
def _encoding(name: str) -> Any:
    import tiktoken

    try:
        return tiktoken.get_encoding(name)
    except (ValueError, KeyError):
        return tiktoken.get_encoding(DEFAULT_ENCODING)


class MeteredModel(BaseLLM):
    """Wraps a provider and counts the tokens flowing through it.

    ``MeteredModel(inner)`` is a drop-in replacement for ``inner`` — it is a
    :class:`BaseLLM` itself, so it can be handed to metrics, stages, or another
    decorator without anything noticing.
    """

    def __init__(self, inner: BaseLLM, *, encoding: str = DEFAULT_ENCODING) -> None:
        self.inner = inner
        self.encoding = encoding
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._instrument()

    def _instrument(self) -> None:
        """Replace the inner model's ``generate`` pair with counting versions.

        **This mutates the wrapped model**, and it has to. Counting from the
        outside — recording in ``MeteredModel.generate`` and delegating
        elsewhere — misses every call the provider makes to *itself*, and that
        is most of them: ``generate_structured`` is implemented in terms of
        ``self.generate``, so a wrapper that intercepts only its own surface
        counts zero for a pipeline that uses structured output for everything,
        and never sees a reask at all.

        The alternative was to reimplement structured output on the wrapper,
        which would drop the provider's native JSON mode the moment metering was
        switched on. A decorator that changes behaviour is not a decorator, so
        the mutation is the lesser cost. :meth:`unwrap` puts it back.
        """
        self._original_generate = self.inner.generate
        self._original_a_generate = self.inner.a_generate

        def metered_generate(prompt: str, **kwargs: Any) -> str:
            response = self._original_generate(prompt, **kwargs)
            self._record(prompt, response)
            return response

        async def metered_a_generate(prompt: str, **kwargs: Any) -> str:
            response = await self._original_a_generate(prompt, **kwargs)
            self._record(prompt, response)
            return response

        # Instance attributes shadow the class's bound methods, so the provider
        # calling ``self.generate`` internally reaches these.
        self.inner.generate = metered_generate  # type: ignore[method-assign]
        self.inner.a_generate = metered_a_generate  # type: ignore[method-assign]

    def unwrap(self) -> BaseLLM:
        """Restore the inner model's own methods and return it."""
        try:
            del self.inner.generate  # type: ignore[attr-defined]
            del self.inner.a_generate  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - already unwrapped
            pass
        return self.inner

    # -- BaseLLM contract ------------------------------------------------------
    #
    # These forward without recording: the counting happens inside the inner
    # model's instrumented methods, so recording here as well would double every
    # call made through the wrapper's own surface.

    def get_model_name(self) -> str:
        return self.inner.get_model_name()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self.inner.generate(prompt, **kwargs)

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        return await self.inner.a_generate(prompt, **kwargs)

    # -- capabilities are forwarded, not reimplemented --------------------------
    #
    # Delegating keeps the provider's own structured-output implementation —
    # Azure's native JSON mode survives metering — and because the inner model's
    # ``generate`` pair is instrumented, every underlying call it makes is
    # counted, reasks included.

    def generate_structured(
        self, prompt: str, schema: Type[BaseModelT], **kwargs: Any
    ) -> BaseModelT:
        return self.inner.generate_structured(prompt, schema, **kwargs)

    async def a_generate_structured(
        self, prompt: str, schema: Type[BaseModelT], **kwargs: Any
    ) -> BaseModelT:
        return await self.inner.a_generate_structured(prompt, schema, **kwargs)

    def ragas_llm(self) -> Any:
        return self.inner.ragas_llm()

    def __getattr__(self, name: str) -> Any:
        """Forward anything else — ``max_workers``, ``client``, provider extras.

        Only called for attributes this class does not define, so the metering
        state above is never shadowed. Without it, wrapping a model would hide
        ``max_workers`` and silently change ``evaluate()``'s batch size.
        """
        return getattr(self.inner, name)

    # -- accounting -------------------------------------------------------------

    def _record(self, prompt: str, response: Any) -> None:
        encoder = _encoding(self.encoding)
        self.calls += 1
        self.prompt_tokens += len(encoder.encode(prompt))
        # A provider returning a non-string (a langchain message that slipped
        # through) is counted on its ``str`` form rather than crashing a run
        # over accounting, which is the least important thing happening here.
        self.completion_tokens += len(encoder.encode(str(response)))

    @property
    def usage(self) -> Dict[str, int]:
        """``{calls, prompt_tokens, completion_tokens, total_tokens}``."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    def reset(self) -> None:
        """Zero the counters, for metering one phase of a longer session."""
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0


def total_usage(models: List[Any]) -> Dict[str, int]:
    """Sum the usage of several metered models, ignoring unmetered ones.

    A run may meter the generating model and the critic separately; the caller
    wants one number.
    """
    totals = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    seen: set = set()
    for model in models:
        usage = getattr(model, "usage", None)
        # Identity, not equality: when critic_model falls back to model they are
        # the same object and must not be counted twice.
        if not isinstance(usage, dict) or id(model) in seen:
            continue
        seen.add(id(model))
        for key in totals:
            totals[key] += usage.get(key, 0)
    return totals
