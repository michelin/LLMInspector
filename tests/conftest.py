"""Shared provider stubs for the multi-call generation pipeline tests.

Until now the suite had no ``conftest.py``: every module defined its own
single-response stub (``StubLLM`` in test_provider_contract.py, ``StubModel`` in
test_ragas_backend.py, …), each of which answers *every* prompt with the same
canned string. That is fine for a metric, which makes one call per test case.

It cannot express a generation pipeline, which makes 4-6 calls per record with a
different prompt template each time. Asserting anything about such a pipeline
needs a stub that (a) returns a *different* response per call, (b) records the
prompts in call order, and (c) fails loudly on an unscripted call. Hence
``ScriptedLLM``. ``FakeEmbedding`` closes the matching gap on the embedding side:
every pre-existing embedding stub returns a constant vector, so nothing in the
suite has ever exercised similarity.

Tests may import the classes directly (``from tests.conftest import ScriptedLLM``
— ``tests`` is a package, so the bare ``conftest`` name is not importable) or
take the thin factory fixtures at the bottom of this module. The classes are the
real API; the fixtures are sugar.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np
import pytest

from llminspector.models.base_model import BaseEmbeddingModel, BaseLLM

# How much of an offending prompt to quote back in the exhaustion error. Long
# enough to identify which template ran, short enough not to bury the assertion.
_PROMPT_EXCERPT = 160


def _truncate(text: str, limit: int = _PROMPT_EXCERPT) -> str:
    """Return ``text`` clipped to ``limit`` characters with an ellipsis marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


class ScriptedLLM(BaseLLM):
    """A provider that answers a *sequence* of prompts, one response each.

    Implements only the three required ``BaseLLM`` methods — no ``.client``, no
    ``ragas_llm`` override, no third-party imports — so any test driven by this
    stub also proves the provider contract stayed at three methods (the same
    guarantee ``tests/test_provider_contract.py::StubLLM`` gives for metrics).

    Two modes:

    * **queue** — ``responses`` is a sequence, consumed strictly in order.
    * **dispatch** — ``responses`` is a callable ``(prompt) -> str``, so a test
      that does not care about call *order* can key off the prompt instead.
    """

    def __init__(
        self,
        responses: Sequence[str] | Callable[[str], str],
        *,
        name: str = "scripted",
    ) -> None:
        self._name = name
        self._dispatch: Callable[[str], str] | None = None
        self._script: List[str] = []
        if callable(responses):
            self._dispatch = responses
        else:
            # Copy: the caller's sequence must survive ``reset()`` unmutated.
            self._script = list(responses)
        self._queue: List[str] = list(self._script)
        self.prompts: List[str] = []

    # -- BaseLLM contract ------------------------------------------------------

    def get_model_name(self) -> str:
        return self._name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self._next(prompt)

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        # Sync and async deliberately share ``_next`` so the two paths can never
        # drift: a pipeline switched from generate to a_generate must see the
        # identical script, and the recorded prompt order must stay comparable.
        return self._next(prompt)

    # -- test inspection -------------------------------------------------------

    @property
    def calls(self) -> int:
        """Number of generate/a_generate calls made so far."""
        return len(self.prompts)

    def reset(self) -> None:
        """Restore the unconsumed script and forget recorded prompts.

        Lets one instance be reused across parametrised cases without rebuilding
        the (often long) response script.
        """
        self._queue = list(self._script)
        self.prompts = []

    # -- internals -------------------------------------------------------------

    def _next(self, prompt: str) -> str:
        index = len(self.prompts)
        self.prompts.append(prompt)
        if self._dispatch is not None:
            return self._dispatch(prompt)
        if not self._queue:
            # Loud failure on purpose. The whole point of these tests is that a
            # pipeline makes a *fixed* number of calls per golden; silently
            # replaying the last response would turn "made one call too many"
            # into a passing test with quietly wrong output.
            raise AssertionError(
                f"ScriptedLLM({self._name!r}) exhausted: call index {index} had "
                f"no scripted response (script length {len(self._script)}). "
                f"Unanswered prompt: {_truncate(prompt)!r}"
            )
        return self._queue.pop(0)


def _first_line_kind(prompt: str, limit: int = 48) -> str:
    """Classify a prompt by its first non-empty line, lowercased and clipped.

    A stand-in for "which prompt template produced this". Real templates start
    with a distinctive instruction line, so this separates them without any test
    having to know the template text verbatim.
    """
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.lower()[:limit]
    return ""


class MeteredScriptedLLM(ScriptedLLM):
    """``ScriptedLLM`` that also tallies calls per prompt kind.

    Lets a test assert "the pipeline makes N calls of kind K per golden" without
    reaching into ``prompts`` and re-deriving the grouping each time.
    """

    def __init__(
        self,
        responses: Sequence[str] | Callable[[str], str],
        *,
        name: str = "metered",
        classifier: Callable[[str], str] = _first_line_kind,
    ) -> None:
        super().__init__(responses, name=name)
        self._classifier = classifier
        self.counts: Dict[str, int] = Counter()

    def _next(self, prompt: str) -> str:
        # Count before delegating: an exhausted script still tells the test which
        # kind of prompt overran its budget.
        self.counts[self._classifier(prompt)] += 1
        return super()._next(prompt)


class FakeEmbedding(BaseEmbeddingModel):
    """Deterministic embeddings that actually *distinguish* their inputs.

    Every pre-existing embedding stub in this suite returns a constant vector,
    so no test has ever exercised similarity — a chunker that returned the same
    chunk twice, or a de-duplicator that dropped a distinct one, would pass.
    This fake gives each distinct string its own pseudo-random unit vector:
    identical texts embed identically, different texts land far apart.

    The seed comes from ``hashlib.blake2b``, **not** the builtin ``hash()``:
    Python randomises string hashing per process (PYTHONHASHSEED), which would
    make the vectors — and therefore any similarity threshold asserted against
    them — differ between runs.
    """

    def __init__(self, dim: int = 16, *, name: str = "fake-embedding") -> None:
        self.dim = dim
        self._name = name
        self.texts: List[str] = []
        # Calls, not texts: lets a test assert batching ("one embed_texts call
        # per document, not one per chunk").
        self.embed_calls = 0

    # -- BaseEmbeddingModel contract -------------------------------------------

    def get_model_name(self) -> str:
        return self._name

    def embed_text(self, text: str) -> List[float]:
        self.embed_calls += 1
        self.texts.append(text)
        return self._vector(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self.embed_calls += 1
        self.texts.extend(texts)
        return [self._vector(text) for text in texts]

    # -- async convenience -----------------------------------------------------
    #
    # ``BaseEmbeddingModel`` now supplies these, defaulting to ``asyncio.to_thread``
    # around the sync call. They are overridden here to answer directly: a thread
    # hop per embed buys nothing for an in-memory fake and the suite has to stay
    # fast. The ABC's default is exercised on a bare stub in
    # ``tests/test_structured_output.py``, which is where it belongs — testing it
    # through a fake that overrides it would prove nothing.

    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

    # -- internals -------------------------------------------------------------

    def _vector(self, text: str) -> List[float]:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "big")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dim)
        vector /= np.linalg.norm(vector)
        return [float(value) for value in vector]


# --------------------------------------------------------------------------- #
# fixtures — thin factories over the classes above
# --------------------------------------------------------------------------- #


@pytest.fixture
def scripted_llm() -> Callable[..., ScriptedLLM]:
    """Factory: ``scripted_llm(["a", "b"])`` -> a fresh :class:`ScriptedLLM`."""

    def _make(
        responses: Iterable[str] | Callable[[str], str],
        **kwargs: Any,
    ) -> ScriptedLLM:
        if not callable(responses):
            responses = list(responses)
        return ScriptedLLM(responses, **kwargs)

    return _make


@pytest.fixture
def fake_embedding() -> Callable[..., FakeEmbedding]:
    """Factory: ``fake_embedding(dim=8)`` -> a fresh :class:`FakeEmbedding`."""

    def _make(dim: int = 16, **kwargs: Any) -> FakeEmbedding:
        return FakeEmbedding(dim, **kwargs)

    return _make
