"""``GoldenSource`` — where a generation run gets its raw goldens.

A source is the *front* of the pipeline: it produces goldens from something
(a curated bank, a list of contexts, a directory of documents, a styling
description, an existing golden set). Everything after that is a
:class:`~llminspector.generation.stage.Stage`.

The split matters because sources are what differ between use cases and stages
are what they share. Adding a way to seed a run means writing one
``GoldenSource``; the five stages that filter, evolve, style and answer it are
inherited unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, List, Tuple, TypeVar

from ..dataset.golden import Golden, _BaseGolden

if TYPE_CHECKING:  # pragma: no cover
    from .config import GenerationConfig

__all__ = ["GoldenSource", "GoldenT"]

#: Every pipeline type is generic over the golden shape. Today the only shape is
#: :class:`~llminspector.dataset.golden.Golden`; the multi-turn
#: ``ConversationalGolden`` slots in here without the ``Generator``, the stage
#: chain, or the export path changing at all. That is the whole reason this
#: TypeVar exists rather than a hardcoded ``Golden``.
GoldenT = TypeVar("GoldenT", bound=_BaseGolden)


class GoldenSource(ABC, Generic[GoldenT]):
    """Produces the initial goldens for a generation run.

    Async because every non-trivial source is I/O bound — an LLM call per
    context, an embedding call per document. There is one implementation per
    source and it is the async one; a source with nothing to await simply
    returns, which costs nothing.
    """

    #: Keys this source puts in ``Golden.metadata``, in export order. Declared
    #: so the output column set is knowable **without running the source** —
    #: which, for an LLM-backed pipeline, means without paying for a run.
    metadata_keys: Tuple[str, ...] = ()

    @abstractmethod
    async def a_produce(self, config: "GenerationConfig") -> List[GoldenT]:
        """Produce the run's starting goldens."""


class SyncGoldenSource(GoldenSource[Golden]):
    """A source whose work is purely local — no I/O, nothing to await.

    Subclasses implement :meth:`produce`. The curated attack bank is the case
    this exists for: it is a pandas filter over an in-memory frame, and forcing
    it to be written as a coroutine would be theatre.
    """

    @abstractmethod
    def produce(self, config: "GenerationConfig") -> List[Golden]:
        """Produce the run's starting goldens synchronously."""

    async def a_produce(self, config: "GenerationConfig") -> List[Golden]:
        return self.produce(config)
