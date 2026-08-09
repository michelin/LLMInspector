"""``Stage`` — one transformation applied to every golden in a run.

The pipeline is a source followed by an ordered list of stages. Each stage takes
one golden and returns a golden (possibly the same one, possibly rewritten), or
``None`` to discard it.

Stages are deliberately independent of each other: the chain order lives in the
:class:`~llminspector.generation.generator.Generator`, not in the stages, so
reordering the chain — or dropping a stage for a source that cannot support it —
is a caller's decision rather than a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple

from .source import GoldenT

if TYPE_CHECKING:  # pragma: no cover
    from .config import GenerationConfig

__all__ = ["Stage", "StageContext"]


@dataclass
class StageContext:
    """Everything a stage needs beyond the golden itself.

    Passed rather than stored on the stage so one stage instance is safe to use
    across concurrent goldens — the same reason the metric layer clones per row.
    """

    config: "GenerationConfig"
    #: The context chunks the golden was grounded in, when it came from a
    #: grounded source. Empty for scratch generation.
    context: List[str] = field(default_factory=list)
    #: Files those chunks came from, for lineage.
    source_files: List[str] = field(default_factory=list)
    #: Reasons recorded by :meth:`reject`, drained by the generator.
    rejections: List[str] = field(default_factory=list)

    @property
    def model(self) -> Any:
        """The generating model."""
        return self.config.model

    @property
    def critic(self) -> Any:
        """The judging model — falls back to the generating model."""
        return self.config.critic

    def reject(self, reason: str) -> None:
        """Record why a golden is being discarded.

        A stage returning ``None`` says *that* a golden was dropped; this says
        *why*. The generator moves the reason onto
        ``GenerationResult.rejected`` so a run that silently halved its output
        can be explained without re-running it.
        """
        self.rejections.append(reason)


class Stage(ABC, Generic[GoldenT]):
    """One step of the generation chain."""

    #: Stage name, used in lineage records and error/rejection reports.
    name: str = ""

    #: Keys this stage adds to ``Golden.metadata``, in export order. The union
    #: across the source and every stage is the run's full column set, knowable
    #: before paying for a run. ``tests`` assert the declaration stays true.
    metadata_keys: Tuple[str, ...] = ()

    @abstractmethod
    async def a_apply(self, golden: GoldenT, ctx: StageContext) -> Optional[GoldenT]:
        """Transform ``golden``, or return ``None`` to discard it.

        Returning ``None`` **must** be paired with ``ctx.reject(reason)`` so the
        discard is explainable. Raising is also allowed — the generator records
        the error against the golden's slot and carries on — but a raise means
        "this stage broke", not "this golden did not make the cut".
        """

    def record(self, golden: GoldenT, **fields: Any) -> None:
        """Append one compact lineage record to ``golden.metadata["lineage"]``.

        Every stage that changes a golden calls this, so the provenance of a
        generated row is readable off the row itself rather than reconstructed
        from logs.
        """
        entry: Dict[str, Any] = {"stage": self.name}
        entry.update({k: v for k, v in fields.items() if v is not None})
        golden.metadata.setdefault("lineage", []).append(entry)
