"""``ContextSource`` — goldens grounded in contexts the caller supplies.

The simplest grounded source: hand it ``List[List[str]]`` and it writes inputs
answerable from each one. No document loading, no chunking, no embeddings — so
it runs on a core install with no optional extra, and it is the source the
document pipeline reuses once it has built its contexts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

from ...dataset.golden import Golden
from ...utils.concurrency import a_map
from ..source import GoldenSource
from ..stages.generate import generate_inputs

if TYPE_CHECKING:  # pragma: no cover
    from ..config import GenerationConfig, StylingConfig

__all__ = ["ContextSource"]


class ContextSource(GoldenSource[Golden]):
    """Generates grounded inputs from caller-supplied context chunks.

    Parameters
    ----------
    contexts:
        One entry per context; each is a list of chunks that belong together.
        A bare string is accepted and treated as a one-chunk context.
    max_goldens_per_context:
        How many inputs to ask for per context.
    styling:
        Optional voice/shape guidance folded into the generation prompt.
    """

    #: ``context_size`` is promoted flat so an export can show how much source
    #: material each golden was grounded in without parsing the lineage.
    metadata_keys = ("lineage", "context_size")

    def __init__(
        self,
        contexts: Sequence[Sequence[str]],
        max_goldens_per_context: int = 2,
        *,
        styling: Optional["StylingConfig"] = None,
    ) -> None:
        if max_goldens_per_context < 1:
            raise ValueError(
                "max_goldens_per_context must be >= 1, got "
                f"{max_goldens_per_context}"
            )
        # Normalise up front so a caller passing ["a", "b"] gets two contexts of
        # one chunk rather than one context whose chunks are single characters —
        # the failure mode of iterating a bare string.
        self.contexts: List[List[str]] = [
            [ctx] if isinstance(ctx, str) else [str(c) for c in ctx] for ctx in contexts
        ]
        self.max_goldens_per_context = max_goldens_per_context
        self.styling = styling
        #: Per-context generation failures from the last :meth:`a_produce`.
        #: Initialised here rather than assigned mid-run so the attribute always
        #: exists — a caller checking it before a run gets an empty list, not an
        #: ``AttributeError``.
        self.errors: List[dict] = []

    async def a_produce(self, config: "GenerationConfig") -> List[Golden]:
        """One model call per context, bounded by ``config.max_concurrent``.

        Contexts are independent, so this is the outer of the pipeline's two
        concurrency levels; the generator's stage chain over the resulting
        goldens is the inner one. Both share the same bound, so the ceiling
        describes the whole run rather than one layer of it.
        """
        if not self.contexts:
            return []

        async def _for_context(context: List[str]) -> List[Golden]:
            inputs = await generate_inputs(
                config.model,
                context,
                self.max_goldens_per_context,
                self.styling,
            )
            return [
                Golden(
                    input=text,
                    context=list(context),
                    metadata={
                        "context_size": len(context),
                        "lineage": [
                            {"stage": "generate", "context_chunks": len(context)}
                        ],
                    },
                )
                for text in inputs
            ]

        batches, errors = await a_map(
            self.contexts,
            _for_context,
            limit=config.max_concurrent,
            desc="Generating inputs" if config.show_progress else None,
        )
        # A context whose generation call failed contributes no goldens. The
        # failure is not swallowed: a_map logs it, and the surviving contexts
        # still produce a usable run rather than one bad context aborting
        # everything upstream of the stage chain.
        self.errors = errors
        return [golden for batch in batches if batch for golden in batch]
