"""``DocumentSource`` — goldens grounded in a corpus of files.

Builds contexts from documents and then delegates to **the same stage chain** as
:class:`~llminspector.generation.sources.contexts.ContextSource`. Document
handling is a way of *obtaining* contexts, not a different kind of generation,
so nothing downstream of the source knows the difference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

from ...dataset.golden import Golden
from ...utils.concurrency import a_map
from ..config import ContextConfig
from ..context.loaders import load_documents
from ..context.selection import build_contexts
from ..source import GoldenSource
from ..stages.generate import generate_inputs

if TYPE_CHECKING:  # pragma: no cover
    from ..config import GenerationConfig, StylingConfig

__all__ = ["DocumentSource"]


class DocumentSource(GoldenSource[Golden]):
    """Loads documents, builds contexts from them, then generates.

    Parameters
    ----------
    paths / directory:
        Files to load, or a directory to walk. Supply one.
    context_config:
        Chunking, retrieval and assembly settings.
    max_goldens_per_context:
        Inputs to request per assembled context.
    styling:
        Optional voice/shape guidance folded into the generation prompt.
    """

    metadata_keys = (
        "lineage",
        "context_size",
        "context_score",
        "source_file",
        "context_source_files",
    )

    def __init__(
        self,
        paths: Optional[Sequence[str] | str] = None,
        directory: Optional[str] = None,
        *,
        context_config: Optional[ContextConfig] = None,
        max_goldens_per_context: int = 2,
        styling: Optional["StylingConfig"] = None,
    ) -> None:
        if max_goldens_per_context < 1:
            raise ValueError(
                f"max_goldens_per_context must be >= 1, got {max_goldens_per_context}"
            )
        self.paths = paths
        self.directory = directory
        self.context_config = context_config or ContextConfig()
        self.max_goldens_per_context = max_goldens_per_context
        self.styling = styling
        self.errors: List[dict] = []
        #: The contexts built by the last run, for inspection without re-paying.
        self.contexts: list = []

    async def a_produce(self, config: "GenerationConfig") -> List[Golden]:
        """Load, build contexts, then generate one batch of inputs per context.

        Loading and validation happen before any embedding or model call, so a
        misconfigured run fails in milliseconds rather than after a corpus has
        been embedded.
        """
        if config.embedding is None:
            raise ValueError(
                "DocumentSource needs an embedding provider. Pass one as "
                "GenerationConfig(embedding=...)."
            )

        documents = load_documents(self.paths, self.directory)
        self.contexts = await build_contexts(documents, config, self.context_config)

        async def _for_context(context) -> List[Golden]:
            inputs = await generate_inputs(
                config.model,
                context.chunks,
                self.max_goldens_per_context,
                self.styling,
            )
            return [
                Golden(
                    input=text,
                    context=list(context.chunks),
                    metadata={
                        "context_size": len(context.chunks),
                        "context_score": round(context.score, 4),
                        # Scalar for the common single-source case; the full list
                        # is kept alongside so a merged context is not misread as
                        # coming from one file.
                        "source_file": (
                            context.source_files[0] if context.source_files else None
                        ),
                        "context_source_files": list(context.source_files),
                        "lineage": [
                            {
                                "stage": "generate",
                                "context_chunks": len(context.chunks),
                                "sources": len(context.source_files),
                            }
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
        self.errors = errors
        return [golden for batch in batches if batch for golden in batch]
