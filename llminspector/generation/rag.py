"""``RagGenerator`` — RAG testset generation from documents, via ragas.

A :class:`~llminspector.generation.generator.Generator` preset over
:class:`~llminspector.generation.sources.ragas_testset.RagasTestsetBackend`.

**Transitional.** This is the last ragas-dependent piece of the generation
layer; the document ingestion, chunking and context-construction pipeline being
built in the later phases replaces it with a source that needs no optional
extra. When that lands, this module and its backend go.

Score a generated dataset the same way as any other::

    from llminspector import a_evaluate

    result = await gen.a_generate()
    dataset = EvaluationDataset(goldens=result.goldens)
    # ...run your RAG system to fill in the answers...
    scored = await a_evaluate(
        EvaluationDataset(test_cases=dataset.to_test_cases(answers)), metrics
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .generator import Generator
from .sources.ragas_testset import RagasTestsetBackend

if TYPE_CHECKING:  # pragma: no cover
    from .config import GenerationConfig

__all__ = ["RagGenerator"]


class RagGenerator(Generator):
    """A generator over a ragas testset backend; see :meth:`from_documents`."""

    def __init__(
        self,
        source: RagasTestsetBackend,
        *,
        config: Optional["GenerationConfig"] = None,
    ) -> None:
        # No stages: ragas already filters and evolves internally, so layering
        # our own chain on top would double-process every golden.
        super().__init__(source, stages=(), config=config)

    @classmethod
    def from_documents(
        cls,
        model: Any,
        embedding: Any,
        *,
        documents: Optional[list] = None,
        document_dir: Optional[str] = None,
        test_size: int = 10,
        refine_prompt: Optional[str] = None,
        config: Optional["GenerationConfig"] = None,
    ) -> "RagGenerator":
        """Build the default backend from pre-loaded ``documents`` or a directory."""
        kwargs = {
            "model": model,
            "embedding": embedding,
            "documents": documents,
            "document_dir": document_dir,
            "test_size": test_size,
        }
        if refine_prompt is not None:
            kwargs["refine_prompt"] = refine_prompt
        return cls(RagasTestsetBackend(**kwargs), config=config)
