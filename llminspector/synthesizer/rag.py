"""``RagSynthesizer`` — RAG testset generation from documents.

Today's default backend is :class:`RagasTestsetBackend`. Inject a different
:class:`~llminspector.synthesizer.engines.base.TestsetBackend` (e.g. the planned
ragas-free custom generator) to change generation without touching this class.

The legacy ``RagEval`` had a never-set ``self.test_df`` gap; here ``generate()``
always stores its output.

Phase 8.6 removed ``rag_evaluation()`` and ``export_eval()``. They forwarded to
:func:`~llminspector.evaluate.evaluate` and ``result.to_excel()`` and added
nothing — they existed for name-compatibility with legacy methods that were
never implemented. Score a generated dataset the same way as any other::

    from llminspector import a_evaluate

    goldens = synth.generate()
    # ...run your RAG system to fill in actual_output...
    result = await a_evaluate(answered_dataset, metrics)
    result.to_excel("rag_eval.xlsx")
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ..dataset.dataset import EvaluationDataset
from .base import BaseSynthesizer
from .engines import RagasTestsetBackend, TestsetBackend


class RagSynthesizer(BaseSynthesizer):
    """Wraps a :class:`TestsetBackend`; see :meth:`from_documents`."""

    def __init__(self, backend: TestsetBackend) -> None:
        super().__init__()
        self.backend = backend

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
    ) -> "RagSynthesizer":
        """Build the default :class:`RagasTestsetBackend`.

        Supply either pre-loaded ``documents`` or a ``document_dir`` to load.
        """
        kwargs = {
            "model": model,
            "embedding": embedding,
            "documents": documents,
            "document_dir": document_dir,
            "test_size": test_size,
        }
        if refine_prompt is not None:
            kwargs["refine_prompt"] = refine_prompt
        return cls(RagasTestsetBackend(**kwargs))

    @property
    def metadata_keys(self) -> Tuple[str, ...]:
        """Metadata columns the backend emits on every golden."""
        return self.backend.metadata_keys

    def generate(self) -> EvaluationDataset:
        # Always stored (fixes the legacy never-set self.test_df gap).
        return self._store(self.backend.generate())
