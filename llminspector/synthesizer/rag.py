"""``RagSynthesizer`` — RAG testset generation + evaluate()-backed scoring.

Today's default backend is :class:`RagasTestsetBackend`. Inject a different
:class:`~llminspector.synthesizer.engines.base.TestsetBackend` (e.g. the planned
ragas-free custom generator) to change generation without touching this class.

The legacy ``RagEval`` had a never-set ``self.test_df`` gap and referenced
``rag_evaluation()`` / ``export_eval()`` that were never implemented. Here
``generate()`` always stores its output, and RAG *scoring* is rebuilt as thin
wrappers over the Phase 4 :func:`~llminspector.evaluate.evaluate` engine.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..dataset.dataset import EvaluationDataset
from .base import BaseSynthesizer
from .engines import RagasTestsetBackend, TestsetBackend


class RagSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        *,
        model: Any = None,
        embedding: Any = None,
        documents: Optional[list] = None,
        document_dir: Optional[str] = None,
        test_size: int = 10,
        refine_prompt: Optional[str] = None,
        backend: Optional[TestsetBackend] = None,
    ) -> None:
        super().__init__()
        if backend is None:
            if model is None or embedding is None:
                raise ValueError(
                    "Provide a `backend`, or both `model` and `embedding`."
                )
            kwargs = dict(
                model=model,
                embedding=embedding,
                documents=documents,
                document_dir=document_dir,
                test_size=test_size,
            )
            if refine_prompt is not None:
                kwargs["refine_prompt"] = refine_prompt
            backend = RagasTestsetBackend(**kwargs)
        self.backend = backend

    def generate(self) -> EvaluationDataset:
        # Always stored (fixes the legacy never-set self.test_df gap).
        return self._store(self.backend.generate())

    def rag_evaluation(
        self,
        dataset: EvaluationDataset,
        metrics: Sequence[Any],
        **evaluate_kwargs,
    ):
        """Score an answered RAG dataset via the Phase 4 evaluate engine.

        ``dataset`` should carry test cases whose ``actual_output`` was produced
        by the RAG system under test.
        """
        from ..evaluate import evaluate

        return evaluate(dataset, metrics, **evaluate_kwargs)

    def export_eval(self, result, path: str) -> None:
        """Write an evaluation result to ``path`` (thin wrapper)."""
        result.to_excel(path)
