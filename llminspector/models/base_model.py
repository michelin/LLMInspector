"""Provider abstraction — the ABCs every concrete model implements.

``BaseLLM`` and ``BaseEmbeddingModel`` decouple the metrics / evaluate / synth
layers from any single vendor. Azure OpenAI is the first (and, for now, only)
concrete provider (see :mod:`llminspector.models.azure_openai`); OpenAI /
Anthropic implementations are future work.

The required contract is deliberately small: ``get_model_name`` plus
``generate`` / ``a_generate``. Every metric except the five ragas-backed context
metrics, and every synthesizer engine except the ragas testset backend, needs
nothing more — so a new provider is a class with three methods.

Ragas support is an **optional capability**, not part of the required contract.
:meth:`BaseLLM.ragas_llm` / :meth:`BaseEmbeddingModel.ragas_embeddings` are
concrete and raise :class:`NotImplementedError` by default; providers that can
supply the ragas wrappers override them, and
:class:`~llminspector.metrics.base_metric.RagasBackedMetric` is the only thing
that asks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class BaseLLM(ABC):
    """Abstract chat/completions model."""

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the human-readable model name (e.g. ``"gpt-5-mini"``)."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Synchronously generate a completion for ``prompt``."""

    @abstractmethod
    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate a completion for ``prompt``."""

    # -- optional capability ---------------------------------------------------

    def ragas_llm(self) -> Any:
        """Return the ragas-compatible LLM wrapper for this model.

        Optional: only :class:`~llminspector.metrics.base_metric.RagasBackedMetric`
        and the ragas testset backend call it. Providers that cannot supply one
        simply do not override it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a ragas LLM wrapper. "
            "Ragas-backed metrics and the ragas testset backend are unavailable "
            "with this provider."
        )


class BaseEmbeddingModel(ABC):
    """Abstract text-embedding model."""

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the human-readable embedding model name."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single string into a vector."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings into vectors."""

    # -- optional capability ---------------------------------------------------

    def ragas_embeddings(self) -> Any:
        """Return the ragas-compatible embeddings wrapper for this model.

        Optional, for the same reason as :meth:`BaseLLM.ragas_llm`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a ragas embeddings wrapper."
        )
