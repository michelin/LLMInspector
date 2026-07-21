"""Provider abstraction — the ABCs every concrete model implements.

``BaseLLM`` and ``BaseEmbeddingModel`` decouple the metrics / evaluate / synth
layers from any single vendor. Azure OpenAI is the first (and, for now, only)
concrete provider (see :mod:`llminspector.models.azure_openai`); OpenAI /
Anthropic implementations are future work.

The legacy code reached ragas via ``LangchainLLMWrapper`` /
``LangchainEmbeddingsWrapper``. Those wrappers (and the ragas ``RunConfig``)
are what the metric layer consumes, so concrete providers expose them via
:meth:`BaseLLM.ragas_llm` / :meth:`BaseEmbeddingModel.ragas_embeddings`.
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

    @abstractmethod
    def ragas_llm(self) -> Any:
        """Return the ragas-compatible LLM wrapper for this model."""


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

    @abstractmethod
    def ragas_embeddings(self) -> Any:
        """Return the ragas-compatible embeddings wrapper for this model."""
