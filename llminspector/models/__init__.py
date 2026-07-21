"""Model provider abstraction — ``BaseLLM`` / Azure OpenAI (Phase 2)."""

from .azure_openai import AzureOpenAIEmbedding, AzureOpenAIModel
from .base_model import BaseEmbeddingModel, BaseLLM

__all__ = [
    "BaseLLM",
    "BaseEmbeddingModel",
    "AzureOpenAIModel",
    "AzureOpenAIEmbedding",
]
