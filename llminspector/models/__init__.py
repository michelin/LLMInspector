"""Model provider abstraction — ``BaseLLM`` / Azure OpenAI (Phase 2)."""

from .azure_openai import AzureOpenAIEmbedding, AzureOpenAIModel
from .base_model import BaseEmbeddingModel, BaseLLM
from .errors import StructuredOutputError
from .metered import MeteredModel

__all__ = [
    "BaseLLM",
    "BaseEmbeddingModel",
    "AzureOpenAIModel",
    "AzureOpenAIEmbedding",
    "StructuredOutputError",
    "MeteredModel",
]
