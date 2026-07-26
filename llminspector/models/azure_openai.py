"""Azure OpenAI concrete provider — unifies **both** legacy auth styles.

The legacy package had two divergent setups:

* ``EvalMetrics.initialize_core_models`` (``eval_metrics.py``) — Azure AD
  ``azure_ad_token_provider`` auth, ``gpt-5-mini-dzs`` / ``gpt-5-mini``,
  ``bypass_temperature=True`` on the ragas wrapper.
* ``RagEval.initialize_models`` (``rag_eval.py``) — ``api_key`` auth,
  ``gpt-4o-mini``, plain ragas wrapper, ``timeout=120`` on the client.

Both wrapped :class:`AzureChatOpenAI` / :class:`AzureOpenAIEmbeddings` in the
ragas ``LangchainLLMWrapper`` / ``LangchainEmbeddingsWrapper`` with
``RunConfig(max_workers=6, timeout=120)``. This module collapses the two into
one class each, selecting auth by which credential is supplied (exactly one of
``api_key`` / ``azure_ad_token_provider``), and promoting the hard-coded model /
deployment names to constructor args (defaulted from :class:`AzureSettings`).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..config.settings import AzureSettings
from ..utils.optional import optional_dependency
from .base_model import BaseEmbeddingModel, BaseLLM
from .retry import (
    DEFAULT_MAX_RETRIES,
    a_with_rate_limit_retry,
    with_rate_limit_retry,
)

# ragas RunConfig values were identical in both legacy setups.
_DEFAULT_MAX_WORKERS = 6
_DEFAULT_TIMEOUT = 120


def _resolve_auth(
    api_key: Optional[str],
    azure_ad_token_provider: Optional[Callable[[], str]],
) -> None:
    """Validate that exactly one auth credential is supplied."""
    if bool(api_key) == bool(azure_ad_token_provider):
        raise ValueError(
            "Provide exactly one of `api_key` or `azure_ad_token_provider` "
            "(the two legacy auth styles are mutually exclusive)."
        )


def _run_config(
    max_workers: int = _DEFAULT_MAX_WORKERS, timeout: int = _DEFAULT_TIMEOUT
):
    with optional_dependency("ragas", extra="ragas", feature="RunConfig"):
        from ragas.run_config import RunConfig

    return RunConfig(max_workers=max_workers, timeout=timeout)


class AzureOpenAIModel(BaseLLM):
    """Azure OpenAI chat model.

    Supply connection details via an :class:`AzureSettings` instance (or explicit
    kwargs, which override the settings) plus exactly one credential:
    ``api_key`` **or** ``azure_ad_token_provider``.
    """

    def __init__(
        self,
        settings: Optional[AzureSettings] = None,
        *,
        api_key: Optional[str] = None,
        azure_ad_token_provider: Optional[Callable[[], str]] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        model_name: Optional[str] = None,
        bypass_temperature: bool = True,
        timeout: int = _DEFAULT_TIMEOUT,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        settings = settings or AzureSettings()
        api_key = api_key if api_key is not None else settings.api_key
        _resolve_auth(api_key, azure_ad_token_provider)

        self.azure_endpoint = azure_endpoint or settings.azure_endpoint
        self.api_version = api_version or settings.api_version
        self.azure_deployment = azure_deployment or settings.azure_deployment
        self.model_name = model_name or settings.model_name
        #: Concurrency ceiling for this endpoint. Feeds the ragas ``RunConfig``
        #: *and* ``evaluate()``'s default batch size, so the two knobs that used
        #: to drift apart now come from one number.
        self.max_workers = max_workers
        #: Rate-limit retries per call; 0 disables backoff.
        self.max_retries = max_retries
        self._api_key = api_key
        self._azure_ad_token_provider = azure_ad_token_provider
        self._bypass_temperature = bypass_temperature
        self._timeout = timeout

        self._client = self._build_client()
        # Built on first use, not here: ragas is an optional extra, and a
        # provider that eagerly wrapped itself would make `pip install
        # llminspector` unable to construct a model at all.
        self._ragas_llm: Any = None

    def _build_client(self):
        from langchain_openai import AzureChatOpenAI

        kwargs: dict = {
            "openai_api_version": self.api_version,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.azure_deployment,
            "model": self.model_name,
            "validate_base_url": False,
            "timeout": self._timeout,
        }
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        else:
            kwargs["azure_ad_token_provider"] = self._azure_ad_token_provider
        return AzureChatOpenAI(**kwargs)

    def _build_ragas_llm(self):
        with optional_dependency(
            "ragas", extra="ragas", feature="the ragas LLM wrapper"
        ):
            from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(
            self._client, bypass_temperature=self._bypass_temperature
        )

    # -- BaseLLM --------------------------------------------------------------

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        response = with_rate_limit_retry(
            self._client.invoke, prompt, max_retries=self.max_retries, **kwargs
        )
        return getattr(response, "content", response)

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        response = await a_with_rate_limit_retry(
            self._client.ainvoke, prompt, max_retries=self.max_retries, **kwargs
        )
        return getattr(response, "content", response)

    def ragas_llm(self) -> Any:
        """The ragas wrapper around this client, built on first use.

        Raises ``ImportError`` naming the ``llminspector[ragas]`` extra when
        ragas is absent — only the five ``RagasBackedMetric`` context metrics
        and the RAG testset engine ever reach here.
        """
        if self._ragas_llm is None:
            self._ragas_llm = self._build_ragas_llm()
        return self._ragas_llm

    @property
    def client(self):
        """The underlying ``AzureChatOpenAI`` langchain client.

        Azure-specific escape hatch — **not** part of the ``BaseLLM`` contract.
        """
        return self._client

    @property
    def run_config(self):
        """A fresh ragas ``RunConfig`` built from this model's limits."""
        return _run_config(max_workers=self.max_workers, timeout=self._timeout)


class AzureOpenAIEmbedding(BaseEmbeddingModel):
    """Azure OpenAI embedding model.

    Same unified auth contract as :class:`AzureOpenAIModel`. Defaults to the
    legacy ``text-embedding-ada-002`` deployment / model.
    """

    def __init__(
        self,
        settings: Optional[AzureSettings] = None,
        *,
        api_key: Optional[str] = None,
        azure_ad_token_provider: Optional[Callable[[], str]] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        embedding_deployment: Optional[str] = None,
        embedding_name: Optional[str] = None,
    ) -> None:
        settings = settings or AzureSettings()
        api_key = api_key if api_key is not None else settings.api_key
        _resolve_auth(api_key, azure_ad_token_provider)

        self.azure_endpoint = azure_endpoint or settings.azure_endpoint
        self.api_version = api_version or settings.api_version
        self.embedding_deployment = (
            embedding_deployment or settings.embedding_deployment
        )
        self.embedding_name = embedding_name or settings.embedding_name
        self._api_key = api_key
        self._azure_ad_token_provider = azure_ad_token_provider

        self._client = self._build_client()
        # Lazy for the same reason as AzureOpenAIModel._ragas_llm.
        self._ragas_embeddings: Any = None

    def _build_client(self):
        from langchain_openai import AzureOpenAIEmbeddings

        kwargs: dict = {
            "openai_api_version": self.api_version,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.embedding_deployment,
            "model": self.embedding_name,
        }
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        else:
            kwargs["azure_ad_token_provider"] = self._azure_ad_token_provider
        return AzureOpenAIEmbeddings(**kwargs)

    def _build_ragas_embeddings(self):
        with optional_dependency(
            "ragas", extra="ragas", feature="the ragas embeddings wrapper"
        ):
            from ragas.embeddings import LangchainEmbeddingsWrapper

        return LangchainEmbeddingsWrapper(self._client)

    # -- BaseEmbeddingModel ---------------------------------------------------

    def get_model_name(self) -> str:
        return self.embedding_name

    def embed_text(self, text: str) -> List[float]:
        return self._client.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)

    def ragas_embeddings(self) -> Any:
        """The ragas wrapper around this client, built on first use."""
        if self._ragas_embeddings is None:
            self._ragas_embeddings = self._build_ragas_embeddings()
        return self._ragas_embeddings

    @property
    def client(self):
        """The underlying ``AzureOpenAIEmbeddings`` langchain client."""
        return self._client
