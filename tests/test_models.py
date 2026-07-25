"""Phase 2 — model provider abstraction.

Auth-selection + construction are tested without network by monkeypatching the
lazily-imported langchain client classes. A live smoke test runs only when real
Azure credentials are present in the environment.
"""

import asyncio
import os
import sys
import types

import pytest

from llminspector.config import AzureSettings
from llminspector.models import (
    AzureOpenAIEmbedding,
    AzureOpenAIModel,
    BaseEmbeddingModel,
    BaseLLM,
)

# --------------------------------------------------------------------------- #
# Fakes for the langchain / ragas classes the provider imports lazily.
# --------------------------------------------------------------------------- #


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChatClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, prompt, **kwargs):
        return _FakeMessage(f"echo: {prompt}")

    async def ainvoke(self, prompt, **kwargs):
        return _FakeMessage(f"aecho: {prompt}")


class _FakeEmbeddingsClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeLangchainLLMWrapper:
    def __init__(self, client, bypass_temperature=False):
        self.client = client
        self.bypass_temperature = bypass_temperature


class _FakeLangchainEmbeddingsWrapper:
    def __init__(self, client):
        self.client = client


@pytest.fixture
def patched_langchain(monkeypatch):
    """Stub the lazily-imported langchain_openai + ragas symbols."""
    lc = types.ModuleType("langchain_openai")
    lc.AzureChatOpenAI = _FakeChatClient
    lc.AzureOpenAIEmbeddings = _FakeEmbeddingsClient
    monkeypatch.setitem(sys.modules, "langchain_openai", lc)

    ragas_llms = types.ModuleType("ragas.llms")
    ragas_llms.LangchainLLMWrapper = _FakeLangchainLLMWrapper
    monkeypatch.setitem(sys.modules, "ragas.llms", ragas_llms)

    ragas_emb = types.ModuleType("ragas.embeddings")
    ragas_emb.LangchainEmbeddingsWrapper = _FakeLangchainEmbeddingsWrapper
    monkeypatch.setitem(sys.modules, "ragas.embeddings", ragas_emb)


@pytest.fixture
def settings():
    return AzureSettings(
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-02-01",
    )


# --------------------------------------------------------------------------- #
# Auth selection
# --------------------------------------------------------------------------- #


def test_api_key_auth_builds_client(patched_langchain, settings):
    model = AzureOpenAIModel(settings, api_key="secret")
    assert isinstance(model, BaseLLM)
    assert model.client.kwargs["api_key"] == "secret"
    assert "azure_ad_token_provider" not in model.client.kwargs


def test_token_provider_auth_builds_client(patched_langchain, settings):
    provider = lambda: "token"
    model = AzureOpenAIModel(settings, azure_ad_token_provider=provider)
    assert model.client.kwargs["azure_ad_token_provider"] is provider
    assert "api_key" not in model.client.kwargs


def test_both_credentials_raises(patched_langchain, settings):
    with pytest.raises(ValueError):
        AzureOpenAIModel(
            settings, api_key="secret", azure_ad_token_provider=lambda: "t"
        )


def test_no_credentials_raises(patched_langchain, settings):
    with pytest.raises(ValueError):
        AzureOpenAIModel(settings)


def test_api_key_from_settings(patched_langchain):
    settings = AzureSettings(
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-02-01",
        api_key="from-settings",
    )
    model = AzureOpenAIModel(settings)
    assert model.client.kwargs["api_key"] == "from-settings"


# --------------------------------------------------------------------------- #
# Model / deployment names + wrappers
# --------------------------------------------------------------------------- #


def test_default_model_names_promoted(patched_langchain, settings):
    model = AzureOpenAIModel(settings, api_key="k")
    assert model.get_model_name() == "gpt-5-mini"
    assert model.client.kwargs["azure_deployment"] == "gpt-5-mini-dzs"


def test_model_name_override(patched_langchain, settings):
    model = AzureOpenAIModel(
        settings,
        api_key="k",
        model_name="gpt-4o-mini",
        azure_deployment="gpt-4o-mini",
    )
    assert model.get_model_name() == "gpt-4o-mini"
    assert model.client.kwargs["azure_deployment"] == "gpt-4o-mini"


def test_ragas_llm_wrapper_bypass_temperature(patched_langchain, settings):
    model = AzureOpenAIModel(settings, api_key="k")
    assert model.ragas_llm().bypass_temperature is True


def test_run_config_matches_legacy(patched_langchain, settings):
    class _RC:
        def __init__(self, max_workers, timeout):
            self.max_workers = max_workers
            self.timeout = timeout

    rc_mod = types.ModuleType("ragas.run_config")
    rc_mod.RunConfig = _RC
    sys.modules["ragas.run_config"] = rc_mod
    try:
        model = AzureOpenAIModel(settings, api_key="k")
        rc = model.run_config
        assert (rc.max_workers, rc.timeout) == (6, 120)
    finally:
        del sys.modules["ragas.run_config"]


# --------------------------------------------------------------------------- #
# generate() / embed_text() return types
# --------------------------------------------------------------------------- #


def test_generate_returns_str(patched_langchain, settings):
    model = AzureOpenAIModel(settings, api_key="k")
    out = model.generate("hi")
    assert isinstance(out, str)
    assert out == "echo: hi"


def test_a_generate_returns_str(patched_langchain, settings):
    model = AzureOpenAIModel(settings, api_key="k")
    out = asyncio.run(model.a_generate("hi"))
    assert out == "aecho: hi"


def test_embedding_defaults_and_embed_text(patched_langchain, settings):
    emb = AzureOpenAIEmbedding(settings, api_key="k")
    assert isinstance(emb, BaseEmbeddingModel)
    assert emb.get_model_name() == "text-embedding-ada-002"
    vec = emb.embed_text("hello")
    assert isinstance(vec, list) and all(isinstance(x, float) for x in vec)


def test_embed_texts_batch(patched_langchain, settings):
    emb = AzureOpenAIEmbedding(settings, api_key="k")
    vecs = emb.embed_texts(["a", "b"])
    assert len(vecs) == 2 and all(isinstance(v, list) for v in vecs)


def test_embedding_token_provider_auth(patched_langchain, settings):
    provider = lambda: "token"
    emb = AzureOpenAIEmbedding(settings, azure_ad_token_provider=provider)
    assert emb.client.kwargs["azure_ad_token_provider"] is provider


def test_embedding_ragas_wrapper(patched_langchain, settings):
    emb = AzureOpenAIEmbedding(settings, api_key="k")
    assert emb.ragas_embeddings().client is emb.client


# --------------------------------------------------------------------------- #
# Optional live smoke test
# --------------------------------------------------------------------------- #

_LIVE = all(
    os.getenv(f"LLMINSPECTOR_{k.upper()}")
    for k in ("azure_endpoint", "api_version", "api_key")
)


@pytest.mark.skipif(not _LIVE, reason="no live Azure credentials in env")
def test_live_generate_and_embed():
    settings = AzureSettings.from_env()
    model = AzureOpenAIModel(settings)
    emb = AzureOpenAIEmbedding(settings)
    assert isinstance(model.generate("Say 'ok'."), str)
    assert isinstance(emb.embed_text("hello"), list)
