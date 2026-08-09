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
from pydantic import BaseModel

from llminspector.config import AzureSettings
from llminspector.models import (
    AzureOpenAIEmbedding,
    AzureOpenAIModel,
    BaseEmbeddingModel,
    BaseLLM,
)
from llminspector.models import retry as retry_module
from llminspector.models.retry import DEFAULT_MAX_RETRIES

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


# --------------------------------------------------------------------------- #
# Phase 2 — Azure-specific structured output
#
# The generic prompt-and-parse machinery (schema block, reask, error) belongs to
# BaseLLM and is covered in tests/test_structured_output.py. What is tested here
# is only what the Azure subclass adds: the `response_format` kwarg reaching the
# langchain client, and the fact that it adds *nothing else*.
# --------------------------------------------------------------------------- #

JSON_OBJECT = {"type": "json_object"}


class _Answer(BaseModel):
    """Minimal schema for the structured-output calls below."""

    answer: str


class _RecordingChatClient(_FakeChatClient):
    """`_FakeChatClient` that logs every call and every failed attribute lookup.

    Subclassed rather than written fresh so the provider is still constructed
    through exactly the same fake-langchain path as the tests above.

    Two logs matter:

    * ``calls`` — the prompt and kwargs of each ``invoke`` / ``ainvoke``, which
      is where the ``response_format`` assertions look.
    * ``unknown_attribute_lookups`` — anything the provider reached for that
      this fake does not implement. A real ``AzureChatOpenAI`` has
      ``with_structured_output``, so an override that quietly grew a dependency
      on it (or on any other client method) would pass a ``hasattr`` check
      against the real class and silently change behaviour. Recording the misses
      instead of relying on ``hasattr`` makes that impossible to miss here.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []
        self.unknown_attribute_lookups = []
        #: Responses to hand back in order; falls back to a parseable object.
        self.script = []

    def __getattr__(self, name):
        # Only reached when normal lookup fails. Goes through __dict__ so a miss
        # during __init__ cannot recurse.
        if not name.startswith("__"):
            self.__dict__.setdefault("unknown_attribute_lookups", []).append(name)
        raise AttributeError(name)

    def _respond(self, prompt, kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        if self.script:
            return _FakeMessage(self.script.pop(0))
        return _FakeMessage('{"answer": "ok"}')

    def invoke(self, prompt, **kwargs):
        return self._respond(prompt, kwargs)

    async def ainvoke(self, prompt, **kwargs):
        return self._respond(prompt, kwargs)


@pytest.fixture
def recording_chat(monkeypatch, patched_langchain):  # pylint: disable=unused-argument
    """Swap only the chat class inside the already-stubbed langchain module."""
    monkeypatch.setattr(
        sys.modules["langchain_openai"], "AzureChatOpenAI", _RecordingChatClient
    )


def test_generate_structured_passes_json_object_response_format(
    recording_chat, settings
):
    model = AzureOpenAIModel(settings, api_key="k")
    out = model.generate_structured("Give me an answer.", _Answer)

    assert out.answer == "ok"
    assert len(model.client.calls) == 1
    assert model.client.calls[0]["kwargs"]["response_format"] == JSON_OBJECT


def test_a_generate_structured_passes_json_object_response_format(
    recording_chat, settings
):
    model = AzureOpenAIModel(settings, api_key="k")
    out = asyncio.run(model.a_generate_structured("Give me an answer.", _Answer))

    assert out.answer == "ok"
    assert len(model.client.calls) == 1
    assert model.client.calls[0]["kwargs"]["response_format"] == JSON_OBJECT


def test_structured_prompt_mentions_json(recording_chat, settings):
    """Azure's ``json_object`` mode 400s unless the prompt contains "JSON".

    The inherited instruction block satisfies that, which is why the override
    can enable the mode unconditionally.
    """
    model = AzureOpenAIModel(settings, api_key="k")
    model.generate_structured("Give me an answer.", _Answer)
    assert "JSON" in model.client.calls[0]["prompt"]


def test_explicit_response_format_is_not_clobbered(recording_chat, settings):
    """``setdefault``, not assignment — a caller asking for a stricter mode
    (``json_schema``, say) must keep it."""
    model = AzureOpenAIModel(settings, api_key="k")
    explicit = {"type": "json_schema", "json_schema": {"name": "answer"}}

    model.generate_structured("q", _Answer, response_format=explicit)

    assert model.client.calls[0]["kwargs"]["response_format"] == explicit


def test_explicit_response_format_is_not_clobbered_async(recording_chat, settings):
    model = AzureOpenAIModel(settings, api_key="k")
    explicit = {"type": "json_schema", "json_schema": {"name": "answer"}}

    asyncio.run(model.a_generate_structured("q", _Answer, response_format=explicit))

    assert model.client.calls[0]["kwargs"]["response_format"] == explicit


def test_other_kwargs_still_reach_the_client(recording_chat, settings):
    model = AzureOpenAIModel(settings, api_key="k")

    model.generate_structured("q", _Answer, temperature=0, seed=7)

    kwargs = model.client.calls[0]["kwargs"]
    assert kwargs == {"response_format": JSON_OBJECT, "temperature": 0, "seed": 7}


def test_reask_also_carries_response_format(recording_chat, settings):
    """The override adds the kwarg and nothing else: the reask is inherited.

    Only the call count and the kwargs on *both* calls are asserted — the reask
    semantics themselves belong to the base class's tests.
    """
    model = AzureOpenAIModel(settings, api_key="k")
    model.client.script = ["sorry, no JSON for you", '{"answer": "42"}']

    out = model.generate_structured("q", _Answer)

    assert out.answer == "42"
    assert len(model.client.calls) == 2
    assert all(
        c["kwargs"]["response_format"] == JSON_OBJECT for c in model.client.calls
    )


def test_a_reask_also_carries_response_format(recording_chat, settings):
    model = AzureOpenAIModel(settings, api_key="k")
    model.client.script = ["sorry, no JSON for you", '{"answer": "42"}']

    out = asyncio.run(model.a_generate_structured("q", _Answer))

    assert out.answer == "42"
    assert len(model.client.calls) == 2
    assert all(
        c["kwargs"]["response_format"] == JSON_OBJECT for c in model.client.calls
    )


def test_structured_output_touches_no_new_client_api(recording_chat, settings):
    """No ``with_structured_output``, no other langchain surface.

    The fake implements exactly the two methods the provider is allowed to use;
    the recorded misses must stay empty for sync, async and the reask path.
    """
    model = AzureOpenAIModel(settings, api_key="k")
    model.client.script = ["not json", '{"answer": "42"}']

    model.generate_structured("q", _Answer)
    asyncio.run(model.a_generate_structured("q", _Answer))

    assert model.client.unknown_attribute_lookups == []


# --------------------------------------------------------------------------- #
# Phase 2 — rate-limit retry on the embedding provider
# --------------------------------------------------------------------------- #


class _TooManyRequests(Exception):
    """A 429 carrying nothing but a status code.

    ``is_rate_limit_error`` checks ``status_code`` before the class name and the
    message text; the empty message and neutral class name keep this test on
    that first branch. Deliberately not a vendor exception class — the retry
    layer is provider-agnostic and must never need one.
    """

    status_code = 429


class _Boom(Exception):
    """A plain failure with no rate-limit tell of any kind."""


class _FlakyEmbeddingsClient(_FakeEmbeddingsClient):
    """`_FakeEmbeddingsClient` that fails the first ``failures`` calls.

    ``failures`` / ``error`` are set by the test after the provider has built
    the client, so each test scripts its own failure run without class-level
    state leaking between tests.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.failures = 0
        self.error = _TooManyRequests
        self.query_calls = 0
        self.document_calls = 0

    def embed_query(self, text):
        self.query_calls += 1
        if self.query_calls <= self.failures:
            raise self.error()
        return super().embed_query(text)

    def embed_documents(self, texts):
        self.document_calls += 1
        if self.document_calls <= self.failures:
            raise self.error()
        return super().embed_documents(texts)


@pytest.fixture
def flaky_embeddings(monkeypatch, patched_langchain):  # pylint: disable=unused-argument
    """Swap only the embeddings class inside the stubbed langchain module."""
    monkeypatch.setattr(
        sys.modules["langchain_openai"], "AzureOpenAIEmbeddings", _FlakyEmbeddingsClient
    )


@pytest.fixture
def no_backoff_sleep(monkeypatch):
    """Record the backoff delays instead of sleeping them.

    Patched where ``retry.py`` looks the name up, so nothing outside the retry
    layer is affected. The tests below assert the retry *policy* — how many
    calls happen and what propagates — never the delay, which is jittered and
    would otherwise put seconds on a ~4s suite.
    """
    slept = []
    monkeypatch.setattr(retry_module, "time", types.SimpleNamespace(sleep=slept.append))
    return slept


def test_embed_text_retries_rate_limit_then_succeeds(
    flaky_embeddings, no_backoff_sleep, settings
):
    emb = AzureOpenAIEmbedding(settings, api_key="k")
    emb.client.failures = 2

    assert emb.embed_text("hello") == [0.1, 0.2, 0.3]
    assert emb.client.query_calls == 3
    assert len(no_backoff_sleep) == 2


def test_embed_texts_retries_rate_limit_then_succeeds(
    flaky_embeddings, no_backoff_sleep, settings
):
    emb = AzureOpenAIEmbedding(settings, api_key="k")
    emb.client.failures = 1

    vecs = emb.embed_texts(["a", "b"])
    assert len(vecs) == 2
    assert emb.client.document_calls == 2
    assert len(no_backoff_sleep) == 1


def test_non_rate_limit_error_propagates_immediately(
    flaky_embeddings, no_backoff_sleep, settings
):
    """embed_text must not become a general-purpose retry."""
    emb = AzureOpenAIEmbedding(settings, api_key="k")
    emb.client.failures = 99
    emb.client.error = _Boom

    with pytest.raises(_Boom):
        emb.embed_text("hello")
    assert emb.client.query_calls == 1
    assert no_backoff_sleep == []


def test_non_rate_limit_error_propagates_immediately_for_batches(
    flaky_embeddings, no_backoff_sleep, settings
):
    emb = AzureOpenAIEmbedding(settings, api_key="k")
    emb.client.failures = 99
    emb.client.error = _Boom

    with pytest.raises(_Boom):
        emb.embed_texts(["a"])
    assert emb.client.document_calls == 1


def test_max_retries_zero_disables_backoff(
    flaky_embeddings, no_backoff_sleep, settings
):
    emb = AzureOpenAIEmbedding(settings, api_key="k", max_retries=0)
    emb.client.failures = 1

    with pytest.raises(_TooManyRequests):
        emb.embed_text("hello")
    assert emb.client.query_calls == 1
    assert no_backoff_sleep == []


def test_embedding_max_retries_defaults_to_the_shared_constant(
    flaky_embeddings, no_backoff_sleep, settings
):
    """One number governs the budget: retry.DEFAULT_MAX_RETRIES."""
    emb = AzureOpenAIEmbedding(settings, api_key="k")
    assert emb.max_retries == DEFAULT_MAX_RETRIES

    emb.client.failures = 99
    with pytest.raises(_TooManyRequests):
        emb.embed_text("hello")
    assert emb.client.query_calls == DEFAULT_MAX_RETRIES + 1


def test_embedding_max_retries_argument_is_honoured(
    flaky_embeddings, no_backoff_sleep, settings
):
    emb = AzureOpenAIEmbedding(settings, api_key="k", max_retries=2)
    assert emb.max_retries == 2

    emb.client.failures = 99
    with pytest.raises(_TooManyRequests):
        emb.embed_texts(["a"])
    assert emb.client.document_calls == 3
    assert len(no_backoff_sleep) == 2


# --------------------------------------------------------------------------- #
# Phase 2 — the concrete provider inherits working async embedding methods
#
# The thread-offload mechanics are the base class's business; what is pinned
# here is that AzureOpenAIEmbedding actually gets them, and that they inherit
# the retry because they delegate to the sync calls.
# --------------------------------------------------------------------------- #


def test_a_embed_text_matches_the_sync_call(patched_langchain, settings):
    emb = AzureOpenAIEmbedding(settings, api_key="k")
    assert asyncio.run(emb.a_embed_text("hello")) == emb.embed_text("hello")


def test_a_embed_texts_matches_the_sync_call(patched_langchain, settings):
    emb = AzureOpenAIEmbedding(settings, api_key="k")
    assert asyncio.run(emb.a_embed_texts(["a", "b"])) == emb.embed_texts(["a", "b"])


def test_a_embed_texts_gets_the_retry_too(flaky_embeddings, no_backoff_sleep, settings):
    emb = AzureOpenAIEmbedding(settings, api_key="k")
    emb.client.failures = 1

    vecs = asyncio.run(emb.a_embed_texts(["a", "b"]))
    assert len(vecs) == 2
    assert emb.client.document_calls == 2


def test_a_embed_text_gets_the_retry_too(flaky_embeddings, no_backoff_sleep, settings):
    emb = AzureOpenAIEmbedding(settings, api_key="k")
    emb.client.failures = 1

    assert asyncio.run(emb.a_embed_text("hello")) == [0.1, 0.2, 0.3]
    assert emb.client.query_calls == 2


# --------------------------------------------------------------------------- #
# Phase 2 — regression guard: max_retries did not disturb auth selection
# --------------------------------------------------------------------------- #


def test_embedding_auth_rules_survive_max_retries(patched_langchain, settings):
    """Exactly one of api_key / azure_ad_token_provider, still."""
    with pytest.raises(ValueError):
        AzureOpenAIEmbedding(
            settings,
            api_key="k",
            azure_ad_token_provider=lambda: "t",
            max_retries=3,
        )
    with pytest.raises(ValueError):
        AzureOpenAIEmbedding(settings, max_retries=3)

    emb = AzureOpenAIEmbedding(settings, api_key="k", max_retries=0)
    assert emb.client.kwargs["api_key"] == "k"
    assert "azure_ad_token_provider" not in emb.client.kwargs
    assert emb.max_retries == 0
