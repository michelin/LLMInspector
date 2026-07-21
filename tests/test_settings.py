"""Phase 1 tests for the ``Settings`` dataclass + ``from_env``."""

from llminspector.config import Settings


def test_defaults_promote_legacy_hardcoded_names():
    s = Settings()
    assert s.azure_endpoint is None
    assert s.api_key is None
    assert s.azure_deployment == "gpt-5-mini-dzs"
    assert s.model_name == "gpt-5-mini"
    assert s.embedding_deployment == "text-embedding-ada-002"
    assert s.embedding_name == "text-embedding-ada-002"


def test_direct_construction_overrides():
    s = Settings(azure_endpoint="https://x", api_key="k", model_name="gpt-4o-mini")
    assert s.azure_endpoint == "https://x"
    assert s.api_key == "k"
    assert s.model_name == "gpt-4o-mini"


def test_from_env_reads_legacy_variable_names(monkeypatch):
    monkeypatch.setenv("azure_endpoint", "https://env-endpoint")
    monkeypatch.setenv("api_version", "2024-01-01")
    monkeypatch.setenv("api_key", "secret")
    s = Settings.from_env()
    assert s.azure_endpoint == "https://env-endpoint"
    assert s.api_version == "2024-01-01"
    assert s.api_key == "secret"
    # unset model names fall back to defaults
    assert s.model_name == "gpt-5-mini"


def test_from_env_missing_vars_are_none(monkeypatch):
    for var in ("azure_endpoint", "api_version", "api_key"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert s.azure_endpoint is None
    assert s.api_version is None
    assert s.api_key is None
