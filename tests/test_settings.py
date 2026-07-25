"""Tests for ``AzureSettings`` + ``from_env`` (Phase 1, revised in Phase 8.5).

8.5 namespaced the environment variables and renamed the class. Both the new
behaviour and the one-release deprecation path are pinned here.
"""

import pytest

from llminspector.config import AzureSettings, Settings

_ALL_VARS = [
    "azure_endpoint",
    "api_version",
    "api_key",
    "azure_deployment",
    "model_name",
    "embedding_deployment",
    "embedding_name",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Neither naming scheme leaks in from the developer's shell."""
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(f"LLMINSPECTOR_{var.upper()}", raising=False)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def test_defaults_promote_legacy_hardcoded_names():
    s = AzureSettings()
    assert s.azure_endpoint is None
    assert s.api_key is None
    assert s.azure_deployment == "gpt-5-mini-dzs"
    assert s.model_name == "gpt-5-mini"
    assert s.embedding_deployment == "text-embedding-ada-002"
    assert s.embedding_name == "text-embedding-ada-002"


def test_direct_construction_overrides():
    s = AzureSettings(azure_endpoint="https://x", api_key="k", model_name="gpt-4o-mini")
    assert s.azure_endpoint == "https://x"
    assert s.api_key == "k"
    assert s.model_name == "gpt-4o-mini"


def test_settings_are_not_re_exported_at_the_package_root():
    """Config is owned by llminspector.config; the root does not mirror it."""
    import llminspector

    assert not hasattr(llminspector, "AzureSettings")
    assert not hasattr(llminspector, "Settings")
    assert llminspector.config.AzureSettings is AzureSettings


# --------------------------------------------------------------------------- #
# namespaced environment variables
# --------------------------------------------------------------------------- #


def test_from_env_reads_namespaced_variables(monkeypatch):
    monkeypatch.setenv("LLMINSPECTOR_AZURE_ENDPOINT", "https://env-endpoint")
    monkeypatch.setenv("LLMINSPECTOR_API_VERSION", "2024-01-01")
    monkeypatch.setenv("LLMINSPECTOR_API_KEY", "secret")
    monkeypatch.setenv("LLMINSPECTOR_MODEL_NAME", "gpt-4o-mini")

    s = AzureSettings.from_env()
    assert s.azure_endpoint == "https://env-endpoint"
    assert s.api_version == "2024-01-01"
    assert s.api_key == "secret"
    assert s.model_name == "gpt-4o-mini"
    # unset names still fall back to the class defaults
    assert s.embedding_name == "text-embedding-ada-002"


def test_namespaced_variables_do_not_need_a_deprecation_warning(recwarn):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        AzureSettings.from_env()  # nothing set at all -> no warning


def test_from_env_missing_vars_are_none():
    s = AzureSettings.from_env()
    assert s.azure_endpoint is None
    assert s.api_version is None
    assert s.api_key is None


# --------------------------------------------------------------------------- #
# the one-release legacy fallback
# --------------------------------------------------------------------------- #


def test_legacy_lowercase_names_still_work_but_warn(monkeypatch):
    monkeypatch.setenv("azure_endpoint", "https://legacy")
    monkeypatch.setenv("api_key", "legacy-secret")

    with pytest.warns(DeprecationWarning, match="LLMINSPECTOR_AZURE_ENDPOINT"):
        s = AzureSettings.from_env()

    assert s.azure_endpoint == "https://legacy"
    assert s.api_key == "legacy-secret"


def test_namespaced_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("azure_endpoint", "https://legacy")
    monkeypatch.setenv("LLMINSPECTOR_AZURE_ENDPOINT", "https://namespaced")
    s = AzureSettings.from_env()
    assert s.azure_endpoint == "https://namespaced"


def test_generic_api_key_collision_is_the_reason_for_the_prefix(monkeypatch):
    """`api_key` is generic enough to be set by something unrelated."""
    monkeypatch.setenv("api_key", "belongs-to-another-tool")
    monkeypatch.setenv("LLMINSPECTOR_API_KEY", "ours")
    assert AzureSettings.from_env().api_key == "ours"


# --------------------------------------------------------------------------- #
# the deprecated alias
# --------------------------------------------------------------------------- #


def test_settings_alias_still_constructs_but_warns():
    with pytest.warns(DeprecationWarning, match="AzureSettings"):
        s = Settings(azure_endpoint="https://x", api_key="k")
    assert isinstance(s, AzureSettings)
    assert s.azure_endpoint == "https://x"


def test_settings_alias_is_accepted_wherever_azure_settings_is():
    with pytest.warns(DeprecationWarning):
        s = Settings(api_key="k")
    assert isinstance(s, AzureSettings)


def test_importing_the_alias_does_not_warn():
    """The warning fires on use, not on import — re-exporting it costs nothing."""
    import importlib
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.reload(importlib.import_module("llminspector.config"))
