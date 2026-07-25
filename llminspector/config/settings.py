"""``AzureSettings`` — the code-driven replacement for the legacy ``config.ini``
+ ``.env`` pair.

Holds Azure OpenAI connection details and the model / deployment names that
were hard-coded in ``eval_metrics.py``. ``from_env()`` is an optional
convenience; it is never required — construct :class:`AzureSettings` directly in
code instead.

Two things changed in Phase 8.5:

* **Environment variables are namespaced.** The legacy names were lowercase and
  unnamespaced (``azure_endpoint``, ``api_key``), which is case-sensitive on
  Linux and generic enough to collide with anything else in the environment —
  ``api_key`` in particular. The names are now ``LLMINSPECTOR_AZURE_ENDPOINT``,
  ``LLMINSPECTOR_API_KEY``, and so on. The old names still work for one release
  and emit a :class:`DeprecationWarning` when used.
* **The class is named for what it is.** It was called ``Settings`` while
  hardcoding Azure fields and a Michelin-specific deployment name, promising a
  generality it did not have. ``Settings`` remains as a deprecated alias.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

#: Prefix for this package's environment variables.
ENV_PREFIX = "LLMINSPECTOR_"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read ``LLMINSPECTOR_<NAME>``, falling back to the legacy lowercase name.

    The fallback is scheduled for removal one release after 8.5.
    """
    value = os.getenv(f"{ENV_PREFIX}{name.upper()}")
    if value is not None:
        return value

    legacy = os.getenv(name)
    if legacy is not None:
        namespaced_name = f"{ENV_PREFIX}{name.upper()}"
        warnings.warn(
            f"Reading configuration from the unnamespaced environment variable "
            f"{name!r} is deprecated and will be removed in a future release; "
            f"rename it to {namespaced_name!r}.",
            DeprecationWarning,
            stacklevel=3,
        )
        return legacy
    return default


@dataclass
class AzureSettings:
    """Connection + model configuration for the Azure OpenAI provider.

    Model / deployment defaults are promoted verbatim from the legacy
    ``EvalMetrics.initialize_core_models`` hard-coded values. They are
    deployment names from one specific Azure resource, not package-wide
    defaults — override them for any other tenant.
    """

    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    api_key: Optional[str] = None

    azure_deployment: str = "gpt-5-mini-dzs"
    model_name: str = "gpt-5-mini"
    embedding_deployment: str = "text-embedding-ada-002"
    embedding_name: str = "text-embedding-ada-002"

    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "AzureSettings":
        """Build settings from environment variables.

        Optionally loads a ``.env`` file first (if ``python-dotenv`` is
        installed); missing dotenv is silently ignored. Each field is read from
        ``LLMINSPECTOR_<FIELD>``, falling back to the deprecated lowercase name.
        Model / deployment names fall back to the class defaults.
        """
        try:  # optional convenience only
            from dotenv import load_dotenv

            if env_path:
                load_dotenv(env_path)
            else:
                load_dotenv()
        except ImportError:
            pass

        defaults = {f.name: getattr(cls, f.name, None) for f in fields(cls)}
        values: Dict[str, Any] = {
            name: _env(name, defaults[name])
            for name in (
                "azure_endpoint",
                "api_version",
                "api_key",
                "azure_deployment",
                "model_name",
                "embedding_deployment",
                "embedding_name",
            )
        }
        return cls(**values)


class Settings(AzureSettings):
    """Deprecated alias for :class:`AzureSettings`.

    Kept so existing code keeps working; it warns on construction rather than on
    import, so simply having it re-exported costs nothing.
    """

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "Settings is deprecated; use AzureSettings instead. The class was "
            "never provider-neutral — every field on it is Azure-specific.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
