"""``Settings`` — the code-driven replacement for the legacy ``config.ini`` +
``.env`` pair.

Holds Azure OpenAI connection details and the model / deployment names that
were hard-coded in ``eval_metrics.py``. ``from_env()`` is an optional
convenience that reads the same environment variable names the legacy
``rag_eval.py`` used (``api_version`` / ``azure_endpoint`` / ``api_key``);
it is never required — construct :class:`Settings` directly in code instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Connection + model configuration for the Azure OpenAI provider.

    Model / deployment defaults are promoted verbatim from the legacy
    ``EvalMetrics.initialize_core_models`` hard-coded values.
    """

    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    api_key: Optional[str] = None

    azure_deployment: str = "gpt-5-mini-dzs"
    model_name: str = "gpt-5-mini"
    embedding_deployment: str = "text-embedding-ada-002"
    embedding_name: str = "text-embedding-ada-002"

    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "Settings":
        """Build :class:`Settings` from environment variables.

        Optionally loads a ``.env`` file first (if ``python-dotenv`` is
        installed); missing dotenv is silently ignored. Reads the legacy
        variable names; model/deployment names fall back to the class
        defaults unless overridden via env.
        """
        try:  # optional convenience only
            from dotenv import load_dotenv

            if env_path:
                load_dotenv(env_path)
            else:
                load_dotenv()
        except ImportError:
            pass

        return cls(
            azure_endpoint=os.getenv("azure_endpoint"),
            api_version=os.getenv("api_version"),
            api_key=os.getenv("api_key"),
            azure_deployment=os.getenv("azure_deployment", cls.azure_deployment),
            model_name=os.getenv("model_name", cls.model_name),
            embedding_deployment=os.getenv(
                "embedding_deployment", cls.embedding_deployment
            ),
            embedding_name=os.getenv("embedding_name", cls.embedding_name),
        )
