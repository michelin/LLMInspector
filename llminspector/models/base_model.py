"""Provider abstraction — the ABCs every concrete model implements.

``BaseLLM`` and ``BaseEmbeddingModel`` decouple the metrics / evaluate / synth
layers from any single vendor. Azure OpenAI is the first (and, for now, only)
concrete provider (see :mod:`llminspector.models.azure_openai`); OpenAI /
Anthropic implementations are future work.

The required contract is deliberately small: ``get_model_name`` plus
``generate`` / ``a_generate``. Every metric except the five ragas-backed context
metrics, and every golden source except the ragas testset backend, needs
nothing more — so a new provider is a class with three methods.

Capabilities beyond those three are **concrete methods with working defaults**,
never ``@abstractmethod`` — that is what keeps "a new provider is a class with
three methods" true as the package grows:

* ``generate_structured`` / ``a_generate_structured`` — prompt-and-parse into a
  pydantic schema, with one reask. A provider with a native JSON mode overrides
  to pass it through and inherits the parsing.
* ``a_embed_text`` / ``a_embed_texts`` — the sync call offloaded to a worker
  thread, so embedding a corpus does not stall the event loop.

Ragas support is an **optional capability**, not part of the required contract.
:meth:`BaseLLM.ragas_llm` / :meth:`BaseEmbeddingModel.ragas_embeddings` are
concrete and raise :class:`NotImplementedError` by default; providers that can
supply the ragas wrappers override them, and
:class:`~llminspector.metrics.base_metric.RagasBackedMetric` is the only thing
that asks.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, List, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ..utils.json_utils import extract_json_object
from .errors import StructuredOutputError

#: The pydantic model a caller asked ``generate_structured`` to produce.
BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


# The wording here is load-bearing — it is the whole of the default structured
# output contract for providers with no native JSON mode. Do not reflow it.
# pylint: disable=line-too-long
STRUCTURED_OUTPUT_INSTRUCTION = """

Respond with a single JSON object matching this JSON Schema:

{schema}

Return only the JSON object. Do not include any explanation, preamble, or markdown code fences.
"""

STRUCTURED_OUTPUT_REASK = """

Your previous response could not be parsed:

{response}

The error was: {error}

Try again. Return only a single JSON object matching the schema above, with no explanation, preamble, or markdown code fences.
"""
# pylint: enable=line-too-long


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

    # -- structured output -----------------------------------------------------
    #
    # Concrete with a working default, deliberately **not** abstract: a provider
    # is still three methods. The default is prompt-and-parse, which works on
    # any model that can follow an instruction; providers with a native JSON
    # mode override to pass it through and inherit the parsing below.

    def _structured_prompt(self, prompt: str, schema: Type[BaseModelT]) -> str:
        """``prompt`` plus the schema and the return-only-JSON directive."""
        return prompt + STRUCTURED_OUTPUT_INSTRUCTION.format(
            schema=json.dumps(schema.model_json_schema(), indent=2)
        )

    def _reask_prompt(
        self, structured_prompt: str, response: str, error: BaseException
    ) -> str:
        """The one retry prompt: the original, plus what went wrong."""
        return structured_prompt + STRUCTURED_OUTPUT_REASK.format(
            response=response, error=f"{type(error).__name__}: {error}"
        )

    @staticmethod
    def _parse_structured(response: str, schema: Type[BaseModelT]) -> BaseModelT:
        """Parse and validate, or raise ``JSONDecodeError`` / ``ValidationError``."""
        return schema.model_validate(extract_json_object(response))

    def generate_structured(
        self, prompt: str, schema: Type[BaseModelT], **kwargs: Any
    ) -> BaseModelT:
        """Generate a response parsed and validated into ``schema``.

        Asks once, and on a parse or validation failure **reasks exactly once**
        with the error text appended. A second failure raises
        :class:`~llminspector.models.errors.StructuredOutputError`.

        This is a different retry axis from :mod:`llminspector.models.retry`,
        which is rate-limit-only and deliberately so. A malformed response is
        not a transient condition: retrying it five times with backoff costs
        five calls to get the same prose back. Do not merge the two, and do not
        teach ``is_rate_limit_error`` about parse failures.
        """
        structured = self._structured_prompt(prompt, schema)
        response = self.generate(structured, **kwargs)
        try:
            return self._parse_structured(response, schema)
        except (json.JSONDecodeError, ValidationError) as first_error:
            retry_prompt = self._reask_prompt(structured, response, first_error)
            response = self.generate(retry_prompt, **kwargs)
            try:
                return self._parse_structured(response, schema)
            except (json.JSONDecodeError, ValidationError) as second_error:
                raise StructuredOutputError(
                    schema.__name__, response, second_error
                ) from second_error

    async def a_generate_structured(
        self, prompt: str, schema: Type[BaseModelT], **kwargs: Any
    ) -> BaseModelT:
        """Async form of :meth:`generate_structured`; identical semantics."""
        structured = self._structured_prompt(prompt, schema)
        response = await self.a_generate(structured, **kwargs)
        try:
            return self._parse_structured(response, schema)
        except (json.JSONDecodeError, ValidationError) as first_error:
            retry_prompt = self._reask_prompt(structured, response, first_error)
            response = await self.a_generate(retry_prompt, **kwargs)
            try:
                return self._parse_structured(response, schema)
            except (json.JSONDecodeError, ValidationError) as second_error:
                raise StructuredOutputError(
                    schema.__name__, response, second_error
                ) from second_error

    # -- optional capability ---------------------------------------------------

    def ragas_llm(self) -> Any:
        """Return the ragas-compatible LLM wrapper for this model.

        Optional: only :class:`~llminspector.metrics.base_metric.RagasBackedMetric`
        and the ragas testset backend call it. Providers that cannot supply one
        simply do not override it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a ragas LLM wrapper. "
            "Ragas-backed metrics and the ragas testset backend are unavailable "
            "with this provider."
        )


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

    # -- async ------------------------------------------------------------------
    #
    # Concrete with a default, like the structured-output methods: the required
    # contract stays at three methods. The default offloads the blocking call to
    # a worker thread rather than pretending to be async — without it, embedding
    # a document corpus stalls the whole event loop, which is exactly the
    # workload the generation pipeline runs concurrently. A provider with a
    # genuinely async client should override both.

    async def a_embed_text(self, text: str) -> List[float]:
        """Async :meth:`embed_text`; runs the sync call off the event loop."""
        return await asyncio.to_thread(self.embed_text, text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Async :meth:`embed_texts`; runs the sync call off the event loop."""
        return await asyncio.to_thread(self.embed_texts, texts)

    # -- optional capability ---------------------------------------------------

    def ragas_embeddings(self) -> Any:
        """Return the ragas-compatible embeddings wrapper for this model.

        Optional, for the same reason as :meth:`BaseLLM.ragas_llm`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a ragas embeddings wrapper."
        )
