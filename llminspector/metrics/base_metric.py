"""``BaseMetric`` — the ABC every metric subclasses.

A metric is a self-contained object built against a model (the Phase 2
:class:`~llminspector.models.azure_openai.AzureOpenAIModel`): construct it, then
call :meth:`measure` / :meth:`a_measure` with an
:class:`~llminspector.test_case.LLMTestCase`. The result is stored on the
instance (``score`` / ``reason`` / ``success``) and also returned.

The legacy ``EvalMetrics`` bundled the Azure chat client, the ragas evaluator
wrapper, and the ``prompt | llm`` chain boilerplate as shared state. That
plumbing lives here so subclasses only carry their own prompt + parsing:

* ``self.llm``           -> the langchain chat client (legacy ``self.azure_llm``)
* ``self.evaluator_llm`` -> the ragas wrapper (legacy ``self.evaluator_llm``)
* ``self._arun_prompt``  -> ``(PromptTemplate | llm).ainvoke(values).content``

All heavy imports (langchain) are deferred to first use.
"""

from __future__ import annotations

import asyncio
import copy
from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Optional, Set


class BaseMetric(ABC):
    """Abstract metric.

    Parameters
    ----------
    model:
        A model exposing ``.client`` (langchain chat client, for LLM-judge
        metrics) and ``.ragas_llm()`` (ragas wrapper, for the context metrics).
        Local-only metrics (BERTScore, PII, tokens, language, readability) do
        not use it. May be ``None`` for those.
    threshold:
        Optional pass/fail threshold used by :meth:`is_successful`.
    """

    #: Human-readable metric name (subclasses set this, or override ``name``).
    metric_name: str = ""
    #: LLMTestCase attributes this metric needs (evaluate uses this to filter).
    required_inputs: Set[str] = set()
    #: True for LLM-judge metrics that also expose a ``{name}_reasoning`` string.
    produces_reasoning: bool = False

    def __init__(self, model: Any = None, threshold: Optional[float] = None) -> None:
        self.model = model
        self.threshold = threshold
        self.score: Any = None
        self.reason: Optional[str] = None
        self.success: Optional[bool] = None

    @property
    def name(self) -> str:
        return self.metric_name

    # -- model handles --------------------------------------------------------

    @property
    def llm(self) -> Any:
        """The langchain chat client for ``prompt | llm`` chains."""
        return getattr(self.model, "client", None)

    @property
    def evaluator_llm(self) -> Any:
        """The ragas ``LangchainLLMWrapper`` for context metrics."""
        return self.model.ragas_llm() if self.model is not None else None

    # -- prompt-chain helpers (verbatim mechanics of the legacy metrics) ------

    def _prompt_template(self, template, input_variables, partial_variables=None):
        from langchain_core.prompts import PromptTemplate

        kwargs = {"template": template, "input_variables": input_variables}
        if partial_variables:
            kwargs["partial_variables"] = partial_variables
        return PromptTemplate(**kwargs)

    def _run_prompt(self, template, input_variables, values, partial_variables=None) -> str:
        prompt = self._prompt_template(template, input_variables, partial_variables)
        chain = prompt | self.llm
        return chain.invoke(values).content

    async def _arun_prompt(
        self, template, input_variables, values, partial_variables=None
    ) -> str:
        prompt = self._prompt_template(template, input_variables, partial_variables)
        chain = prompt | self.llm
        result = await chain.ainvoke(values)
        return result.content

    # -- measure contract -----------------------------------------------------

    @abstractmethod
    async def a_measure(self, test_case: Any) -> Any:
        """Asynchronously compute the metric for ``test_case``.

        Implementations must set ``self.score`` (and ``self.reason`` where a
        rationale exists), call :meth:`is_successful`, and return ``self.score``.
        """

    def measure(self, test_case: Any) -> Any:
        """Synchronous wrapper around :meth:`a_measure`."""
        return asyncio.run(self.a_measure(test_case))

    def clone(self) -> "BaseMetric":
        """Return a fresh copy with reset result state, sharing the model.

        ``evaluate`` clones each metric per row so concurrently-processed rows
        never race on the shared ``score`` / ``reason`` / ``success`` state.
        """
        new = copy.copy(self)
        new.score = None
        new.reason = None
        new.success = None
        return new

    def is_successful(self) -> Optional[bool]:
        """Pass/fail against ``threshold``.

        Returns ``None`` when there is no threshold or the score is not numeric
        (many metrics return labels / lists / dicts); otherwise ``score >=
        threshold``. Subclasses with risk semantics may override.
        """
        if self.threshold is None or not isinstance(self.score, Number):
            self.success = None
        else:
            self.success = self.score >= self.threshold
        return self.success


#: Maps an LLMTestCase text attribute to the legacy metric-name prefix.
_TARGET_PREFIX = {"input": "question", "actual_output": "answer"}


class DualTargetMetric(BaseMetric):
    """Base for metrics that run identically on the question or the answer.

    The legacy pipeline wired one method to both ``question_*`` and ``answer_*``
    keys (e.g. ``sentiment_analysis_async(question)`` and ``(answer)``). Here a
    single class is parameterized by ``target`` — the ``LLMTestCase`` attribute
    to read (``"input"`` or ``"actual_output"``) — and derives its ``name`` and
    ``required_inputs`` from it.
    """

    #: Suffix appended to the target prefix to form the metric name.
    name_suffix: str = ""

    def __init__(
        self,
        model: Any = None,
        threshold: Optional[float] = None,
        target: str = "actual_output",
    ) -> None:
        super().__init__(model=model, threshold=threshold)
        if target not in _TARGET_PREFIX:
            raise ValueError(
                f"target must be one of {sorted(_TARGET_PREFIX)}, got {target!r}"
            )
        self.target = target
        self.metric_name = f"{_TARGET_PREFIX[target]}_{self.name_suffix}"
        self.required_inputs = {target}

    def _text(self, test_case: Any) -> Any:
        return getattr(test_case, self.target)
