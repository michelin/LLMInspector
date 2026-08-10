"""``BaseMetric`` — the ABC every metric subclasses.

A metric is a self-contained object built against a model (the Phase 2
:class:`~llminspector.models.azure_openai.AzureOpenAIModel`): construct it, then
call :meth:`measure` / :meth:`a_measure` with an
:class:`~llminspector.test_case.LLMTestCase`. The result is stored on the
instance (``score`` / ``reason`` / ``success``) and also returned.

The legacy ``EvalMetrics`` bundled the Azure chat client, the ragas evaluator
wrapper, and the ``prompt | llm`` chain boilerplate as shared state. That
plumbing lives here so subclasses only carry their own prompt + parsing:

* ``self._arun_prompt`` -> render the template, then ``model.a_generate(...)``

Judge metrics talk to the model through :class:`~llminspector.models.base_model.BaseLLM`
alone — ``generate`` / ``a_generate`` — so a provider satisfies the contract by
implementing that ABC and nothing else. They used to reach through to
``model.client`` (a raw langchain ``BaseChatModel``) and build a ``prompt | llm``
chain, which quietly made "expose a langchain client" the real contract and left
``BaseLLM.a_generate`` used by nobody.

Metrics that genuinely need ragas subclass :class:`RagasBackedMetric`, which
declares that requirement in the type rather than hiding it behind an attribute
lookup. That confines ragas to the five context metrics, which are now its
only consumers anywhere in the package.

No langchain import remains in this module.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Dict, Optional, Sequence, Set, Tuple

from ..utils.prompting import render_prompt

logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """Abstract metric.

    Parameters
    ----------
    model:
        A :class:`~llminspector.models.base_model.BaseLLM` — anything with
        ``generate`` / ``a_generate``. Local-only metrics (BERTScore, PII,
        tokens, language, readability) do not use it and may pass ``None``.
    threshold:
        Optional pass/fail threshold used by :meth:`is_successful`.
    """

    #: Human-readable metric name (subclasses set this, or override ``name``).
    metric_name: str = ""
    #: LLMTestCase attributes this metric needs (evaluate uses this to filter).
    required_inputs: Set[str] = set()
    #: True for LLM-judge metrics that also expose a ``{name}_reasoning`` string.
    produces_reasoning: bool = False
    #: Where this metric's columns sit in the exported table. Lower comes first;
    #: metrics sharing a key are ordered by name. The values are spaced so a new
    #: metric can be slotted between two existing ones without renumbering.
    sort_key: int = 1000

    def __init__(self, model: Any = None, threshold: Optional[float] = None) -> None:
        self.model = model
        self.threshold = threshold
        self.score: Any = None
        self.reason: Optional[str] = None
        self.success: Optional[bool] = None
        #: Set when this metric's own error path fired. ``evaluate`` collects it
        #: onto ``EvaluationResult.errors`` so a run that failed is visible in
        #: the output, not just in the logs.
        self.error: Optional[str] = None

    def record_failure(self, exc: BaseException) -> None:
        """Log a metric failure and remember it for the result object.

        Metrics isolate their own exceptions and score ``None``. Without this,
        a run where every call returned 401 produced a clean table of ``None``
        values indistinguishable from *skipped for missing input*.
        """
        self.error = f"{type(exc).__name__}: {exc}"
        logger.warning("Metric %r failed: %s", self.name, exc, exc_info=exc)

    @property
    def name(self) -> str:
        return self.metric_name

    # -- prompt helpers -------------------------------------------------------

    def _render(
        self,
        template: str,
        input_variables: Sequence[str],
        values: Dict[str, Any],
        partial_variables: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Substitute ``values`` into ``template``.

        Delegates to :func:`~llminspector.utils.prompting.render_prompt`, which
        holds the actual implementation so the generation layer can render its
        own prompts without importing ``metrics``. ``caller`` carries this
        metric's class name into the error message, so a prompt missing a
        variable still names the metric that owns it.
        """
        return render_prompt(
            template,
            input_variables,
            values,
            partial_variables,
            caller=type(self).__name__,
        )

    def _run_prompt(
        self, template, input_variables, values, partial_variables=None
    ) -> str:
        return self.model.generate(
            self._render(template, input_variables, values, partial_variables)
        )

    async def _arun_prompt(
        self, template, input_variables, values, partial_variables=None
    ) -> str:
        return await self.model.a_generate(
            self._render(template, input_variables, values, partial_variables)
        )

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

    # -- output contract ------------------------------------------------------
    #
    # A metric owns its result columns. The evaluate engine asks for them
    # through ``expand`` / ``output_columns`` / ``sort_key`` and contains no
    # metric names of its own, so adding a metric touches one file.

    def expand(self, score: Any) -> Dict[str, Any]:
        """Map this metric's score onto the result columns it owns.

        The default is the one-column case: ``{name: score}``. Metrics whose
        score is a structured value (moderation flag sets, code detection,
        policy verdicts) or that expose sub-judgements override this so the
        evaluate engine never has to know their column names.

        Must return the same key set for **every** score, including ``None`` —
        the engine calls ``expand(None)`` up front to reserve columns for rows
        where the metric is skipped.

        ``{name}_reasoning`` is handled separately by ``produces_reasoning``.
        """
        return {self.name: score}

    @property
    def success_column(self) -> Optional[str]:
        """Name of this metric's pass/fail column, or ``None`` when it has none.

        The column exists **only when a threshold is set**. With thresholds
        defaulting to ``None``, always emitting it would add one all-blank
        column per metric to every export — 20-plus columns of nothing on a
        default run. Give a metric a threshold and its verdict appears.

        The value can still be blank *with* a threshold: :meth:`is_successful`
        returns ``None`` for a non-numeric score (sentiment labels, moderation
        flag dicts), since ``>=`` means nothing there.
        """
        return f"{self.name}_success" if self.threshold is not None else None

    @property
    def output_columns(self) -> Tuple[str, ...]:
        """Every column this metric contributes, in export order.

        Derived from :meth:`expand` so the two can never drift, with the
        reasoning column slotted directly behind the headline score and the
        pass/fail verdict (when there is a threshold) closing the block.
        """
        columns = list(self.expand(None))
        if self.produces_reasoning:
            columns.insert(1, f"{self.name}_reasoning")
        if self.success_column is not None:
            columns.append(self.success_column)
        return tuple(columns)

    def clone(self) -> "BaseMetric":
        """Return a fresh copy with reset result state, sharing the model.

        ``evaluate`` clones each metric per row so concurrently-processed rows
        never race on the shared ``score`` / ``reason`` / ``success`` / ``error``
        state.

        .. warning::

           This is a **shallow** copy (``copy.copy``). Every attribute other
           than the four reset below is *shared* between the clone and the
           original, so a subclass that keeps mutable per-run state in an
           attribute will silently share it across concurrent rows.

           Subclasses holding mutable state must override ``clone`` and reset
           it — see
           :meth:`~llminspector.metrics.rag.AnswerCorrectnessMetric.clone`,
           which re-initialises its ``sub_scores`` dict. The ``model`` is shared
           deliberately: rebuilding a provider client per row would be
           pathological.
        """
        new = copy.copy(self)
        new.score = None
        new.reason = None
        new.success = None
        new.error = None
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
            self.success = float(self.score) >= self.threshold  # type: ignore[arg-type]
        return self.success


class RagasBackedMetric(BaseMetric):
    """Base for metrics that delegate scoring to a ragas scorer.

    These cannot run on a bare :class:`~llminspector.models.base_model.BaseLLM`:
    they need the provider's ragas wrapper, so the requirement is declared in the
    type rather than discovered at runtime. Every other metric in the package
    needs only ``a_generate``, which is what makes a non-langchain provider
    droppable.
    """

    #: Marker for callers wanting to filter a metric set by provider capability.
    requires_ragas = True

    @property
    def evaluator_llm(self) -> Any:
        """The provider's ragas LLM wrapper.

        Raises a directed error rather than ``AttributeError`` when handed a
        provider that does not support ragas.
        """
        if self.model is None:
            return None
        factory = getattr(self.model, "ragas_llm", None)
        if factory is None:
            raise TypeError(
                f"{type(self).__name__} is ragas-backed and needs a provider "
                f"exposing ragas_llm(); {type(self.model).__name__} does not. "
                "Use a ragas-capable provider or drop this metric from the set."
            )
        return factory()  # pylint: disable=not-callable


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
    #: Per-target column ordering. The question and answer variants of a metric
    #: do not always sit next to each other in the exported table (the readability
    #: pair is split across two blocks; the moderation pair is adjacent), so each
    #: subclass states both positions rather than deriving one from the other.
    sort_key_by_target: Dict[str, int] = {}

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
        self.target_prefix = _TARGET_PREFIX[target]
        self.metric_name = f"{self.target_prefix}_{self.name_suffix}"
        self.required_inputs = {target}
        self.sort_key = self.sort_key_by_target.get(target, self.sort_key)

    def _text(self, test_case: Any) -> Any:
        return getattr(test_case, self.target)
