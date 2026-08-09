"""The ``Golden`` — a seed row consumed by the generation pipeline.

A golden carries the ``input`` (and optionally an ``expected_output`` /
``context``) from which derived test cases are produced, whether it was authored
by hand or generated.

``_BaseGolden`` holds the fields that describe a golden's *identity and
provenance* rather than its turn shape: ``id``, ``context``, ``metadata``.
``Golden`` adds the single-turn pair ``input`` / ``expected_output``. That split
is the seam for multi-turn work: a ``ConversationalGolden`` becomes a sibling
subclass substituting ``scenario`` / ``expected_outcome``, and everything that
consumes goldens generically — the export path, the generator, the stage
chain — keeps working against ``_BaseGolden`` untouched. The sibling class is
deliberately **not** written yet; an unused class is worse than a clean
extension point.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..test_case.test_case import LLMTestCase


class _BaseGolden(BaseModel):
    """Fields shared by every golden shape: identity, grounding, provenance."""

    model_config = ConfigDict(extra="ignore")

    #: Stable per-golden identity. Survives the Excel round trip and is copied
    #: onto ``LLMTestCase.golden_id``, so a scored row can be traced back to the
    #: golden — and to the generation lineage in ``metadata`` — that produced it.
    #: Generated rather than derived from the input, because two goldens may
    #: legitimately share an input (different evolutions of the same seed).
    id: str = Field(default_factory=lambda: uuid4().hex)

    #: Grounding passages. A bare string coerces to a one-element list.
    context: Optional[List[str]] = None

    #: The uniformity escape hatch: generation-specific columns (lineage,
    #: quality scores, ``source_file``, attack capability, …) so every producer
    #: emits the same golden shape while carrying its own extra columns.
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("context", mode="before")
    @classmethod
    def _coerce_context(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        cleaned = [str(item) for item in value if str(item).strip() != ""]
        return cleaned or None


class Golden(_BaseGolden):
    """Single-turn seed row: an input plus optional reference answer and context.

    Unlike :class:`~llminspector.test_case.LLMTestCase` it has no
    ``actual_output`` — the answer is what evaluation produces. Use
    :meth:`to_test_case` to promote one once an answer exists.
    """

    input: str
    expected_output: Optional[str] = None

    @field_validator("input")
    @classmethod
    def _input_must_be_nonempty(cls, value: str) -> str:
        if value is None or str(value).strip() == "":
            raise ValueError("Golden.input must be a non-empty string")
        return value

    def to_test_case(
        self,
        actual_output: Optional[str] = None,
        policy: Optional[str] = None,
    ) -> "LLMTestCase":
        """Promote this golden to an evaluable :class:`LLMTestCase`.

        ``context`` becomes ``retrieval_context`` — the same passages, named for
        what they are on each side of the pipeline. ``id`` lands on
        ``golden_id`` and ``metadata`` is **deep-copied**, so mutating the test
        case afterwards cannot corrupt the golden it came from.

        The copy is deep rather than shallow because metadata is not flat: the
        generation stages append records to a ``metadata["lineage"]`` list, and
        a shallow copy would leave that list shared between the golden and every
        test case promoted from it.

        Parameters
        ----------
        actual_output:
            The system's answer, if one has been produced yet.
        policy:
            Policy text for :class:`PolicyComplianceMetric`. Goldens have no
            policy field of their own; it is supplied at promotion time because
            the policy belongs to the evaluation, not to the seed.
        """
        from ..test_case.test_case import LLMTestCase

        return LLMTestCase(
            input=self.input,
            actual_output=actual_output,
            expected_output=self.expected_output,
            retrieval_context=self.context,
            policy=policy,
            golden_id=self.id,
            metadata=deepcopy(self.metadata),
        )
