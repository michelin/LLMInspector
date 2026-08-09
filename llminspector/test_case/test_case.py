"""The ``LLMTestCase`` — a single evaluable unit.

Maps the five per-row spreadsheet columns (``question / answer / ground_truth /
contexts / policy``) onto explicit attribute names:

    question     -> input
    answer       -> actual_output
    ground_truth -> expected_output
    contexts     -> retrieval_context
    policy       -> policy
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMTestCase(BaseModel):
    """One row of an evaluation: a prompt and (optionally) the model's
    answer, a reference answer, retrieval context, and a policy to check.

    ``input`` is the only required field; every metric that needs more than
    the prompt filters itself out when its inputs are missing (see the
    availability logic ported in the evaluate engine).
    """

    model_config = ConfigDict(extra="ignore")

    input: str
    actual_output: Optional[str] = None
    expected_output: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    policy: Optional[str] = None

    #: The ``Golden.id`` this case was promoted from, when it came from one.
    #: This is the only link back from a scored row to the golden — and to the
    #: generation lineage on its metadata — that produced it. ``None`` for cases
    #: read straight from a spreadsheet.
    golden_id: Optional[str] = None

    #: Carried over from the originating golden. Deliberately **not** exported
    #: by ``EvaluationDataset.to_pandas`` or ``EvaluationResult.to_pandas``,
    #: both of which name their columns explicitly: lineage is for tracing a row
    #: back, not for widening every result table.
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("input")
    @classmethod
    def _input_must_be_nonempty(cls, value: str) -> str:
        if value is None or str(value).strip() == "":
            raise ValueError("LLMTestCase.input must be a non-empty string")
        return value

    @field_validator("retrieval_context", mode="before")
    @classmethod
    def _coerce_retrieval_context(cls, value):
        """Accept a single string or an iterable of strings; drop blanks."""
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        cleaned = [str(item) for item in value if str(item).strip() != ""]
        return cleaned or None
