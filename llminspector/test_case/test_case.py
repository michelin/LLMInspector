"""The ``LLMTestCase`` — a single evaluable unit.

Maps the five per-row spreadsheet columns (``question / answer / ground_truth /
contexts / policy``) onto explicit attribute names:

    question     -> input
    answer       -> actual_output
    ground_truth -> expected_output
    contexts     -> retrieval_context
    policy       -> policy
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, field_validator


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
