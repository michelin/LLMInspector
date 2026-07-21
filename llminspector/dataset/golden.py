"""The ``Golden`` — a seed row consumed by the synthesizers.

A golden carries the human-authored ``input`` (and optionally an
``expected_output`` / ``context``) from which alignment, adversarial, and RAG
synthesis produce derived test cases.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Golden(BaseModel):
    """Seed row for synthesis: an input plus optional reference answer and
    context. Unlike :class:`~llminspector.test_case.LLMTestCase` it has no
    ``actual_output`` — the answer is what synthesis/evaluation produces.

    ``metadata`` carries synthesizer-specific columns (e.g. ``augmentation_type``,
    ``Capability``, ``synthesizer_name``) so every synthesizer — legacy or a
    future custom engine — emits the same uniform :class:`Golden` shape.
    """

    model_config = ConfigDict(extra="ignore")

    input: str
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("input")
    @classmethod
    def _input_must_be_nonempty(cls, value: str) -> str:
        if value is None or str(value).strip() == "":
            raise ValueError("Golden.input must be a non-empty string")
        return value

    @field_validator("context", mode="before")
    @classmethod
    def _coerce_context(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        cleaned = [str(item) for item in value if str(item).strip() != ""]
        return cleaned or None
