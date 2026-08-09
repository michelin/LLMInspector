"""``StylingStage`` — rewrite an input into the caller's voice.

Separate from generation on purpose. Generation is about *what* is asked and
whether it is answerable; styling is about *how* it is phrased. Folding the two
together makes a styling change a reason to regenerate — and regenerating costs
a fresh grounded input, a fresh filtration pass, and a fresh set of evolutions.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ...dataset.golden import Golden
from ...utils.prompting import render_prompt
from ..config import StylingConfig
from ..stage import Stage, StageContext

__all__ = ["StylingStage", "StyledInput", "STYLE_INPUT_PROMPT"]


class StyledInput(BaseModel):
    """The model's reply: the restyled input."""

    input: str


# "Preserve exactly what is being asked" is the load-bearing clause: without it
# a styling pass quietly changes the question and the expected output no longer
# answers it. Do not reflow.
# pylint: disable=line-too-long
STYLE_INPUT_PROMPT = """\
Rewrite the input below so it reads as it would in the setting described, without changing what it asks.

Input:
{input}

Setting:
- Scenario: {scenario}
- Task the system performs: {task}
- Shape the input should take: {input_format}

Rules:
1. Preserve exactly what is being asked. Same subject, same objective, same level of difficulty.
2. Preserve self-containment: never refer to "the text" or "the passage".
3. Change only the voice, register, and shape.

Return JSON of the form:
{{"input": "<the restyled input>"}}
"""
# pylint: enable=line-too-long

_UNSPECIFIED = "unspecified"


class StylingStage(Stage):
    """Restyles the input, when there is any styling to apply."""

    name = "style"
    metadata_keys = ("lineage", "styled")

    def __init__(self, config: Optional[StylingConfig] = None) -> None:
        self.config = config or StylingConfig()

    def _is_configured(self) -> bool:
        return any(
            (
                self.config.scenario,
                self.config.task,
                self.config.input_format,
            )
        )

    async def a_apply(self, golden: Golden, ctx: StageContext) -> Optional[Golden]:
        """Restyle in place, or do nothing at all.

        An unconfigured styling config makes **zero** model calls. A stage left
        in the default chain but never configured must be free, otherwise every
        run pays for a no-op rewrite per golden.
        """
        if not self._is_configured():
            return golden

        prompt = render_prompt(
            STYLE_INPUT_PROMPT,
            ["input", "scenario", "task", "input_format"],
            {
                "input": golden.input,
                "scenario": self.config.scenario or _UNSPECIFIED,
                "task": self.config.task or _UNSPECIFIED,
                "input_format": self.config.input_format or _UNSPECIFIED,
            },
            caller="StylingStage",
        )
        reply = await ctx.model.a_generate_structured(prompt, StyledInput)
        styled = (reply.input or "").strip()
        if not styled:
            # An empty restyle is not a restyle. Keep the original rather than
            # blanking an input that was already valid.
            return golden

        golden.input = styled
        golden.metadata["styled"] = True
        self.record(golden, scenario=self.config.scenario)
        return golden
