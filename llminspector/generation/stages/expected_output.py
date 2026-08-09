"""``ExpectedOutputStage`` — the reference answer for a golden.

Runs last. Everything before it can still change the input, and an expected
output written against a pre-evolution question is worse than none at all: it
looks like ground truth and quietly scores the wrong thing.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from ...dataset.golden import Golden
from ...utils.prompting import render_prompt
from ..config import StylingConfig
from ..stage import Stage, StageContext

__all__ = ["ExpectedOutputStage", "ExpectedOutput", "EXPECTED_OUTPUT_PROMPT"]


class ExpectedOutput(BaseModel):
    """The model's reply: the reference answer."""

    expected_output: str


# "Answer using only the source material" is what makes this a ground truth
# rather than a second opinion. Do not reflow.
# pylint: disable=line-too-long
EXPECTED_OUTPUT_PROMPT = """\
Write the reference answer to the input below.

Input:
{input}

Source material:
{context}

Rules:
1. Answer using **only** the source material. Do not add outside knowledge.
2. If the source material does not fully answer the input, say what it does support and no more.
3. Answer directly. Do not restate the question or explain your reasoning.
{format_rule}
Return JSON of the form:
{{"expected_output": "<the reference answer>"}}
"""

FORMAT_RULE = "4. Shape the answer like this: {expected_output_format}\n"
# pylint: enable=line-too-long

_NO_CONTEXT = "(no source material was supplied)"


class ExpectedOutputStage(Stage):
    """Generates ``expected_output`` from the golden's context."""

    name = "expected_output"
    metadata_keys = ("lineage", "expected_output_grounded")

    def __init__(self, styling: Optional[StylingConfig] = None) -> None:
        self.styling = styling or StylingConfig()

    async def a_apply(self, golden: Golden, ctx: StageContext) -> Optional[Golden]:
        """Write the reference answer, unless the run has them switched off.

        A golden with no context still gets an answer, flagged
        ``expected_output_grounded=False``. Skipping instead would leave the
        column silently empty for scratch-generated goldens, which reads as "the
        stage failed" rather than "there was nothing to ground it in"; and
        refusing outright would make the scratch source unable to produce
        reference answers at all.
        """
        if not ctx.config.include_expected_output:
            return golden

        grounded = bool([c for c in ctx.context if c and c.strip()])
        prompt = render_prompt(
            EXPECTED_OUTPUT_PROMPT,
            ["input", "context", "format_rule"],
            {
                "input": golden.input,
                "context": self._context_block(ctx.context),
                "format_rule": self._format_rule(),
            },
            caller="ExpectedOutputStage",
        )
        reply = await ctx.model.a_generate_structured(prompt, ExpectedOutput)
        answer = (reply.expected_output or "").strip()
        if not answer:
            return golden

        golden.expected_output = answer
        if not grounded:
            golden.metadata["expected_output_grounded"] = False
        self.record(golden, grounded=grounded)
        return golden

    def _format_rule(self) -> str:
        """The optional shape instruction, or nothing.

        Rendered as an empty string when unset so the prompt does not carry a
        dangling rule 4 telling the model the format is ``None``.
        """
        wanted = self.styling.expected_output_format
        if not wanted:
            return ""
        return render_prompt(
            FORMAT_RULE,
            ["expected_output_format"],
            {"expected_output_format": wanted},
            caller="ExpectedOutputStage",
        )

    @staticmethod
    def _context_block(context: List[str]) -> str:
        joined = "\n\n".join(c for c in context if c and c.strip())
        return joined or _NO_CONTEXT
