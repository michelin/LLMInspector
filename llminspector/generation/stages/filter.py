"""``FiltrationStage`` — score an input against a rubric, rewrite it, re-score.

The quality gate of the pipeline. It asks the *critic* model how good an input
is, and what to do about an input that is not good enough is a policy the caller
sets on :class:`~llminspector.generation.config.FiltrationConfig` rather than
something this module decides.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, Field

from ...dataset.golden import Golden
from ...utils.prompting import render_prompt
from ..config import FiltrationConfig
from ..stage import Stage, StageContext

__all__ = [
    "FiltrationStage",
    "InputQuality",
    "RewrittenInput",
    "INPUT_QUALITY_PROMPT",
    "REWRITE_INPUT_PROMPT",
]

#: What the prompts say when a golden came from a source with no grounding at
#: all. Rendered instead of an empty block because a heading followed by nothing
#: reads to a model like source material that was lost, not source material that
#: never existed.
_NO_CONTEXT = "(none — this input was written without source material)"


class InputQuality(BaseModel):
    """The critic's rubric judgement of one input."""

    score: float = Field(
        default=0.0, description="Quality from 0.0 (unusable) to 1.0 (excellent)."
    )
    feedback: str = Field(
        default="", description="What specifically to fix, in one or two sentences."
    )


class RewrittenInput(BaseModel):
    """The critic's repaired input — the rewritten text and nothing else."""

    input: str = Field(default="", description="The rewritten input.")


# The rubric wording is load-bearing: it is the other half of the contract
# ``SYNTHETIC_INPUTS_PROMPT`` states when it generates ("self-contained", "one
# clear objective"), so a generated input is scored against the same two rules
# it was written under. Do not reflow.
# pylint: disable=line-too-long
INPUT_QUALITY_PROMPT = """\
You are reviewing an evaluation input written for a question-answering system.

Score it from 0.0 to 1.0 on exactly two criteria, weighted equally:

1. **Self-containment** — does it make sense to a reader who cannot see the source material? An input that says "according to the text", "in the passage above", "this document", or that leans on an unstated antecedent ("what did he decide?") is not self-contained.
2. **Clear objective** — does it ask for one answerable thing? An input that bundles several questions together, or that is so vague that several different answers would all be correct, does not have a clear objective.

Scoring guide: 1.0 fully satisfies both. 0.5 satisfies one. 0.0 satisfies neither.

Do not score the input on how interesting, difficult or well-punctuated it is. Do not reward or penalise length.

Input under review:
{input}

Source material the input was written from (for your reference only — the input must stand without it):
{context}

Return JSON of the form:
{{"score": 0.0, "feedback": "<what to fix, in one or two sentences>"}}
"""

REWRITE_INPUT_PROMPT = """\
You are repairing an evaluation input that failed review.

Rewrite it so that it is fully self-contained — understandable without the source material — and asks for exactly one answerable thing.

Rules:
1. Keep the original intent. You are repairing the input, not replacing it with a different question.
2. The rewritten input must still be answerable using **only** the source material below.
3. Fold any information the reader needs into the input itself instead of referring to the source material. Never write "according to the text", "in the passage above", or "this document".
4. Return the rewritten input alone. No preamble, no explanation, no quotation marks around it.

Input to repair:
{input}

Reviewer feedback to address:
{feedback}

Source material:
{context}

Return JSON of the form:
{{"input": "<the rewritten input>"}}
"""
# pylint: enable=line-too-long


class FiltrationStage(Stage[Golden]):
    """Score an input, rewrite it while it fails, re-score after every rewrite.

    The re-score is the point. The design this is adapted from scored once,
    rewrote, and then stored the *original* score — so the quality number in the
    export described a string that no longer existed, and the rewrite made the
    number look better than the text it was attached to. Here the loop is
    score → rewrite → **score again**, so the stored ``quality`` always
    describes the stored ``input``.

    What happens to an input that never clears the bar is
    :attr:`~llminspector.generation.config.FiltrationConfig.on_reject`:

    * ``"discard"`` — return ``None`` and record the reason on the context.
    * ``"rewrite"`` — keep the best text we reached, **flagged** below threshold.
    * ``"keep"`` — keep it, unflagged.

    "Kept but flagged" is deliberately distinct from "kept". Both leave the
    golden in the run, but the flag says the pipeline *tried* and failed to get
    this input over the bar, which is a different claim from "nobody was asked to
    care". A consumer filtering on ``below_threshold`` is dropping known-weak
    rows; a consumer under ``"keep"`` has said in advance that the score is
    reporting, not a gate — so flagging there would be noise, not information.

    The judging model is ``ctx.critic``, which falls back to the generating
    model. The rewrite goes to the critic too: it is repairing against the
    critic's own feedback, and splitting the two across models would mean a
    caller who set a cheap ``critic_model`` still paid the expensive model for
    the repair calls — the opposite of why the split exists.

    Parameters
    ----------
    config:
        The threshold, the rewrite budget, and the reject policy.
    """

    name = "filter"

    #: ``quality`` and ``rewrites`` are promoted flat so an export can be sorted
    #: and filtered on them without parsing lineage. ``below_threshold`` is only
    #: *written* on the kept-but-failing path, but it is declared unconditionally
    #: because the column set has to be knowable before the run, not after it.
    metadata_keys: Tuple[str, ...] = (
        "lineage",
        "quality",
        "rewrites",
        "below_threshold",
    )

    def __init__(self, config: Optional[FiltrationConfig] = None) -> None:
        self.config = config or FiltrationConfig()

    async def a_apply(self, golden: Golden, ctx: StageContext) -> Optional[Golden]:
        """Score ``golden.input``, rewriting it up to ``max_rewrites`` times.

        The golden is mutated in place and returned, rather than copied. Every
        other stage in the chain does the same: the generator tracks goldens by
        positional slot and ``Golden.id`` is the handle lineage and rejections
        are read against, so a stage that swapped in a fresh object would break
        the identity that provenance depends on.
        """
        critic = ctx.critic
        context = self._context_block(ctx.context)

        score, feedback = await self._a_score(critic, golden.input, context)
        rewrites = 0
        # ``max_rewrites=0`` never enters the loop, so the input is scored once
        # and the policy below still applies — "score but do not repair".
        while score < self.config.quality_threshold and rewrites < (
            self.config.max_rewrites
        ):
            rewritten = await self._a_rewrite(critic, golden.input, feedback, context)
            if not rewritten:
                # A model that returns nothing has not repaired anything, and
                # ``Golden.input`` may not be blank. Asking again with the same
                # feedback would only spend the rest of the budget on the same
                # answer, so stop with the text and score we already have.
                break
            golden.input = rewritten
            rewrites += 1
            score, feedback = await self._a_score(critic, golden.input, context)

        below = score < self.config.quality_threshold
        # ``quality`` is overwritten: the latest score is the one that describes
        # the text now on the golden, which is the whole point of re-scoring.
        golden.metadata["quality"] = score
        # ``rewrites`` accumulates instead. The default chain runs this stage
        # twice — once to repair, once to re-check after evolution — and the
        # second pass does no rewriting, so overwriting would erase the record
        # of a repair that actually happened.
        golden.metadata["rewrites"] = golden.metadata.get("rewrites", 0) + rewrites
        if below and self.config.on_reject == "rewrite":
            golden.metadata["below_threshold"] = True
        elif not below:
            # A later pass that clears the bar retires an earlier pass's flag;
            # leaving it set would mark a golden as sub-threshold on the
            # strength of a score that has since been superseded.
            golden.metadata.pop("below_threshold", None)

        # One lineage entry per stage application, written before the policy
        # branch so a discarded golden still carries the record of why it was
        # judged — the reason on the context and the record on the golden say
        # the same thing to two different readers.
        self.record(
            golden,
            score=score,
            rewrites=rewrites,
            below_threshold=True if below else None,
        )

        if below and self.config.on_reject == "discard":
            ctx.reject(
                f"quality {score:.2f} below threshold "
                f"{self.config.quality_threshold:.2f} after {rewrites} rewrite(s)"
            )
            return None
        return golden

    # -- model calls -----------------------------------------------------------

    async def _a_score(self, model: Any, text: str, context: str) -> Tuple[float, str]:
        """One rubric call: the clamped score and the critic's feedback."""
        prompt = render_prompt(
            INPUT_QUALITY_PROMPT,
            ["input", "context"],
            {"input": text, "context": context},
            caller="FiltrationStage",
        )
        reply = await model.a_generate_structured(prompt, InputQuality)
        return self._clamp(reply.score), (reply.feedback or "").strip()

    async def _a_rewrite(
        self, model: Any, text: str, feedback: str, context: str
    ) -> str:
        """One repair call, grounded in the same context the input came from."""
        prompt = render_prompt(
            REWRITE_INPUT_PROMPT,
            ["input", "feedback", "context"],
            {
                "input": text,
                # The feedback is the whole reason a rewrite beats a re-roll: it
                # tells the model which of the two rubric criteria it missed.
                "feedback": feedback or "The input failed the rubric.",
                "context": context,
            },
            caller="FiltrationStage",
        )
        reply = await model.a_generate_structured(prompt, RewrittenInput)
        return (reply.input or "").strip()

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _clamp(score: float) -> float:
        """Force a model-supplied score into ``[0.0, 1.0]``.

        Models return 1.7, or 85, or -0.2 for a rubric declared as 0-1 often
        enough that trusting the number would let an unusable input clear the
        threshold — and would put a value in the ``quality`` column that no
        downstream consumer can interpret against the documented range.
        """
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def _context_block(context: List[str]) -> str:
        """The grounding passages as one block, or the no-context placeholder."""
        joined = "\n\n".join(chunk for chunk in context if chunk and chunk.strip())
        return joined or _NO_CONTEXT
