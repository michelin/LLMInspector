"""``SeedGoldenSource`` — grow an existing golden set.

Takes goldens you already have and produces more in the same vein. Seeds that
carry context go down the grounded path; seeds that do not go down the scratch
path. Both, when both are present.

That partition is the fix for a real defect in the design this is adapted from,
where the routing is all-or-nothing: *any* context-bearing golden sends the
whole batch down the grounded path, and every context-less seed is silently
dropped on the way. A caller mixing the two gets fewer goldens than they asked
for and no indication why.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

from pydantic import BaseModel

from ...dataset.golden import Golden
from ...utils.prompting import render_prompt
from ..config import StylingConfig
from ..source import GoldenSource
from .contexts import ContextSource
from .scratch import ScratchSource

if TYPE_CHECKING:  # pragma: no cover
    from ..config import GenerationConfig

__all__ = ["SeedGoldenSource", "ExtractedStyling", "STYLING_EXTRACTION_PROMPT"]

#: Seeds sampled when reverse-engineering a styling config. Ten is plenty to
#: characterise a voice and keeps the prompt small.
_STYLE_SAMPLE = 10


class ExtractedStyling(BaseModel):
    """The model's reading of what the seed inputs have in common."""

    scenario: str = ""
    task: str = ""
    input_format: str = ""


# Asking for a description rather than a label keeps the output usable directly
# as prompt text for the generation call. Do not reflow.
# pylint: disable=line-too-long
STYLING_EXTRACTION_PROMPT = """\
Below are real inputs sent to a question-answering system.

Infer the setting they come from and describe it in three short phrases:

- **scenario** — what kind of product or domain is this, and who is writing?
- **task** — what does the system these were sent to actually do?
- **input_format** — what shape do these messages take? (length, register, punctuation, whether they are questions or statements)

Describe what you observe. Do not invent detail the inputs do not support.

Inputs:
{inputs}

Return JSON of the form:
{{"scenario": "<...>", "task": "<...>", "input_format": "<...>"}}
"""
# pylint: enable=line-too-long


class SeedGoldenSource(GoldenSource[Golden]):
    """Augments an existing golden set, preserving input order.

    Parameters
    ----------
    goldens:
        The seeds. Must be non-empty.
    max_per_golden:
        Inputs to generate per seed.
    styling:
        Optional. When omitted it is **reverse-engineered** from the seeds with
        one model call, so an augmented set sounds like the set it grew from
        rather than like the model's default register.
    """

    metadata_keys = ("lineage", "context_size", "generated_from", "seed_id")

    def __init__(
        self,
        goldens: Sequence[Golden],
        max_per_golden: int = 2,
        *,
        styling: Optional[StylingConfig] = None,
    ) -> None:
        if not goldens:
            raise ValueError("SeedGoldenSource needs at least one seed golden.")
        if max_per_golden < 1:
            raise ValueError(f"max_per_golden must be >= 1, got {max_per_golden}")
        self.goldens = list(goldens)
        self.max_per_golden = max_per_golden
        self.styling = styling
        self.errors: List[dict] = []

    def _partition(self) -> tuple[List[Golden], List[Golden]]:
        """Split the seeds into grounded and context-free.

        Both halves are kept. Routing the whole batch on whether *any* seed has
        context is the bug this exists to avoid.
        """
        grounded = [g for g in self.goldens if g.context]
        bare = [g for g in self.goldens if not g.context]
        return grounded, bare

    async def _a_infer_styling(self, config: "GenerationConfig") -> StylingConfig:
        """One call to describe the seeds' shared voice."""
        sample = [g.input for g in self.goldens[:_STYLE_SAMPLE]]
        prompt = render_prompt(
            STYLING_EXTRACTION_PROMPT,
            ["inputs"],
            {"inputs": "\n".join(f"- {text}" for text in sample)},
            caller="SeedGoldenSource",
        )
        reply = await config.critic.a_generate_structured(prompt, ExtractedStyling)
        return StylingConfig(
            scenario=reply.scenario or None,
            task=reply.task or None,
            input_format=reply.input_format or None,
        )

    async def a_produce(self, config: "GenerationConfig") -> List[Golden]:
        styling = self.styling
        if styling is None:
            styling = await self._a_infer_styling(config)

        grounded, bare = self._partition()
        produced: List[Golden] = []

        if grounded:
            source = ContextSource(
                [list(g.context or []) for g in grounded],
                max_goldens_per_context=self.max_per_golden,
                styling=styling,
            )
            batch = await source.a_produce(config)
            self.errors.extend(source.errors)
            # Tag each generated golden with the seed that produced it, so an
            # augmented set can be traced back to what it grew from.
            per_seed = self.max_per_golden
            for position, golden in enumerate(batch):
                seed = grounded[min(position // per_seed, len(grounded) - 1)]
                golden.metadata["seed_id"] = seed.id
                golden.metadata["generated_from"] = "seed:context"
            produced.extend(batch)

        if bare:
            missing = styling.missing_fields()
            if missing:
                raise ValueError(
                    "Some seed goldens carry no context, so they are augmented "
                    "from their styling alone — but the styling is incomplete "
                    f"(missing: {', '.join(missing)}). Supply a full "
                    "StylingConfig, or drop the context-free seeds."
                )
            source = ScratchSource(styling, num_goldens=len(bare) * self.max_per_golden)
            batch = await source.a_produce(config)
            self.errors.extend(source.errors)
            for position, golden in enumerate(batch):
                seed = bare[min(position // self.max_per_golden, len(bare) - 1)]
                golden.metadata["seed_id"] = seed.id
                golden.metadata["generated_from"] = "seed:scratch"
            produced.extend(batch)

        return produced
