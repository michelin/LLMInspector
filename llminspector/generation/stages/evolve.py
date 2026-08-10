"""``EvolutionStage`` — deepen an input along a named strategy.

Evolution rewrites an input to be harder while keeping it answerable from the
same context. The reply is **schema-constrained**, not free text: the model says
which strategy it applied and whether the result is still grounded. The design
this replaces asked for a rewritten string and nothing else, so a chain of
evolutions could drift off its context with nothing able to notice.

Grounding is reported here and *acted on* by the filtration pass that follows.
That division is deliberate — this stage's job is to evolve, and concentrating
the keep/discard policy in one place keeps `FiltrationConfig.on_reject` the
single answer to "what happens to a bad input".
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from ...dataset.golden import Golden
from ...utils.prompting import render_prompt
from ..config import EvolutionConfig
from ..stage import Stage, StageContext

__all__ = ["EvolutionStage", "EvolvedInput", "STRATEGIES", "CONTEXT_STRATEGIES"]


class EvolvedInput(BaseModel):
    """The model's reply: the evolved input plus what it did to it."""

    input: str
    applied: str = ""
    still_grounded: bool = True


#: Strategy name -> the instruction handed to the model. Each one makes the
#: input harder along a different axis, so a run drawing from several of them
#: produces a varied testset rather than N variations of one difficulty.
STRATEGIES: Dict[str, str] = {
    # Require the answerer to combine two or more facts rather than look one up.
    "reasoning": (
        "Rewrite it so answering requires combining at least two separate facts "
        "from the source material, instead of retrieving a single one."
    ),
    # Swap a general term for a specific one drawn from the source material.
    "concretising": (
        "Rewrite it to refer to a specific entity, figure, or term that appears "
        "in the source material, instead of a general one."
    ),
    # Add a qualifying condition that narrows what a correct answer looks like.
    "constrained": (
        "Rewrite it to add one qualifying condition or constraint that narrows "
        "what a correct answer would be."
    ),
    # Ask for a relationship between two things rather than a single value.
    "comparative": (
        "Rewrite it to ask for a comparison or a relationship between two things "
        "described in the source material."
    ),
    # Frame it as a situation the user is in, not a direct question.
    "hypothetical": (
        "Rewrite it as a realistic situation the user is in, so the question is "
        "implied by the circumstances rather than asked directly."
    ),
}

#: Strategies whose instruction refers to the source material, and which are
#: therefore impossible without one. Dropped from the distribution for
#: context-free goldens rather than picked and silently half-followed.
CONTEXT_STRATEGIES = frozenset({"concretising", "comparative"})

# The "still answerable using only the source material" clause is what makes
# ``still_grounded`` meaningful; without it the model has no stated bar to
# report against. Do not reflow.
# pylint: disable=line-too-long
EVOLVE_INPUT_PROMPT = """\
You are making an evaluation input harder without making it unanswerable.

Current input:
{input}

Source material the input must remain answerable from:
{context}

Apply exactly this transformation:
{strategy}

Rules:
1. The result must still be answerable using **only** the source material.
2. The result must stay self-contained: never refer to "the text", "the passage", or "the source material".
3. Change the difficulty, not the subject. It must still be about the same thing.

Set "still_grounded" to false if you could not satisfy rule 1.

Return JSON of the form:
{{"input": "<the evolved input>", "applied": "<strategy name>", "still_grounded": true}}
"""
# pylint: enable=line-too-long

_NO_CONTEXT = "(no source material was supplied)"


class EvolutionStage(Stage):
    """Applies ``num_evolutions`` rounds of strategy-driven rewriting."""

    name = "evolve"
    metadata_keys = ("lineage", "evolutions", "ungrounded")

    def __init__(self, config: Optional[EvolutionConfig] = None) -> None:
        self.config = config or EvolutionConfig()
        unknown = sorted(set(self.config.strategies) - set(STRATEGIES))
        if unknown:
            raise ValueError(
                f"Unknown evolution strateg{'y' if len(unknown) == 1 else 'ies'} "
                f"{unknown}. Valid strategies: {sorted(STRATEGIES)}"
            )

    def _weights(self, has_context: bool) -> Tuple[List[str], List[float]]:
        """The strategy names and their weights, uniform when unconfigured.

        Zero-weighted strategies are dropped rather than left in with weight 0,
        so ``random.choices`` cannot return one through floating-point noise.

        Strategies that need source material are dropped when there is none, and
        the remaining weights simply renormalise (``random.choices`` normalises
        for us). A scratch-generated golden has no context, so asking a model to
        "refer to a specific entity that appears in the source material" is an
        instruction it cannot follow — the reference design picks such strategies
        anyway and takes whatever comes back.
        """
        configured = {k: v for k, v in self.config.strategies.items() if v > 0}
        if not configured:
            configured = {name: 1.0 for name in STRATEGIES}
        if not has_context:
            configured = {
                k: v for k, v in configured.items() if k not in CONTEXT_STRATEGIES
            }
        if not configured:
            raise ValueError(
                "No evolution strategy is possible: every configured strategy "
                f"({sorted(CONTEXT_STRATEGIES)}) needs source material, and this "
                "golden has none. Configure a context-free strategy such as "
                f"{sorted(set(STRATEGIES) - CONTEXT_STRATEGIES)} instead."
            )
        names = sorted(configured)
        return names, [configured[n] for n in names]

    async def a_apply(self, golden: Golden, ctx: StageContext) -> Optional[Golden]:
        """Evolve in place, once per configured round.

        ``num_evolutions=0`` makes no model call at all — an evolution stage
        left in the chain but switched off must not cost anything.
        """
        if self.config.num_evolutions < 1:
            return golden

        rng = self._rng(ctx.config.seed, golden.input)
        has_context = bool([c for c in ctx.context if c and c.strip()])
        names, weights = self._weights(has_context)
        context = self._context_block(ctx.context)

        applied: List[str] = []
        ungrounded = False
        for _ in range(self.config.num_evolutions):
            strategy = rng.choices(names, weights=weights, k=1)[0]
            prompt = render_prompt(
                EVOLVE_INPUT_PROMPT,
                ["input", "context", "strategy"],
                {
                    "input": golden.input,
                    "context": context,
                    "strategy": STRATEGIES[strategy],
                },
                caller="EvolutionStage",
            )
            reply = await ctx.model.a_generate_structured(prompt, EvolvedInput)
            evolved = (reply.input or "").strip()
            if not evolved:
                # Nothing came back; keep the input we have rather than blanking
                # it, and stop — the remaining rounds would ask the same thing.
                break
            golden.input = evolved
            applied.append(strategy)
            if not reply.still_grounded:
                ungrounded = True
            self.record(golden, strategy=strategy, grounded=reply.still_grounded)

        golden.metadata["evolutions"] = len(applied)
        if ungrounded:
            # Flagged, not discarded: the filtration pass that follows owns the
            # keep/discard decision, so there is one place to configure it.
            golden.metadata["ungrounded"] = True
        return golden

    @staticmethod
    def _rng(seed: Optional[int], text: str) -> random.Random:
        """A per-golden generator, reproducible across runs.

        Two properties are needed at once, and seeding from the config alone
        gives only the first:

        * **Reproducible** — the same seed must replay the same strategies.
        * **Varied** — goldens in one run must not all evolve the same way. A
          single ``Random(seed)`` per golden hands every golden an identical
          draw sequence, which would collapse a whole run onto one strategy
          pattern and defeat the point of evolving at all.

        Mixing the input text into the seed gives both. ``blake2b`` rather than
        ``hash()`` because Python randomises string hashing per process, which
        would make a "seeded" run differ between invocations. ``Golden.id``
        would be the obvious discriminator but is a fresh uuid on every run, so
        it is reproducible within a run and useless across them.
        """
        if seed is None:
            # No seed means the caller wants a nondeterministic run; system
            # entropy, not a derived-but-fixed value.
            return random.Random()
        digest = hashlib.blake2b(
            text.encode("utf-8"), digest_size=8, key=str(seed).encode("utf-8")
        ).digest()
        return random.Random(int.from_bytes(digest, "big"))

    @staticmethod
    def _context_block(context: List[str]) -> str:
        joined = "\n\n".join(c for c in context if c and c.strip())
        return joined or _NO_CONTEXT
