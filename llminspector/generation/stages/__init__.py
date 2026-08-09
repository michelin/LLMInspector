"""The stage chain — one transformation per module.

Order matters, and :func:`default_stages` encodes the order that is correct
rather than the one that is obvious.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional

from ..config import EvolutionConfig, FiltrationConfig, StylingConfig
from .evolve import EvolutionStage
from .expected_output import ExpectedOutputStage
from .filter import FiltrationStage
from .generate import SyntheticInputs, generate_inputs
from .perturb import PerturbationStage
from .style import StylingStage

if TYPE_CHECKING:  # pragma: no cover
    from ..stage import Stage

__all__ = [
    "FiltrationStage",
    "EvolutionStage",
    "StylingStage",
    "ExpectedOutputStage",
    "PerturbationStage",
    "SyntheticInputs",
    "generate_inputs",
    "default_stages",
]


def default_stages(
    *,
    filtration: Optional[FiltrationConfig] = None,
    evolution: Optional[EvolutionConfig] = None,
    styling: Optional[StylingConfig] = None,
) -> List["Stage"]:
    """The standard chain: filter, evolve, re-check, style, answer.

    Generation itself is not here — it is 1-to-N and therefore a source's job;
    see :mod:`llminspector.generation.stages.generate`.

    Two orderings in this list are deliberate and easy to get backwards:

    **Evolution runs before a final filter pass, not after.** Evolving last and
    never re-checking is what the design this replaces did, and it means a
    compounding chain of rewrites can leave an input unanswerable from its
    context with nothing able to notice. The second pass is the cheap version —
    ``max_rewrites=0``, so it scores once and applies the reject policy rather
    than re-running the whole repair loop on text that has already been through
    it.

    **Expected output runs last.** Every stage before it can still change the
    input, and a reference answer written against a pre-evolution question is
    worse than no answer at all: it looks like ground truth and scores the wrong
    thing.

    ``PerturbationStage`` is deliberately absent — it is opt-in, and it belongs
    *after* everything here, since filtration would score its own noise as a
    defect.
    """
    filtration = filtration or FiltrationConfig()
    return [
        FiltrationStage(filtration),
        EvolutionStage(evolution or EvolutionConfig()),
        FiltrationStage(replace(filtration, max_rewrites=0)),
        StylingStage(styling or StylingConfig()),
        ExpectedOutputStage(styling or StylingConfig()),
    ]
