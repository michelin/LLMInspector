"""``PerturbationStage`` — surface noise, no model call.

Wraps :mod:`llminspector.generation.perturbations`. Opt-in and off by default:
roughening an input into a near-miss variant is what you want for adversarial
robustness testing and exactly what you do not want for a clean RAG testset,
where a typo would score the retriever rather than the answer.

The only stage in the chain that costs nothing. It is also the only one whose
output is not a fluent input, so it belongs last in any chain that includes it —
filtration would score its own noise as a defect.
"""

from __future__ import annotations

import hashlib
import random
from typing import Callable, Dict, List, Optional, Sequence

from ...dataset.golden import Golden
from .. import perturbations
from ..stage import Stage, StageContext

__all__ = ["PerturbationStage", "TRANSFORMS"]

#: Name -> the transform. Each takes a list of strings and returns a list, which
#: is the shape ``perturbations`` exposes; the stage adapts the single-input
#: case rather than reaching past that module's public surface.
TRANSFORMS: Dict[str, Callable[[List[str]], List[str]]] = {
    "typo": perturbations.add_typo,
    "ocr_typo": perturbations.add_ocr_typo,
    "contraction": perturbations.add_contraction,
    "abbreviation": perturbations.add_abbreviation,
    "uppercase": perturbations.uppercase_transform,
    "lowercase": perturbations.lowercase_transform,
    "titlecase": perturbations.titlecase_transform,
    "punctuation": perturbations.add_punctuation,
    "strip_punctuation": perturbations.strip_punctuation,
}


class PerturbationStage(Stage):
    """Applies one randomly chosen transform to the input."""

    name = "perturb"
    metadata_keys = ("lineage", "perturbation")

    def __init__(self, transforms: Optional[Sequence[str]] = None) -> None:
        """``transforms`` names which are eligible; ``None`` means all of them."""
        names = list(transforms) if transforms is not None else sorted(TRANSFORMS)
        if not names:
            raise ValueError("PerturbationStage needs at least one transform name.")
        unknown = sorted(set(names) - set(TRANSFORMS))
        if unknown:
            raise ValueError(
                f"Unknown perturbation(s) {unknown}. "
                f"Valid transforms: {sorted(TRANSFORMS)}"
            )
        self.transforms = names

    async def a_apply(self, golden: Golden, ctx: StageContext) -> Optional[Golden]:
        """Perturb in place. Makes no model call — ``async`` only to fit the ABC."""
        derived = self._seed(ctx.config.seed, golden.input)
        rng = random.Random(derived)
        choice = rng.choice(self.transforms)

        # Every transform takes and returns a list; a transform that declines to
        # fire (they are probabilistic) returns the input unchanged, which is a
        # valid outcome and not an error.
        #
        # ``perturbations`` draws from the *global* ``random`` module, so seeding
        # only the choice above would leave the transform itself unreproducible
        # and make this stage's seeding promise half true. The global state is
        # therefore seeded and restored around the call. There is no ``await``
        # inside the window, so concurrent goldens cannot interleave with it.
        state = random.getstate()
        try:
            if derived is not None:
                random.seed(derived)
            result = TRANSFORMS[choice]([golden.input])
        finally:
            random.setstate(state)
        perturbed = result[0] if result else golden.input
        if not perturbed or not perturbed.strip():
            # Golden.input may not be blank, and a blank attack is not an attack.
            return golden

        golden.input = perturbed
        golden.metadata["perturbation"] = choice
        self.record(golden, transform=choice)
        return golden

    @staticmethod
    def _seed(seed: Optional[int], text: str) -> Optional[int]:
        """A per-golden seed derived from the run seed and the input text.

        ``None`` in, ``None`` out — an unseeded run stays nondeterministic. See
        ``EvolutionStage._rng`` for why the text is mixed in and why this uses
        ``blake2b`` rather than ``hash()``.
        """
        if seed is None:
            return None
        digest = hashlib.blake2b(
            text.encode("utf-8"), digest_size=8, key=str(seed).encode("utf-8")
        ).digest()
        return int.from_bytes(digest, "big")
