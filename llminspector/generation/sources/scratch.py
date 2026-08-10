"""``ScratchSource`` — goldens from a description, with no source material.

For evaluating a system you have no corpus for: describe the scenario, the task
and the shape of a user message, and the model writes inputs that fit.

Nothing here is grounded, so there is no expected output by default. A reference
answer invented without source material is not ground truth — it is a second
opinion wearing ground truth's column name, and scoring against it silently
measures agreement between two models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ...dataset.golden import Golden
from ...utils.concurrency import a_map
from ..config import StylingConfig
from ..source import GoldenSource
from ..stages.generate import generate_from_styling

if TYPE_CHECKING:  # pragma: no cover
    from ..config import GenerationConfig

__all__ = ["ScratchSource"]

#: Inputs requested per model call. Asking for hundreds in one reply reliably
#: degrades into near-duplicates, so a large run is several smaller calls.
BATCH_SIZE = 10


class ScratchSource(GoldenSource[Golden]):
    """Generates ungrounded inputs from a :class:`StylingConfig`.

    Parameters
    ----------
    styling:
        Must have ``scenario``, ``task`` and ``input_format`` set — without
        source material they are the only description the model has.
    num_goldens:
        How many inputs to produce in total.
    """

    metadata_keys = ("lineage", "generated_from")

    def __init__(
        self,
        styling: StylingConfig,
        num_goldens: int = 10,
    ) -> None:
        missing = styling.missing_fields()
        if missing:
            # Every missing field at once. Reporting the first one means the
            # caller fixes it, reruns, and is told about the second — three
            # round trips to learn what one message could have said.
            raise ValueError(
                "ScratchSource has no source material to work from, so it needs "
                f"a full StylingConfig. Missing: {', '.join(missing)}. "
                "Supply StylingConfig(scenario=..., task=..., input_format=...)."
            )
        if num_goldens < 1:
            raise ValueError(f"num_goldens must be >= 1, got {num_goldens}")

        self.styling = styling
        self.num_goldens = num_goldens
        self.errors: List[dict] = []

    def _batches(self) -> List[int]:
        """Sizes of each generation call, summing to ``num_goldens``."""
        full, remainder = divmod(self.num_goldens, BATCH_SIZE)
        sizes = [BATCH_SIZE] * full
        if remainder:
            sizes.append(remainder)
        return sizes

    async def a_produce(self, config: "GenerationConfig") -> List[Golden]:
        async def _batch(size: int) -> List[Golden]:
            texts = await generate_from_styling(config.model, size, self.styling)
            return [
                Golden(
                    input=text,
                    metadata={
                        "generated_from": "scratch",
                        "lineage": [{"stage": "generate", "grounded": False}],
                    },
                )
                for text in texts
            ]

        batches, errors = await a_map(
            self._batches(),
            _batch,
            limit=config.max_concurrent,
            desc="Generating inputs" if config.show_progress else None,
        )
        self.errors = errors
        goldens = [g for batch in batches if batch for g in batch]
        # Batches are generated independently and can overlap; de-duplicate so a
        # run asked for 50 inputs does not quietly return 50 with 12 repeats.
        seen: set = set()
        unique: List[Golden] = []
        for golden in goldens:
            key = golden.input.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(golden)
        return unique[: self.num_goldens]
