"""``AdversarialGenerator`` — a :class:`Generator` preset for attack banks.

Adversarial generation is a plain :class:`~llminspector.generation.generator.Generator`
over a :class:`~llminspector.generation.sources.curated_bank.CuratedBankSource`
with **no stages**: the bank's prompts are the goldens, and nothing is filtered,
evolved or restyled. There is no special machinery, only a preset.

This class exists purely so ``from_dataframe`` / ``from_excel`` keep working —
without it every caller would have to assemble the source and the generator by
hand at the point of use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd

from .generator import Generator
from .sources.curated_bank import CuratedBankSource

if TYPE_CHECKING:  # pragma: no cover
    from .config import GenerationConfig

__all__ = ["AdversarialGenerator"]


class AdversarialGenerator(Generator):
    """A generator over a curated attack bank; see :meth:`from_dataframe`."""

    def __init__(
        self,
        source: CuratedBankSource,
        *,
        config: Optional["GenerationConfig"] = None,
    ) -> None:
        # No stages: an attack prompt is used verbatim. Rewriting or evolving it
        # would change the attack, which is the one thing a curated bank exists
        # to hold constant.
        super().__init__(source, stages=(), config=config)

    @classmethod
    def from_dataframe(
        cls,
        bank_df: pd.DataFrame,
        *,
        capability: Optional[str] = None,
        subcapability: Optional[str] = None,
        sample_size: int = 1000,
        config: Optional["GenerationConfig"] = None,
    ) -> "AdversarialGenerator":
        """Build the default :class:`CuratedBankSource` from an attack bank."""
        return cls(
            CuratedBankSource(
                bank_df,
                capability=capability,
                subcapability=subcapability,
                sample_size=sample_size,
            ),
            config=config,
        )

    @classmethod
    def from_excel(cls, path: str, **kwargs) -> "AdversarialGenerator":
        return cls.from_dataframe(pd.read_excel(path), **kwargs)
