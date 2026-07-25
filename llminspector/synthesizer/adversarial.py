"""``AdversarialSynthesizer`` — adversarial test data from an attack source.

Today's default source is the static :class:`CuratedBankSource`. Inject a
different :class:`~llminspector.synthesizer.engines.base.AttackSource` (e.g. a
future red-teaming generator) to change *how* attacks are produced without
touching this class or its callers.

Construction is split in two (Phase 8.6) — see
:mod:`llminspector.synthesizer.alignment` for the reasoning.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from ..dataset.dataset import EvaluationDataset
from .base import BaseSynthesizer
from .engines import AttackSource, CuratedBankSource


class AdversarialSynthesizer(BaseSynthesizer):
    """Wraps an :class:`AttackSource`; see :meth:`from_dataframe`."""

    def __init__(self, source: AttackSource) -> None:
        super().__init__()
        self.source = source

    @classmethod
    def from_dataframe(
        cls,
        bank_df: pd.DataFrame,
        *,
        capability: Optional[str] = None,
        subcapability: Optional[str] = None,
        sample_size: int = 1000,
    ) -> "AdversarialSynthesizer":
        """Build the default :class:`CuratedBankSource` from an attack bank."""
        return cls(
            CuratedBankSource(
                bank_df,
                capability=capability,
                subcapability=subcapability,
                sample_size=sample_size,
            )
        )

    @classmethod
    def from_excel(cls, path: str, **kwargs) -> "AdversarialSynthesizer":
        return cls.from_dataframe(pd.read_excel(path), **kwargs)

    @property
    def metadata_keys(self) -> Tuple[str, ...]:
        """Metadata columns the source emits on every golden."""
        return self.source.metadata_keys

    def generate(self) -> EvaluationDataset:
        return self._store(self.source.generate())
