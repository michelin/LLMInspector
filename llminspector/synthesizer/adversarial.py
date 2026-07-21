"""``AdversarialSynthesizer`` — adversarial test data from an attack source.

Today's default source is the static :class:`CuratedBankSource`. Inject a
different :class:`~llminspector.synthesizer.engines.base.AttackSource` (e.g. a
future red-teaming generator) to change *how* attacks are produced without
touching this class or its callers.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..dataset.dataset import EvaluationDataset
from .base import BaseSynthesizer
from .engines import AttackSource, CuratedBankSource


class AdversarialSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        bank_df: Optional[pd.DataFrame] = None,
        *,
        capability: Optional[str] = None,
        subcapability: Optional[str] = None,
        sample_size: int = 1000,
        source: Optional[AttackSource] = None,
    ) -> None:
        super().__init__()
        if source is None:
            if bank_df is None:
                raise ValueError("Provide either `bank_df` or a `source`.")
            source = CuratedBankSource(
                bank_df,
                capability=capability,
                subcapability=subcapability,
                sample_size=sample_size,
            )
        self.source = source

    @classmethod
    def from_excel(cls, path: str, **kwargs) -> "AdversarialSynthesizer":
        return cls(pd.read_excel(path), **kwargs)

    def generate(self) -> EvaluationDataset:
        return self._store(self.source.generate())
