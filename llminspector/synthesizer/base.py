"""``BaseSynthesizer`` — the stable synthesizer contract.

Every synthesizer (legacy or a future custom engine) implements ``generate()``
returning an :class:`~llminspector.dataset.dataset.EvaluationDataset` of
:class:`~llminspector.dataset.golden.Golden`. Callers depend only on this, so
swapping the internal engine never requires a caller-side refactor.

Goldens carry synthesizer-specific columns in ``metadata``; :meth:`to_pandas`
flattens core fields + metadata back into a DataFrame via
:func:`~llminspector.dataset.dataset.goldens_to_dataframe`, which lives in
``dataset/`` because flattening goldens is a dataset concern — the synthesizer
layer only happens to have been its first caller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from ..dataset.dataset import EvaluationDataset, goldens_to_dataframe
from ..dataset.golden import Golden


class BaseSynthesizer(ABC):
    """Abstract synthesizer producing an ``EvaluationDataset`` of goldens."""

    def __init__(self) -> None:
        self.dataset: Optional[EvaluationDataset] = None

    @abstractmethod
    def generate(self) -> EvaluationDataset:
        """Produce and return an ``EvaluationDataset`` (also stored on ``self``)."""

    def to_pandas(self) -> pd.DataFrame:
        """Flatten the last-generated goldens (core fields + metadata columns)."""
        dataset = self.dataset if self.dataset is not None else self.generate()
        return goldens_to_dataframe(dataset.goldens)

    def to_excel(self, path: str) -> None:
        self.to_pandas().to_excel(path, index=False)

    def _store(self, goldens: List[Golden]) -> EvaluationDataset:
        self.dataset = EvaluationDataset(goldens=goldens)
        return self.dataset
