"""``AlignmentSynthesizer`` — alignment test data via an alignment engine.

Today's default engine is :class:`LegacyTagT5Engine` (tag-augment -> HF-T5
paraphrase -> perturb). Inject a different
:class:`~llminspector.synthesizer.engines.base.AlignmentEngine` (e.g. a future
custom generator) to change *how* prompts are produced without touching this
class or its callers.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from ..dataset.dataset import EvaluationDataset
from .base import BaseSynthesizer
from .engines import AlignmentEngine, LegacyTagT5Engine


class AlignmentSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        alignment_df: Optional[pd.DataFrame] = None,
        *,
        tag_keyword_dict: Optional[Dict[str, list]] = None,
        augmentation_dict: Optional[Dict[str, list]] = None,
        augmentations: Optional[Dict[str, Tuple[str, float]]] = None,
        paraphrase_count: int = 3,
        engine: Optional[AlignmentEngine] = None,
    ) -> None:
        super().__init__()
        if engine is None:
            if alignment_df is None:
                raise ValueError("Provide either `alignment_df` or an `engine`.")
            engine = LegacyTagT5Engine(
                alignment_df=alignment_df,
                tag_keyword_dict=tag_keyword_dict or {},
                augmentation_dict=augmentation_dict or {},
                augmentations=augmentations or {},
                paraphrase_count=paraphrase_count,
            )
        self.engine = engine

    @classmethod
    def from_excel(cls, path: str, **kwargs) -> "AlignmentSynthesizer":
        return cls(pd.read_excel(path), **kwargs)

    def generate(self) -> EvaluationDataset:
        return self._store(self.engine.generate())
