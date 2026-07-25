"""``AlignmentSynthesizer`` — alignment test data via an alignment engine.

Today's default engine is :class:`LegacyTagT5Engine` (tag-augment -> HF-T5
paraphrase -> perturb). Inject a different
:class:`~llminspector.synthesizer.engines.base.AlignmentEngine` (e.g. a future
custom generator) to change *how* prompts are produced without touching this
class or its callers.

Construction is split in two (Phase 8.6). The constructor takes an engine and
nothing else; the ``from_*`` classmethods build the default engine from raw
data. Previously one signature accepted *either* a DataFrame plus five
engine-config kwargs *or* a pre-built engine, and raised ``ValueError`` at
runtime when given neither — so the type signature described a call that was
never valid.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from ..dataset.dataset import EvaluationDataset
from .base import BaseSynthesizer
from .engines import AlignmentEngine, LegacyTagT5Engine


class AlignmentSynthesizer(BaseSynthesizer):
    """Wraps an :class:`AlignmentEngine`; see :meth:`from_dataframe`."""

    def __init__(self, engine: AlignmentEngine) -> None:
        super().__init__()
        self.engine = engine

    @classmethod
    def from_dataframe(
        cls,
        alignment_df: pd.DataFrame,
        *,
        tag_keyword_dict: Optional[Dict[str, list]] = None,
        augmentation_dict: Optional[Dict[str, list]] = None,
        augmentations: Optional[Dict[str, Tuple[str, float]]] = None,
        paraphrase_count: int = 3,
    ) -> "AlignmentSynthesizer":
        """Build the default :class:`LegacyTagT5Engine` from a seed frame."""
        return cls(
            LegacyTagT5Engine(
                alignment_df=alignment_df,
                tag_keyword_dict=tag_keyword_dict or {},
                augmentation_dict=augmentation_dict or {},
                augmentations=augmentations or {},
                paraphrase_count=paraphrase_count,
            )
        )

    @classmethod
    def from_excel(cls, path: str, **kwargs) -> "AlignmentSynthesizer":
        return cls.from_dataframe(pd.read_excel(path), **kwargs)

    @property
    def metadata_keys(self) -> Tuple[str, ...]:
        """Metadata columns the engine emits on every golden."""
        return self.engine.metadata_keys

    def generate(self) -> EvaluationDataset:
        return self._store(self.engine.generate())
