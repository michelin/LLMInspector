"""Default adversarial attack source — the static curated bank.

Ports the legacy ``Adversarial`` sample/melt/filter algorithm unchanged. This is
the seam a future **red-teaming** engine (active attack generation against a
model) will replace: implement :class:`AttackSource` and inject it into
``AdversarialSynthesizer`` instead of this class.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ...dataset.golden import Golden
from .base import AttackSource

_ID_VARS = ["Capability", "Sub Capability", "Prompt"]


class CuratedBankSource(AttackSource):
    """Samples / filters a curated bank DataFrame of adversarial prompts.

    Parameters
    ----------
    bank_df:
        The curated adversarial bank (columns include ``Capability``,
        ``Sub Capability``, ``Prompt`` plus per-attack variant columns).
    capability / subcapability:
        Filter to a capability / sub-capability. ``None`` (or ``"all"``) selects
        everything (a random sample of ``sample_size`` when both are unset).
    sample_size:
        Rows to sample when no capability/sub-capability filter is applied.
    """

    def __init__(
        self,
        bank_df: pd.DataFrame,
        capability: Optional[str] = None,
        subcapability: Optional[str] = None,
        sample_size: int = 1000,
    ) -> None:
        self.bank_df = bank_df
        self.sample_size = sample_size

        if capability is not None and capability.lower() == "all":
            capability = None
        if subcapability is not None and subcapability.lower() == "all":
            subcapability = None

        self.capability = capability if capability is not None else []
        self.subcapability = subcapability if subcapability is not None else []

    def _random_selection(self) -> pd.DataFrame:
        sample_record_df = self.bank_df.sample(n=self.sample_size)
        return sample_record_df

    def _filtered(self) -> pd.DataFrame:
        cap, subcap = self.capability, self.subcapability
        if cap != [] and subcap == []:
            return self.bank_df[self.bank_df["Capability"].str.lower() == cap.lower()]
        if cap == [] and subcap != []:
            return self.bank_df[
                self.bank_df["Sub Capability"].str.lower() == subcap.lower()
            ]
        if cap != [] and subcap != []:
            return self.bank_df[
                (self.bank_df["Capability"].str.lower() == cap.lower())
                & (self.bank_df["Sub Capability"].str.lower() == subcap.lower())
            ]
        return self._random_selection()

    def to_dataframe(self) -> pd.DataFrame:
        """The legacy melted/filtered output (Capability / Sub Capability /
        Prompt / Char Len)."""
        filtered = self._filtered()
        df_melt = filtered.melt(id_vars=_ID_VARS)
        transformed = df_melt.filter([*_ID_VARS, "value"])
        transformed = transformed.rename(columns={"value": "Char Len"})
        return transformed

    def generate(self) -> List[Golden]:
        df = self.to_dataframe()
        goldens: List[Golden] = []
        for _, row in df.iterrows():
            prompt = row["Prompt"]
            if prompt is None or str(prompt).strip() == "":
                continue
            goldens.append(
                Golden(
                    input=str(prompt),
                    metadata={
                        "Capability": row.get("Capability"),
                        "Sub Capability": row.get("Sub Capability"),
                        "Char Len": row.get("Char Len"),
                    },
                )
            )
        return goldens
