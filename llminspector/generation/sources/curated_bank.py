"""Adversarial goldens from a static curated attack bank.

This is the seam a future **red-teaming** source (active attack generation
against a live model) replaces: implement
:class:`~llminspector.generation.source.GoldenSource` and hand it to a
:class:`~llminspector.generation.generator.Generator` instead of this class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from ...dataset.golden import Golden
from ..source import SyncGoldenSource

if TYPE_CHECKING:  # pragma: no cover
    from ..config import GenerationConfig

__all__ = ["CuratedBankSource"]

_CORE_COLUMNS = ["Capability", "Sub Capability", "Prompt"]
_CHAR_LEN = "Char Len"


class CuratedBankSource(SyncGoldenSource):
    """Samples / filters a curated bank DataFrame of adversarial prompts.

    Parameters
    ----------
    bank_df:
        The curated adversarial bank: ``Capability``, ``Sub Capability``,
        ``Prompt``, ``Char Len``.
    capability / subcapability:
        Filter to a capability / sub-capability. ``None`` (or ``"all"``) selects
        everything — a random sample of ``sample_size`` when both are unset.
    sample_size:
        Rows to sample when no capability/sub-capability filter is applied.
    """

    #: Declared so the output column set is knowable without running the source.
    metadata_keys = ("Capability", "Sub Capability", "Char Len")

    def __init__(
        self,
        bank_df: pd.DataFrame,
        capability: Optional[str] = None,
        subcapability: Optional[str] = None,
        sample_size: int = 1000,
    ) -> None:
        self.bank_df = bank_df
        self.sample_size = sample_size

        # "all" means "no filter", same as omitting it. The legacy code then
        # converted None to [] and used `!= []` as the unset sentinel, which
        # made the attribute's type str-or-list for no benefit; None is the
        # sentinel throughout now.
        if capability is not None and capability.lower() == "all":
            capability = None
        if subcapability is not None and subcapability.lower() == "all":
            subcapability = None

        self.capability: Optional[str] = capability
        self.subcapability: Optional[str] = subcapability

    def _random_selection(self, seed: Optional[int] = None) -> pd.DataFrame:
        # ``random_state`` is what makes GenerationConfig.seed mean something on
        # the unfiltered path. Without it this is the one branch of the whole
        # pipeline that stays unreproducible, which would make the config's
        # promise ("threads into every random draw") false.
        return self.bank_df.sample(n=self.sample_size, random_state=seed)

    def _filtered(self, seed: Optional[int] = None) -> pd.DataFrame:
        cap, subcap = self.capability, self.subcapability
        if cap is not None and subcap is None:
            return self.bank_df[self.bank_df["Capability"].str.lower() == cap.lower()]
        if cap is None and subcap is not None:
            return self.bank_df[
                self.bank_df["Sub Capability"].str.lower() == subcap.lower()
            ]
        if cap is not None and subcap is not None:
            return self.bank_df[
                (self.bank_df["Capability"].str.lower() == cap.lower())
                & (self.bank_df["Sub Capability"].str.lower() == subcap.lower())
            ]
        return self._random_selection(seed)

    def to_dataframe(self, seed: Optional[int] = None) -> pd.DataFrame:
        """The filtered bank: Capability / Sub Capability / Prompt / Char Len.

        This used to route through ``df.melt(id_vars=[...])``, on the strength
        of a docstring claiming the bank carried per-attack *variant* columns.
        No shipped bank does — the real schema has exactly one non-id column,
        ``Char Len`` — so the melt was a no-op that produced the right answer
        for the wrong reason, and was wrong for both of the shapes it claimed to
        support:

        * with several variant columns it emitted one golden **per variant with
          the same ``Prompt``**, i.e. duplicates, and stuffed each variant's
          value into a column labelled ``Char Len`` regardless of what it held;
        * with no non-id columns at all it melted to an empty frame, silently
          producing **zero goldens**.

        Selecting the columns directly is byte-identical for the real schema and
        has neither failure mode. ``Char Len`` is reindexed rather than required
        so a bank missing it yields ``NaN`` instead of a ``KeyError``.

        ``seed`` only affects the unfiltered path, which samples; the
        capability filters are deterministic already.
        """
        return self._filtered(seed).reindex(columns=[*_CORE_COLUMNS, _CHAR_LEN])

    def produce(self, config: "GenerationConfig") -> List[Golden]:
        goldens: List[Golden] = []
        for _, row in self.to_dataframe(seed=config.seed).iterrows():
            prompt = row["Prompt"]
            if prompt is None or str(prompt).strip() == "":
                continue
            goldens.append(
                Golden(
                    input=str(prompt),
                    metadata={
                        "Capability": row.get("Capability"),
                        "Sub Capability": row.get("Sub Capability"),
                        "Char Len": row.get(_CHAR_LEN),
                    },
                )
            )
        return goldens
