"""The ``EvaluationDataset`` — a container of test cases and/or goldens with
pandas / Excel (de)serialization and configurable column-name mapping.

Column defaults reproduce the legacy ``helper.py`` ``*_col`` arguments
(``question / answer / ground_truth / contexts / policy``) so existing
spreadsheets load unchanged.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd

from ..test_case.test_case import LLMTestCase
from .golden import Golden


@dataclass
class ColumnMapping:
    """Maps :class:`LLMTestCase` attributes to spreadsheet column names.

    Defaults mirror the legacy ``helper.process_batch_metrics`` column args.
    """

    input_col: str = "question"
    actual_output_col: str = "answer"
    expected_output_col: str = "ground_truth"
    retrieval_context_col: str = "contexts"
    policy_col: str = "policy"


@dataclass
class GoldenColumnMapping:
    """Maps :class:`Golden` attributes to spreadsheet column names."""

    input_col: str = "question"
    expected_output_col: str = "ground_truth"
    context_col: str = "contexts"


def _cell(value: Any) -> Optional[str]:
    """Normalize a scalar spreadsheet cell to ``str`` or ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _parse_context(value: Any) -> Optional[List[str]]:
    """Parse a context cell into a list of strings.

    Handles native lists/tuples, ``repr``-style ``"['a', 'b']"`` strings
    (as written back by :meth:`to_excel`), and plain single-context strings.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value if str(v).strip() != ""]
        return items or None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text == "":
        return None
    if text[:1] in "[(":
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                items = [str(v) for v in parsed if str(v).strip() != ""]
                return items or None
        except (ValueError, SyntaxError):
            pass
    return [text]


class EvaluationDataset:
    """Holds evaluation ``test_cases`` and synthesis ``goldens``.

    Construct directly, or via the ``from_pandas`` / ``from_excel`` (test
    cases) and ``goldens_from_pandas`` / ``goldens_from_excel`` classmethods.
    """

    def __init__(
        self,
        test_cases: Optional[List[LLMTestCase]] = None,
        goldens: Optional[List[Golden]] = None,
    ) -> None:
        self.test_cases: List[LLMTestCase] = list(test_cases or [])
        self.goldens: List[Golden] = list(goldens or [])

    def __len__(self) -> int:
        return len(self.test_cases)

    def __repr__(self) -> str:
        return (
            f"EvaluationDataset(test_cases={len(self.test_cases)}, "
            f"goldens={len(self.goldens)})"
        )

    # -- test cases -----------------------------------------------------------

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        mapping: Optional[ColumnMapping] = None,
        **col_overrides: str,
    ) -> "EvaluationDataset":
        """Build a dataset of :class:`LLMTestCase` from a DataFrame."""
        mapping = _resolve_mapping(ColumnMapping, mapping, col_overrides)
        test_cases: List[LLMTestCase] = []
        for _, row in df.iterrows():
            input_value = _cell(row.get(mapping.input_col))
            if input_value is None:
                continue
            test_cases.append(
                LLMTestCase(
                    input=input_value,
                    actual_output=_cell(row.get(mapping.actual_output_col)),
                    expected_output=_cell(row.get(mapping.expected_output_col)),
                    retrieval_context=_parse_context(
                        row.get(mapping.retrieval_context_col)
                    ),
                    policy=_cell(row.get(mapping.policy_col)),
                )
            )
        return cls(test_cases=test_cases)

    @classmethod
    def from_excel(
        cls,
        path: str,
        sheet_name: Any = 0,
        mapping: Optional[ColumnMapping] = None,
        **col_overrides: str,
    ) -> "EvaluationDataset":
        """Build a dataset of :class:`LLMTestCase` from an ``.xlsx`` file."""
        df = pd.read_excel(path, sheet_name=sheet_name)
        return cls.from_pandas(df, mapping=mapping, **col_overrides)

    def to_pandas(
        self,
        mapping: Optional[ColumnMapping] = None,
        **col_overrides: str,
    ) -> pd.DataFrame:
        """Serialize ``test_cases`` back to a DataFrame using the mapping."""
        mapping = _resolve_mapping(ColumnMapping, mapping, col_overrides)
        rows = []
        for tc in self.test_cases:
            rows.append(
                {
                    mapping.input_col: tc.input,
                    mapping.actual_output_col: tc.actual_output,
                    mapping.expected_output_col: tc.expected_output,
                    mapping.retrieval_context_col: tc.retrieval_context,
                    mapping.policy_col: tc.policy,
                }
            )
        return pd.DataFrame(rows, columns=_test_case_columns(mapping))

    def to_excel(
        self,
        path: str,
        mapping: Optional[ColumnMapping] = None,
        **col_overrides: str,
    ) -> None:
        """Write ``test_cases`` to an ``.xlsx`` file."""
        self.to_pandas(mapping=mapping, **col_overrides).to_excel(path, index=False)

    # -- goldens --------------------------------------------------------------

    @classmethod
    def goldens_from_pandas(
        cls,
        df: pd.DataFrame,
        mapping: Optional[GoldenColumnMapping] = None,
        **col_overrides: str,
    ) -> "EvaluationDataset":
        """Build a dataset of :class:`Golden` from a DataFrame."""
        mapping = _resolve_mapping(GoldenColumnMapping, mapping, col_overrides)
        goldens: List[Golden] = []
        for _, row in df.iterrows():
            input_value = _cell(row.get(mapping.input_col))
            if input_value is None:
                continue
            goldens.append(
                Golden(
                    input=input_value,
                    expected_output=_cell(row.get(mapping.expected_output_col)),
                    context=_parse_context(row.get(mapping.context_col)),
                )
            )
        return cls(goldens=goldens)

    @classmethod
    def goldens_from_excel(
        cls,
        path: str,
        sheet_name: Any = 0,
        mapping: Optional[GoldenColumnMapping] = None,
        **col_overrides: str,
    ) -> "EvaluationDataset":
        """Build a dataset of :class:`Golden` from an ``.xlsx`` file."""
        df = pd.read_excel(path, sheet_name=sheet_name)
        return cls.goldens_from_pandas(df, mapping=mapping, **col_overrides)

    def goldens_to_pandas(
        self,
        mapping: Optional[GoldenColumnMapping] = None,
        **col_overrides: str,
    ) -> pd.DataFrame:
        """Serialize ``goldens`` back to a DataFrame using the mapping."""
        mapping = _resolve_mapping(GoldenColumnMapping, mapping, col_overrides)
        rows = []
        for g in self.goldens:
            rows.append(
                {
                    mapping.input_col: g.input,
                    mapping.expected_output_col: g.expected_output,
                    mapping.context_col: g.context,
                }
            )
        columns = [
            mapping.input_col,
            mapping.expected_output_col,
            mapping.context_col,
        ]
        return pd.DataFrame(rows, columns=columns)

    def goldens_to_excel(
        self,
        path: str,
        mapping: Optional[GoldenColumnMapping] = None,
        **col_overrides: str,
    ) -> None:
        """Write ``goldens`` to an ``.xlsx`` file."""
        self.goldens_to_pandas(mapping=mapping, **col_overrides).to_excel(
            path, index=False
        )


def _test_case_columns(mapping: ColumnMapping) -> List[str]:
    return [
        mapping.input_col,
        mapping.actual_output_col,
        mapping.expected_output_col,
        mapping.retrieval_context_col,
        mapping.policy_col,
    ]


def _resolve_mapping(mapping_cls, mapping, col_overrides):
    """Merge an explicit mapping with per-column keyword overrides."""
    base = mapping if mapping is not None else mapping_cls()
    if col_overrides:
        valid = set(base.__dataclass_fields__)
        unknown = set(col_overrides) - valid
        if unknown:
            raise TypeError(
                f"Unknown column argument(s): {sorted(unknown)}. "
                f"Valid: {sorted(valid)}"
            )
        base = mapping_cls(**{**base.__dict__, **col_overrides})
    return base
