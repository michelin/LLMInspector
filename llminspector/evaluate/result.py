"""``EvaluationResult`` — the object returned by :func:`evaluate`.

Holds the per-row metric dictionaries (already expanded + reordered to match the
legacy ``reorder_results`` contract) alongside the source test cases, and
serializes them to a pandas DataFrame that prepends the original
question/answer/ground_truth/contexts/policy columns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..dataset.dataset import ColumnMapping
from ..test_case.test_case import LLMTestCase


class EvaluationResult:
    """Container for evaluation output.

    Parameters
    ----------
    rows:
        One ordered metric dict per test case (metric name -> value), already
        column-expanded and reordered.
    test_cases:
        The evaluated test cases, aligned positionally with ``rows``.
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        test_cases: List[LLMTestCase],
    ) -> None:
        self.rows: List[Dict[str, Any]] = rows
        self.test_cases: List[LLMTestCase] = test_cases

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return f"EvaluationResult(rows={len(self.rows)})"

    def _metric_columns(self) -> List[str]:
        """Union of metric keys across rows, preserving first-seen order."""
        columns: List[str] = []
        seen = set()
        for row in self.rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
        return columns

    def to_pandas(self, mapping: Optional[ColumnMapping] = None) -> pd.DataFrame:
        """Source columns (question/answer/...) followed by metric columns."""
        mapping = mapping or ColumnMapping()
        records = []
        for tc, row in zip(self.test_cases, self.rows):
            record = {
                mapping.input_col: tc.input,
                mapping.actual_output_col: tc.actual_output,
                mapping.expected_output_col: tc.expected_output,
                mapping.retrieval_context_col: tc.retrieval_context,
                mapping.policy_col: tc.policy,
            }
            record.update(row)
            records.append(record)

        source_cols = [
            mapping.input_col,
            mapping.actual_output_col,
            mapping.expected_output_col,
            mapping.retrieval_context_col,
            mapping.policy_col,
        ]
        columns = source_cols + self._metric_columns()
        return pd.DataFrame(records, columns=columns)

    def to_excel(self, path: str, mapping: Optional[ColumnMapping] = None) -> None:
        """Write :meth:`to_pandas` to an ``.xlsx`` file."""
        self.to_pandas(mapping=mapping).to_excel(path, index=False)
