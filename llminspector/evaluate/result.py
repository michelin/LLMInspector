"""``EvaluationResult`` — the object returned by :func:`evaluate`.

Holds the per-row metric dictionaries (already expanded + reordered) alongside
the source test cases, and serializes them to a pandas DataFrame that prepends
the original question/answer/ground_truth/contexts/policy columns.

Export column names come from :class:`ResultColumns`, not from the dataset's
``ColumnMapping``. They looked interchangeable — the same five names with the
same defaults — but one describes *how to read* a spreadsheet and the other
*how to name* the output, and only the read side has to match the file on disk.
Reusing one type for both meant renaming an input column silently renamed an
output column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ..dataset.dataset import ColumnMapping
from ..test_case.test_case import LLMTestCase


@dataclass
class ResultColumns:
    """Names for the source columns prepended to the exported result table.

    The **write** schema. Defaults match :class:`ColumnMapping`'s so existing
    exports are unchanged.
    """

    input_col: str = "question"
    actual_output_col: str = "answer"
    expected_output_col: str = "ground_truth"
    retrieval_context_col: str = "contexts"
    policy_col: str = "policy"

    @classmethod
    def from_column_mapping(cls, mapping: ColumnMapping) -> "ResultColumns":
        """Mirror a dataset's read schema onto the write side, when wanted."""
        return cls(
            input_col=mapping.input_col,
            actual_output_col=mapping.actual_output_col,
            expected_output_col=mapping.expected_output_col,
            retrieval_context_col=mapping.retrieval_context_col,
            policy_col=mapping.policy_col,
        )


class EvaluationResult:
    """Container for evaluation output.

    Parameters
    ----------
    rows:
        One ordered metric dict per test case (metric name -> value), already
        column-expanded and reordered.
    test_cases:
        The evaluated test cases, aligned positionally with ``rows``.
    errors:
        Metric failures collected during the run, as
        ``{"row": int, "error": "metric_name: ExcType: message"}``. A metric
        that fails scores ``None``, which is indistinguishable in the table from
        *skipped for missing input* — this is where the difference is visible.
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        test_cases: List[LLMTestCase],
        errors: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.rows: List[Dict[str, Any]] = rows
        self.test_cases: List[LLMTestCase] = test_cases
        self.errors: List[Dict[str, Any]] = errors or []

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        errors = f", errors={len(self.errors)}" if self.errors else ""
        return f"EvaluationResult(rows={len(self.rows)}{errors})"

    def error_summary(self) -> Dict[str, int]:
        """Failure count per metric name, most frequent first.

        A run where every call returned 401 shows up here as one entry with a
        count equal to the number of rows.
        """
        counts: Dict[str, int] = {}
        for entry in self.errors:
            metric = str(entry.get("error", "")).split(":", 1)[0]
            counts[metric] = counts.get(metric, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

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

    def to_pandas(self, columns: Optional[ResultColumns] = None) -> pd.DataFrame:
        """Source columns (question/answer/...) followed by metric columns.

        ``columns`` names the source columns in the output. Pass a
        :class:`ResultColumns`, or ``ResultColumns.from_column_mapping(m)`` to
        echo the names a dataset was read with.
        """
        columns = columns or ResultColumns()
        records = []
        for tc, row in zip(self.test_cases, self.rows):
            record = {
                columns.input_col: tc.input,
                columns.actual_output_col: tc.actual_output,
                columns.expected_output_col: tc.expected_output,
                columns.retrieval_context_col: tc.retrieval_context,
                columns.policy_col: tc.policy,
            }
            record.update(row)
            records.append(record)

        source_cols = [
            columns.input_col,
            columns.actual_output_col,
            columns.expected_output_col,
            columns.retrieval_context_col,
            columns.policy_col,
        ]
        return pd.DataFrame(records, columns=source_cols + self._metric_columns())

    def to_excel(self, path: str, columns: Optional[ResultColumns] = None) -> None:
        """Write :meth:`to_pandas` to an ``.xlsx`` file."""
        self.to_pandas(columns=columns).to_excel(path, index=False)
