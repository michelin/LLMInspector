"""Result exporters — the public reporting surface over ``EvaluationResult``.

Thin, stable functions the public API re-exports. They wrap
:class:`~llminspector.evaluate.result.EvaluationResult` so downstream code has a
single import point for turning results into DataFrames / Excel / a numeric
summary, independent of the result object's internals.
"""

from __future__ import annotations

from numbers import Number
from typing import Any, Dict

import pandas as pd

from ..evaluate.result import EvaluationResult


def to_dataframe(result: EvaluationResult) -> pd.DataFrame:
    """Return the evaluation result as a DataFrame (source + metric columns)."""
    return result.to_pandas()


def to_excel(result: EvaluationResult, path: str) -> None:
    """Write the evaluation result to an ``.xlsx`` file."""
    result.to_excel(path)


def summary(result: EvaluationResult) -> Dict[str, Dict[str, float]]:
    """Aggregate stats (count / mean / min / max) per numeric metric column.

    Non-numeric metric columns (labels, lists, dicts) are skipped. Useful for a
    quick at-a-glance report over an evaluation run.
    """
    df = result.to_pandas()
    stats: Dict[str, Dict[str, float]] = {}
    for column in df.columns:
        numeric = [v for v in df[column] if isinstance(v, Number) and not _is_bool(v)]
        if not numeric:
            continue
        stats[column] = {
            "count": len(numeric),
            "mean": sum(numeric) / len(numeric),
            "min": min(numeric),
            "max": max(numeric),
        }
    return stats


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)
