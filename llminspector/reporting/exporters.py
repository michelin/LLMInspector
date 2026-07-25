"""Reporting — analysis over an ``EvaluationResult``.

This module holds what reporting *adds*. Serializing a result is the result
object's own job:

    result.to_pandas()     -> DataFrame (source + metric columns)
    result.to_excel(path)  -> .xlsx

``reporting.to_dataframe`` / ``reporting.to_excel`` used to forward to those
identically-named methods and add nothing, so every export had two doors. Phase
8.6 removed the forwarders — call the methods on the result.
"""

from __future__ import annotations

from numbers import Number
from typing import Any, Dict

import pandas as pd

from ..evaluate.result import EvaluationResult


def summary(result: EvaluationResult) -> Dict[str, Dict[str, float]]:
    """Aggregate stats (count / mean / min / max) per numeric metric column.

    Non-numeric metric columns (labels, lists, dicts) are skipped. Useful for a
    quick at-a-glance report over an evaluation run.
    """
    df = result.to_pandas()
    stats: Dict[str, Dict[str, float]] = {}
    for column in df.columns:
        numeric = [
            float(v)  # type: ignore[arg-type]
            for v in df[column]
            if isinstance(v, Number) and not _is_bool(v)
        ]
        if not numeric:
            continue
        stats[column] = {
            "count": len(numeric),
            "mean": sum(numeric) / len(numeric),
            "min": min(numeric),
            "max": max(numeric),
        }
    return stats


def errors(result: EvaluationResult) -> pd.DataFrame:
    """Metric failures as a DataFrame (``row`` / ``metric`` / ``error``).

    Empty when the run was clean. Pair with :func:`summary`: a metric with a
    suspiciously low ``count`` there usually has rows here.
    """
    records = []
    for entry in result.errors:
        message = str(entry.get("error", ""))
        metric, _, detail = message.partition(": ")
        records.append(
            {"row": entry.get("row"), "metric": metric, "error": detail or message}
        )
    return pd.DataFrame(records, columns=["row", "metric", "error"])


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)
