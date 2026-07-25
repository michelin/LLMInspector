"""Reporting — analysis over an ``EvaluationResult`` (Phase 6).

Serialization lives on the result object (``result.to_pandas()`` /
``result.to_excel(path)``); this namespace holds what reporting adds on top.
"""

from .exporters import errors, summary

__all__ = ["summary", "errors"]
