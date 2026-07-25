"""Evaluation engine — ``evaluate()`` / ``a_evaluate()`` (Phase 4)."""

from .evaluate import a_evaluate, evaluate
from .result import EvaluationResult, ResultColumns

__all__ = ["evaluate", "a_evaluate", "EvaluationResult", "ResultColumns"]
