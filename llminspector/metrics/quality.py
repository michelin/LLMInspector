"""Quality metrics — BERTScore.

Ported from ``EvalMetrics.bertscore_async``. The BERTScorer model is loaded
once per process and shared across instances via an ``lru_cache`` loader.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any

from .base_metric import BaseMetric

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_bert_scorer():
    from bert_score import BERTScorer

    logger.info("Loading BERTScore model...")
    return BERTScorer(model_type="microsoft/deberta-base-mnli")


class BertScoreMetric(BaseMetric):
    """BERTScore F1 between the actual output (prediction) and expected output
    (reference). Local model; no LLM required.
    """

    metric_name = "bert_score"
    sort_key = 300
    required_inputs = {"actual_output", "expected_output"}

    def _scorer(self):
        return _get_bert_scorer()

    async def a_measure(self, test_case: Any) -> Any:
        prediction = test_case.actual_output
        reference = test_case.expected_output
        try:
            loop = asyncio.get_event_loop()
            model = self._scorer()
            _, _, f1 = await loop.run_in_executor(
                None, lambda: model.score([prediction], [reference])
            )
            self.score = float(f1.mean())
        except Exception as e:  # noqa: BLE001 - mirror legacy behavior
            self.record_failure(e)
            self.score = None
        self.is_successful()
        return self.score
