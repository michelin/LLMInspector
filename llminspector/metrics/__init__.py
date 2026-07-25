"""Class-based metrics — ``BaseMetric`` and concrete metrics (Phase 3)."""

from .aggregate import calculate_total_tokens
from .base_metric import BaseMetric, DualTargetMetric, RagasBackedMetric
from .nlp import (
    EmotionMetric,
    LanguageDetectionMetric,
    ReadabilityMetric,
    SentimentMetric,
    TokenCountMetric,
)
from .policy import PolicyComplianceMetric
from .quality import BertScoreMetric
from .rag import (
    AnswerCorrectnessMetric,
    AnswerRelevancyMetric,
    ConcisenessMetric,
    ContextEntityRecallMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    ContextRelevanceMetric,
    ContextUtilisationMetric,
    FaithfulnessMetric,
)
from .safety import (
    AnswerJailbreakMetric,
    CodeDetectMetric,
    ContentModerationMetric,
    HallucinationMetric,
    PIIDetectionMetric,
    QuestionJailbreakMetric,
    RefusalMetric,
)

__all__ = [
    "BaseMetric",
    "DualTargetMetric",
    "RagasBackedMetric",
    # quality
    "BertScoreMetric",
    # rag
    "FaithfulnessMetric",
    "AnswerCorrectnessMetric",
    "AnswerRelevancyMetric",
    "ConcisenessMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "ContextUtilisationMetric",
    "ContextRelevanceMetric",
    "ContextEntityRecallMetric",
    # safety
    "PIIDetectionMetric",
    "ContentModerationMetric",
    "QuestionJailbreakMetric",
    "AnswerJailbreakMetric",
    "RefusalMetric",
    "HallucinationMetric",
    "CodeDetectMetric",
    # nlp
    "SentimentMetric",
    "EmotionMetric",
    "LanguageDetectionMetric",
    "ReadabilityMetric",
    "TokenCountMetric",
    # policy
    "PolicyComplianceMetric",
    # aggregate
    "calculate_total_tokens",
]
