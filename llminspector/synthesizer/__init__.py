"""Synthetic data generation — alignment / adversarial / RAG (Phase 5).

Public surface: the three synthesizers (stable ``generate() -> EvaluationDataset``
contract) plus the swappable engine ABCs + default implementations. See
``engines/`` for the seams that future custom / red-teaming / ragas-free
implementations replace.
"""

from .adversarial import AdversarialSynthesizer
from .alignment import AlignmentSynthesizer
from .base import BaseSynthesizer
from .engines import (
    AlignmentEngine,
    AttackSource,
    CuratedBankSource,
    LegacyTagT5Engine,
    RagasTestsetBackend,
    TestsetBackend,
)
from .rag import RagSynthesizer

__all__ = [
    "BaseSynthesizer",
    "AlignmentSynthesizer",
    "AdversarialSynthesizer",
    "RagSynthesizer",
    # engine seams
    "AlignmentEngine",
    "TestsetBackend",
    "AttackSource",
    "LegacyTagT5Engine",
    "RagasTestsetBackend",
    "CuratedBankSource",
]
