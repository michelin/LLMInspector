"""Swappable synthesis engines (Phase 5).

The ABCs are the stable seams; the ``Legacy*`` / ``Ragas*`` / ``Curated*``
classes are today's default implementations, each isolating its heavy dependency
(transformers / ragas / pandas) so future replacements are drop-in.
"""

from .base import AlignmentEngine, AttackSource, TestsetBackend
from .curated_bank import CuratedBankSource
from .legacy_alignment import LegacyTagT5Engine
from .ragas_testset import RagasTestsetBackend

__all__ = [
    "AlignmentEngine",
    "TestsetBackend",
    "AttackSource",
    "LegacyTagT5Engine",
    "RagasTestsetBackend",
    "CuratedBankSource",
]
