"""Swappable synthesis engines — the seams slated for replacement.

Each synthesizer delegates its *replaceable core* to one of these strategies,
defaulting to today's legacy implementation. Future work (custom testset
generation, dropping ragas, red-teaming instead of static filtering) means
writing a new subclass and injecting it — with **no change** to the synthesizer
classes, their callers, or the :class:`~llminspector.dataset.golden.Golden`
output contract.

    AlignmentEngine   tag-augment -> paraphrase -> perturb   (LegacyTagT5Engine)
    TestsetBackend    document -> RAG testset + GT refine     (RagasTestsetBackend)
    AttackSource      adversarial prompts                     (CuratedBankSource)

Every engine returns ``List[Golden]`` so the synthesizer can wrap it in an
``EvaluationDataset`` uniformly, regardless of how the goldens were produced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ...dataset.golden import Golden


class AlignmentEngine(ABC):
    """Produces alignment goldens (augmented/paraphrased/perturbed prompts)."""

    @abstractmethod
    def generate(self) -> List[Golden]:
        ...


class TestsetBackend(ABC):
    """Produces RAG goldens (question / ground_truth / context) from documents."""

    @abstractmethod
    def generate(self) -> List[Golden]:
        ...


class AttackSource(ABC):
    """Produces adversarial goldens (attack prompts by capability)."""

    @abstractmethod
    def generate(self) -> List[Golden]:
        ...
