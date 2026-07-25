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

``Golden.metadata`` is the escape hatch that makes that uniformity possible —
each engine hangs its own extra columns there. The cost is that the output shape
was undiscoverable without running the engine, so every engine now declares
``metadata_keys`` and ``to_pandas()``'s column set is knowable up front.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from ...dataset.golden import Golden


class _GoldenSource(ABC):
    """Shared contract: produce goldens, and declare the columns they carry."""

    #: Keys this engine puts in ``Golden.metadata``, in export order. These
    #: become the trailing columns of ``BaseSynthesizer.to_pandas()``.
    metadata_keys: Tuple[str, ...] = ()

    @abstractmethod
    def generate(self) -> List[Golden]: ...


class AlignmentEngine(_GoldenSource):
    """Produces alignment goldens (augmented/paraphrased/perturbed prompts)."""


class TestsetBackend(_GoldenSource):
    """Produces RAG goldens (question / ground_truth / context) from documents."""


class AttackSource(_GoldenSource):
    """Produces adversarial goldens (attack prompts by capability)."""
