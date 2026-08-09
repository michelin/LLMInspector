"""Dataset generation — sources, stages, and the generator that runs them.

The pipeline is one shape regardless of what is being generated::

    Generator(source, stages) -> GenerationResult

A :class:`GoldenSource` produces the starting goldens (a curated attack bank, a
set of contexts, a corpus of documents, a styling description). An ordered list
of :class:`Stage` objects then transforms each one — filter, evolve, style,
answer — with a stage returning ``None`` to discard a golden and recording why.

Presets (:class:`AdversarialGenerator`, :class:`RagGenerator`) are thin
subclasses that pick a source; they add no machinery of their own.
"""

from .adversarial import AdversarialGenerator
from .config import (
    EvolutionConfig,
    FiltrationConfig,
    GenerationConfig,
    StylingConfig,
)
from .generator import GenerationResult, Generator
from .rag import RagGenerator
from .source import GoldenSource, SyncGoldenSource
from .sources import ContextSource, CuratedBankSource
from .sources.ragas_testset import RagasTestsetBackend
from .stage import Stage, StageContext
from .stages import (
    EvolutionStage,
    ExpectedOutputStage,
    FiltrationStage,
    PerturbationStage,
    StylingStage,
    default_stages,
)

__all__ = [
    # the pipeline
    "Generator",
    "GenerationResult",
    "GoldenSource",
    "SyncGoldenSource",
    "Stage",
    "StageContext",
    # the stage chain
    "default_stages",
    "FiltrationStage",
    "EvolutionStage",
    "StylingStage",
    "ExpectedOutputStage",
    "PerturbationStage",
    # configuration
    "GenerationConfig",
    "FiltrationConfig",
    "EvolutionConfig",
    "StylingConfig",
    # presets and the sources behind them
    "AdversarialGenerator",
    "RagGenerator",
    "ContextSource",
    "CuratedBankSource",
    "RagasTestsetBackend",
]
