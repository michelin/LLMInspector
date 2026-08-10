"""Dataset generation — sources, stages, and the generator that runs them.

The pipeline is one shape regardless of what is being generated::

    Generator(source, stages) -> GenerationResult

A :class:`GoldenSource` produces the starting goldens (a curated attack bank, a
set of contexts, a corpus of documents, a styling description). An ordered list
of :class:`Stage` objects then transforms each one — filter, evolve, style,
answer — with a stage returning ``None`` to discard a golden and recording why.

:class:`AdversarialGenerator` is a thin subclass that picks a source; it adds
no machinery of its own.
"""

from .adversarial import AdversarialGenerator
from .config import (
    ContextConfig,
    EvolutionConfig,
    FiltrationConfig,
    GenerationConfig,
    StylingConfig,
)
from .generator import GenerationResult, Generator
from .source import GoldenSource, SyncGoldenSource
from .sources import (
    ContextSource,
    CuratedBankSource,
    DocumentSource,
    ScratchSource,
    SeedGoldenSource,
)
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
    "ContextConfig",
    # presets and the sources behind them
    "AdversarialGenerator",
    "ContextSource",
    "DocumentSource",
    "ScratchSource",
    "SeedGoldenSource",
    "CuratedBankSource",
]
