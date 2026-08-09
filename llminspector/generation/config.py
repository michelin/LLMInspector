"""Configuration for the generation pipeline.

Plain ``@dataclass``, deliberately **not** in :mod:`llminspector.config`. That
package is provider-connection config — endpoints and credentials read from the
environment. These are per-run knobs that live next to the pipeline they steer,
and mixing the two would put an LLM temperature next to an API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

__all__ = [
    "GenerationConfig",
    "FiltrationConfig",
    "EvolutionConfig",
    "StylingConfig",
    "ContextConfig",
]


@dataclass
class GenerationConfig:
    """Run-wide settings: the models, concurrency, and determinism.

    Parameters
    ----------
    model:
        The :class:`~llminspector.models.base_model.BaseLLM` that produces
        goldens.
    critic_model:
        A second model for the judging stages (filtration, context scoring).
        Falls back to ``model``; see :meth:`critic`. Splitting them lets a run
        generate with a large model and judge with a cheap one, which is where
        most of the cost sits.
    embedding:
        Embedding provider, needed only by the document/context sources.
    max_concurrent:
        Ceiling on in-flight model calls, passed to
        :func:`~llminspector.utils.concurrency.a_map` at every level.
    show_progress:
        Show a progress bar per pipeline stage.
    seed:
        Threads into every ``random`` / ``numpy`` draw in the pipeline —
        evolution strategy choice, chunk sampling. ``None`` means
        nondeterministic. Without a seed a generation run cannot be reproduced
        or cheaply tested, which is the single biggest gap in the design this
        one is adapted from.
    include_expected_output:
        Run the expected-output stage.
    """

    model: Any = None
    critic_model: Any = None
    embedding: Any = None
    max_concurrent: int = 5
    show_progress: bool = True
    seed: Optional[int] = None
    include_expected_output: bool = True

    def __post_init__(self) -> None:
        if self.max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {self.max_concurrent}")

    @property
    def critic(self) -> Any:
        """The judging model — ``critic_model`` when set, else ``model``.

        A property rather than a ``__post_init__`` assignment so that setting
        ``config.model`` afterwards still updates the fallback, and so the
        config keeps reporting what the caller actually asked for.
        """
        return self.critic_model if self.critic_model is not None else self.model


@dataclass
class FiltrationConfig:
    """What to do with an input that scores below the quality bar.

    ``on_reject`` is the policy the reference design never had — it always kept
    everything, and recorded a score belonging to the *pre-rewrite* text:

    * ``"rewrite"`` — rewrite up to ``max_rewrites`` times, then keep whatever
      we ended with, flagged as below threshold.
    * ``"discard"`` — rewrite, then drop the golden if it still fails.
    * ``"keep"`` — keep it unflagged; the score is still recorded.

    In every case the stored score describes the *stored* text, because the
    input is re-scored after each rewrite.
    """

    quality_threshold: float = 0.5
    max_rewrites: int = 3
    on_reject: Literal["rewrite", "discard", "keep"] = "rewrite"

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError(
                "quality_threshold must be within [0.0, 1.0], got "
                f"{self.quality_threshold}"
            )
        if self.max_rewrites < 0:
            raise ValueError(f"max_rewrites must be >= 0, got {self.max_rewrites}")


@dataclass
class EvolutionConfig:
    """How many times to evolve an input, and with what strategy mix.

    ``strategies`` maps a strategy name to a relative weight; weights need not
    sum to 1 and are normalised at draw time. A source that cannot support a
    strategy (scratch generation has no context, so multi-context evolution is
    impossible) drops it from the map and renormalises rather than silently
    picking something it cannot do.
    """

    num_evolutions: int = 1
    strategies: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_evolutions < 0:
            raise ValueError(f"num_evolutions must be >= 0, got {self.num_evolutions}")
        negative = sorted(k for k, v in self.strategies.items() if v < 0)
        if negative:
            raise ValueError(f"strategy weights must be >= 0; negative: {negative}")


@dataclass
class StylingConfig:
    """The voice and shape of generated inputs.

    All four are free-text descriptions handed to the model, not enumerations —
    the scenario of a tyre-retail support bot is not drawn from a fixed list.
    """

    scenario: Optional[str] = None
    task: Optional[str] = None
    input_format: Optional[str] = None
    expected_output_format: Optional[str] = None

    def missing_fields(self) -> list[str]:
        """Which of the three generation-critical fields are unset.

        Sources that generate without context (scratch) need all three and
        report them **together** — being told about one missing field at a time
        across three failed runs is the behaviour this replaces.
        """
        return [
            name
            for name in ("scenario", "task", "input_format")
            if not getattr(self, name)
        ]


@dataclass
class ContextConfig:
    """How documents become contexts: chunking, retrieval, assembly.

    Parameters
    ----------
    chunk_size / chunk_overlap:
        Token counts for the splitter. ``chunk_overlap`` must be smaller than
        ``chunk_size`` or chunking never advances.
    max_contexts:
        How many contexts to build across the whole corpus.
    chunks_per_context:
        Seed chunk plus up to this many neighbours.
    similarity_threshold:
        Minimum cosine similarity for a neighbour to join a context. Defaults to
        **0.5, not 0.0**. The design this is adapted from defaults to 0.0, which
        accepts every neighbour including orthogonal ones and quietly defeats the
        point of the similarity check.
    candidate_pool:
        How many chunks to score with the critic before taking the best
        ``max_contexts``. Scoring is a model call per chunk, so this is the main
        cost dial for context construction.
    index_backend:
        ``"numpy"`` (default), ``"faiss"``, or ``"auto"``. Never silent: ``auto``
        picks faiss only when it is installed *and* the corpus exceeds
        :data:`FAISS_AUTO_THRESHOLD`.
    cross_file:
        Merge contexts drawn from different files into multi-source contexts, so
        generated inputs require combining documents.
    max_files_per_context:
        Ceiling on distinct source files in one merged context.
    """

    chunk_size: int = 1024
    chunk_overlap: int = 0
    max_contexts: int = 10
    chunks_per_context: int = 3
    similarity_threshold: float = 0.5
    candidate_pool: int = 30
    index_backend: Literal["numpy", "faiss", "auto"] = "numpy"
    cross_file: bool = False
    max_files_per_context: int = 2

    def __post_init__(self) -> None:
        # Validation runs before any embedding call — see
        # ``context/selection.py``. An error naming the actual numbers is worth
        # a great deal more than a ZeroDivisionError three hundred API calls in.
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}), otherwise chunking never "
                f"advances. Try chunk_size={self.chunk_size}, "
                f"chunk_overlap={self.chunk_size // 8}."
            )
        if self.max_contexts < 1:
            raise ValueError(f"max_contexts must be >= 1, got {self.max_contexts}")
        if self.chunks_per_context < 1:
            raise ValueError(
                f"chunks_per_context must be >= 1, got {self.chunks_per_context}"
            )
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be within [0.0, 1.0], got "
                f"{self.similarity_threshold}"
            )
        if self.index_backend not in ("numpy", "faiss", "auto"):
            raise ValueError(
                "index_backend must be 'numpy', 'faiss' or 'auto', got "
                f"{self.index_backend!r}"
            )
        if self.max_files_per_context < 1:
            raise ValueError(
                "max_files_per_context must be >= 1, got "
                f"{self.max_files_per_context}"
            )
