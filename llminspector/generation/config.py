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
