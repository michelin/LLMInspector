"""``Generator`` — the one engine that runs a generation pipeline.

A run is a :class:`~llminspector.generation.source.GoldenSource` followed by an
ordered list of :class:`~llminspector.generation.stage.Stage` objects. This
module owns the chain order, the fan-out, and the bookkeeping that makes a
shrunken output explainable; the source and the stages own everything else.

Two things here are deliberate departures from the ``BaseSynthesizer`` this
replaces, and both are about *not lying to the caller*:

1. **A dropped golden leaves a record.** The old synthesizers returned a shorter
   list than they were given and said nothing about the difference. Here every
   golden that does not survive lands on
   :attr:`GenerationResult.rejected` or :attr:`GenerationResult.errors` with the
   index it occupied in the source's output, so a run that halved its output can
   be explained without paying for a second one.
2. **:meth:`Generator.to_pandas` raises before a run instead of starting one.**
   See the comment on that method — it is the single most important behavioural
   difference in this file.
"""

from __future__ import annotations

import asyncio
import logging
from typing import (
    Any,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import pandas as pd

from ..dataset.dataset import goldens_to_dataframe
from ..utils.concurrency import a_map
from .config import GenerationConfig
from .source import GoldenSource, GoldenT
from .stage import Stage, StageContext

logger = logging.getLogger(__name__)

__all__ = ["Generator", "GenerationResult"]


# --------------------------------------------------------------------------- #
# result
# --------------------------------------------------------------------------- #


class GenerationResult(Generic[GoldenT]):
    """What a generation run produced, and what it lost on the way.

    Mirrors :class:`~llminspector.evaluate.result.EvaluationResult`'s
    partial-failure shape — *the run always returns; the reasons live on the
    result object* — because that is the reading habit the package already
    teaches. The one structural difference: an evaluation keeps every row and
    scores the failures ``None``, whereas a generation genuinely has fewer
    goldens afterwards. Hence :attr:`errors` and :attr:`rejected` carry the
    ``index`` the golden held **in the source's output**, which is the only
    remaining way to line a missing golden up against what went in.

    Parameters
    ----------
    goldens:
        The goldens that made it through every stage.
    errors:
        Stage failures, as ``{"index": int, "stage": str, "error": "Exc: msg"}``.
        A stage *raised* — the stage broke.
    rejected:
        Deliberate discards, as ``{"index": int, "stage": str, "reason": str}``.
        A stage returned ``None`` — the golden did not make the cut. Kept apart
        from ``errors`` because the two demand opposite responses: a rejection
        is the filter working, an error is something to go and fix.
    usage:
        Token/cost accounting, populated by the metering decorator in a later
        phase. Carried, never computed here.
    """

    def __init__(
        self,
        goldens: List[GoldenT],
        errors: Optional[List[Dict[str, Any]]] = None,
        rejected: Optional[List[Dict[str, Any]]] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.goldens: List[GoldenT] = goldens
        self.errors: List[Dict[str, Any]] = errors or []
        self.rejected: List[Dict[str, Any]] = rejected or []
        self.usage: Optional[Dict[str, Any]] = usage

    def __len__(self) -> int:
        return len(self.goldens)

    def __repr__(self) -> str:
        # All three counts, always — unlike EvaluationResult, which hides an
        # empty errors list. "produced=3" on its own is ambiguous about whether
        # anything was dropped, and that ambiguity is the whole defect this
        # class exists to close.
        return (
            f"GenerationResult(produced={len(self.goldens)}, "
            f"rejected={len(self.rejected)}, failed={len(self.errors)})"
        )

    def error_summary(self) -> str:
        """One line naming the stages that lost goldens, worst first.

        A string rather than ``EvaluationResult``'s dict because there are two
        distinct populations to report (broken vs. filtered) and a single dict
        cannot say which is which. The counting idiom is the same: grouped by
        the name that caused it, ordered by descending count then name.

        Returns ``""`` for a clean run — falsy, exactly as
        ``EvaluationResult.error_summary()``'s empty dict is.
        """
        parts: List[str] = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s) [{_by_stage(self.errors)}]")
        if self.rejected:
            parts.append(f"{len(self.rejected)} rejected [{_by_stage(self.rejected)}]")
        return "; ".join(parts)

    def to_pandas(self) -> pd.DataFrame:
        """The surviving goldens as a table.

        Delegates to :func:`~llminspector.dataset.dataset.goldens_to_dataframe`
        with no mapping, so the columns are the ``Golden`` attribute names plus
        the metadata keys the run actually produced. Flattening goldens is a
        dataset concern; this layer must not grow a second implementation of it.
        """
        return goldens_to_dataframe(self.goldens)

    def to_excel(self, path: str) -> None:
        """Write :meth:`to_pandas` to an ``.xlsx`` file."""
        self.to_pandas().to_excel(path, index=False)


def _by_stage(entries: Sequence[Dict[str, Any]]) -> str:
    """``"filter: 2, evolve: 1"`` — counts per stage, most frequent first."""
    counts: Dict[str, int] = {}
    for entry in entries:
        stage = str(entry.get("stage"))
        counts[stage] = counts.get(stage, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{stage}: {count}" for stage, count in ordered)


# --------------------------------------------------------------------------- #
# per-golden chain
# --------------------------------------------------------------------------- #


class _Outcome(NamedTuple):
    """What running the stage chain over one golden produced.

    At most one of the three is set: a survivor, a failure, or a discard.
    """

    golden: Optional[Any]
    error: Optional[Dict[str, Any]]
    rejection: Optional[Dict[str, Any]]


def _stage_name(stage: Stage[Any]) -> str:
    """``stage.name``, falling back to the class name.

    ``Stage.name`` defaults to ``""``, and a report saying ``{"stage": ""}`` is
    worse than no report at all — the fallback keeps every record attributable.
    """
    return stage.name or type(stage).__name__


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #


class Generator(Generic[GoldenT]):
    """Runs ``source`` then ``stages`` over every golden it produced.

    Parameters
    ----------
    source:
        Produces the run's starting goldens.
    stages:
        Applied in order to each golden. An empty chain is legitimate: it makes
        the generator a thin, uniformly-reported wrapper around the source.
    config:
        Run-wide settings. ``None`` builds a default
        :class:`~llminspector.generation.config.GenerationConfig`, so a source
        that needs no model is usable without constructing one.

    Examples
    --------
    >>> generator = Generator(source, [filter_stage, evolve_stage])
    >>> result = generator.generate()          # await a_generate() in a loop
    >>> generator.to_excel("goldens.xlsx")
    """

    def __init__(
        self,
        source: GoldenSource[GoldenT],
        stages: Sequence[Stage[GoldenT]] = (),
        *,
        config: Optional[GenerationConfig] = None,
    ) -> None:
        self.source = source
        self.stages: List[Stage[GoldenT]] = list(stages)
        self.config = config if config is not None else GenerationConfig()
        # None means "never ran", which is not the same state as "ran and
        # produced nothing" — see to_pandas().
        self._result: Optional[GenerationResult[GoldenT]] = None

    # -- introspection ---------------------------------------------------------

    @property
    def metadata_keys(self) -> Tuple[str, ...]:
        """Every metadata column this pipeline can emit, in declaration order.

        The ordered union of the source's keys and each stage's, de-duplicated
        first-seen. Answerable **without running anything**, which for an
        LLM-backed pipeline means without paying for a run — the reason the
        declaration exists on the source and stage ABCs at all.
        """
        keys: List[str] = []
        seen = set()
        for producer in (self.source, *self.stages):
            for key in producer.metadata_keys:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        return tuple(keys)

    # -- the run ---------------------------------------------------------------

    async def a_generate(self) -> GenerationResult[GoldenT]:
        """Produce the goldens and run them through the stage chain.

        **The only implementation.** :meth:`generate` is an ``asyncio.run``
        wrapper over this one; there is no parallel synchronous pipeline to
        drift out of step with it.

        Goldens are processed concurrently, at most ``config.max_concurrent`` in
        flight, via :func:`~llminspector.utils.concurrency.a_map`. Failures are
        collected per golden and never abort the run.
        """
        config = self.config
        goldens: List[GoldenT] = list(await self.source.a_produce(config))

        # desc=None is a_map's "no bar at all" signal; it never imports tqdm in
        # that case, which is what show_progress=False has to mean in a library.
        desc = "Generating" if config.show_progress else None

        outcomes, map_errors = await a_map(
            list(enumerate(goldens)),
            self._a_run_chain,
            limit=config.max_concurrent,
            desc=desc,
        )

        survivors: List[GoldenT] = []
        errors: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        # a_map preserves input order, so walking the results in order keeps
        # both report lists sorted by index without a second sort.
        for outcome in outcomes:
            if outcome is None:
                # The chain itself raised rather than one of its stages — a bug
                # in this module, not in a stage. Recorded from a_map's own
                # error list below so it cannot vanish silently.
                continue
            if outcome.error is not None:
                errors.append(outcome.error)
            elif outcome.rejection is not None:
                rejected.append(outcome.rejection)
            else:
                survivors.append(outcome.golden)

        # Backstop: anything a_map caught that _a_run_chain did not, attributed
        # to no stage because no stage owns it.
        for entry in map_errors:
            errors.append(
                {"index": entry["index"], "stage": None, "error": entry["error"]}
            )
        errors.sort(key=lambda e: e["index"])

        result: GenerationResult[GoldenT] = GenerationResult(
            goldens=survivors, errors=errors, rejected=rejected
        )
        if errors or rejected:
            logger.warning(
                "generation produced %d of %d golden(s): %s",
                len(survivors),
                len(goldens),
                result.error_summary(),
            )
        self._result = result
        return result

    def generate(self) -> GenerationResult[GoldenT]:
        """Synchronous form of :meth:`a_generate`.

        Inside a running event loop (Jupyter, FastAPI) use
        ``await a_generate()`` instead — ``asyncio.run`` cannot be called there,
        and the bare ``RuntimeError`` it raises names neither this function nor
        the one to use.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "generate() cannot be called from a running event loop "
                "(Jupyter, FastAPI, ...). Use `await a_generate()` instead."
            )

        return asyncio.run(self.a_generate())

    async def _a_run_chain(self, item: Tuple[int, GoldenT]) -> _Outcome:
        """Run every stage over one golden, in order, until one stops it.

        Two ways a golden stops short, kept apart on purpose:

        * a stage returns ``None`` — a **discard**. The filter did its job; the
          reason goes on ``rejected`` and the remaining stages are skipped
          (there is nothing left to apply them to, and running an evolution
          over a golden that already failed the quality bar would bill for it).
        * a stage **raises** — a *failure*. The stage broke, so nothing it would
          have produced can be trusted; the golden is dropped from the output
          and the exception goes on ``errors``. Folding this into ``rejected``
          would report a broken judge as a working filter.
        """
        index, golden = item
        ctx = self._make_context(golden)

        current: GoldenT = golden
        for stage in self.stages:
            name = _stage_name(stage)
            # Where this stage's reasons start, so a discard reports only its
            # own — an earlier stage may have rejected and still returned.
            mark = len(ctx.rejections)
            try:
                produced = await stage.a_apply(current, ctx)
            except Exception as exc:  # noqa: BLE001 - isolate per-golden failures
                # Exception, never BaseException: CancelledError must propagate
                # (same reasoning as a_map's own handler).
                return _Outcome(
                    None,
                    {
                        "index": index,
                        "stage": name,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    None,
                )
            if produced is None:
                reasons = ctx.rejections[mark:]
                reason = (
                    "; ".join(reasons)
                    if reasons
                    # A stage that returns None without calling ctx.reject is
                    # the silent drop this whole result shape exists to
                    # prevent; give it a placeholder rather than an empty
                    # string so the entry still reads as an explanation gap.
                    else f"discarded by {name} (no reason given)"
                )
                return _Outcome(
                    None, None, {"index": index, "stage": name, "reason": reason}
                )
            current = produced

        return _Outcome(current, None, None)

    def _make_context(self, golden: GoldenT) -> StageContext:
        """A fresh :class:`StageContext` for one golden.

        Per golden, never shared: goldens run concurrently and stages write to
        ``ctx.rejections`` (and may stash their own state on the context), so a
        shared instance would let one golden's run read another's. Same
        reasoning as ``BaseMetric.clone()`` per row in the evaluate engine —
        the mutable per-item state is cloned, the stage object is not.

        ``context`` and ``source_files`` are *copies* of the golden's own, so a
        stage editing the context list cannot corrupt the golden it came from.
        """
        metadata = golden.metadata
        # Both spellings are honoured: grounded sources record a list under
        # "source_files", the RAG chunker records a single "source_file" per
        # chunk. Normalising here keeps every stage reading one attribute.
        files = metadata.get("source_files")
        if files is None:
            single = metadata.get("source_file")
            files = [single] if single is not None else []
        return StageContext(
            config=self.config,
            context=list(golden.context or []),
            source_files=[str(f) for f in files],
        )

    # -- export ----------------------------------------------------------------

    def to_pandas(self) -> pd.DataFrame:
        """The last run's goldens as a table.

        **Raises if nothing has been generated yet.** ``BaseSynthesizer.to_pandas()``
        used to call ``generate()`` for you, which on an LLM-backed pipeline
        turns an innocuous-looking export line into a full paid generation run —
        and a second one on the next call. This is the single most important
        behavioural difference in this file: exporting reads a result, it never
        creates one.

        A run that produced *nothing* is a different state and returns an empty
        frame rather than raising: the pipeline ran and the answer was zero
        goldens, which the caller needs to be able to see.
        """
        if self._result is None:
            raise RuntimeError(
                "Nothing has been generated yet. Call generate() (or await "
                "a_generate()) first, then export — to_pandas() deliberately "
                "does not start a run."
            )
        return self._result.to_pandas()

    def to_excel(self, path: str) -> None:
        """Write :meth:`to_pandas` to an ``.xlsx`` file."""
        self.to_pandas().to_excel(path, index=False)
