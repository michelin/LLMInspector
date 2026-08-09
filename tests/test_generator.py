"""Phase 3 — the generation engine: ``Generator`` and ``GenerationResult``.

Everything the pipeline guarantees is here, exercised through tiny in-file fakes
rather than real sources and stages. That is deliberate: the guarantees belong
to the *generator*, and a test that needed an LLM-backed source to demonstrate
"a stage that raises does not take the run down with it" would be testing the
source instead.

Three defects from ``BaseSynthesizer`` are pinned as behaviour here:

1. A dropped golden used to leave no trace — the list simply came back shorter.
2. A stage returning ``None`` and a stage *raising* were the same outcome.
3. ``to_pandas()`` silently called ``generate()``, so an export line on an
   LLM-backed pipeline was an accidental paid run.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import pytest

from llminspector.dataset.golden import Golden
from llminspector.generation.config import GenerationConfig
from llminspector.generation.generator import GenerationResult, Generator
from llminspector.generation.source import GoldenSource, SyncGoldenSource
from llminspector.generation.stage import Stage, StageContext

# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


class ListSource(SyncGoldenSource):
    """Hands back a fixed list of goldens; the pipeline's degenerate front end."""

    metadata_keys = ("origin",)

    def __init__(self, inputs: Sequence[str], **metadata: Any) -> None:
        self.inputs = list(inputs)
        self.metadata = metadata

    def produce(self, config: GenerationConfig) -> List[Golden]:
        return [
            Golden(input=text, metadata=dict(self.metadata)) for text in self.inputs
        ]


class Suffix(Stage):
    """Appends to ``golden.input``. Ordering is readable straight off the text."""

    def __init__(self, suffix: str, *, name: Optional[str] = None) -> None:
        self.suffix = suffix
        self.name = name or f"suffix{suffix}"

    async def a_apply(self, golden: Golden, ctx: StageContext) -> Optional[Golden]:
        golden.input = golden.input + self.suffix
        return golden


class DropIf(Stage):
    """Discards goldens matching ``predicate``; optionally without saying why."""

    name = "dropper"

    def __init__(self, predicate, *, reason: Optional[str] = "below threshold") -> None:
        self.predicate = predicate
        self.reason = reason

    async def a_apply(self, golden: Golden, ctx: StageContext) -> Optional[Golden]:
        if self.predicate(golden):
            if self.reason is not None:
                ctx.reject(self.reason)
            return None
        return golden


class BoomIf(Stage):
    """Raises on the goldens matching ``predicate`` — a *broken* stage."""

    name = "boom"

    def __init__(self, predicate, exc: Optional[BaseException] = None) -> None:
        self.predicate = predicate
        self.exc = exc or ValueError("judge returned garbage")

    async def a_apply(self, golden: Golden, ctx: StageContext) -> Optional[Golden]:
        if self.predicate(golden):
            raise self.exc
        return golden


class Recorder(Stage):
    """Records which goldens reached it, so 'later stages did not run' is testable."""

    name = "recorder"

    def __init__(self) -> None:
        self.seen: List[str] = []

    async def a_apply(self, golden: Golden, ctx: StageContext) -> Optional[Golden]:
        self.seen.append(golden.input)
        return golden


def _config(**kwargs: Any) -> GenerationConfig:
    """A config with the progress bar off — tests must not print tqdm noise."""
    kwargs.setdefault("show_progress", False)
    return GenerationConfig(**kwargs)


def _run(generator: Generator) -> GenerationResult:
    return asyncio.run(generator.a_generate())


def _inputs(result: GenerationResult) -> List[str]:
    return [g.input for g in result.goldens]


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #


def test_a_source_with_no_stages_passes_its_goldens_straight_through():
    """An empty chain is legitimate — the generator is then pure bookkeeping."""
    result = _run(Generator(ListSource(["a", "b"]), config=_config()))
    assert _inputs(result) == ["a", "b"]
    assert result.errors == [] and result.rejected == []
    assert repr(result) == "GenerationResult(produced=2, rejected=0, failed=0)"


def test_stages_apply_in_chain_order():
    """The chain order lives in the generator, so it has to be the list order."""
    generator = Generator(
        ListSource(["seed"]), [Suffix("-one"), Suffix("-two")], config=_config()
    )
    assert _inputs(_run(generator)) == ["seed-one-two"]


def test_config_defaults_when_none_is_given():
    """A source needing no model must be usable without constructing a config."""
    generator = Generator(ListSource(["a"]))
    assert isinstance(generator.config, GenerationConfig)
    assert generator.config.max_concurrent == GenerationConfig().max_concurrent


def test_a_clean_run_summarises_to_nothing():
    """Falsy for a clean run, mirroring EvaluationResult's empty dict."""
    result = _run(Generator(ListSource(["a"]), config=_config()))
    assert result.error_summary() == ""
    assert len(result) == 1


# --------------------------------------------------------------------------- #
# discards: the stage said no
# --------------------------------------------------------------------------- #


def test_a_rejected_golden_leaves_a_record_instead_of_vanishing():
    generator = Generator(
        ListSource(["keep", "drop"]),
        [DropIf(lambda g: g.input == "drop", reason="below threshold")],
        config=_config(),
    )
    result = _run(generator)

    assert _inputs(result) == ["keep"]
    assert result.errors == []
    assert result.rejected == [
        {"index": 1, "stage": "dropper", "reason": "below threshold"}
    ]


def test_later_stages_do_not_run_on_a_rejected_golden():
    """Nothing is left to apply them to, and an LLM stage would still bill."""
    recorder = Recorder()
    generator = Generator(
        ListSource(["keep", "drop"]),
        [DropIf(lambda g: g.input == "drop"), recorder],
        config=_config(),
    )
    _run(generator)
    assert recorder.seen == ["keep"]


def test_a_silent_drop_still_records_a_reason():
    """A stage that forgets ctx.reject is exactly what this shape prevents."""
    generator = Generator(
        ListSource(["drop"]),
        [DropIf(lambda g: True, reason=None)],
        config=_config(),
    )
    result = _run(generator)

    assert result.goldens == []
    assert len(result.rejected) == 1
    entry = result.rejected[0]
    assert entry["index"] == 0 and entry["stage"] == "dropper"
    assert "no reason given" in entry["reason"]


def test_only_the_rejecting_stage_s_reasons_are_reported():
    """An earlier stage may reject and still return; its reason is not this one."""

    class NoteThenPass(Stage):
        name = "noter"

        async def a_apply(self, golden, ctx):
            ctx.reject("stale note from an earlier stage")
            return golden

    generator = Generator(
        ListSource(["x"]),
        [NoteThenPass(), DropIf(lambda g: True, reason="the real reason")],
        config=_config(),
    )
    assert _run(generator).rejected[0]["reason"] == "the real reason"


# --------------------------------------------------------------------------- #
# failures: the stage broke
# --------------------------------------------------------------------------- #


def test_a_raising_stage_is_recorded_as_an_error_not_a_rejection():
    """A broken judge must never read as a working filter."""
    generator = Generator(
        ListSource(["ok", "bad"]),
        [BoomIf(lambda g: g.input == "bad", ValueError("judge returned garbage"))],
        config=_config(),
    )
    result = _run(generator)

    assert _inputs(result) == ["ok"]
    assert result.rejected == []
    assert result.errors == [
        {
            "index": 1,
            "stage": "boom",
            "error": "ValueError: judge returned garbage",
        }
    ]


def test_one_broken_golden_does_not_stop_the_others():
    generator = Generator(
        ListSource(["a", "b", "c"]),
        [BoomIf(lambda g: g.input == "b"), Suffix("!")],
        config=_config(),
    )
    result = _run(generator)
    assert _inputs(result) == ["a!", "c!"]
    assert len(result.errors) == 1


def test_indices_refer_to_positions_in_the_source_output():
    """The middle golden fails; its index is 2, not 0 and not 1.

    This is the only remaining link from a missing golden back to what went in,
    since rejected/failed goldens are gone from ``goldens``.
    """
    generator = Generator(
        ListSource(["g0", "g1", "g2", "g3", "g4"]),
        [
            BoomIf(lambda g: g.input == "g2"),
            DropIf(lambda g: g.input == "g3", reason="too short"),
        ],
        config=_config(max_concurrent=5),
    )
    result = _run(generator)

    assert _inputs(result) == ["g0", "g1", "g4"]
    assert [e["index"] for e in result.errors] == [2]
    assert [r["index"] for r in result.rejected] == [3]


def test_error_summary_names_the_failing_stage():
    generator = Generator(
        ListSource(["a", "b", "c"]),
        [
            BoomIf(lambda g: g.input in {"a", "b"}),
            DropIf(lambda g: g.input == "c", reason="nope"),
        ],
        config=_config(),
    )
    summary = _run(generator).error_summary()
    assert "boom" in summary and "2" in summary
    assert "dropper" in summary


def test_a_run_with_failures_still_returns_and_logs(caplog, capsys):
    """Collected, never raised — and reported through logging, not stdout."""
    generator = Generator(ListSource(["a"]), [BoomIf(lambda g: True)], config=_config())
    with caplog.at_level("WARNING"):
        result = _run(generator)
    assert result.goldens == []
    assert "boom" in caplog.text
    assert capsys.readouterr().out == ""


def test_an_unnamed_stage_is_attributed_to_its_class():
    """``Stage.name`` defaults to ""; ``{"stage": ""}`` would be unreadable."""

    class Anonymous(Stage):
        async def a_apply(self, golden, ctx):
            raise RuntimeError("boom")

    result = _run(Generator(ListSource(["a"]), [Anonymous()], config=_config()))
    assert result.errors[0]["stage"] == "Anonymous"


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def test_to_pandas_refuses_to_start_a_run():
    """The headline behaviour change: exporting reads a result, never makes one.

    ``BaseSynthesizer.to_pandas()`` called ``generate()`` for you, which on an
    LLM-backed pipeline is a full paid run per export call.
    """
    generator = Generator(ListSource(["a"]), config=_config())
    with pytest.raises(RuntimeError, match="generate"):
        generator.to_pandas()


def test_to_pandas_works_after_a_run():
    generator = Generator(ListSource(["a", "b"], origin="bank"), config=_config())
    _run(generator)
    frame = generator.to_pandas()

    assert list(frame["input"]) == ["a", "b"]
    # metadata is carried into columns by goldens_to_dataframe, not re-derived
    assert list(frame["origin"]) == ["bank", "bank"]
    assert list(frame.columns)[:4] == ["id", "input", "expected_output", "context"]


def test_to_excel_round_trips(tmp_path):
    generator = Generator(ListSource(["a", "b"]), [Suffix("!")], config=_config())
    _run(generator)
    path = tmp_path / "goldens.xlsx"
    generator.to_excel(str(path))

    assert list(pd.read_excel(path)["input"]) == ["a!", "b!"]


def test_the_result_exports_the_same_table_as_the_generator():
    generator = Generator(ListSource(["a"]), config=_config())
    result = _run(generator)
    pd.testing.assert_frame_equal(result.to_pandas(), generator.to_pandas())


# --------------------------------------------------------------------------- #
# the empty run — "produced nothing" is not "never ran"
# --------------------------------------------------------------------------- #


def test_an_empty_source_produces_an_empty_result_without_crashing():
    result = _run(Generator(ListSource([]), [Suffix("!")], config=_config()))
    assert result.goldens == [] and result.errors == [] and result.rejected == []
    assert len(result) == 0


def test_generated_nothing_exports_an_empty_frame_rather_than_raising():
    """The two empty states are distinguishable, and this is the pinned split.

    *Never ran* raises (see ``test_to_pandas_refuses_to_start_a_run``); *ran and
    produced nothing* is a real answer the caller needs to see, so it returns an
    empty frame.
    """
    generator = Generator(ListSource([]), config=_config())
    with pytest.raises(RuntimeError):
        generator.to_pandas()

    _run(generator)
    frame = generator.to_pandas()
    assert frame.empty and len(frame) == 0


def test_a_run_that_rejected_everything_still_exports_empty():
    generator = Generator(
        ListSource(["a", "b"]), [DropIf(lambda g: True)], config=_config()
    )
    result = _run(generator)
    assert generator.to_pandas().empty
    assert len(result.rejected) == 2


# --------------------------------------------------------------------------- #
# metadata_keys — knowable without running anything
# --------------------------------------------------------------------------- #


def test_metadata_keys_union_source_then_stages_in_order():
    class Keyed(Stage):
        def __init__(self, name, keys):
            self.name = name
            self.metadata_keys = keys

        async def a_apply(self, golden, ctx):
            return golden

    generator = Generator(
        ListSource([]),  # metadata_keys = ("origin",)
        [Keyed("a", ("quality", "lineage")), Keyed("b", ("lineage", "style"))],
        config=_config(),
    )
    # first-seen order, de-duplicated: "lineage" keeps stage a's position
    assert generator.metadata_keys == ("origin", "quality", "lineage", "style")


def test_metadata_keys_needs_no_run():
    """The whole point: the column set is knowable before paying for a run."""
    generator = Generator(ListSource(["a"]), config=_config())
    assert generator.metadata_keys == ("origin",)
    assert generator._result is None


# --------------------------------------------------------------------------- #
# sync / async entry points
# --------------------------------------------------------------------------- #


def test_sync_generate_inside_a_loop_says_what_to_use_instead():
    """Mirrors tests/test_evaluate_runtime.py's check for ``evaluate()``."""
    generator = Generator(ListSource(["a"]), config=_config())

    async def main():
        with pytest.raises(RuntimeError, match="a_generate"):
            generator.generate()

    asyncio.run(main())


def test_a_generate_runs_inside_a_running_loop():
    generator = Generator(ListSource(["a"]), [Suffix("!")], config=_config())

    async def main():
        return await generator.a_generate()

    assert _inputs(asyncio.run(main())) == ["a!"]


def test_sync_and_async_entry_points_agree():
    def build():
        return Generator(ListSource(["a", "b"]), [Suffix("!")], config=_config())

    assert _inputs(build().generate()) == _inputs(asyncio.run(build().a_generate()))


# --------------------------------------------------------------------------- #
# concurrency and isolation
# --------------------------------------------------------------------------- #


class PeakCounter(Stage):
    """Tracks how many goldens are inside the stage at once."""

    name = "peak"

    def __init__(self) -> None:
        self.live = 0
        self.peak = 0

    async def a_apply(self, golden: Golden, ctx: StageContext) -> Optional[Golden]:
        self.live += 1
        self.peak = max(self.peak, self.live)
        # Yields control, so every admitted golden piles up here before any of
        # them finishes; without this the awaitless coroutines would run to
        # completion one at a time and peak would always read 1.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.live -= 1
        return golden


def test_concurrency_is_bounded_by_max_concurrent():
    counter = PeakCounter()
    generator = Generator(
        ListSource([f"g{i}" for i in range(10)]),
        [counter],
        config=_config(max_concurrent=3),
    )
    result = _run(generator)

    assert len(result.goldens) == 10
    assert counter.peak <= 3


def test_a_higher_limit_actually_overlaps():
    """Guards the test above from passing because nothing ran concurrently."""
    counter = PeakCounter()
    generator = Generator(
        ListSource([f"g{i}" for i in range(10)]),
        [counter],
        config=_config(max_concurrent=10),
    )
    _run(generator)
    assert counter.peak > 1


def test_each_golden_gets_its_own_stage_context():
    """Concurrent goldens must never share mutable state — as with metric clones."""
    # The contexts themselves, not their ``id()``: a freed object's address is
    # reused by CPython, so an id-based check can pass or fail on GC timing
    # rather than on whether the contexts were actually distinct.
    contexts: List[StageContext] = []

    class Marker(Stage):
        name = "marker"

        async def a_apply(self, golden, ctx):
            contexts.append(ctx)
            ctx.reject(f"note for {golden.input}")
            # A leak would show up as another golden's note on this context.
            assert ctx.rejections == [f"note for {golden.input}"]
            return golden

    generator = Generator(
        ListSource(["a", "b", "c"]), [Marker()], config=_config(max_concurrent=3)
    )
    result = _run(generator)

    assert len(result.goldens) == 3
    assert len({id(ctx) for ctx in contexts}) == 3


def test_the_context_copies_the_golden_s_grounding():
    """A stage editing ctx.context must not corrupt the golden it came from."""

    class Grounded(SyncGoldenSource):
        def produce(self, config):
            return [
                Golden(
                    input="q",
                    context=["chunk one"],
                    metadata={"source_files": ["a.pdf"]},
                )
            ]

    seen: Dict[str, Any] = {}

    class Mutate(Stage):
        name = "mutate"

        async def a_apply(self, golden, ctx):
            seen["context"] = list(ctx.context)
            seen["files"] = list(ctx.source_files)
            ctx.context.append("injected")
            return golden

    result = _run(Generator(Grounded(), [Mutate()], config=_config()))

    assert seen == {"context": ["chunk one"], "files": ["a.pdf"]}
    assert result.goldens[0].context == ["chunk one"]


def test_a_single_source_file_is_normalised_to_a_list():
    """RAG chunks carry one ``source_file``; stages read one attribute either way."""

    class OneFile(SyncGoldenSource):
        def produce(self, config):
            return [Golden(input="q", metadata={"source_file": "a.pdf"})]

    captured: List[List[str]] = []

    class Peek(Stage):
        name = "peek"

        async def a_apply(self, golden, ctx):
            captured.append(list(ctx.source_files))
            return golden

    _run(Generator(OneFile(), [Peek()], config=_config()))
    assert captured == [["a.pdf"]]


# --------------------------------------------------------------------------- #
# the result object on its own
# --------------------------------------------------------------------------- #


def test_usage_is_carried_not_computed():
    """A later phase's metering decorator fills this; the engine only passes it."""
    assert GenerationResult(goldens=[]).usage is None
    assert GenerationResult(goldens=[], usage={"total_tokens": 12}).usage == {
        "total_tokens": 12
    }


def test_an_async_source_is_awaited():
    """Sources are async first; SyncGoldenSource is the convenience wrapper."""

    class AsyncSource(GoldenSource):
        async def a_produce(self, config):
            await asyncio.sleep(0)
            return [Golden(input="from async")]

    assert _inputs(_run(Generator(AsyncSource(), config=_config()))) == ["from async"]
