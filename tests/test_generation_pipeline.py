"""The whole pipeline: ContextSource through the default five-stage chain.

The stage modules each have unit tests; this asserts they compose — that the
chain runs in the documented order, makes the documented number of calls, and
that a failure or a rejection anywhere in it leaves the rest of the run intact.

Every model reply is dispatched on the prompt rather than queued in order, so a
test breaks when the *chain order* changes rather than when an unrelated stage
gains a call.
"""

import asyncio
import json

import pytest

from llminspector.generation import (
    ContextSource,
    FiltrationConfig,
    GenerationConfig,
    Generator,
    StylingConfig,
    default_stages,
)
from llminspector.generation.config import EvolutionConfig
from tests.conftest import ScriptedLLM

# Each prompt's opening line, which is how a reply is routed to the right stage.
KIND_MARKERS = [
    ("generate", "You are writing evaluation inputs"),
    ("score", "You are reviewing an evaluation input"),
    ("rewrite", "You are repairing an evaluation input"),
    ("evolve", "You are making an evaluation input harder"),
    ("style", "Rewrite the input below so it reads"),
    ("expected_output", "Write the reference answer"),
]


def kind_of(prompt):
    for kind, marker in KIND_MARKERS:
        if marker in prompt:
            return kind
    raise AssertionError(f"unrecognised prompt: {prompt[:120]!r}")


class ChainLLM(ScriptedLLM):
    """Answers by stage kind and records the order the stages ran in."""

    def __init__(self, *, scores=(1.0,), inputs=("q1",), **overrides):
        super().__init__(self._dispatch_reply)
        self.kinds = []
        self._scores = list(scores)
        self._inputs = list(inputs)
        self._overrides = overrides

    def _dispatch_reply(self, prompt):
        kind = kind_of(prompt)
        self.kinds.append(kind)
        if kind in self._overrides:
            return self._overrides[kind]
        if kind == "generate":
            return json.dumps({"inputs": self._inputs})
        if kind == "score":
            score = self._scores[0] if len(self._scores) == 1 else self._scores.pop(0)
            return json.dumps({"score": score, "feedback": "needs work"})
        if kind == "rewrite":
            return json.dumps({"input": "repaired"})
        if kind == "evolve":
            return json.dumps(
                {"input": "evolved", "applied": "reasoning", "still_grounded": True}
            )
        if kind == "style":
            return json.dumps({"input": "styled"})
        return json.dumps({"expected_output": "the answer"})


def _run(generator):
    return asyncio.run(generator.a_generate())


def _generator(model, *, contexts=None, seed=1, **stage_kwargs):
    config = GenerationConfig(model=model, show_progress=False, seed=seed)
    return Generator(
        ContextSource(contexts or [["the API allows 500 requests per minute"]]),
        default_stages(**stage_kwargs),
        config=config,
    )


# --------------------------------------------------------------------------- #
# the chain runs in the documented order
# --------------------------------------------------------------------------- #


def test_the_default_chain_runs_in_order_and_costs_five_calls():
    """generate -> filter -> evolve -> re-filter -> expected_output.

    Styling is unconfigured by default and must therefore cost nothing, so a
    clean golden is five calls: one to write it, one to score it, one to evolve
    it, one to re-score it after evolution, one to answer it.
    """
    model = ChainLLM(inputs=["q1"])
    result = _run(_generator(model))

    assert model.kinds == [
        "generate",
        "score",
        "evolve",
        "score",
        "expected_output",
    ]
    assert len(result.goldens) == 1


def test_styling_adds_exactly_one_call_when_configured():
    model = ChainLLM(inputs=["q1"])
    styling = StylingConfig(
        scenario="tyre retail", task="support", input_format="a chat message"
    )
    result = _run(_generator(model, styling=styling))

    assert model.kinds.count("style") == 1
    assert model.kinds.index("style") > model.kinds.index("evolve")
    assert result.goldens[0].input == "styled"


def test_expected_output_runs_last():
    """A reference answer written before the input settled would score the wrong thing."""
    model = ChainLLM(inputs=["q1"])
    _run(_generator(model))

    assert model.kinds[-1] == "expected_output"


def test_evolution_is_re_checked_rather_than_trusted():
    """The second filter pass exists because evolution can break grounding.

    Evolving last and never re-checking is what the design this replaces did.
    The re-check is the cheap variant: it scores once and does not re-run the
    repair loop, so it must add exactly one call.
    """
    model = ChainLLM(inputs=["q1"])
    _run(_generator(model))

    assert model.kinds.count("score") == 2
    assert model.kinds.count("rewrite") == 0


def test_the_run_produces_a_full_golden():
    model = ChainLLM(inputs=["q1"])
    result = _run(_generator(model))

    golden = result.goldens[0]
    assert golden.input == "evolved"
    assert golden.expected_output == "the answer"
    assert golden.context == ["the API allows 500 requests per minute"]
    assert golden.metadata["quality"] == 1.0
    assert golden.metadata["evolutions"] == 1


# --------------------------------------------------------------------------- #
# filtration policy end to end
# --------------------------------------------------------------------------- #


def test_a_persistently_bad_golden_is_discarded_under_discard():
    model = ChainLLM(inputs=["q1"], scores=[0.1])
    generator = _generator(
        model,
        filtration=FiltrationConfig(
            quality_threshold=0.5, max_rewrites=1, on_reject="discard"
        ),
    )
    result = _run(generator)

    assert result.goldens == []
    assert len(result.rejected) == 1
    assert result.rejected[0]["stage"] == "filter"
    assert "0.10" in result.rejected[0]["reason"]
    # Discarded at the first filter, so evolution never ran.
    assert "evolve" not in model.kinds


def test_the_same_golden_is_kept_and_flagged_under_rewrite():
    model = ChainLLM(inputs=["q1"], scores=[0.1])
    generator = _generator(
        model,
        filtration=FiltrationConfig(
            quality_threshold=0.5, max_rewrites=1, on_reject="rewrite"
        ),
    )
    result = _run(generator)

    assert len(result.goldens) == 1
    assert result.goldens[0].metadata["below_threshold"] is True
    assert result.rejected == []


def test_the_stored_quality_describes_the_stored_text():
    """The score must belong to the text that survived, not the one that failed.

    The design this replaces never re-scored after a rewrite, so the number in
    the ``quality`` column described a string that no longer existed.
    """
    model = ChainLLM(inputs=["q1"], scores=[0.1, 0.9, 0.9])
    generator = _generator(
        model,
        filtration=FiltrationConfig(
            quality_threshold=0.5, max_rewrites=2, on_reject="discard"
        ),
    )
    result = _run(generator)

    golden = result.goldens[0]
    assert golden.metadata["quality"] == 0.9
    assert golden.metadata["rewrites"] == 1
    assert "rewrite" in model.kinds


# --------------------------------------------------------------------------- #
# partial failure
# --------------------------------------------------------------------------- #


def test_a_stage_failing_mid_run_records_the_error_and_spares_the_rest():
    """One broken golden must not cost the run."""

    class Flaky(ChainLLM):
        def _dispatch_reply(self, prompt):
            if "poison" in prompt and kind_of(prompt) == "evolve":
                return "not json and not recoverable"
            return super()._dispatch_reply(prompt)

    model = Flaky(inputs=["poison", "fine"])
    result = _run(_generator(model))

    assert [g.input for g in result.goldens] == ["evolved"]
    assert len(result.errors) == 1
    assert result.errors[0]["stage"] == "evolve"
    assert "StructuredOutputError" in result.errors[0]["error"]
    assert result.error_summary()


def test_a_failing_context_does_not_stop_the_others():
    def dispatch(prompt):
        if kind_of(prompt) == "generate" and "poison" in prompt:
            return "not json"
        return ChainLLM._dispatch_reply(model, prompt)

    model = ChainLLM(inputs=["q1"])
    model._dispatch = dispatch
    generator = _generator(model, contexts=[["fine"], ["poison"], ["also fine"]])

    result = _run(generator)

    assert len(result.goldens) == 2


# --------------------------------------------------------------------------- #
# determinism and the metadata declaration
# --------------------------------------------------------------------------- #


def test_the_same_seed_produces_the_same_evolution_choices():
    def strategies(seed):
        model = ChainLLM(inputs=["q1"])
        result = _run(_generator(model, seed=seed))
        return [
            entry.get("strategy")
            for entry in result.goldens[0].metadata["lineage"]
            if entry["stage"] == "evolve"
        ]

    assert strategies(11) == strategies(11)


def test_the_metadata_declaration_stays_true_for_the_whole_pipeline():
    """The export column set must be knowable without paying for a run.

    This is the pipeline-level version of the per-stage check: whatever the
    source and every stage actually wrote must be covered by what they declared.
    """
    model = ChainLLM(inputs=["q1"])
    styling = StylingConfig(scenario="retail", task="support", input_format="chat")
    generator = _generator(model, styling=styling)

    result = _run(generator)

    assert set(result.goldens[0].metadata) <= set(generator.metadata_keys)


def test_the_exported_frame_carries_the_lineage_columns():
    model = ChainLLM(inputs=["q1"])
    generator = _generator(model)
    _run(generator)

    columns = list(generator.to_pandas().columns)

    assert columns[:4] == ["id", "input", "expected_output", "context"]
    for expected in ("quality", "evolutions", "lineage"):
        assert expected in columns


def test_to_pandas_refuses_to_start_a_paid_run():
    model = ChainLLM(inputs=["q1"])
    generator = _generator(model)

    with pytest.raises(RuntimeError):
        generator.to_pandas()
    assert model.calls == 0


def test_two_contexts_produce_two_independent_goldens():
    model = ChainLLM(inputs=["q1"])
    generator = _generator(model, contexts=[["ctx one"], ["ctx two"]])

    result = _run(generator)

    assert len(result.goldens) == 2
    assert {g.context[0] for g in result.goldens} == {"ctx one", "ctx two"}


def test_no_evolutions_shortens_the_chain():
    model = ChainLLM(inputs=["q1"])
    generator = _generator(model, evolution=EvolutionConfig(num_evolutions=0))

    _run(generator)

    assert "evolve" not in model.kinds
    assert model.kinds.count("score") == 2, "the re-check still runs"


def test_rewrites_accumulate_across_both_filter_passes():
    """The re-check must not erase the repair the first pass performed.

    ``quality`` is deliberately overwritten — the latest score describes the
    text now on the golden. ``rewrites`` is not: the default chain filters
    twice, and the second pass does no rewriting, so overwriting would report
    zero repairs on a golden that was repaired.
    """
    model = ChainLLM(inputs=["q1"], scores=[0.1, 0.9, 0.9])
    generator = _generator(
        model,
        filtration=FiltrationConfig(
            quality_threshold=0.5, max_rewrites=2, on_reject="rewrite"
        ),
    )
    result = _run(generator)

    golden = result.goldens[0]
    assert golden.metadata["rewrites"] == 1
    assert golden.metadata["quality"] == 0.9


def test_a_later_pass_clears_an_earlier_below_threshold_flag():
    """A flag set on a superseded score must not outlive it."""
    model = ChainLLM(inputs=["q1"], scores=[0.1, 0.95])
    generator = _generator(
        model,
        filtration=FiltrationConfig(
            quality_threshold=0.5, max_rewrites=0, on_reject="rewrite"
        ),
    )
    result = _run(generator)

    golden = result.goldens[0]
    assert golden.metadata["quality"] == 0.95
    assert "below_threshold" not in golden.metadata
