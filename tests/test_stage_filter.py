"""Phase 4 — the filtration stage's policy, call budget, and score bookkeeping.

Every assertion about call *count* here is an assertion about money: each call is
one real request in production, so "one rewrite then passes" costing three calls
rather than four is a behavioural guarantee, not an implementation detail.

The load-bearing test in this module is the re-scoring one. The design this stage
is adapted from scored once and stored that score even after rewriting, so the
exported ``quality`` described text that no longer existed. Several tests below
script sharply different scores specifically so a regression to storing the first
score fails loudly rather than by a rounding-sized margin.
"""

import asyncio
import json
from typing import Any, List, Optional

import pytest

from llminspector.dataset.golden import Golden
from llminspector.generation.config import FiltrationConfig, GenerationConfig
from llminspector.generation.stage import StageContext
from llminspector.generation.stages.filter import FiltrationStage
from tests.conftest import ScriptedLLM

CONTEXT = ["Tyres must be replaced below 1.6mm of tread depth."]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _quality(score: float, feedback: str = "not self-contained") -> str:
    """One scripted rubric reply, as the JSON string a real provider returns."""
    return json.dumps({"score": score, "feedback": feedback})


def _rewrite(text: str) -> str:
    """One scripted repair reply."""
    return json.dumps({"input": text})


def _ctx(
    model: Any,
    *,
    critic: Any = None,
    context: Optional[List[str]] = None,
) -> StageContext:
    config = GenerationConfig(model=model, critic_model=critic, show_progress=False)
    return StageContext(
        config=config, context=list(CONTEXT if context is None else context)
    )


def _apply(
    stage: FiltrationStage, golden: Golden, ctx: StageContext
) -> Optional[Golden]:
    """Run the stage. ``asyncio.run`` rather than pytest-asyncio, per tests/CLAUDE.md."""
    return asyncio.run(stage.a_apply(golden, ctx))


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #


def test_an_input_above_the_threshold_passes_on_one_call():
    """No rewrite means exactly one model call — the cheapest possible path."""
    model = ScriptedLLM([_quality(0.9, "")])
    stage = FiltrationStage(FiltrationConfig(quality_threshold=0.5))
    golden = Golden(input="What is the legal minimum tread depth?")

    result = _apply(stage, golden, _ctx(model))

    assert result is golden
    assert golden.input == "What is the legal minimum tread depth?"
    assert golden.metadata["quality"] == pytest.approx(0.9)
    assert golden.metadata["rewrites"] == 0
    assert "below_threshold" not in golden.metadata
    assert model.calls == 1


def test_the_stage_mutates_in_place_and_returns_the_same_golden():
    """Identity survives the stage: lineage and rejections key off ``Golden.id``."""
    model = ScriptedLLM([_quality(0.8)])
    golden = Golden(input="How thin can a tyre get before it is illegal?")
    original_id = golden.id

    result = _apply(FiltrationStage(), golden, _ctx(model))

    assert result is golden and golden.id == original_id


def test_lineage_records_one_entry_naming_the_stage_and_the_score():
    model = ScriptedLLM([_quality(0.75)])
    golden = Golden(input="What is the minimum tread depth?")

    _apply(FiltrationStage(), golden, _ctx(model))

    assert golden.metadata["lineage"] == [
        {"stage": "filter", "score": pytest.approx(0.75), "rewrites": 0}
    ]


# --------------------------------------------------------------------------- #
# rewriting, and the re-score that follows it
# --------------------------------------------------------------------------- #


def test_one_rewrite_then_passing_costs_three_calls_and_stores_the_second_score():
    """evaluate → rewrite → evaluate. The stored score is the *second* one.

    Both halves are asserted explicitly: the golden must hold the rewritten text
    *and* the score that was measured against it.
    """
    model = ScriptedLLM(
        [
            _quality(0.2, "refers to 'the passage above'"),
            _rewrite("What is the legal minimum tyre tread depth?"),
            _quality(0.9, ""),
        ]
    )
    stage = FiltrationStage(FiltrationConfig(quality_threshold=0.5, max_rewrites=3))
    golden = Golden(input="What does the passage above say about tread?")

    result = _apply(stage, golden, _ctx(model))

    assert result is golden
    assert model.calls == 3
    assert golden.input == "What is the legal minimum tyre tread depth?"
    assert golden.metadata["quality"] == pytest.approx(0.9)
    assert golden.metadata["quality"] != pytest.approx(0.2)
    assert golden.metadata["rewrites"] == 1
    assert "below_threshold" not in golden.metadata


def test_the_stored_quality_always_describes_the_stored_input():
    """The regression guard, with scores far apart and two rewrites in play.

    A stage that stored the first score would report 0.05 for text scoring 0.95;
    a stage that stored the score from *before* the last rewrite would report
    0.30. Both are pinned out here, against the text each score belongs to.
    """
    model = ScriptedLLM(
        [
            _quality(0.05, "bundles three questions"),
            _rewrite("first rewrite"),
            _quality(0.30, "still vague"),
            _rewrite("second rewrite"),
            _quality(0.95, ""),
        ]
    )
    stage = FiltrationStage(FiltrationConfig(quality_threshold=0.5, max_rewrites=3))
    golden = Golden(input="original")

    _apply(stage, golden, _ctx(model))

    assert golden.input == "second rewrite"
    assert golden.metadata["quality"] == pytest.approx(0.95)
    assert golden.metadata["rewrites"] == 2
    assert model.calls == 5


def test_the_failing_feedback_reaches_the_rewrite_prompt():
    """A rewrite without the feedback is a re-roll; the feedback is what steers it."""
    model = ScriptedLLM(
        [
            _quality(0.1, "the pronoun 'he' has no antecedent"),
            _rewrite("Which mechanic signed off the inspection?"),
            _quality(0.9, ""),
        ]
    )
    golden = Golden(input="What did he decide?")

    _apply(FiltrationStage(), golden, _ctx(model))

    rewrite_prompt = model.prompts[1]
    assert "the pronoun 'he' has no antecedent" in rewrite_prompt
    assert "What did he decide?" in rewrite_prompt


def test_the_context_reaches_both_prompts_so_rewrites_stay_grounded():
    """The rubric sees the source material, and so does the repair."""
    model = ScriptedLLM(
        [_quality(0.1, "vague"), _rewrite("rewritten"), _quality(0.9, "")]
    )
    golden = Golden(input="original")

    _apply(FiltrationStage(), golden, _ctx(model))

    assert model.calls == 3
    assert all(CONTEXT[0] in prompt for prompt in model.prompts)


# --------------------------------------------------------------------------- #
# the policy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("on_reject", ["discard", "rewrite", "keep"])
def test_exhausting_the_rewrite_budget_applies_the_reject_policy(on_reject):
    """Two rewrites, all failing: 1 + 2*max_rewrites calls, then the policy."""
    max_rewrites = 2
    model = ScriptedLLM(
        [
            _quality(0.1, "vague"),
            _rewrite("attempt one"),
            _quality(0.2, "still vague"),
            _rewrite("attempt two"),
            _quality(0.3, "still vague"),
        ]
    )
    stage = FiltrationStage(
        FiltrationConfig(
            quality_threshold=0.5, max_rewrites=max_rewrites, on_reject=on_reject
        )
    )
    ctx = _ctx(model)
    golden = Golden(input="original")

    result = _apply(stage, golden, ctx)

    assert model.calls == 1 + 2 * max_rewrites

    if on_reject == "discard":
        assert result is None
        assert len(ctx.rejections) == 1
        reason = ctx.rejections[0]
        assert "0.30" in reason and "0.50" in reason
    elif on_reject == "rewrite":
        assert result is golden
        assert golden.metadata["below_threshold"] is True
        assert ctx.rejections == []
    else:
        assert result is golden
        assert "below_threshold" not in golden.metadata
        assert ctx.rejections == []

    # In every case the recorded numbers describe the text we ended with.
    assert golden.input == "attempt two"
    assert golden.metadata["quality"] == pytest.approx(0.3)
    assert golden.metadata["rewrites"] == max_rewrites


@pytest.mark.parametrize(
    ("on_reject", "discarded"),
    [("discard", True), ("rewrite", False), ("keep", False)],
)
def test_max_rewrites_zero_scores_once_and_still_honours_the_policy(
    on_reject, discarded
):
    """``max_rewrites=0`` is "score, never repair" — not "skip the policy"."""
    model = ScriptedLLM([_quality(0.1, "vague")])
    stage = FiltrationStage(
        FiltrationConfig(quality_threshold=0.5, max_rewrites=0, on_reject=on_reject)
    )
    ctx = _ctx(model)
    golden = Golden(input="original")

    result = _apply(stage, golden, ctx)

    assert model.calls == 1
    assert golden.input == "original"
    assert golden.metadata["rewrites"] == 0
    assert (result is None) is discarded
    assert (len(ctx.rejections) == 1) is discarded
    assert ("below_threshold" in golden.metadata) is (on_reject == "rewrite")


def test_a_discarded_golden_names_the_score_and_the_threshold():
    """A run that halved its output has to be explainable without re-running it."""
    model = ScriptedLLM([_quality(0.25)])
    stage = FiltrationStage(
        FiltrationConfig(quality_threshold=0.8, max_rewrites=0, on_reject="discard")
    )
    ctx = _ctx(model)

    assert _apply(stage, Golden(input="original"), ctx) is None
    assert ctx.rejections == ["quality 0.25 below threshold 0.80 after 0 rewrite(s)"]


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "clamped"), [(1.7, 1.0), (-0.2, 0.0), (42.0, 1.0), (0.6, 0.6)]
)
def test_out_of_range_model_scores_are_clamped(raw, clamped):
    """A 1.7 would clear any threshold and land uninterpretable in the export."""
    model = ScriptedLLM([_quality(raw)])
    stage = FiltrationStage(FiltrationConfig(quality_threshold=0.5, max_rewrites=0))
    golden = Golden(input="original")

    _apply(stage, golden, _ctx(model))

    assert golden.metadata["quality"] == pytest.approx(clamped)


def test_a_below_range_score_is_clamped_to_zero_and_still_fails_the_bar():
    """Clamping must not accidentally rescue an input: -0.2 becomes 0.0, not 0.5."""
    model = ScriptedLLM([_quality(-0.2)])
    stage = FiltrationStage(
        FiltrationConfig(quality_threshold=0.5, max_rewrites=0, on_reject="discard")
    )
    ctx = _ctx(model)

    assert _apply(stage, Golden(input="original"), ctx) is None
    assert ctx.rejections and "0.00" in ctx.rejections[0]


def test_a_blank_rewrite_stops_the_loop_instead_of_burning_the_budget():
    """``Golden.input`` may not be blank, and re-asking would only repeat it."""
    model = ScriptedLLM([_quality(0.1, "vague"), _rewrite("   ")])
    stage = FiltrationStage(
        FiltrationConfig(quality_threshold=0.5, max_rewrites=3, on_reject="keep")
    )
    golden = Golden(input="original")

    result = _apply(stage, golden, _ctx(model))

    assert result is golden
    assert golden.input == "original"
    assert golden.metadata["rewrites"] == 0
    assert model.calls == 2


def test_a_golden_with_no_context_is_still_scorable():
    """Scratch sources have no grounding; the prompt says so rather than lying."""
    model = ScriptedLLM([_quality(0.9)])
    golden = Golden(input="original")

    result = _apply(FiltrationStage(), golden, _ctx(model, context=[]))

    assert result is golden
    assert "none" in model.prompts[0].lower()


# --------------------------------------------------------------------------- #
# declarations
# --------------------------------------------------------------------------- #


def test_the_critic_model_does_the_judging_not_the_generating_model():
    """The whole point of ``critic_model`` is that the judging calls go elsewhere."""
    generator_model = ScriptedLLM(["unused"], name="generator")
    critic_model = ScriptedLLM(
        [_quality(0.1, "vague"), _rewrite("rewritten"), _quality(0.9, "")],
        name="critic",
    )
    golden = Golden(input="original")

    _apply(FiltrationStage(), golden, _ctx(generator_model, critic=critic_model))

    assert critic_model.calls == 3
    assert generator_model.calls == 0


def test_metadata_keys_covers_every_key_the_stage_actually_writes():
    """The declaration stays true: the column set is knowable without a run."""
    model = ScriptedLLM(
        [
            _quality(0.1, "vague"),
            _rewrite("attempt one"),
            _quality(0.2, "still vague"),
        ]
    )
    stage = FiltrationStage(
        FiltrationConfig(quality_threshold=0.5, max_rewrites=1, on_reject="rewrite")
    )
    golden = Golden(input="original")

    _apply(stage, golden, _ctx(model))

    # The flagged path writes the most keys, so it is the one worth checking.
    assert set(golden.metadata) == {
        "quality",
        "rewrites",
        "below_threshold",
        "lineage",
    }
    assert set(golden.metadata) <= set(stage.metadata_keys)
