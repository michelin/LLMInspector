"""The stages that evolve, style, answer, and perturb a golden.

Filtration has its own module (``tests/test_stage_filter.py``) because its
rewrite policy is substantial enough to warrant one.

Call counts are asserted everywhere. Each model call is a real API call in
production, and a stage that quietly makes two where it should make one is a
cost regression that no output assertion would catch. ``ScriptedLLM`` returns
raw strings and ``a_generate_structured`` parses them, so scripted replies are
JSON strings.
"""

import asyncio
import json

import pytest

from llminspector.dataset.golden import Golden
from llminspector.generation.config import (
    EvolutionConfig,
    GenerationConfig,
    StylingConfig,
)
from llminspector.generation.stage import StageContext
from llminspector.generation.stages.evolve import STRATEGIES, EvolutionStage
from llminspector.generation.stages.expected_output import ExpectedOutputStage
from llminspector.generation.stages.perturb import TRANSFORMS, PerturbationStage
from llminspector.generation.stages.style import StylingStage
from tests.conftest import ScriptedLLM


class ExplodingLLM:
    """A model that fails the test if anything calls it."""

    def get_model_name(self):
        return "exploding"

    def generate(self, prompt, **kwargs):
        raise AssertionError("this stage must not call a model")

    async def a_generate(self, prompt, **kwargs):
        raise AssertionError("this stage must not call a model")

    async def a_generate_structured(self, prompt, schema, **kwargs):
        raise AssertionError("this stage must not call a model")


def _ctx(model=None, context=("some source material",), **kwargs):
    kwargs.setdefault("show_progress", False)
    return StageContext(
        config=GenerationConfig(model=model, **kwargs), context=list(context)
    )


def _apply(stage, golden, ctx):
    return asyncio.run(stage.a_apply(golden, ctx))


def _evolved(text, applied="reasoning", grounded=True):
    return json.dumps({"input": text, "applied": applied, "still_grounded": grounded})


# --------------------------------------------------------------------------- #
# EvolutionStage
# --------------------------------------------------------------------------- #


def test_evolution_applies_once_per_round():
    model = ScriptedLLM([_evolved("harder v1"), _evolved("harder v2")])
    stage = EvolutionStage(EvolutionConfig(num_evolutions=2))
    golden = Golden(input="original")

    out = _apply(stage, golden, _ctx(model, seed=7))

    assert out.input == "harder v2"
    assert out.metadata["evolutions"] == 2
    assert model.calls == 2


def test_zero_evolutions_makes_no_call():
    """A stage left in the chain but switched off must be free."""
    stage = EvolutionStage(EvolutionConfig(num_evolutions=0))
    golden = Golden(input="untouched")

    out = _apply(stage, golden, _ctx(ExplodingLLM()))

    assert out.input == "untouched"


def test_the_strategy_is_recorded_in_the_lineage():
    model = ScriptedLLM([_evolved("harder")])
    stage = EvolutionStage(EvolutionConfig(num_evolutions=1))

    out = _apply(stage, Golden(input="q"), _ctx(model, seed=1))

    entry = out.metadata["lineage"][-1]
    assert entry["stage"] == "evolve"
    assert entry["strategy"] in STRATEGIES


def test_the_same_seed_replays_the_same_strategies():
    def run(seed):
        model = ScriptedLLM([_evolved(f"v{i}") for i in range(6)])
        stage = EvolutionStage(EvolutionConfig(num_evolutions=6))
        out = _apply(stage, Golden(input="q"), _ctx(model, seed=seed))
        return [e["strategy"] for e in out.metadata["lineage"]]

    assert run(42) == run(42)
    assert run(42) != run(99), "a different seed should draw differently"


def test_different_goldens_evolve_differently_under_one_seed():
    """Reproducible must not mean identical.

    Seeding a single ``Random(seed)`` per golden hands every golden the same
    draw sequence, collapsing a whole run onto one strategy pattern — which
    defeats the point of evolving at all. The seed is mixed with the input text
    so runs stay replayable while goldens stay varied.
    """

    def strategies_for(text):
        model = ScriptedLLM([_evolved(f"v{i}") for i in range(6)])
        stage = EvolutionStage(EvolutionConfig(num_evolutions=6))
        out = _apply(stage, Golden(input=text), _ctx(model, seed=42))
        return [e["strategy"] for e in out.metadata["lineage"]]

    assert strategies_for("first question") != strategies_for("second question")


def test_zero_weighted_strategies_are_never_drawn():
    model = ScriptedLLM([_evolved(f"v{i}") for i in range(30)])
    stage = EvolutionStage(
        EvolutionConfig(
            num_evolutions=30,
            strategies={"reasoning": 1.0, "comparative": 0.0},
        )
    )

    out = _apply(stage, Golden(input="q"), _ctx(model, seed=3))

    drawn = {e["strategy"] for e in out.metadata["lineage"]}
    assert drawn == {"reasoning"}


def test_an_unknown_strategy_is_rejected_with_the_valid_list():
    with pytest.raises(ValueError, match="reasoning"):
        EvolutionStage(EvolutionConfig(strategies={"nonsense": 1.0}))


def test_an_ungrounded_evolution_is_flagged_not_discarded():
    """Filtration owns the keep/discard decision, so this only reports."""
    model = ScriptedLLM([_evolved("drifted", grounded=False)])
    stage = EvolutionStage(EvolutionConfig(num_evolutions=1))

    out = _apply(stage, Golden(input="q"), _ctx(model, seed=1))

    assert out is not None
    assert out.metadata["ungrounded"] is True
    assert out.metadata["lineage"][-1]["grounded"] is False


def test_an_empty_evolution_keeps_the_previous_input():
    model = ScriptedLLM([_evolved("")])
    stage = EvolutionStage(EvolutionConfig(num_evolutions=1))

    out = _apply(stage, Golden(input="original"), _ctx(model, seed=1))

    assert out.input == "original"
    assert out.metadata["evolutions"] == 0


def test_the_context_reaches_the_evolve_prompt():
    model = ScriptedLLM([_evolved("harder")])
    stage = EvolutionStage(EvolutionConfig(num_evolutions=1))

    _apply(stage, Golden(input="q"), _ctx(model, context=["rate limit is 500"]))

    assert "rate limit is 500" in model.prompts[0]


def test_evolution_metadata_keys_declaration_stays_true():
    model = ScriptedLLM([_evolved("harder", grounded=False)])
    stage = EvolutionStage(EvolutionConfig(num_evolutions=1))

    out = _apply(stage, Golden(input="q"), _ctx(model, seed=1))

    assert set(out.metadata) <= set(stage.metadata_keys)


# --------------------------------------------------------------------------- #
# StylingStage
# --------------------------------------------------------------------------- #


def _styling():
    return StylingConfig(
        scenario="tyre retail support",
        task="answer billing questions",
        input_format="a short customer message",
    )


def test_styling_rewrites_the_input():
    model = ScriptedLLM([json.dumps({"input": "hiya, quick billing q"})])
    stage = StylingStage(_styling())

    out = _apply(stage, Golden(input="What is the invoice policy?"), _ctx(model))

    assert out.input == "hiya, quick billing q"
    assert out.metadata["styled"] is True
    assert model.calls == 1


def test_unconfigured_styling_makes_no_call():
    """A default-constructed styling stage in the chain must cost nothing."""
    stage = StylingStage(StylingConfig())
    golden = Golden(input="untouched")

    out = _apply(stage, golden, _ctx(ExplodingLLM()))

    assert out.input == "untouched"
    assert "styled" not in out.metadata


def test_the_setting_reaches_the_style_prompt():
    model = ScriptedLLM([json.dumps({"input": "restyled"})])

    _apply(stage := StylingStage(_styling()), Golden(input="q"), _ctx(model))

    assert stage.name == "style"
    prompt = model.prompts[0]
    for expected in ("tyre retail support", "answer billing questions"):
        assert expected in prompt


def test_an_empty_restyle_keeps_the_original():
    model = ScriptedLLM([json.dumps({"input": "   "})])
    stage = StylingStage(_styling())

    out = _apply(stage, Golden(input="original"), _ctx(model))

    assert out.input == "original"


def test_styling_metadata_keys_declaration_stays_true():
    model = ScriptedLLM([json.dumps({"input": "restyled"})])
    stage = StylingStage(_styling())

    out = _apply(stage, Golden(input="q"), _ctx(model))

    assert set(out.metadata) <= set(stage.metadata_keys)


# --------------------------------------------------------------------------- #
# ExpectedOutputStage
# --------------------------------------------------------------------------- #


def test_expected_output_is_written_from_the_context():
    model = ScriptedLLM([json.dumps({"expected_output": "500 per minute."})])
    stage = ExpectedOutputStage()

    out = _apply(
        stage, Golden(input="q"), _ctx(model, context=["the limit is 500/min"])
    )

    assert out.expected_output == "500 per minute."
    assert "the limit is 500/min" in model.prompts[0]
    assert model.calls == 1


def test_expected_output_is_skipped_when_switched_off():
    stage = ExpectedOutputStage()
    golden = Golden(input="q")

    out = _apply(stage, golden, _ctx(ExplodingLLM(), include_expected_output=False))

    assert out.expected_output is None


def test_a_golden_without_context_is_answered_but_flagged():
    """Silently empty would read as "the stage failed" rather than "no context".

    Scratch-generated goldens have no context by construction; refusing to
    answer them would leave that source unable to produce reference answers at
    all.
    """
    model = ScriptedLLM([json.dumps({"expected_output": "an answer"})])
    stage = ExpectedOutputStage()

    out = _apply(stage, Golden(input="q"), _ctx(model, context=[]))

    assert out.expected_output == "an answer"
    assert out.metadata["expected_output_grounded"] is False


def test_the_output_format_rule_appears_only_when_configured():
    model = ScriptedLLM([json.dumps({"expected_output": "a"})])
    _apply(ExpectedOutputStage(), Golden(input="q"), _ctx(model))
    assert "Shape the answer" not in model.prompts[0]

    model = ScriptedLLM([json.dumps({"expected_output": "a"})])
    stage = ExpectedOutputStage(StylingConfig(expected_output_format="one sentence"))
    _apply(stage, Golden(input="q"), _ctx(model))
    assert "one sentence" in model.prompts[0]


def test_expected_output_metadata_keys_declaration_stays_true():
    model = ScriptedLLM([json.dumps({"expected_output": "a"})])
    stage = ExpectedOutputStage()

    out = _apply(stage, Golden(input="q"), _ctx(model, context=[]))

    assert set(out.metadata) <= set(stage.metadata_keys)


# --------------------------------------------------------------------------- #
# PerturbationStage
# --------------------------------------------------------------------------- #


def test_perturbation_makes_no_model_call():
    stage = PerturbationStage(["uppercase"])

    out = _apply(stage, Golden(input="hello there"), _ctx(ExplodingLLM(), seed=1))

    assert out.input == "HELLO THERE"
    assert out.metadata["perturbation"] == "uppercase"


def test_perturbation_is_recorded_in_the_lineage():
    stage = PerturbationStage(["lowercase"])

    out = _apply(stage, Golden(input="LOUD"), _ctx(ExplodingLLM(), seed=1))

    assert out.metadata["lineage"][-1] == {
        "stage": "perturb",
        "transform": "lowercase",
    }


def test_perturbation_is_reproducible_under_a_seed():
    """Covers the transform's own randomness, not just the choice of transform.

    ``perturbations`` draws from the global ``random`` module, so seeding only
    the transform choice would make this stage's seeding promise half true.
    """

    def run():
        stage = PerturbationStage(["typo", "ocr_typo", "contraction"])
        out = _apply(
            stage,
            Golden(input="the quick brown fox jumps over the lazy dog"),
            _ctx(ExplodingLLM(), seed=1234),
        )
        return out.input, out.metadata["perturbation"]

    assert run() == run()


def test_perturbation_does_not_leak_its_seed_into_the_global_random():
    """The global RNG state is restored, so the rest of the process is unaffected."""
    import random

    random.seed(0)
    before = [random.random() for _ in range(3)]

    random.seed(0)
    _apply(
        PerturbationStage(["typo"]),
        Golden(input="a sentence to perturb"),
        _ctx(ExplodingLLM(), seed=99),
    )
    after = [random.random() for _ in range(3)]

    assert before == after


def test_an_unknown_transform_is_rejected_with_the_valid_list():
    with pytest.raises(ValueError, match="uppercase"):
        PerturbationStage(["nonsense"])


def test_an_empty_transform_list_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        PerturbationStage([])


def test_every_declared_transform_resolves():
    """``TRANSFORMS`` names must match what ``perturbations`` actually exports."""
    for name, fn in TRANSFORMS.items():
        assert callable(fn), name
        assert fn(["a sample sentence"]), name


def test_perturbation_metadata_keys_declaration_stays_true():
    stage = PerturbationStage(["uppercase"])

    out = _apply(stage, Golden(input="q"), _ctx(ExplodingLLM(), seed=1))

    assert set(out.metadata) <= set(stage.metadata_keys)
