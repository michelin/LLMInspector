"""``ScratchSource`` and ``SeedGoldenSource`` — the two ungrounded-capable seeds.

The substance here is the routing. ``SeedGoldenSource`` **partitions** its seeds
into grounded and context-free and runs both paths, where the design it is
adapted from routes all-or-nothing on whether any seed has context and silently
drops the rest.
"""

import asyncio
import json

import pytest

from llminspector.dataset.golden import Golden
from llminspector.generation.config import GenerationConfig, StylingConfig
from llminspector.generation.sources.scratch import ScratchSource
from llminspector.generation.sources.seed import SeedGoldenSource
from tests.conftest import ScriptedLLM


def _inputs(*texts):
    return json.dumps({"inputs": list(texts)})


def _styling():
    return StylingConfig(
        scenario="tyre retail support",
        task="answer billing and delivery questions",
        input_format="a short customer chat message",
    )


def _config(model, **kwargs):
    kwargs.setdefault("show_progress", False)
    return GenerationConfig(model=model, **kwargs)


def _produce(source, model, **kwargs):
    return asyncio.run(source.a_produce(_config(model, **kwargs)))


# --------------------------------------------------------------------------- #
# ScratchSource
# --------------------------------------------------------------------------- #


def test_scratch_generates_from_styling_alone():
    model = ScriptedLLM([_inputs("q1", "q2", "q3")])
    source = ScratchSource(_styling(), num_goldens=3)

    goldens = _produce(source, model)

    assert [g.input for g in goldens] == ["q1", "q2", "q3"]
    assert all(g.context is None for g in goldens)
    assert all(g.metadata["generated_from"] == "scratch" for g in goldens)


def test_every_missing_styling_field_is_reported_at_once():
    """One message, not three round trips.

    Reporting only the first missing field means the caller fixes it, reruns,
    and is told about the second.
    """
    with pytest.raises(ValueError) as excinfo:
        ScratchSource(StylingConfig(scenario="retail"), num_goldens=2)

    message = str(excinfo.value)
    assert "task" in message and "input_format" in message
    assert "scenario" not in message.split("Missing:")[1].split(".")[0]


def test_a_completely_empty_styling_names_all_three():
    with pytest.raises(ValueError) as excinfo:
        ScratchSource(StylingConfig(), num_goldens=2)

    for field in ("scenario", "task", "input_format"):
        assert field in str(excinfo.value)


def test_scratch_rejects_a_nonsense_count():
    with pytest.raises(ValueError, match="num_goldens"):
        ScratchSource(_styling(), num_goldens=0)


def test_the_styling_reaches_the_scratch_prompt():
    model = ScriptedLLM([_inputs("q1")])
    _produce(ScratchSource(_styling(), num_goldens=1), model)

    prompt = model.prompts[0]
    assert "tyre retail support" in prompt
    assert "a short customer chat message" in prompt


def test_scratch_makes_no_grounding_claim_in_its_lineage():
    model = ScriptedLLM([_inputs("q1")])
    goldens = _produce(ScratchSource(_styling(), num_goldens=1), model)

    assert goldens[0].metadata["lineage"] == [{"stage": "generate", "grounded": False}]


def test_large_runs_are_batched_rather_than_asked_for_in_one_reply():
    """Asking for hundreds in one reply degrades into near-duplicates."""
    model = ScriptedLLM(lambda p: _inputs(*[f"q{i}" for i in range(10)]))
    source = ScratchSource(_styling(), num_goldens=25)

    _produce(source, model)

    assert model.calls == 3, "10 + 10 + 5"


def test_duplicate_inputs_across_batches_are_dropped():
    """A run asked for 20 must not return 20 with repeats."""
    model = ScriptedLLM(lambda p: _inputs("same", "also same"))
    source = ScratchSource(_styling(), num_goldens=20)

    goldens = _produce(source, model)

    assert len(goldens) == 2


def test_scratch_metadata_keys_declaration_stays_true():
    model = ScriptedLLM([_inputs("q1")])
    source = ScratchSource(_styling(), num_goldens=1)

    goldens = _produce(source, model)

    assert set(goldens[0].metadata) <= set(source.metadata_keys)


# --------------------------------------------------------------------------- #
# SeedGoldenSource — routing
# --------------------------------------------------------------------------- #


def _seed_dispatch(prompt):
    if "Infer the setting" in prompt:
        return json.dumps(
            {
                "scenario": "tyre retail",
                "task": "support",
                "input_format": "short chat message",
            }
        )
    return _inputs("new1", "new2")


def test_mixed_seeds_run_both_paths_and_nothing_is_dropped():
    """The partition fix.

    All-or-nothing routing sends the whole batch down the grounded path on the
    strength of one context-bearing seed, and every context-free seed vanishes
    without a word.
    """
    seeds = [
        Golden(input="grounded one", context=["ctx a"]),
        Golden(input="bare one"),
        Golden(input="grounded two", context=["ctx b"]),
        Golden(input="bare two"),
    ]
    model = ScriptedLLM(_seed_dispatch)
    source = SeedGoldenSource(seeds, max_per_golden=2, styling=_styling())

    goldens = _produce(source, model)

    origins = {g.metadata["generated_from"] for g in goldens}
    assert origins == {"seed:context", "seed:scratch"}


def test_only_grounded_seeds_use_only_the_context_path():
    seeds = [Golden(input="a", context=["ctx"])]
    model = ScriptedLLM(_seed_dispatch)
    source = SeedGoldenSource(seeds, styling=_styling())

    goldens = _produce(source, model)

    assert all(g.metadata["generated_from"] == "seed:context" for g in goldens)
    assert all(g.context for g in goldens)


def test_only_bare_seeds_use_only_the_scratch_path():
    seeds = [Golden(input="a"), Golden(input="b")]
    model = ScriptedLLM(_seed_dispatch)
    source = SeedGoldenSource(seeds, styling=_styling())

    goldens = _produce(source, model)

    assert all(g.metadata["generated_from"] == "seed:scratch" for g in goldens)


def test_generated_goldens_are_traceable_to_their_seed():
    seeds = [
        Golden(input="grounded", context=["ctx a"]),
        Golden(input="bare"),
    ]
    model = ScriptedLLM(_seed_dispatch)
    source = SeedGoldenSource(seeds, max_per_golden=2, styling=_styling())

    goldens = _produce(source, model)

    seed_ids = {g.id for g in seeds}
    assert all(g.metadata["seed_id"] in seed_ids for g in goldens)


# --------------------------------------------------------------------------- #
# SeedGoldenSource — styling extraction
# --------------------------------------------------------------------------- #


def test_styling_is_reverse_engineered_when_not_supplied():
    """So an augmented set sounds like the set it grew from."""
    seeds = [Golden(input="hiya, wheres my invoice?", context=["ctx"])]
    model = ScriptedLLM(_seed_dispatch)
    source = SeedGoldenSource(seeds)

    _produce(source, model)

    assert "Infer the setting" in model.prompts[0]
    # The inferred description reaches the generation prompt that follows.
    assert "tyre retail" in model.prompts[1]


def test_supplied_styling_skips_the_extraction_call():
    seeds = [Golden(input="a", context=["ctx"])]
    model = ScriptedLLM([_inputs("new1")])
    source = SeedGoldenSource(seeds, max_per_golden=1, styling=_styling())

    _produce(source, model)

    assert model.calls == 1
    assert "Infer the setting" not in model.prompts[0]


def test_only_a_sample_of_seeds_is_sent_for_extraction():
    seeds = [Golden(input=f"question {i}", context=["ctx"]) for i in range(40)]
    model = ScriptedLLM(_seed_dispatch)

    _produce(SeedGoldenSource(seeds, max_per_golden=1), model)

    extraction = model.prompts[0]
    assert "question 0" in extraction
    assert "question 39" not in extraction, "the whole set should not be sent"


def test_incomplete_inferred_styling_is_reported_for_bare_seeds():
    """Bare seeds have nothing but the styling, so an incomplete one is fatal."""

    def dispatch(prompt):
        if "Infer the setting" in prompt:
            return json.dumps({"scenario": "retail", "task": "", "input_format": ""})
        return _inputs("new1")

    model = ScriptedLLM(dispatch)
    source = SeedGoldenSource([Golden(input="bare")])

    with pytest.raises(ValueError, match="task"):
        _produce(source, model)


# --------------------------------------------------------------------------- #
# SeedGoldenSource — validation
# --------------------------------------------------------------------------- #


def test_an_empty_seed_set_is_rejected():
    with pytest.raises(ValueError, match="at least one seed"):
        SeedGoldenSource([])


def test_a_nonsense_per_golden_count_is_rejected():
    with pytest.raises(ValueError, match="max_per_golden"):
        SeedGoldenSource([Golden(input="a")], max_per_golden=0)


def test_seed_metadata_keys_declaration_stays_true():
    seeds = [Golden(input="a", context=["ctx"]), Golden(input="b")]
    model = ScriptedLLM(_seed_dispatch)
    source = SeedGoldenSource(seeds, styling=_styling())

    goldens = _produce(source, model)

    for golden in goldens:
        assert set(golden.metadata) <= set(source.metadata_keys)
