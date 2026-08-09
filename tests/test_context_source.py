"""``ContextSource`` — grounded generation from caller-supplied contexts.

The first source that actually calls a model, and the one the document pipeline
will reuse once it has built contexts of its own. These tests pin the two things
that are easy to get wrong: how many goldens come out of one context, and what
happens to the rest of the run when one context's call fails.

Model replies are JSON strings because ``a_generate_structured`` parses them —
``ScriptedLLM`` returns raw text and the schema layer does the rest.
"""

import asyncio
import json

import pytest

from llminspector.generation.config import GenerationConfig, StylingConfig
from llminspector.generation.sources.contexts import ContextSource
from tests.conftest import ScriptedLLM


def _inputs(*texts):
    """A scripted SyntheticInputs reply."""
    return json.dumps({"inputs": list(texts)})


def _config(model, **kwargs):
    kwargs.setdefault("show_progress", False)
    return GenerationConfig(model=model, **kwargs)


def _produce(source, model, **kwargs):
    return asyncio.run(source.a_produce(_config(model, **kwargs)))


# --------------------------------------------------------------------------- #
# how many goldens come out
# --------------------------------------------------------------------------- #


def test_one_call_per_context_and_one_golden_per_input():
    model = ScriptedLLM([_inputs("q1", "q2"), _inputs("q3", "q4")])
    source = ContextSource([["ctx a"], ["ctx b"]], max_goldens_per_context=2)

    goldens = _produce(source, model)

    assert [g.input for g in goldens] == ["q1", "q2", "q3", "q4"]
    assert model.calls == 2, "one generation call per context, not per golden"


def test_extra_inputs_are_truncated():
    """A model asked for two inputs routinely returns three.

    The design this replaces truncated on its async path and not on its sync
    one, so the same call produced a different number of goldens depending on
    which entry point the caller used. There is one path here and it truncates.
    """
    model = ScriptedLLM([_inputs("q1", "q2", "q3", "q4")])
    source = ContextSource([["ctx"]], max_goldens_per_context=2)

    assert [g.input for g in _produce(source, model)] == ["q1", "q2"]


def test_blank_inputs_do_not_eat_the_quota():
    """Padding the list with empty strings must not cost real goldens."""
    model = ScriptedLLM([_inputs("", "  ", "q1", "q2")])
    source = ContextSource([["ctx"]], max_goldens_per_context=2)

    assert [g.input for g in _produce(source, model)] == ["q1", "q2"]


def test_inputs_are_stripped():
    model = ScriptedLLM([_inputs("  q1  ")])
    source = ContextSource([["ctx"]], max_goldens_per_context=1)

    assert _produce(source, model)[0].input == "q1"


def test_no_contexts_makes_no_calls():
    model = ScriptedLLM([])
    assert _produce(ContextSource([]), model) == []
    assert model.calls == 0


def test_max_goldens_per_context_must_be_positive():
    with pytest.raises(ValueError, match="max_goldens_per_context"):
        ContextSource([["ctx"]], max_goldens_per_context=0)


# --------------------------------------------------------------------------- #
# context normalisation
# --------------------------------------------------------------------------- #


def test_a_bare_string_context_is_one_chunk_not_many_characters():
    """``["abc"]`` is one context of one chunk, not one context of 3 letters.

    Iterating a bare string is the classic failure here, and it fails quietly:
    the model gets a context of single characters and still returns something.
    """
    model = ScriptedLLM([_inputs("q1")])
    source = ContextSource(["abc"], max_goldens_per_context=1)

    golden = _produce(source, model)[0]

    assert golden.context == ["abc"]
    assert golden.metadata["context_size"] == 1


def test_context_is_carried_onto_the_golden():
    model = ScriptedLLM([_inputs("q1", "q2")])
    source = ContextSource([["chunk one", "chunk two"]], max_goldens_per_context=2)

    goldens = _produce(source, model)

    for golden in goldens:
        assert golden.context == ["chunk one", "chunk two"]
        assert golden.metadata["context_size"] == 2


def test_each_golden_gets_its_own_context_list():
    """A stage editing one golden's context must not touch its siblings'."""
    model = ScriptedLLM([_inputs("q1", "q2")])
    source = ContextSource([["shared"]], max_goldens_per_context=2)

    first, second = _produce(source, model)
    first.context.append("mutated")

    assert second.context == ["shared"]


# --------------------------------------------------------------------------- #
# the prompt
# --------------------------------------------------------------------------- #


def test_the_context_reaches_the_prompt():
    model = ScriptedLLM([_inputs("q1")])
    source = ContextSource([["the API allows 500 requests per minute"]])

    _produce(source, model)

    assert "500 requests per minute" in model.prompts[0]


def test_styling_reaches_the_prompt_when_set():
    model = ScriptedLLM([_inputs("q1")])
    source = ContextSource(
        [["ctx"]],
        styling=StylingConfig(
            scenario="tyre retail support",
            task="answer billing questions",
            input_format="a short customer message",
        ),
    )

    _produce(source, model)

    prompt = model.prompts[0]
    assert "tyre retail support" in prompt
    assert "answer billing questions" in prompt
    assert "a short customer message" in prompt


def test_unset_styling_does_not_tell_the_model_the_scenario_is_none():
    """An empty styling block is omitted, not rendered as "Scenario: None".

    Telling a model the scenario is ``None`` is worse than not raising the
    subject — it invites the model to reason about the absence.
    """
    model = ScriptedLLM([_inputs("q1")])
    _produce(ContextSource([["ctx"]]), model)

    assert "None" not in model.prompts[0]
    assert "Scenario" not in model.prompts[0]


def test_partially_filled_styling_still_renders():
    model = ScriptedLLM([_inputs("q1")])
    source = ContextSource([["ctx"]], styling=StylingConfig(scenario="banking"))

    _produce(source, model)

    prompt = model.prompts[0]
    assert "banking" in prompt
    # The unset fields say "unspecified" rather than "None".
    assert "unspecified" in prompt


# --------------------------------------------------------------------------- #
# lineage
# --------------------------------------------------------------------------- #


def test_generation_is_recorded_in_the_lineage():
    model = ScriptedLLM([_inputs("q1")])
    source = ContextSource([["a", "b"]], max_goldens_per_context=1)

    golden = _produce(source, model)[0]

    assert golden.metadata["lineage"] == [{"stage": "generate", "context_chunks": 2}]


def test_metadata_keys_declaration_stays_true():
    """The export column set must be knowable without paying for a run."""
    model = ScriptedLLM([_inputs("q1")])
    source = ContextSource([["ctx"]], max_goldens_per_context=1)

    golden = _produce(source, model)[0]

    assert set(golden.metadata) <= set(source.metadata_keys)


# --------------------------------------------------------------------------- #
# partial failure
# --------------------------------------------------------------------------- #


def test_one_failing_context_does_not_abort_the_others():
    """A bad context costs its own goldens and nothing else.

    Contexts are independent, so one unparseable reply must not take down a run
    over hundreds of them. The failure is recorded rather than swallowed.
    """

    def dispatch(prompt):
        if "poison" in prompt:
            return "not json, and not recoverable either"
        return _inputs("ok")

    model = ScriptedLLM(dispatch)
    source = ContextSource([["fine one"], ["poison"], ["fine two"]])

    goldens = _produce(source, model)

    assert [g.input for g in goldens] == ["ok", "ok"]
    assert [e["index"] for e in source.errors] == [1]
    assert "StructuredOutputError" in source.errors[0]["error"]


def test_errors_is_empty_before_a_run():
    """The attribute always exists, so checking it early is not an error."""
    assert ContextSource([["ctx"]]).errors == []


def test_errors_are_reset_between_runs():
    model = ScriptedLLM(lambda prompt: _inputs("q1"))
    source = ContextSource([["ctx"]])

    _produce(source, model)
    _produce(source, model)

    assert source.errors == []


# --------------------------------------------------------------------------- #
# concurrency
# --------------------------------------------------------------------------- #


def test_generation_is_bounded_by_max_concurrent():
    live = 0
    peak = 0

    class CountingLLM(ScriptedLLM):
        async def a_generate(self, prompt, **kwargs):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0)
            try:
                return self._next(prompt)
            finally:
                live -= 1

    model = CountingLLM(lambda prompt: _inputs("q"))
    source = ContextSource([[f"ctx {i}"] for i in range(12)])

    _produce(source, model, max_concurrent=3)

    assert peak <= 3
    assert peak > 1, "nothing overlapped, so the bound was not really tested"
