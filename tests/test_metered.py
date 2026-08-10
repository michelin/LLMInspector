"""``MeteredModel`` — token accounting without a contract change.

The point of the class is that it is invisible: a wrapped model must behave
exactly like the model it wraps, including its optional capabilities. Most of
these tests are about what metering must *not* change.
"""

import asyncio
import json

import pytest
from pydantic import BaseModel

from llminspector.generation import ContextSource, GenerationConfig, Generator
from llminspector.models.base_model import BaseLLM
from llminspector.models.metered import MeteredModel, total_usage
from tests.conftest import ScriptedLLM


class Answer(BaseModel):
    text: str


def _inputs(*texts):
    return json.dumps({"inputs": list(texts)})


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #


def test_calls_and_tokens_are_counted():
    model = MeteredModel(ScriptedLLM(["a reply", "another reply"]))

    model.generate("first prompt")
    asyncio.run(model.a_generate("second prompt"))

    usage = model.usage
    assert usage["calls"] == 2
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == (
        usage["prompt_tokens"] + usage["completion_tokens"]
    )


def test_a_longer_prompt_costs_more():
    """Sanity that this measures something rather than counting calls twice."""
    short = MeteredModel(ScriptedLLM(["x"]))
    long = MeteredModel(ScriptedLLM(["x"]))

    short.generate("hi")
    long.generate("hi " * 200)

    assert long.usage["prompt_tokens"] > short.usage["prompt_tokens"]


def test_reset_zeroes_the_counters():
    model = MeteredModel(ScriptedLLM(["a", "b"]))
    model.generate("one")

    model.reset()
    model.generate("two")

    assert model.usage["calls"] == 1


def test_a_non_string_response_is_counted_rather_than_crashing():
    """Accounting is the least important thing happening in a run."""

    class OddProvider(BaseLLM):
        def get_model_name(self):
            return "odd"

        def generate(self, prompt, **kwargs):
            return 12345  # not a str

        async def a_generate(self, prompt, **kwargs):
            return 12345

    model = MeteredModel(OddProvider())
    assert model.generate("p") == 12345
    assert model.usage["completion_tokens"] > 0


# --------------------------------------------------------------------------- #
# the decorator must be invisible
# --------------------------------------------------------------------------- #


def test_a_metered_model_is_a_base_llm():
    model = MeteredModel(ScriptedLLM(["x"]))
    assert isinstance(model, BaseLLM)
    assert model.get_model_name() == "scripted"


def test_structured_output_delegates_to_the_inner_provider():
    """A wrapped Azure model must not lose its native JSON mode.

    Falling back to ``BaseLLM``'s default implementation would silently drop the
    ``response_format`` the provider's own override adds — a decorator that
    changes behaviour is not a decorator.
    """
    calls = []

    class NativeProvider(ScriptedLLM):
        def generate_structured(self, prompt, schema, **kwargs):
            calls.append("sync")
            return schema(text="native")

        async def a_generate_structured(self, prompt, schema, **kwargs):
            calls.append("async")
            return schema(text="native")

    model = MeteredModel(NativeProvider([]))

    assert model.generate_structured("p", Answer).text == "native"
    assert asyncio.run(model.a_generate_structured("p", Answer)).text == "native"
    assert calls == ["sync", "async"]


def test_structured_output_still_works_on_a_bare_three_method_provider():
    """Delegation must not require the inner model to override anything."""
    model = MeteredModel(ScriptedLLM([json.dumps({"text": "parsed"})]))

    assert model.generate_structured("p", Answer).text == "parsed"


def test_unknown_attributes_are_forwarded():
    """Wrapping must not hide ``max_workers`` and change evaluate's batch size."""

    class WithLimit(ScriptedLLM):
        max_workers = 3

    model = MeteredModel(WithLimit(["x"]))

    assert model.max_workers == 3


def test_forwarding_does_not_shadow_the_metering_state():
    model = MeteredModel(ScriptedLLM(["x"]))
    model.generate("p")

    assert model.calls == 1, "the wrapper's own counter, not the inner model's"


def test_prompts_still_reach_the_inner_model():
    inner = ScriptedLLM(["x"])
    MeteredModel(inner).generate("the actual prompt")

    assert inner.prompts == ["the actual prompt"]


# --------------------------------------------------------------------------- #
# totals
# --------------------------------------------------------------------------- #


def test_total_usage_sums_across_models():
    a = MeteredModel(ScriptedLLM(["x"]))
    b = MeteredModel(ScriptedLLM(["y"]))
    a.generate("one")
    b.generate("two")

    totals = total_usage([a, b])

    assert totals["calls"] == 2


def test_the_same_model_is_not_counted_twice():
    """``critic`` falls back to ``model``; they are then the same object."""
    model = MeteredModel(ScriptedLLM(["x"]))
    model.generate("p")

    assert total_usage([model, model])["calls"] == 1


def test_unmetered_models_are_ignored():
    metered = MeteredModel(ScriptedLLM(["x"]))
    metered.generate("p")

    assert total_usage([metered, ScriptedLLM(["y"]), None])["calls"] == 1


# --------------------------------------------------------------------------- #
# wiring into a run
# --------------------------------------------------------------------------- #


def test_track_usage_wraps_the_models():
    config = GenerationConfig(model=ScriptedLLM(["x"]), track_usage=True)
    assert isinstance(config.model, MeteredModel)


def test_wrapping_is_idempotent():
    """Re-wrapping would stack decorators and double-count every token."""
    model = MeteredModel(ScriptedLLM(["x"]))
    config = GenerationConfig(model=model, track_usage=True)

    assert config.model is model


def test_an_untracked_run_reports_no_usage_rather_than_zeroes():
    """ "Not measured" and "measured, cost nothing" must not look alike."""
    config = GenerationConfig(model=ScriptedLLM([]), track_usage=False)
    assert config.usage() is None


def test_a_tracked_run_reports_totals_on_the_result():
    model = ScriptedLLM(lambda prompt: _inputs("q1"))
    config = GenerationConfig(
        model=model, show_progress=False, track_usage=True, seed=1
    )
    generator = Generator(ContextSource([["some context"]]), (), config=config)

    result = asyncio.run(generator.a_generate())

    assert result.usage is not None
    assert result.usage["calls"] == 1
    assert result.usage["total_tokens"] > 0


def test_an_untracked_generation_leaves_usage_none():
    model = ScriptedLLM(lambda prompt: _inputs("q1"))
    config = GenerationConfig(model=model, show_progress=False)
    generator = Generator(ContextSource([["ctx"]]), (), config=config)

    assert asyncio.run(generator.a_generate()).usage is None


def test_the_critic_is_metered_separately_but_summed_once():
    generating = ScriptedLLM(lambda p: "gen")
    critic = ScriptedLLM(lambda p: "crit")
    config = GenerationConfig(model=generating, critic_model=critic, track_usage=True)

    config.model.generate("a")
    config.critic_model.generate("b")

    assert config.usage()["calls"] == 2


def test_a_reask_is_counted_as_two_calls():
    """The reason metering instruments the inner model rather than the wrapper.

    ``generate_structured`` is implemented in terms of ``self.generate``, so a
    wrapper that counted only its own surface would report one call for a
    structured request that actually cost two — and a reask is precisely when
    you most want to know.
    """
    model = MeteredModel(
        ScriptedLLM(["not json at all", json.dumps({"text": "repaired"})])
    )

    assert model.generate_structured("p", Answer).text == "repaired"
    assert model.usage["calls"] == 2


def test_structured_output_through_the_wrapper_is_counted_at_all():
    model = MeteredModel(ScriptedLLM([json.dumps({"text": "ok"})]))

    model.generate_structured("a prompt", Answer)

    assert model.usage["calls"] == 1
    assert model.usage["prompt_tokens"] > 0


def test_unwrap_restores_the_inner_model():
    inner = ScriptedLLM(["a", "b"])
    metered = MeteredModel(inner)
    metered.generate("one")

    restored = metered.unwrap()
    restored.generate("two")

    assert restored is inner
    assert metered.usage["calls"] == 1, "the second call is no longer counted"


def test_calls_made_directly_on_the_wrapper_are_not_double_counted():
    model = MeteredModel(ScriptedLLM(["a"]))

    model.generate("one")

    assert model.usage["calls"] == 1
