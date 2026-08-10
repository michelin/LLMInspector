"""Synthetic input generation — the prompt and schema, shared by every source.

**This module holds no ``Stage``.** Generation turns one context into *many*
inputs, and :meth:`~llminspector.generation.stage.Stage.a_apply` is one golden
in, one golden out. Expressing 1-to-N as a stage would mean either a stage that
returns lists (breaking every other stage's signature) or a stage that mutates
the run's golden list behind the generator's back.

Producing the initial goldens is a *source*'s job, so that is where it lives.
What sits here is the part that would otherwise be copied into every source that
generates from scratch: the prompt, the response schema, and the call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from pydantic import BaseModel, Field

from ...utils.prompting import render_prompt

if TYPE_CHECKING:  # pragma: no cover
    from ..config import StylingConfig

__all__ = [
    "SyntheticInputs",
    "generate_inputs",
    "generate_from_styling",
    "SYNTHETIC_INPUTS_PROMPT",
    "SCRATCH_INPUTS_PROMPT",
]


class SyntheticInputs(BaseModel):
    """The model's reply: a list of standalone inputs."""

    inputs: List[str] = Field(default_factory=list)


# The wording is load-bearing — "answerable using only the information above" is
# what keeps generated inputs grounded, and the self-containment clause is what
# the filtration rubric then scores against. Do not reflow.
# pylint: disable=line-too-long
SYNTHETIC_INPUTS_PROMPT = """\
You are writing evaluation inputs for a question-answering system.

Read the source material below and write {num_inputs} distinct input(s) that a real user might send.
{styling}
Rules:
1. Each input must be answerable using **only** the information in the source material.
2. Each input must be self-contained: it must make sense to someone who cannot see the source material. Never write "according to the text", "in the passage above", or "this document".
3. Each input must have one clear objective. Do not bundle several questions together.
4. Vary the phrasing and the angle across the inputs. Do not paraphrase one question {num_inputs} times.

Source material:
{context}

Return JSON of the form:
{{"inputs": ["<input 1>", "<input 2>"]}}
"""

STYLING_BLOCK = """
Write them for this setting:
- Scenario: {scenario}
- Task the system performs: {task}
- Shape the input should take: {input_format}
"""
# pylint: enable=line-too-long


def _styling_block(styling: Optional["StylingConfig"]) -> str:
    """The styling paragraph, or nothing at all when styling is unset.

    Rendered as an empty string rather than "Scenario: None" — telling a model
    the scenario is ``None`` is worse than not raising the subject.
    """
    if styling is None:
        return ""
    fields = {
        "scenario": styling.scenario,
        "task": styling.task,
        "input_format": styling.input_format,
    }
    if not any(fields.values()):
        return ""
    return render_prompt(
        STYLING_BLOCK,
        list(fields),
        {k: v or "unspecified" for k, v in fields.items()},
        caller="generate_inputs",
    )


async def generate_inputs(
    model: Any,
    context: List[str],
    num_inputs: int,
    styling: Optional["StylingConfig"] = None,
) -> List[str]:
    """Ask ``model`` for ``num_inputs`` grounded inputs over ``context``.

    Truncates to ``num_inputs``. A model asked for two inputs routinely returns
    three; the reference implementation truncated on its async path and not on
    its sync one, so the same call produced different numbers of goldens
    depending on which entry point the caller used. There is one path here, and
    it truncates.

    Blank entries are dropped before truncation, so a model that pads its list
    with empty strings does not eat the quota.
    """
    prompt = render_prompt(
        SYNTHETIC_INPUTS_PROMPT,
        ["num_inputs", "styling", "context"],
        {
            "num_inputs": num_inputs,
            "styling": _styling_block(styling),
            "context": "\n\n".join(context),
        },
        caller="generate_inputs",
    )
    reply = await model.a_generate_structured(prompt, SyntheticInputs)
    cleaned = [text.strip() for text in reply.inputs if text and text.strip()]
    return cleaned[:num_inputs]


# Scratch generation has no source material, so the grounding clause is replaced
# by the styling description doing all the work. Do not reflow.
# pylint: disable=line-too-long
SCRATCH_INPUTS_PROMPT = """\
You are writing evaluation inputs for a question-answering system.

Write {num_inputs} distinct input(s) that a real user might send in this setting:
- Scenario: {scenario}
- Task the system performs: {task}
- Shape the input should take: {input_format}

Rules:
1. Each input must be self-contained and make sense on its own.
2. Each input must have one clear objective. Do not bundle several questions together.
3. Vary the phrasing, the angle, and the difficulty across the inputs.
4. Write what a user would actually type, not a polished specification of it.

Return JSON of the form:
{{"inputs": ["<input 1>", "<input 2>"]}}
"""
# pylint: enable=line-too-long


async def generate_from_styling(
    model: Any,
    num_inputs: int,
    styling: "StylingConfig",
) -> List[str]:
    """Ask ``model`` for ``num_inputs`` inputs described only by ``styling``.

    The ungrounded counterpart of :func:`generate_inputs`. Same truncation and
    blank-dropping rules, for the same reason: one path, and it truncates.
    """
    prompt = render_prompt(
        SCRATCH_INPUTS_PROMPT,
        ["num_inputs", "scenario", "task", "input_format"],
        {
            "num_inputs": num_inputs,
            "scenario": styling.scenario,
            "task": styling.task,
            "input_format": styling.input_format,
        },
        caller="generate_from_styling",
    )
    reply = await model.a_generate_structured(prompt, SyntheticInputs)
    cleaned = [text.strip() for text in reply.inputs if text and text.strip()]
    return cleaned[:num_inputs]
