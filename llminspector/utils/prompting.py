"""Prompt template rendering — the f-string substitution every prompt uses.

This is ``BaseMetric._render`` hoisted out of the metric layer. It never had any
metric-specific state; it lived on the ABC only because metrics were the first
callers. The generation pipeline renders prompts too, and importing ``metrics``
from ``generation`` to reach a formatting helper would invert the package's
layering, so it moves down here where both layers can reach it.

Same semantics as the langchain f-string ``PromptTemplate`` that was removed in
Phase 8.3: ``{name}`` interpolates and ``{{`` / ``}}`` escape a literal brace,
which every prompt's JSON output block relies on.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

__all__ = ["render_prompt"]


def render_prompt(
    template: str,
    input_variables: Sequence[str],
    values: Mapping[str, Any],
    partial_variables: Optional[Mapping[str, Any]] = None,
    *,
    caller: str = "render_prompt",
) -> str:
    """Substitute ``values`` into ``template``.

    Parameters
    ----------
    template:
        An f-string-style template. ``{name}`` interpolates; ``{{`` / ``}}``
        escape a literal brace.
    input_variables:
        The template's declared variable list. It is checked against what was
        actually supplied so a prompt edit that adds a placeholder fails loudly
        instead of raising a bare ``KeyError`` from deep inside ``str.format``.
    values:
        The per-call substitutions.
    partial_variables:
        Substitutions fixed at construction time; merged over ``values``.
    caller:
        Name used to prefix the error message. Callers pass their own class name
        so the failure points at the prompt's owner, not at this helper.

    Raises
    ------
    KeyError
        When a declared variable was not supplied.
    """
    merged = dict(values)
    if partial_variables:
        merged.update(partial_variables)
    missing = [name for name in input_variables if name not in merged]
    if missing:
        raise KeyError(
            f"{caller}: prompt variables {missing} declared but "
            f"not supplied (got {sorted(merged)})"
        )
    return template.format(**merged)
