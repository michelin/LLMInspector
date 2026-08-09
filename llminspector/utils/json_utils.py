"""JSON-response parsing shared by the LLM-judge metrics.

The legacy metrics all stripped a fenced ```json block and ``` markers before
``json.loads``. This centralizes that identical inline pattern.
"""

import json
from typing import Any

JSON_BLOCK = "```json"


def parse_json_response(text: str, json_block: str = JSON_BLOCK) -> Any:
    """Strip ```json / ``` fences and parse the remaining text as JSON.

    Mirrors the legacy ``result.replace(JSON_BLOCK, "").replace("```", "")
    .strip()`` then ``json.loads`` sequence used throughout ``eval_metrics.py``.
    """
    cleaned = text.replace(json_block, "").replace("```", "").strip()
    return json.loads(cleaned)


def extract_json_object(text: str, json_block: str = JSON_BLOCK) -> Any:
    """Parse the first JSON object in ``text``, tolerating surrounding prose.

    :func:`parse_json_response` requires the whole (de-fenced) response to be
    valid JSON. That is the right contract for the judge metrics, whose prompts
    all pin the output format tightly and whose failure mode should be a scored
    ``None`` rather than a guess.

    Structured *generation* needs a looser reading: a model that prefixes
    ``Here is the JSON:`` has still answered, and burning a reask on it costs a
    real API call. So this falls back to slicing between the outermost braces.

    Kept separate rather than folded into ``parse_json_response`` so the metrics'
    stricter contract is not loosened by a change made for the generation layer.
    """
    try:
        return parse_json_response(text, json_block)
    except json.JSONDecodeError:
        cleaned = text.replace(json_block, "").replace("```", "")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])
