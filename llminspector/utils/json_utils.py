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
