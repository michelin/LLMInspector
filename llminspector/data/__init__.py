"""Static lookup tables used by the perturbation transforms.

These are the six tables the legacy 148k-line ``constants.py`` actually used
(the rest was dropped). They are stored as JSON data files alongside this module
and loaded lazily / cached on first access:

    TYPO_FREQUENCY     keyboard-adjacency frequency vectors  (add_typo)
    CONTRACTION_MAP    phrase -> contraction                 (add_contraction)
    ocr_typo_dict      word -> OCR-style typo                (add_ocr_typo)
    abbreviation_dict  abbreviation -> [expansions]          (add_abbreviation)
    starting_context   greeting prefixes                     (add_context)
    ending_context     closing suffixes                      (add_context)

Access via the module-level names (e.g. ``from llminspector.data import
CONTRACTION_MAP``) — they are resolved on demand through ``__getattr__``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def _load(filename: str) -> Any:
    with open(_DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


_TABLES = {
    "TYPO_FREQUENCY": ("typo_frequency.json", None),
    "CONTRACTION_MAP": ("contraction_map.json", None),
    "ocr_typo_dict": ("ocr_typo_dict.json", None),
    "abbreviation_dict": ("abbreviation_dict.json", None),
    "starting_context": ("context.json", "starting_context"),
    "ending_context": ("context.json", "ending_context"),
}

__all__ = list(_TABLES)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy module attribute access for the tables."""
    if name in _TABLES:
        filename, key = _TABLES[name]
        data = _load(filename)
        return data[key] if key is not None else data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
