# data/

Six static lookup tables backing the perturbation transforms, split out of the
legacy 148k-line `constants.py` (the rest was dropped) and stored as JSON
alongside this module.

```
TYPO_FREQUENCY     typo_frequency.json     add_typo
CONTRACTION_MAP    contraction_map.json    add_contraction
ocr_typo_dict      ocr_typo_dict.json      add_ocr_typo
abbreviation_dict  abbreviation_dict.json  add_abbreviation
starting_context   context.json            add_context
ending_context     context.json            add_context
```

## Two things to know

**Access is lazy, via PEP 562 `__getattr__`.** `from llminspector.data import
CONTRACTION_MAP` resolves and `@lru_cache`s the file on first touch. That is why
`__all__` names attributes with no module-level binding, and why pylint's
`undefined-all-variable` is disabled package-wide. Adding a table means one entry
in `_TABLES` — nothing else.

**These files must stay in the wheel.** They load at runtime from
`Path(__file__).parent`. `pyproject.toml` declares
`[tool.setuptools.package-data] llminspector = ["data/*.json", "py.typed"]`
explicitly rather than relying on `include-package-data`. Drop that and every
perturbation raises `FileNotFoundError` in an installed environment while
passing every test in the source tree. `.claude/skills/release-check` verifies
the built wheel actually contains them.
