# tests/

553 tests, ~7s with `.venv/bin/python -m pytest -q`. Fast is a feature: a Stop
hook runs the whole suite on any turn that touched `llminspector/` or `tests/`.
Keep it that way — a test that takes seconds belongs behind a marker or a stub.

## Hard rules

**No network.** Providers are stubbed (`StubLLM`, `StubModel`, `FakeMetric`,
`_FakeLangchainLLMWrapper`). A test that would reach Azure, HuggingFace Hub, or
any API is a bug in the test, not a reason to add credentials.

**No real credentials.** `tests/test_settings.py` uses an autouse fixture to
scrub `LLMINSPECTOR_*` from the environment. Never hardcode a key, even a fake
one that looks real.

**Fixtures in `tests/test_sample/`** — two small xlsx workbooks and a config
file. Use them rather than generating spreadsheets at import time.

## The four contract tests

These are the ones that catch structural regressions. When one fails, the
default assumption is that the *change* is wrong, not the expectation.

| File | Pins |
|---|---|
| `test_public_api.py` | The package root exports exactly its 10 names; subpackage `__all__`s are accurate. |
| `test_column_contract.py` | `GOLDEN_HEADER` — the exact exported column order, a frozen artefact of the legacy `_REORDER_KEYS` list. The engine must reproduce it from the metrics alone. |
| `test_provider_contract.py` | What a provider must implement. A new provider passes this file **unmodified** or it isn't done. |
| `test_scaffolding.py` | Every subpackage imports, and `__version__` resolves to a string or gracefully to `None` in a source checkout. |

`test_examples.py` is close kin: it parses every shipped notebook and resolves
every import against the current public API, and asserts the notebooks use
`a_evaluate` (the sync `evaluate()` raises inside Jupyter's loop). Rename a
public name and this fails — that's intended, the notebooks are the front door.

## Conventions

- One test module per component, named after it.
- Cheap metrics are exercised for real; LLM-judge metrics get a stub model whose
  `a_generate` returns a canned JSON string.
- Test the documented behaviour, including the deliberate divergences from
  legacy (input-order reassembly, requested-metrics-only columns, the fixed
  `add_contraction` / `add_abbreviation` perturbations).
- `pyproject.toml` sets `addopts = "--cov-branch"`; pass `--no-cov` for a quick
  run.
