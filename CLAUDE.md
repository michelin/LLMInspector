# LLMInspector

Python package for end-to-end LLM evaluation: build a dataset, score it against
a set of metrics, export the results. Also generates synthetic evaluation
datasets (alignment, adversarial, RAG).

```python
from llminspector import evaluate
from llminspector.dataset import EvaluationDataset
from llminspector.metrics import FaithfulnessMetric

dataset = EvaluationDataset.from_excel("rows.xlsx")
result = evaluate(dataset, metrics=[FaithfulnessMetric(model=model)])
```

## Environment

The project targets Python 3.12 and uses a local `.venv`. **The venv is built
with `uv` and has no pip.**

```bash
.venv/bin/python -m pytest -q          # full suite, a few seconds
uv pip install --python .venv/bin/python <pkg>
uv pip install --python .venv/bin/python -e . --no-build-isolation
.venv/bin/pre-commit run --all-files   # black, isort, pylint, mypy
```

The build backend declared in `pyproject.toml` (`pydnx_packaging`) is internal
and not on public PyPI. That is why editable installs need
`--no-build-isolation` with setuptools/wheel already present, and why
`llminspector/version.py` is absent in a source checkout — it is generated at
build time and `__init__.py` tolerates its absence.

`ragas` is an optional extra: `pip install 'llminspector[ragas]'`. Everything
except the five context metrics and the RAG testset engine works without it.

A PreToolUse hook blocks bare `pip install` and bare `pytest` for these reasons.

## Architecture

Strict one-directional layering. **Never import upward.**

```
test_case → dataset → models → metrics → evaluate → synthesizer → reporting
                        ↑                                ↑
                    config                             data
```

| Package | Owns |
|---|---|
| `test_case/` | `LLMTestCase` — the single row schema |
| `dataset/` | `EvaluationDataset`, `Golden`, Excel/DataFrame mapping |
| `models/` | Provider ABCs (`BaseLLM`, `BaseEmbeddingModel`), Azure OpenAI, retry |
| `metrics/` | `BaseMetric` and the concrete metric classes |
| `evaluate/` | The async batch engine and `EvaluationResult` |
| `synthesizer/` | Dataset generation; a stable shell over swappable `engines/` |
| `reporting/` | `to_dataframe`, `to_excel`, `summary`, `errors` |
| `config/`, `data/`, `utils/` | Settings, static JSON tables, small helpers |

Most of these carry their own `CLAUDE.md` with the local contract. **Read it
before changing anything in that directory** — several of them document
invariants that are not obvious from the code.

### The package root is a contract

`llminspector/__init__.py` exports the seven subpackage namespaces plus
`evaluate`, `a_evaluate`, and `EvaluationResult`. Everything else is imported
from the subpackage that owns it, so every class has exactly one import path.
**Adding a name to the root is a permanent compatibility commitment — don't,
unless asked.**

Wart worth knowing: `llminspector.evaluate` is both a subpackage and the
re-exported function, and the function wins attribute lookup. Reach the module
with `from llminspector.evaluate import EvaluationResult` or `import_module`.

### Two conventions everything else rests on

1. **Heavy imports are deferred to first use.** langchain, ragas, transformers,
   torch, presidio, lingua — imported inside the function that needs them, never
   at module top level. `import llminspector` must stay cheap and must not
   require any optional dependency. This is why `import-outside-toplevel` is
   disabled in pylint: it is a design rule, not an oversight.
2. **Optional dependencies stay confined to one module each.** ragas may only be
   imported by `metrics/` classes deriving from `RagasBackedMetric` and by
   `synthesizer/engines/ragas_testset.py`, always through
   `utils/optional.py::optional_dependency`, which turns a bare
   `ModuleNotFoundError` into install instructions.

## Workflow

Feature work follows `.claude/skills/feature-work`:

1. Present the tasks and affected files. **Wait for approval before writing code.**
2. Create a feature branch off the integration branch (`git checkout -b <name>`).
3. Implement it fully, with tests.
4. Stop and hand back for manual verification. Do not commit yet.
5. Commit and merge **only on explicit go-ahead**.

Commit messages carry **no `Co-Authored-By` trailer** — summary and body only,
merge commits included. A PreToolUse hook enforces this and blocks commits made
directly on `main` or the integration branch.

## Testing

- One test module per component, in `tests/`. See `tests/CLAUDE.md`.
- **No network and no real credentials in tests.** Providers are stubbed; a test
  that would call a live API is a bug in the test.
- Every behaviour change ships with a test in the same commit.
- The contract tests — `test_public_api.py`, `test_column_contract.py`,
  `test_provider_contract.py`, `test_scaffolding.py` — catch layering and export
  regressions. When one fails, the default assumption is that the change is
  wrong, not the expectation.
- A Stop hook runs the suite whenever `llminspector/` or `tests/` changed, so
  don't spend a turn on `pytest` unless you need a specific failure in view.

## Docs and examples are part of the change

A public-API change is not done until the matching guide in `docs/guides/` and
the affected notebooks in `examples/` are updated in the same branch. Runnable
notebooks and markdown usage guides ship alongside the API reference, not after
it. `tests/test_examples.py` fails if a notebook import stops resolving.

## Style

- Black + isort, line length 88. Enforced by a PostToolUse hook and pre-commit.
- Comments explain *why*, particularly where the code diverges from the obvious
  choice. The existing modules set the density — match them.
- Type annotations on public functions: the package ships `py.typed`, so
  annotations are load-bearing for downstream users. mypy runs lenient today and
  is tightened by removing flags from `.pre-commit-config.yaml`.
- `from __future__ import annotations` at the top of new modules.

## Generated — do not edit

`docs/_build/`, `graphify-out/`, `.venv/`, `__pycache__/`, `.coverage`, and
`llminspector/version.py`. The deny-list in `.claude/settings.json` covers them.
