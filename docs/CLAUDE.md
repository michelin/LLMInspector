# docs/

Sphinx site (`index.rst`, `api.rst`, `conf.py`) plus hand-written usage guides in
`guides/`. Built to `_build/` — **never edit anything under `_build/`**, it is
generated and deny-listed.

## The standing rule

A public-API change is not finished until the docs move with it. Every such
change updates, in the same branch:

1. The matching guide in `guides/` —
   `01_datasets` · `02_models` · `03_metrics` · `04_evaluate` ·
   `05_synthesizers` · `06_reporting`. `05_synthesizers.md` is the **generation**
   guide; it keeps the old filename on purpose, because renaming it churns four
   cross-links for no reader benefit. Its title and content are "Generation".
2. Any affected notebook in `examples/` (`test_examples.py` will fail if an
   import there no longer resolves).

Runnable notebooks and markdown usage guides ship *alongside* the API reference,
not after it. The autodoc pages are the reference; the guides are how anyone
actually learns the package.

## Writing guides

- Every code block should be runnable as written against the current API.
- Notebooks use `await a_evaluate(...)`, never `evaluate(...)` — the sync entry
  point raises inside Jupyter's running event loop.
- Show the import path that the package actually exposes: from the owning
  subpackage (`from llminspector.metrics import FaithfulnessMetric`), not the
  root.
- No credentials in examples. Read config from `LLMINSPECTOR_*` env vars.

## Building

```bash
.venv/bin/python -m sphinx -b html docs docs/_build/html
```

The `pages` CI job builds this on the default branch only.
