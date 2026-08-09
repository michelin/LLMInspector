---
name: release-check
description: Pre-release gate for llminspector — build the wheel locally and verify it is actually installable and complete (data files, py.typed, version, imports) before tagging or publishing. Use when preparing a release, tagging, or debugging a wheel that misbehaves once installed.
---

# Release check

The failure mode this exists to catch: **a wheel that passes every test in the
source tree and breaks at runtime once installed**, because a data file or
marker never made it into the archive.

## 1. Build

```bash
.venv/bin/python -m build --wheel --no-isolation
```

`--no-isolation` because the declared build backend (`pydnx_packaging`) is
internal and unreachable outside the network that hosts it; setuptools and wheel
must already be in the venv. On a machine with access, a plain
`python -m build` is the real path.

## 2. Verify the archive contents

```bash
.venv/bin/python -m zipfile -l dist/llminspector-*.whl
```

Must be present:

- [ ] `llminspector/data/*.json` — all six tables. They are loaded at runtime
      from `Path(__file__).parent`; without them every perturbation raises
      `FileNotFoundError`. Guaranteed by `[tool.setuptools.package-data]`.
- [ ] `llminspector/py.typed` — the package ships type information, and
      downstream mypy silently ignores the package without this marker.
- [ ] `llminspector/version.py` — generated at build time.
- [ ] No `tests/`, `docs/`, or `examples/`.
- [ ] No `llminspector/synthesizer/` and no `alignment*` module anywhere. The
      package was renamed to `generation/` and the alignment path deleted; if
      either shows up, the build picked up a stale tree (a leftover
      `build/`/`*.egg-info` from before the rename is the usual cause) and the
      wheel will shadow the real package on install.

## 3. Install clean and smoke-test

In a throwaway venv, not the project one:

```bash
uv venv --python 3.12 /tmp/relcheck && \
uv pip install --python /tmp/relcheck/bin/python dist/llminspector-*.whl
/tmp/relcheck/bin/python -c "
import llminspector as li
print(li.__version__)
from llminspector.data import CONTRACTION_MAP; print(len(CONTRACTION_MAP))
from llminspector.metrics import FaithfulnessMetric
from llminspector.generation import AdversarialGenerator, Generator
"
```

This is the real test: it exercises the lazy JSON loading and the import surface
from an installed package rather than the source tree.

- [ ] Import works **without** the `ragas` extra installed. If it doesn't, an
      optional dependency has leaked to module scope — find it and defer it.
- [ ] Then repeat with `[ragas]` installed and import a context metric.

## 4. Dependency sanity

- [ ] Every runtime import under `llminspector/` is covered by a `dependencies`
      entry in `pyproject.toml`, and nothing there is unused.
- [ ] Ranges, not exact pins — upper bounds at the next known-breaking major so
      security patches land without a release here.
- [ ] `ragas` and `langchain-community` stay in the `ragas` extra, never core.

## 5. Repo state

- [ ] Full suite green: `.venv/bin/python -m pytest -q`.
- [ ] `.venv/bin/pre-commit run --all-files` clean.
- [ ] Docs build: `.venv/bin/python -m sphinx -b html docs docs/_build/html`.
- [ ] Version bumped in `pyproject.toml`.
- [ ] `dist/` cleaned up afterwards — it is gitignored, keep it that way.

## 6. Stop before publishing

Tagging, Artifactory, and the `wheel` / `wheelpush` CI jobs are owned outside
this checklist. Report the results and let the user drive the publish.
