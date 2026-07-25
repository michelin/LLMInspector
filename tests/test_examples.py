"""The shipped example notebooks stay valid.

They are the package's front door, so a rename that breaks them is a real
regression. This does not execute the cells that need a live model — it checks
that every notebook parses, that every import in every cell resolves against the
current public API, and that the notebooks use the async entry point (the sync
``evaluate()`` raises inside Jupyter's running loop).
"""

import ast
import importlib
import json
import pathlib
import textwrap

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))


def _code_cells(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = cell["source"]
            yield index, source if isinstance(source, str) else "".join(source)


def _parse(source, where):
    """Parse a cell the way Jupyter does — tolerating top-level ``await``."""
    try:
        return ast.parse(source)
    except SyntaxError:
        return ast.parse("async def _cell():\n" + textwrap.indent(source, "    "))


def _imports(tree):
    """(module, name) for every `from X import Y` of an llminspector module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "llminspector"
        ):
            for alias in node.names:
                yield node.module, alias.name


def test_examples_directory_ships_notebooks_only():
    stray = [p.name for p in EXAMPLES.glob("*.py")]
    assert stray == [], f"examples/ should contain notebooks, found: {stray}"
    assert NOTEBOOKS, "no notebooks found in examples/"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json_with_the_expected_shape(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook has no cells"
    assert {c["cell_type"] for c in nb["cells"]} <= {"markdown", "code"}


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_parses(path):
    for index, source in _code_cells(path):
        _parse(source, f"{path.name}[{index}]")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_llminspector_import_resolves(path):
    """Catches a rename that left a notebook importing something that is gone."""
    missing = []
    for index, source in _code_cells(path):
        for module, name in _imports(_parse(source, f"{path.name}[{index}]")):
            try:
                if not hasattr(importlib.import_module(module), name):
                    missing.append(f"{path.name}[{index}]: {module}.{name}")
            except ImportError as exc:  # a module that no longer exists
                missing.append(f"{path.name}[{index}]: {module} ({exc})")
    assert missing == [], "\n".join(missing)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_use_the_async_entry_point(path):
    """`evaluate(...)` calls asyncio.run and raises inside Jupyter's loop."""
    for index, source in _code_cells(path):
        tree = _parse(source, f"{path.name}[{index}]")
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "evaluate" not in called, (
            f"{path.name}[{index}] calls the sync evaluate(); "
            "notebooks must `await a_evaluate(...)`"
        )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_metric_is_imported_from_the_package_root(path):
    """Imports follow the hierarchy: metrics come from llminspector.metrics."""
    offenders = []
    for index, source in _code_cells(path):
        for module, name in _imports(_parse(source, f"{path.name}[{index}]")):
            if module == "llminspector" and name.endswith("Metric"):
                offenders.append(f"{path.name}[{index}]: {name}")
    assert offenders == [], "\n".join(offenders)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_sample_data_paths_resolve_from_the_examples_directory(path):
    """Notebooks are opened from examples/, so their relative paths start ../."""
    for _, source in _code_cells(path):
        for line in source.splitlines():
            if "tests/test_sample/" in line and not line.strip().startswith("#"):
                assert "../tests/test_sample/" in line, line.strip()
                referenced = line.split('"')[1]
                assert (EXAMPLES / referenced).exists(), referenced
