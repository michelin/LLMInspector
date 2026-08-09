"""Phase 0 exit-criteria tests: the ``llminspector`` package tree imports.

These are intentionally lightweight — they assert the scaffolding is in place
and importable, and that the version attribute resolves (either to a build-time
string via ``version.py`` or gracefully to ``None`` in a source checkout).
"""

import importlib

import pytest

SUBPACKAGES = [
    "llminspector.config",
    "llminspector.test_case",
    "llminspector.dataset",
    "llminspector.models",
    "llminspector.metrics",
    "llminspector.evaluate",
    "llminspector.generation",
    "llminspector.generation.sources",
    "llminspector.data",
    "llminspector.reporting",
    "llminspector.utils",
]


def test_top_level_import():
    """The top-level package imports."""
    import llminspector  # noqa: F401


def test_version_resolves():
    """``__version__`` is defined (a string when built, else ``None``)."""
    import llminspector

    assert hasattr(llminspector, "__version__")
    assert llminspector.__version__ is None or isinstance(llminspector.__version__, str)


@pytest.mark.parametrize("module", SUBPACKAGES)
def test_subpackage_imports(module):
    """Every scaffolded sub-package is importable."""
    assert importlib.import_module(module) is not None
