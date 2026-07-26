"""Directed errors for optional third-party backends.

``ragas`` is an optional extra (see ``pyproject.toml``), so the imports that
need it are deferred to their point of use. Without help, a user who skipped the
extra meets a bare ``ModuleNotFoundError: No module named 'ragas'`` raised from
inside a metric, with nothing pointing at the fix. Wrapping those imports turns
it into an instruction.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

_MESSAGE = (
    "{package} is needed for {feature} but is not installed. It ships as an "
    "optional extra:\n\n    pip install 'llminspector[{extra}]'\n"
)


@contextmanager
def optional_dependency(package: str, *, extra: str, feature: str) -> Iterator[None]:
    """Re-raise an ``ImportError`` from the wrapped block with install steps.

    Parameters
    ----------
    package:
        The distribution the block imports, as the user would install it.
    extra:
        The ``llminspector[...]`` extra that provides it.
    feature:
        What the caller was trying to do, phrased to complete "X is needed
        for ...".

    Usage::

        with optional_dependency("ragas", extra="ragas", feature="context metrics"):
            from ragas import SingleTurnSample
    """
    try:
        yield
    except ImportError as exc:
        raise ImportError(
            _MESSAGE.format(package=package, extra=extra, feature=feature)
        ) from exc
