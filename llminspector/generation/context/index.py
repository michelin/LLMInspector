"""Vector search over chunk embeddings.

Two backends behind one protocol. Both L2-normalise on insert and score with an
inner product, so they return **identical cosine scores** — a test asserts the
same top-k for the same vectors, which is what makes the choice a performance
decision rather than a behaviour one.
"""

from __future__ import annotations

import logging
from typing import List, Protocol, Sequence, Tuple

import numpy as np

from ...utils.optional import optional_dependency

logger = logging.getLogger(__name__)

__all__ = ["VectorIndex", "NumpyIndex", "FaissIndex", "build_index"]

#: Above this many chunks, ``index_backend="auto"`` prefers faiss *if it is
#: installed*. Below it, numpy wins once index build time is counted — faiss
#: pays a construction cost that a few thousand vectors never earn back.
FAISS_AUTO_THRESHOLD = 5000


class VectorIndex(Protocol):
    """The minimum a backend must provide."""

    def add(self, ids: List[str], vectors: np.ndarray) -> None: ...

    def search(self, vector: np.ndarray, k: int) -> List[Tuple[str, float]]: ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """Unit-length rows, so an inner product *is* cosine similarity.

    Zero vectors would divide by zero; their norm is clamped to 1, leaving them
    as zero vectors that score 0.0 against everything. That is the right answer
    for an empty chunk and avoids NaNs propagating through a whole index.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class NumpyIndex:
    """Exact search over a normalised matrix. The default, and core-only."""

    def __init__(self) -> None:
        self._ids: List[str] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError(
                f"got {len(ids)} id(s) for {len(vectors)} vector(s); they must "
                "correspond one to one"
            )
        if not ids:
            return
        block = _normalise(vectors)
        self._matrix = (
            block if self._matrix is None else np.vstack([self._matrix, block])
        )
        self._ids.extend(ids)

    def search(self, vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """The ``k`` nearest ids with their cosine scores, best first."""
        if self._matrix is None or not self._ids or k < 1:
            return []
        query = _normalise(vector)[0]
        scores = self._matrix @ query
        k = min(k, len(self._ids))
        # argpartition is O(n) to find the top k, then a small sort orders them.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top]


class FaissIndex:
    """``IndexFlatIP`` over normalised vectors — same scores, faster at scale."""

    def __init__(self, dim: int) -> None:
        with optional_dependency(
            "faiss", extra="faiss", feature="the faiss vector index"
        ):
            import faiss

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._ids: List[str] = []

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError(
                f"got {len(ids)} id(s) for {len(vectors)} vector(s); they must "
                "correspond one to one"
            )
        if not ids:
            return
        self._index.add(_normalise(vectors))
        self._ids.extend(ids)

    def search(self, vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if not self._ids or k < 1:
            return []
        k = min(k, len(self._ids))
        scores, indices = self._index.search(_normalise(vector), k)
        return [
            (self._ids[i], float(s)) for s, i in zip(scores[0], indices[0]) if i >= 0
        ]


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401  pylint: disable=unused-import

        return True
    except ImportError:
        return False


def build_index(backend: str, dim: int, expected_size: int = 0) -> VectorIndex:
    """Construct the requested backend.

    Selection is **explicit**. ``"auto"`` is the only adaptive setting, and it
    logs which backend it picked and why — a vector index that silently changes
    implementation between runs is the sort of thing that turns a reproducibility
    question into an afternoon.
    """
    if backend == "numpy":
        return NumpyIndex()
    if backend == "faiss":
        return FaissIndex(dim)
    if backend != "auto":
        raise ValueError(
            f"Unknown index backend {backend!r}; expected 'numpy', 'faiss' or 'auto'."
        )

    if expected_size > FAISS_AUTO_THRESHOLD and _faiss_available():
        logger.info(
            "index_backend='auto': using faiss (%d chunks > %d threshold)",
            expected_size,
            FAISS_AUTO_THRESHOLD,
        )
        return FaissIndex(dim)
    logger.info(
        "index_backend='auto': using numpy (%d chunks, faiss %s)",
        expected_size,
        "installed" if _faiss_available() else "not installed",
    )
    return NumpyIndex()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two raw vectors."""
    va, vb = _normalise(np.asarray(a)), _normalise(np.asarray(b))
    return float(va[0] @ vb[0])
