"""Token-aware text splitting.

Deliberately **not** ``langchain-text-splitters``. That package is importable in
this repo's dev environment, but it arrives only as a transitive dependency of
the ``ragas`` extra — using it here would make a core install fail at runtime in
a way no test currently catches, because the test environment always has ragas.

``tiktoken`` is already a core dependency (the token-count metric uses it), so
splitting on real tokens costs nothing new.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, List

__all__ = ["TokenChunker"]

#: Matches ``TokenCountMetric``'s default deliberately. tiktoken fetches each
#: BPE table once and caches it on disk; using a second encoding here would mean
#: a second download on any machine that has only ever run the metrics — for no
#: benefit, since chunk sizes are approximate by nature and the exact tokeniser
#: matters far less than the package being consistent with itself.
DEFAULT_ENCODING = "o200k_base"


@lru_cache(maxsize=8)
def _encoding(name: str) -> Any:
    """The tiktoken encoding, loaded once per name.

    Imported inside the function: ``import llminspector`` must stay cheap, and
    tiktoken loads a BPE table on first use.
    """
    import tiktoken

    try:
        return tiktoken.get_encoding(name)
    except (ValueError, KeyError):
        return tiktoken.get_encoding(DEFAULT_ENCODING)


class TokenChunker:
    """Splits text into overlapping token windows.

    Parameters
    ----------
    chunk_size:
        Tokens per chunk.
    overlap:
        Tokens each chunk repeats from the previous one. Must be smaller than
        ``chunk_size`` — equal or larger would mean the window never advances,
        which is an infinite loop rather than a bad result.
    encoding:
        A tiktoken encoding name.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        overlap: int = 0,
        *,
        encoding: str = DEFAULT_ENCODING,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than chunk_size "
                f"({chunk_size}), otherwise the window never advances."
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = encoding

    def count_tokens(self, text: str) -> int:
        """How many tokens ``text`` occupies under this chunker's encoding."""
        return len(_encoding(self.encoding).encode(text))

    def split(self, text: str) -> List[str]:
        """Split ``text`` into chunks, dropping any that are blank.

        Text shorter than one chunk comes back as a single chunk rather than
        nothing, so a small document is still usable.
        """
        if not text or not text.strip():
            return []

        enc = _encoding(self.encoding)
        tokens = enc.encode(text)
        if len(tokens) <= self.chunk_size:
            return [text.strip()]

        step = self.chunk_size - self.overlap
        chunks: List[str] = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.chunk_size]
            if not window:
                break
            chunk = enc.decode(window).strip()
            if chunk:
                chunks.append(chunk)
            # The final window is short; stop rather than emitting successive
            # tails that are each a suffix of the one before.
            if start + self.chunk_size >= len(tokens):
                break
        return chunks
