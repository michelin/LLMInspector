"""Reading documents off disk.

Plain text, Markdown and MDX are handled in **core** with ``pathlib.read_text``.
Only PDF and DOCX need the ``[documents]`` extra, and each of those imports its
dependency inside an ``optional_dependency`` block, so a core install gets a
directed "pip install 'llminspector[documents]'" rather than a bare
``ModuleNotFoundError``.

Remote sources (SharePoint, S3, ...) are out of scope. :class:`DocumentLoader`
is the protocol they slot into, so adding one never touches the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Protocol, Sequence

from ...utils.optional import optional_dependency

logger = logging.getLogger(__name__)

__all__ = ["Document", "DocumentLoader", "load_documents", "SUPPORTED_SUFFIXES"]

#: Suffixes readable without the extra.
_PLAIN_SUFFIXES = (".txt", ".md", ".mdx")


class Document:
    """One loaded file: its text and where it came from."""

    __slots__ = ("text", "source")

    def __init__(self, text: str, source: str) -> None:
        self.text = text
        self.source = source

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Document(source={self.source!r}, chars={len(self.text)})"


class DocumentLoader(Protocol):
    """Anything that can turn a path into text.

    The seam for remote sources: implement this and register it in
    :data:`LOADERS` (or pass it explicitly) without touching the pipeline.
    """

    def __call__(self, path: Path) -> str: ...


def _load_plain(path: Path) -> str:
    """UTF-8 with replacement, because a corpus is not always clean.

    Failing a whole run on one mis-encoded byte in one file is worse than
    reading that byte as a replacement character.
    """
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    with optional_dependency("pypdf", extra="documents", feature="PDF loading"):
        from pypdf import PdfReader

    reader = PdfReader(str(path))
    # extract_text() returns None for image-only pages; those become blanks
    # rather than a TypeError halfway through a corpus.
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _load_docx(path: Path) -> str:
    with optional_dependency("docx", extra="documents", feature="DOCX loading"):
        import docx

    document = docx.Document(str(path))
    return "\n\n".join(p.text for p in document.paragraphs)


#: Suffix -> loader. Extend this to support another format.
LOADERS: Dict[str, Callable[[Path], str]] = {
    **{suffix: _load_plain for suffix in _PLAIN_SUFFIXES},
    ".pdf": _load_pdf,
    ".docx": _load_docx,
}

SUPPORTED_SUFFIXES = tuple(sorted(LOADERS))


def _resolve_paths(
    paths: Sequence[str] | str | None,
    directory: str | None,
) -> List[Path]:
    """Expand the caller's input into a concrete, sorted file list.

    Sorted so a run over a directory is reproducible; filesystem order is not.
    """
    resolved: List[Path] = []
    if directory is not None:
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        resolved.extend(
            p
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in LOADERS
        )
        if not resolved:
            raise FileNotFoundError(
                f"No supported documents under {directory!r}. "
                f"Supported suffixes: {', '.join(SUPPORTED_SUFFIXES)}"
            )
    if paths is not None:
        candidates: Iterable[str] = [paths] if isinstance(paths, str) else paths
        for raw in candidates:
            path = Path(raw)
            if not path.is_file():
                raise FileNotFoundError(f"No such file: {raw}")
            if path.suffix.lower() not in LOADERS:
                raise ValueError(
                    f"Unsupported file type {path.suffix!r} for {raw}. "
                    f"Supported suffixes: {', '.join(SUPPORTED_SUFFIXES)}"
                )
            resolved.append(path)
    if not resolved:
        raise ValueError("Supply either `paths` or `directory`.")
    return resolved


def load_documents(
    paths: Sequence[str] | str | None = None,
    directory: str | None = None,
) -> List[Document]:
    """Load every requested file, skipping any that turn out to be empty.

    An empty document is dropped with a warning rather than carried forward:
    it would chunk to nothing, embed to nothing, and show up much later as an
    unexplained shortfall in the context count.
    """
    documents: List[Document] = []
    for path in _resolve_paths(paths, directory):
        text = LOADERS[path.suffix.lower()](path)
        if not text or not text.strip():
            logger.warning("Skipping %s: no extractable text.", path)
            continue
        documents.append(Document(text=text, source=path.name))
    if not documents:
        raise ValueError(
            "None of the supplied documents contained extractable text. "
            "Scanned PDFs need OCR before they can be used here."
        )
    return documents
