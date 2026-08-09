"""Turning documents into contexts: loading, chunking, indexing, selection.

Everything here except the PDF/DOCX loaders and the faiss backend runs on a core
install. The two optional pieces each confine their dependency to one function,
behind ``utils.optional.optional_dependency``.
"""

from .chunking import TokenChunker
from .index import FaissIndex, NumpyIndex, VectorIndex, build_index
from .loaders import Document, DocumentLoader, load_documents
from .selection import Context, ContextScore, build_contexts

__all__ = [
    "TokenChunker",
    "VectorIndex",
    "NumpyIndex",
    "FaissIndex",
    "build_index",
    "Document",
    "DocumentLoader",
    "load_documents",
    "Context",
    "ContextScore",
    "build_contexts",
]
