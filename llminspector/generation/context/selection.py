"""Turning documents into contexts worth generating from.

Four steps, in this order:

1. Chunk every document and embed all of one document's chunks in a single call.
2. Score a sampled pool of candidate chunks with the critic model and keep the
   best. One selection policy, not the sync/async split the reference has.
3. Assemble each context: the seed chunk plus its nearest neighbours **above the
   similarity threshold**.
4. Optionally merge contexts from different files into multi-source contexts.

Validation runs before step 1, because everything after it costs money.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

from ...utils.concurrency import a_map
from ...utils.prompting import render_prompt
from .chunking import TokenChunker
from .index import build_index, cosine
from .loaders import Document

if TYPE_CHECKING:  # pragma: no cover
    from ..config import ContextConfig, GenerationConfig

logger = logging.getLogger(__name__)

__all__ = ["Context", "ContextScore", "build_contexts", "CONTEXT_SCORE_PROMPT"]


class ContextScore(BaseModel):
    """The critic's judgement of one candidate chunk."""

    clarity: float = Field(default=0.0)
    depth: float = Field(default=0.0)
    structure: float = Field(default=0.0)
    relevance: float = Field(default=0.0)

    def mean(self) -> float:
        """The four criteria averaged into ``[0, 1]``, clamped.

        Clamped because models return 7 for a criterion documented as 0-1 often
        enough that trusting it would put an uninterpretable number in front of
        the selection cutoff.
        """
        values = [self.clarity, self.depth, self.structure, self.relevance]
        clamped = [max(0.0, min(1.0, float(v))) for v in values]
        return sum(clamped) / len(clamped)


class Context:
    """A group of chunks that will ground one or more goldens."""

    __slots__ = ("chunks", "chunk_sources", "source_files", "score")

    def __init__(
        self,
        chunks: List[str],
        source_files: List[str],
        score: float = 0.0,
        chunk_sources: Optional[List[str]] = None,
    ) -> None:
        self.chunks = chunks
        #: One source filename per chunk, positionally aligned with ``chunks``.
        #: ``source_files`` is the de-duplicated set of those; keeping both means
        #: the cross-file merge can label each chunk without having to guess
        #: which file it came from, which a de-duplicated list cannot answer.
        self.chunk_sources = (
            list(chunk_sources)
            if chunk_sources is not None
            else [source_files[0] if source_files else ""] * len(chunks)
        )
        self.source_files = source_files
        self.score = score

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Context(chunks={len(self.chunks)}, "
            f"sources={self.source_files}, score={self.score:.2f})"
        )


# "Judge the passage, not the subject" is what stops the critic scoring an
# interesting topic highly regardless of whether the text supports questions.
# Do not reflow.
# pylint: disable=line-too-long
CONTEXT_SCORE_PROMPT = """\
Rate this passage on how well it could support evaluation questions for a question-answering system.

Score each criterion from 0.0 to 1.0:

- **clarity** — is it unambiguous and readable on its own?
- **depth** — does it contain enough substance to ask a non-trivial question about?
- **structure** — is it coherent, rather than a fragment cut mid-thought?
- **relevance** — does it carry real subject-matter content, rather than boilerplate, navigation, headers, or legal notices?

Judge the passage, not the subject. A well-written passage about a dull topic scores well; a fragmentary passage about an interesting one does not.

Passage:
{chunk}

Return JSON of the form:
{{"clarity": 0.0, "depth": 0.0, "structure": 0.0, "relevance": 0.0}}
"""
# pylint: enable=line-too-long

#: Prefix used only when a context actually spans more than one file, so the
#: single-source case is not cluttered with a header that says nothing.
SOURCE_PREFIX = "[SOURCE: {source}]\n{chunk}"


def _validate(
    documents: Sequence[Document],
    context_config: "ContextConfig",
    chunker: TokenChunker,
) -> None:
    """Fail before the first embedding call, naming the actual numbers.

    Every check here is one that would otherwise surface as an empty result or
    a division by zero *after* a corpus had been chunked, embedded and scored —
    which is to say, after the run had already cost real money.
    """
    if not documents:
        raise ValueError("No documents to build contexts from.")

    total_tokens = sum(chunker.count_tokens(d.text) for d in documents)
    estimated_chunks = max(1, total_tokens // chunker.chunk_size)
    if estimated_chunks < context_config.chunks_per_context:
        # Suggest concrete numbers rather than "try a smaller chunk size": the
        # caller has no way to know what smaller means for their corpus, and
        # finding out costs another failed run.
        suggested_size = max(
            64, total_tokens // (context_config.chunks_per_context * 2)
        )
        suggested_overlap = max(0, total_tokens // 32)
        raise ValueError(
            f"The corpus is about {total_tokens} token(s), which splits into "
            f"roughly {estimated_chunks} chunk(s) at chunk_size="
            f"{chunker.chunk_size} — fewer than the "
            f"{context_config.chunks_per_context} chunk(s) each context needs. "
            f"Try chunk_size={suggested_size} with "
            f"chunk_overlap={suggested_overlap}, or lower chunks_per_context."
        )


async def _score_chunks(
    chunks: Sequence[Tuple[str, str]],
    config: "GenerationConfig",
) -> List[float]:
    """One critic call per candidate chunk, bounded by ``max_concurrent``."""

    async def _score(item: Tuple[str, str]) -> float:
        _, text = item
        prompt = render_prompt(
            CONTEXT_SCORE_PROMPT, ["chunk"], {"chunk": text}, caller="build_contexts"
        )
        reply = await config.critic.a_generate_structured(prompt, ContextScore)
        return reply.mean()

    scores, errors = await a_map(
        list(chunks),
        _score,
        limit=config.max_concurrent,
        desc="Scoring chunks" if config.show_progress else None,
    )
    if errors:
        logger.warning(
            "%d chunk(s) could not be scored and are ranked last.", len(errors)
        )
    # A chunk whose scoring call failed sorts to the bottom rather than being
    # dropped: it may still be usable, and silently shrinking the candidate pool
    # would make the resulting context count inexplicable.
    return [0.0 if s is None else s for s in scores]


def _assemble(
    seed_id: str,
    chunk_text: Dict[str, str],
    chunk_source: Dict[str, str],
    index: Any,
    vectors: Dict[str, np.ndarray],
    context_config: "ContextConfig",
) -> Context:
    """Seed chunk plus its nearest neighbours above the similarity threshold."""
    chunks = [chunk_text[seed_id]]
    per_chunk_sources = [chunk_source[seed_id]]
    sources = [chunk_source[seed_id]]

    # +1 because the seed is its own nearest neighbour at similarity 1.0.
    neighbours = index.search(vectors[seed_id], context_config.chunks_per_context + 1)
    for neighbour_id, score in neighbours:
        if neighbour_id == seed_id:
            continue
        if score < context_config.similarity_threshold:
            # Sorted best-first, so the first miss ends the list.
            break
        if len(chunks) >= context_config.chunks_per_context:
            break
        chunks.append(chunk_text[neighbour_id])
        per_chunk_sources.append(chunk_source[neighbour_id])
        if chunk_source[neighbour_id] not in sources:
            sources.append(chunk_source[neighbour_id])

    return Context(chunks=chunks, source_files=sources, chunk_sources=per_chunk_sources)


def _merge_cross_file(
    contexts: List[Context], context_config: "ContextConfig"
) -> List[Context]:
    """Group contexts with disjoint sources into multi-file contexts.

    Each context is consumed by at most one group, so no chunk is duplicated
    across the output — the property that makes a merged corpus still a fair
    sample of the source material.
    """
    merged: List[Context] = []
    used: set = set()

    for i, primary in enumerate(contexts):
        if i in used:
            continue
        group = [primary]
        sources = set(primary.source_files)
        for j in range(i + 1, len(contexts)):
            if j in used:
                continue
            candidate = contexts[j]
            if sources & set(candidate.source_files):
                continue  # same file: merging would not add a second source
            if (
                len(sources | set(candidate.source_files))
                > context_config.max_files_per_context
            ):
                continue
            group.append(candidate)
            sources |= set(candidate.source_files)
            used.add(j)
            if len(sources) >= context_config.max_files_per_context:
                break
        used.add(i)

        if len(group) == 1:
            merged.append(primary)
            continue

        chunks: List[str] = []
        chunk_sources: List[str] = []
        files: List[str] = []
        for member in group:
            # ``chunk_sources`` is positionally aligned with ``chunks``, so each
            # chunk is labelled with the file it actually came from. The prefix
            # is added only here, in the >=2-source case, which is the only place
            # it carries information.
            for chunk, source in zip(member.chunks, member.chunk_sources):
                chunks.append(SOURCE_PREFIX.format(source=source, chunk=chunk))
                chunk_sources.append(source)
            for source in member.source_files:
                if source not in files:
                    files.append(source)
        merged.append(
            Context(chunks=chunks, source_files=files, chunk_sources=chunk_sources)
        )

    return merged


async def build_contexts(
    documents: Sequence[Document],
    config: "GenerationConfig",
    context_config: "ContextConfig",
) -> List[Context]:
    """Chunk, embed, score and assemble ``documents`` into contexts."""
    chunker = TokenChunker(
        chunk_size=context_config.chunk_size, overlap=context_config.chunk_overlap
    )
    _validate(documents, context_config, chunker)

    chunk_text: Dict[str, str] = {}
    chunk_source: Dict[str, str] = {}
    vectors: Dict[str, np.ndarray] = {}
    per_document: List[Tuple[Document, List[str]]] = []

    for document in documents:
        ids = []
        for position, chunk in enumerate(chunker.split(document.text)):
            chunk_id = f"{document.source}#{position}"
            chunk_text[chunk_id] = chunk
            chunk_source[chunk_id] = document.source
            ids.append(chunk_id)
        per_document.append((document, ids))

    if not chunk_text:
        raise ValueError("The corpus produced no chunks; check chunk_size.")

    # One embedding call per document, not per chunk — the whole reason
    # BaseEmbeddingModel grew a batch method.
    for document, ids in per_document:
        if not ids:
            continue
        embedded = await config.embedding.a_embed_texts([chunk_text[i] for i in ids])
        for chunk_id, vector in zip(ids, embedded):
            vectors[chunk_id] = np.asarray(vector, dtype=np.float32)

    dim = len(next(iter(vectors.values())))
    index = build_index(context_config.index_backend, dim, len(vectors))
    ordered_ids = list(vectors)
    index.add(ordered_ids, np.vstack([vectors[i] for i in ordered_ids]))

    # Seeded candidate sampling: the pool a run scores must be reproducible, or
    # the same corpus and seed give different contexts on every invocation.
    rng = np.random.default_rng(config.seed)
    pool_size = min(context_config.candidate_pool, len(ordered_ids))
    sampled = [
        ordered_ids[i]
        for i in rng.choice(len(ordered_ids), size=pool_size, replace=False)
    ]

    scores = await _score_chunks([(cid, chunk_text[cid]) for cid in sampled], config)
    ranked = sorted(zip(sampled, scores), key=lambda pair: -pair[1])
    seeds = ranked[: context_config.max_contexts]

    contexts = [
        _assemble(cid, chunk_text, chunk_source, index, vectors, context_config)
        for cid, _ in seeds
    ]
    for context, (_, score) in zip(contexts, seeds):
        context.score = score

    if context_config.cross_file:
        contexts = _merge_cross_file(contexts, context_config)
    return contexts


def similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Re-exported for callers assembling contexts by hand."""
    return cosine(a, b)
