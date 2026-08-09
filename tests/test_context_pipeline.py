"""Documents to contexts: chunking, indexing, selection, and DocumentSource.

Nothing here touches the network. The corpus under ``tests/test_sample/corpus``
is three small markdown/text files, read in core — PDF and DOCX need the
``[documents]`` extra and are covered by the loader-dispatch tests rather than
by real fixtures, which would add binary blobs to the repo for very little.
"""

import asyncio
import json

import numpy as np
import pytest

from llminspector.generation.config import ContextConfig, GenerationConfig
from llminspector.generation.context.chunking import TokenChunker
from llminspector.generation.context.index import (
    FAISS_AUTO_THRESHOLD,
    NumpyIndex,
    build_index,
    cosine,
)
from llminspector.generation.context.loaders import (
    SUPPORTED_SUFFIXES,
    Document,
    load_documents,
)
from llminspector.generation.context.selection import (
    Context,
    ContextScore,
    build_contexts,
)
from llminspector.generation.sources.documents import DocumentSource
from tests.conftest import FakeEmbedding, ScriptedLLM

CORPUS = "tests/test_sample/corpus"


def _score_reply(value=0.8):
    return json.dumps(
        {
            "clarity": value,
            "depth": value,
            "structure": value,
            "relevance": value,
        }
    )


def _inputs(*texts):
    return json.dumps({"inputs": list(texts)})


def _config(model=None, **kwargs):
    kwargs.setdefault("show_progress", False)
    kwargs.setdefault("embedding", FakeEmbedding(dim=32))
    kwargs.setdefault("seed", 7)
    return GenerationConfig(model=model, **kwargs)


# --------------------------------------------------------------------------- #
# TokenChunker
# --------------------------------------------------------------------------- #


def test_short_text_is_one_chunk():
    """A small document must still be usable, not chunked into nothing."""
    assert TokenChunker(chunk_size=1000).split("a short sentence") == [
        "a short sentence"
    ]


def test_blank_text_produces_no_chunks():
    chunker = TokenChunker()
    assert chunker.split("") == []
    assert chunker.split("   \n  ") == []


def test_long_text_splits_into_multiple_chunks():
    chunker = TokenChunker(chunk_size=16, overlap=0)
    chunks = chunker.split(" ".join(f"word{i}" for i in range(200)))

    assert len(chunks) > 5
    assert all(chunker.count_tokens(c) <= 16 for c in chunks)


def test_overlap_repeats_content_between_chunks():
    chunker = TokenChunker(chunk_size=20, overlap=10)
    text = " ".join(f"word{i}" for i in range(100))

    with_overlap = chunker.split(text)
    without = TokenChunker(chunk_size=20, overlap=0).split(text)

    assert len(with_overlap) > len(without), "overlap should yield more windows"


def test_overlap_must_be_smaller_than_chunk_size():
    """Equal or larger would never advance the window — an infinite loop."""
    with pytest.raises(ValueError, match="never advances"):
        TokenChunker(chunk_size=10, overlap=10)


def test_chunker_rejects_nonsense_sizes():
    with pytest.raises(ValueError, match="chunk_size"):
        TokenChunker(chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        TokenChunker(chunk_size=10, overlap=-1)


# --------------------------------------------------------------------------- #
# vector index
# --------------------------------------------------------------------------- #


def _vectors():
    rng = np.random.default_rng(0)
    return rng.standard_normal((12, 8)).astype(np.float32)


def test_numpy_index_returns_the_seed_first():
    index = NumpyIndex()
    vectors = _vectors()
    index.add([f"c{i}" for i in range(len(vectors))], vectors)

    hits = index.search(vectors[3], k=3)

    assert hits[0][0] == "c3"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert len(hits) == 3


def test_scores_are_cosine_similarities():
    index = NumpyIndex()
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    index.add(["a", "b"], np.vstack([a, b]))

    hits = dict(index.search(a, k=2))

    assert hits["a"] == pytest.approx(1.0, abs=1e-6)
    assert hits["b"] == pytest.approx(0.0, abs=1e-6)


def test_index_rejects_mismatched_ids_and_vectors():
    with pytest.raises(ValueError, match="correspond one to one"):
        NumpyIndex().add(["only-one"], _vectors())


def test_searching_an_empty_index_returns_nothing():
    assert NumpyIndex().search(np.array([1.0, 0.0]), k=3) == []


def test_zero_vectors_do_not_produce_nans():
    """A blank chunk scores 0.0 against everything rather than poisoning the index."""
    index = NumpyIndex()
    index.add(["zero", "real"], np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))

    hits = dict(index.search(np.array([1.0, 0.0]), k=2))

    assert not np.isnan(list(hits.values())).any()
    assert hits["zero"] == pytest.approx(0.0)


faiss_installed = True
try:  # pragma: no cover - depends on the environment
    import faiss  # noqa: F401
except ImportError:  # pragma: no cover
    faiss_installed = False


@pytest.mark.skipif(not faiss_installed, reason="the [faiss] extra is not installed")
def test_both_backends_return_the_same_top_k():
    """The backend choice must be a performance decision, never a behaviour one."""
    from llminspector.generation.context.index import FaissIndex

    vectors = _vectors()
    ids = [f"c{i}" for i in range(len(vectors))]

    numpy_index = NumpyIndex()
    numpy_index.add(ids, vectors)
    faiss_index = FaissIndex(vectors.shape[1])
    faiss_index.add(ids, vectors)

    for probe in (vectors[0], vectors[5]):
        n_hits = numpy_index.search(probe, k=4)
        f_hits = faiss_index.search(probe, k=4)
        assert [i for i, _ in n_hits] == [i for i, _ in f_hits]
        for (_, ns), (_, fs) in zip(n_hits, f_hits):
            assert ns == pytest.approx(fs, abs=1e-5)


def test_build_index_selection_is_explicit():
    assert isinstance(build_index("numpy", 8), NumpyIndex)
    with pytest.raises(ValueError, match="Unknown index backend"):
        build_index("magic", 8)


def test_auto_stays_on_numpy_for_a_small_corpus():
    """Small corpora never pay faiss's build cost, installed or not."""
    assert isinstance(build_index("auto", 8, expected_size=10), NumpyIndex)
    assert FAISS_AUTO_THRESHOLD > 0


def test_cosine_helper_matches_the_index():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #


def test_plain_formats_load_without_the_extra():
    documents = load_documents(directory=CORPUS)

    assert {d.source for d in documents} == {
        "api_limits.md",
        "billing.md",
        "support.txt",
    }
    assert all(d.text.strip() for d in documents)


def test_directory_walk_is_sorted_for_reproducibility():
    """Filesystem order is not stable; a run over a directory must be."""
    first = [d.source for d in load_documents(directory=CORPUS)]
    second = [d.source for d in load_documents(directory=CORPUS)]

    assert first == second == sorted(first)


def test_explicit_paths_load():
    documents = load_documents(paths=[f"{CORPUS}/billing.md"])
    assert [d.source for d in documents] == ["billing.md"]


def test_a_single_path_string_is_accepted():
    assert len(load_documents(paths=f"{CORPUS}/billing.md")) == 1


def test_a_missing_file_is_named():
    with pytest.raises(FileNotFoundError, match="nope.md"):
        load_documents(paths=["nope.md"])


def test_an_unsupported_suffix_lists_what_is_supported(tmp_path):
    bad = tmp_path / "notes.rtf"
    bad.write_text("x")
    with pytest.raises(ValueError, match="Supported suffixes"):
        load_documents(paths=[str(bad)])


def test_an_empty_directory_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="No supported documents"):
        load_documents(directory=str(tmp_path))


def test_neither_paths_nor_directory_is_an_error():
    with pytest.raises(ValueError, match="Supply either"):
        load_documents()


def test_pdf_and_docx_are_the_only_formats_needing_the_extra():
    assert ".pdf" in SUPPORTED_SUFFIXES and ".docx" in SUPPORTED_SUFFIXES
    for plain in (".txt", ".md", ".mdx"):
        assert plain in SUPPORTED_SUFFIXES


def test_a_missing_extra_names_the_install_command(monkeypatch, tmp_path):
    """Not a bare ModuleNotFoundError from inside a loader."""
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "pypdf" or name.startswith("pypdf."):
            raise ImportError("No module named 'pypdf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    fake_pdf = tmp_path / "doc.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 not really")

    with pytest.raises(ImportError, match=r"llminspector\[documents\]"):
        load_documents(paths=[str(fake_pdf)])


# --------------------------------------------------------------------------- #
# context construction
# --------------------------------------------------------------------------- #


def test_contexts_are_built_from_the_corpus():
    model = ScriptedLLM(lambda prompt: _score_reply())
    documents = load_documents(directory=CORPUS)
    config = _config(critic_model=model)

    contexts = asyncio.run(
        build_contexts(
            documents,
            config,
            ContextConfig(chunk_size=64, max_contexts=3, chunks_per_context=2),
        )
    )

    assert 1 <= len(contexts) <= 3
    for context in contexts:
        assert context.chunks
        assert context.source_files
        assert len(context.chunk_sources) == len(context.chunks)


def test_validation_runs_before_any_embedding_call():
    """A misconfigured run must fail in milliseconds, not after a paid corpus pass."""

    class ExplodingEmbedding(FakeEmbedding):
        async def a_embed_texts(self, texts):
            raise AssertionError("validation should have failed before embedding")

    documents = [Document(text="tiny", source="t.md")]
    config = _config(embedding=ExplodingEmbedding())

    with pytest.raises(ValueError, match="chunk_size"):
        asyncio.run(
            build_contexts(
                documents,
                config,
                ContextConfig(chunk_size=4096, chunks_per_context=5),
            )
        )


def test_the_validation_error_names_actual_numbers_and_suggests_values():
    documents = [Document(text="a few words only", source="t.md")]

    with pytest.raises(ValueError) as excinfo:
        asyncio.run(
            build_contexts(
                documents,
                _config(),
                ContextConfig(chunk_size=2048, chunks_per_context=4),
            )
        )

    message = str(excinfo.value)
    assert "2048" in message and "chunk_overlap" in message


def test_one_embedding_call_per_document_not_per_chunk():
    embedding = FakeEmbedding(dim=32)
    model = ScriptedLLM(lambda prompt: _score_reply())
    documents = load_documents(directory=CORPUS)

    asyncio.run(
        build_contexts(
            documents,
            _config(critic_model=model, embedding=embedding),
            ContextConfig(chunk_size=32, max_contexts=2, chunks_per_context=2),
        )
    )

    assert embedding.embed_calls == len(documents)


def test_the_similarity_threshold_excludes_distant_neighbours():
    """Defaulting this to 0.0 accepts every neighbour and defeats the check."""
    model = ScriptedLLM(lambda prompt: _score_reply())
    documents = load_documents(directory=CORPUS)

    strict = asyncio.run(
        build_contexts(
            documents,
            _config(critic_model=model),
            ContextConfig(
                chunk_size=48,
                max_contexts=3,
                chunks_per_context=4,
                similarity_threshold=0.99,
            ),
        )
    )

    # FakeEmbedding gives unrelated texts near-orthogonal vectors, so at a
    # threshold of 0.99 nothing but the seed itself can qualify.
    assert all(len(c.chunks) == 1 for c in strict)


def test_context_score_is_clamped_and_averaged():
    assert (
        ContextScore(clarity=1.0, depth=1.0, structure=1.0, relevance=1.0).mean() == 1.0
    )
    assert (
        ContextScore(clarity=7.0, depth=1.0, structure=1.0, relevance=1.0).mean() == 1.0
    )
    assert (
        ContextScore(clarity=-3.0, depth=0.0, structure=0.0, relevance=0.0).mean()
        == 0.0
    )


def test_a_failed_scoring_call_ranks_last_rather_than_shrinking_the_pool():
    calls = {"n": 0}

    def dispatch(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json"
        return _score_reply()

    model = ScriptedLLM(dispatch)
    documents = load_documents(directory=CORPUS)

    contexts = asyncio.run(
        build_contexts(
            documents,
            _config(critic_model=model),
            ContextConfig(chunk_size=64, max_contexts=2, chunks_per_context=2),
        )
    )

    assert contexts, "a single scoring failure must not empty the run"


def test_the_same_seed_samples_the_same_candidates():
    def run(seed):
        model = ScriptedLLM(lambda prompt: _score_reply())
        contexts = asyncio.run(
            build_contexts(
                load_documents(directory=CORPUS),
                _config(critic_model=model, seed=seed),
                ContextConfig(chunk_size=48, max_contexts=3, chunks_per_context=2),
            )
        )
        return [c.chunks[0] for c in contexts]

    assert run(5) == run(5)


def test_cross_file_merge_never_duplicates_or_merges_one_file():
    model = ScriptedLLM(lambda prompt: _score_reply())

    contexts = asyncio.run(
        build_contexts(
            load_documents(directory=CORPUS),
            _config(critic_model=model),
            ContextConfig(
                chunk_size=64,
                max_contexts=4,
                chunks_per_context=1,
                cross_file=True,
                max_files_per_context=2,
            ),
        )
    )

    seen = []
    for context in contexts:
        assert len(set(context.source_files)) == len(context.source_files)
        for chunk in context.chunks:
            seen.append(chunk)
        if len(context.source_files) >= 2:
            # The [SOURCE: x] prefix appears only when a context really spans
            # more than one file.
            assert all(c.startswith("[SOURCE: ") for c in context.chunks)
    assert len(seen) == len(set(seen)), "a chunk was used in two contexts"


def test_a_single_source_context_carries_no_source_prefix():
    context = Context(chunks=["plain text"], source_files=["a.md"])
    assert not context.chunks[0].startswith("[SOURCE:")


# --------------------------------------------------------------------------- #
# DocumentSource end to end
# --------------------------------------------------------------------------- #


def test_document_source_produces_grounded_goldens():
    def dispatch(prompt):
        if "Rate this passage" in prompt:
            return _score_reply()
        return _inputs("What is the rate limit?", "How is overage billed?")

    model = ScriptedLLM(dispatch)
    source = DocumentSource(
        directory=CORPUS,
        context_config=ContextConfig(
            chunk_size=64, max_contexts=2, chunks_per_context=2
        ),
        max_goldens_per_context=2,
    )

    goldens = asyncio.run(source.a_produce(_config(model, critic_model=model)))

    assert goldens
    for golden in goldens:
        assert golden.context
        assert golden.metadata["source_file"]
        assert golden.metadata["context_source_files"]
        assert set(golden.metadata) <= set(source.metadata_keys)


def test_document_source_requires_an_embedding_provider():
    source = DocumentSource(directory=CORPUS)
    config = GenerationConfig(model=ScriptedLLM([]), show_progress=False)

    with pytest.raises(ValueError, match="embedding"):
        asyncio.run(source.a_produce(config))


def test_document_source_rejects_a_nonsense_golden_count():
    with pytest.raises(ValueError, match="max_goldens_per_context"):
        DocumentSource(directory=CORPUS, max_goldens_per_context=0)


def test_document_source_keeps_its_contexts_for_inspection():
    """So a caller can look at what grounded a run without paying for it twice."""

    def dispatch(prompt):
        if "Rate this passage" in prompt:
            return _score_reply()
        return _inputs("q")

    model = ScriptedLLM(dispatch)
    source = DocumentSource(
        directory=CORPUS,
        context_config=ContextConfig(
            chunk_size=64, max_contexts=2, chunks_per_context=1
        ),
    )

    asyncio.run(source.a_produce(_config(model, critic_model=model)))

    assert source.contexts
    assert all(hasattr(c, "chunks") for c in source.contexts)
