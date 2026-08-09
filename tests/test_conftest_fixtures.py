"""Tests for the shared stubs in ``tests/conftest.py``.

The stubs are test *infrastructure*: later phases assert things like "the
pipeline makes exactly four calls per golden" and "these two chunks are
distinct", and those assertions are only as trustworthy as the stub underneath
them. A ScriptedLLM that quietly replayed its last response, or a FakeEmbedding
whose vectors shifted between runs, would produce green tests that prove
nothing. So the stubs get tests of their own.

Async paths are driven with ``asyncio.run`` inside ordinary sync test functions:
``pytest-asyncio`` is not among this project's dev dependencies, and this is
what the rest of the suite already does (see ``tests/test_models.py``).
"""

from __future__ import annotations

import asyncio
import hashlib

import numpy as np
import pytest

from llminspector.models.base_model import BaseEmbeddingModel, BaseLLM
from tests.conftest import FakeEmbedding, MeteredScriptedLLM, ScriptedLLM

# --------------------------------------------------------------------------- #
# ScriptedLLM
# --------------------------------------------------------------------------- #


def test_scripted_llm_returns_responses_in_order():
    model = ScriptedLLM(["one", "two", "three"])

    assert model.generate("p1") == "one"
    assert model.generate("p2") == "two"
    assert model.generate("p3") == "three"

    assert model.prompts == ["p1", "p2", "p3"]
    assert model.calls == 3


def test_scripted_llm_async_shares_the_same_queue():
    """Sync and async must consume one script, not one each.

    A pipeline half-migrated to ``a_generate`` would otherwise see two
    independent scripts and its call-count assertions would silently double.
    """
    model = ScriptedLLM(["one", "two", "three"])

    assert model.generate("p1") == "one"
    assert asyncio.run(model.a_generate("p2")) == "two"
    assert asyncio.run(model.a_generate("p3")) == "three"

    assert model.prompts == ["p1", "p2", "p3"]


def test_scripted_llm_exhaustion_names_the_call_index():
    model = ScriptedLLM(["only"])
    model.generate("first")

    with pytest.raises(AssertionError, match=r"call index 1"):
        model.generate("the unexpected fifth call about widgets")


def test_scripted_llm_exhaustion_quotes_the_unanswered_prompt():
    """The message must identify *which* prompt overran the script."""
    model = ScriptedLLM([])

    with pytest.raises(AssertionError, match=r"summarise the document"):
        model.generate("summarise the document")


def test_scripted_llm_exhaustion_also_fires_on_the_async_path():
    model = ScriptedLLM([])

    with pytest.raises(AssertionError, match=r"call index 0"):
        asyncio.run(model.a_generate("nothing scripted"))


def test_scripted_llm_callable_dispatch_ignores_order():
    """Dispatch mode keys off the prompt, so call order is irrelevant."""
    answers = {"question?": "A", "context?": "B"}
    model = ScriptedLLM(lambda prompt: answers[prompt])

    assert model.generate("context?") == "B"
    assert model.generate("question?") == "A"
    assert model.generate("context?") == "B"

    # Dispatch never exhausts — three calls against two known prompts.
    assert model.calls == 3


def test_scripted_llm_is_a_bare_provider():
    """Mirrors ``test_metrics_never_touch_a_client_attribute``.

    Only the three required contract methods; nothing driven by this stub can
    be quietly depending on a langchain ``.client`` or a ragas wrapper.
    """
    model = ScriptedLLM(["x"])

    assert isinstance(model, BaseLLM)
    assert not hasattr(model, "client")
    assert model.get_model_name() == "scripted"
    with pytest.raises(NotImplementedError):
        model.ragas_llm()


def test_scripted_llm_reset_restores_the_queue():
    model = ScriptedLLM(["one", "two"])
    model.generate("a")
    model.generate("b")

    model.reset()

    assert model.calls == 0
    assert model.prompts == []
    assert model.generate("a") == "one"
    assert model.generate("b") == "two"


def test_scripted_llm_name_is_configurable():
    assert ScriptedLLM(["x"], name="judge").get_model_name() == "judge"


def test_scripted_llm_fixture_factory_builds_independent_models(scripted_llm):
    first = scripted_llm(["a"])
    second = scripted_llm(["b"])

    assert first.generate("p") == "a"
    assert second.generate("p") == "b"
    assert first.calls == 1


# --------------------------------------------------------------------------- #
# MeteredScriptedLLM
# --------------------------------------------------------------------------- #


def test_metered_scripted_llm_tallies_per_kind():
    model = MeteredScriptedLLM(["1", "2", "3", "4"])

    model.generate("Extract entities\n\ndoc A")
    model.generate("Extract entities\n\ndoc B")
    model.generate("Write a question\n\ndoc A")
    asyncio.run(model.a_generate("Write a question\n\ndoc B"))

    assert dict(model.counts) == {"extract entities": 2, "write a question": 2}


def test_metered_scripted_llm_accepts_a_custom_classifier():
    model = MeteredScriptedLLM(
        ["a", "b", "c"],
        classifier=lambda prompt: prompt.split(":", 1)[0],
    )

    model.generate("qa: one")
    model.generate("qa: two")
    model.generate("summary: three")

    assert dict(model.counts) == {"qa": 2, "summary": 1}


def test_metered_scripted_llm_counts_the_call_that_overran():
    """The overrunning call is counted before the failure, so the tally shows it."""
    model = MeteredScriptedLLM(["only"])
    model.generate("Extract entities\nx")

    with pytest.raises(AssertionError):
        model.generate("Extract entities\ny")

    assert model.counts["extract entities"] == 2


def test_metered_scripted_llm_is_still_a_scripted_llm():
    model = MeteredScriptedLLM(["one"])

    assert isinstance(model, ScriptedLLM)
    assert model.generate("Kind\nbody") == "one"
    assert model.prompts == ["Kind\nbody"]


# --------------------------------------------------------------------------- #
# FakeEmbedding
# --------------------------------------------------------------------------- #


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def test_fake_embedding_is_a_bare_provider():
    embedding = FakeEmbedding()

    assert isinstance(embedding, BaseEmbeddingModel)
    assert embedding.get_model_name() == "fake-embedding"
    with pytest.raises(NotImplementedError):
        embedding.ragas_embeddings()


def test_fake_embedding_same_text_gives_the_identical_vector():
    embedding = FakeEmbedding()

    assert embedding.embed_text("rate limits") == embedding.embed_text("rate limits")


def test_fake_embedding_distinguishes_different_texts():
    """The property no previous stub had: unrelated texts are not identical.

    Constant-vector stubs score 1.0 for everything, so a chunker returning a
    duplicate chunk would have looked correct.
    """
    embedding = FakeEmbedding()

    similarity = _cosine(
        embedding.embed_text("rate limits are 500 rpm"),
        embedding.embed_text("the cat sat on the mat"),
    )

    assert similarity < 0.9


def test_fake_embedding_vectors_are_unit_norm():
    embedding = FakeEmbedding(dim=32)

    for text in ("alpha", "beta", "gamma"):
        vector = embedding.embed_text(text)
        assert len(vector) == 32
        assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_fake_embedding_is_stable_across_instances():
    """blake2b, not ``hash()``.

    Python randomises string hashing per process, so a builtin-``hash`` seed
    would give different vectors on every run and any similarity threshold
    asserted against them would be flaky. Two independent instances agreeing is
    the in-process half of that guarantee; the hard-coded expectation below
    pins the cross-process half.
    """
    assert FakeEmbedding().embed_text("stable") == FakeEmbedding().embed_text("stable")


def test_fake_embedding_seed_is_process_independent():
    first = FakeEmbedding(dim=4).embed_text("determinism")

    # Regenerating the same vector from the documented recipe — blake2b digest
    # as the RNG seed — rather than from another FakeEmbedding, so this fails if
    # the seeding strategy is ever swapped back to something process-local.
    seed = int.from_bytes(
        hashlib.blake2b(b"determinism", digest_size=8).digest(), "big"
    )
    rng = np.random.default_rng(seed)
    expected = rng.standard_normal(4)
    expected /= np.linalg.norm(expected)

    assert first == pytest.approx(list(expected))


def test_fake_embedding_batches_count_as_one_call():
    """``embed_calls`` counts calls, ``texts`` counts texts.

    Lets a test assert "one embed_texts call per document, not one per chunk".
    """
    embedding = FakeEmbedding()

    vectors = embedding.embed_texts(["a", "b", "c"])

    assert len(vectors) == 3
    assert embedding.embed_calls == 1
    assert embedding.texts == ["a", "b", "c"]


def test_fake_embedding_batch_matches_single():
    embedding = FakeEmbedding()

    batched = embedding.embed_texts(["a", "b"])

    assert batched[0] == embedding.embed_text("a")
    assert batched[1] == embedding.embed_text("b")
    assert embedding.embed_calls == 3


def test_fake_embedding_async_wrappers_delegate():
    """Async lives on the fake for now; the ABC gains it in a later phase.

    Deliberately *not* asserting that ``BaseEmbeddingModel`` lacks the async
    methods: adding them is planned, so such an assertion would fail by design
    and read as "the wrappers broke".
    """
    embedding = FakeEmbedding()

    single = asyncio.run(embedding.a_embed_text("a"))
    batch = asyncio.run(embedding.a_embed_texts(["a", "b"]))

    assert single == batch[0]
    assert embedding.embed_calls == 2


def test_fake_embedding_fixture_factory_honours_dim(fake_embedding):
    assert len(fake_embedding(dim=8).embed_text("x")) == 8
    assert len(fake_embedding().embed_text("x")) == 16
