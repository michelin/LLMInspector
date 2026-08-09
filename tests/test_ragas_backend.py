"""Coverage for the ragas testset backend.

``ragas`` and ``langchain_community`` are not installed in the test environment
(deliberately — see tests/test_provider_contract.py), so both are stubbed at
their lazy import seams. What is exercised is this module's own plumbing: the
document-loading branch, the ground-truth refinement pass, the column
renaming/reordering, and the DataFrame -> Golden mapping.

**Transitional, like the code it covers.** ``RagasTestsetBackend`` and
``RagGenerator`` are deleted once the ragas-free document pipeline lands, and
this module goes with them — so it was migrated to the ``generation`` API
(``GoldenSource.produce(config)``, ``Generator``) without growing any new
coverage.
"""

import sys
import types

import pandas as pd
import pytest

from llminspector.dataset.golden import Golden
from llminspector.generation.config import GenerationConfig
from llminspector.generation.rag import RagGenerator
from llminspector.generation.sources.ragas_testset import (
    _DEFAULT_REFINE_PROMPT,
    RagasTestsetBackend,
)


class StubModel:
    """A BaseLLM-shaped provider: generate + the ragas capability."""

    def __init__(self):
        self.prompts = []
        self.run_config = object()

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return f"refined({prompt[-20:]})"

    def ragas_llm(self):
        return "ragas-llm"


class StubEmbedding:
    def ragas_embeddings(self):
        return "ragas-embeddings"


def _backend(**kwargs):
    defaults = dict(model=StubModel(), embedding=StubEmbedding(), test_size=2)
    defaults.update(kwargs)
    return RagasTestsetBackend(**defaults)


def _testset_df():
    return pd.DataFrame(
        {
            "question": ["q1", "q2"],
            "ground_truth": ["draft1", "draft2"],
            "reference_contexts": [["c1"], ["c2a", "c2b"]],
            "synthesizer_name": ["single_hop", "multi_hop"],
        }
    )


# --------------------------------------------------------------------------- #
# refine_answer — goes through BaseLLM.generate, not a langchain client
# --------------------------------------------------------------------------- #


def test_refine_answer_uses_the_provider_contract():
    model = StubModel()
    backend = _backend(model=model)
    out = backend.refine_answer("q", ["ctx"], "draft")
    assert out.startswith("refined(")
    assert len(model.prompts) == 1


def test_refine_answer_renders_all_three_placeholders():
    model = StubModel()
    _backend(model=model).refine_answer("QQ", ["CC"], "AA")
    prompt = model.prompts[0]
    assert "QQ" in prompt and "CC" in prompt and "AA" in prompt


def test_refine_answer_honours_a_custom_prompt():
    model = StubModel()
    backend = _backend(model=model, refine_prompt="only {question}")
    backend.refine_answer("QQ", ["c"], "a")
    assert model.prompts[0] == "only QQ"


def test_the_default_refine_prompt_declares_its_placeholders():
    for name in ("{question}", "{context}", "{answer}"):
        assert name in _DEFAULT_REFINE_PROMPT


def test_backend_never_touches_a_client_attribute():
    """The stub has no `.client`; reaching for one would AttributeError."""
    model = StubModel()
    assert not hasattr(model, "client")
    _backend(model=model).refine_answer("q", ["c"], "a")


# --------------------------------------------------------------------------- #
# enhance_ground_truth
# --------------------------------------------------------------------------- #


def test_enhance_ground_truth_replaces_the_draft_and_fixes_column_order():
    out = _backend().enhance_ground_truth(_testset_df())
    assert list(out.columns) == [
        "question",
        "ground_truth",
        "reference_contexts",
        "synthesizer_name",
    ]
    assert all(gt.startswith("refined(") for gt in out["ground_truth"])


def test_enhance_ground_truth_refines_every_row():
    model = StubModel()
    _backend(model=model).enhance_ground_truth(_testset_df())
    assert len(model.prompts) == 2


# --------------------------------------------------------------------------- #
# document loading
# --------------------------------------------------------------------------- #


@pytest.fixture
def stub_langchain_community(monkeypatch):
    loaded = []

    class _DirectoryLoader:
        def __init__(self, path, **kwargs):
            loaded.append((path, kwargs))

        def load(self):
            return ["doc1", "doc2"]

    community = types.ModuleType("langchain_community")
    loaders = types.ModuleType("langchain_community.document_loaders")
    loaders.DirectoryLoader = _DirectoryLoader
    community.document_loaders = loaders
    monkeypatch.setitem(sys.modules, "langchain_community", community)
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", loaders)
    return loaded


def test_load_documents_reads_the_directory(stub_langchain_community):
    backend = _backend(document_dir="/docs")
    assert backend._load_documents() == ["doc1", "doc2"]
    path, kwargs = stub_langchain_community[0]
    assert path == "/docs"
    assert kwargs["glob"] == ["**/*.pdf", "**/*.docx", "**/*.txt"]


# --------------------------------------------------------------------------- #
# generation (ragas stubbed)
# --------------------------------------------------------------------------- #


@pytest.fixture
def stub_ragas(monkeypatch):
    seen = {}

    class _Dataset:
        def to_pandas(self):
            return pd.DataFrame(
                {
                    "user_input": ["q1", "q2"],
                    "reference": ["draft1", "draft2"],
                    "reference_contexts": [["c1"], ["c2"]],
                    "synthesizer_name": ["single_hop", "multi_hop"],
                }
            )

    class _TestsetGenerator:
        def __init__(self, llm, embedding_model):
            seen["llm"] = llm
            seen["embedding_model"] = embedding_model

        def generate_with_langchain_docs(self, documents, **kwargs):
            seen["documents"] = documents
            seen.update(kwargs)
            return _Dataset()

    ragas = types.ModuleType("ragas")
    testset = types.ModuleType("ragas.testset")
    testset.TestsetGenerator = _TestsetGenerator
    ragas.testset = testset
    monkeypatch.setitem(sys.modules, "ragas", ragas)
    monkeypatch.setitem(sys.modules, "ragas.testset", testset)
    return seen


def test_generate_maps_ragas_output_to_goldens(stub_ragas):
    backend = _backend(documents=["doc"])
    goldens = backend.produce(GenerationConfig())

    assert len(goldens) == 2
    assert all(isinstance(g, Golden) for g in goldens)
    assert goldens[0].input == "q1"
    assert goldens[0].expected_output.startswith("refined(")
    assert goldens[0].context == ["c1"]
    assert set(goldens[0].metadata) == set(RagasTestsetBackend.metadata_keys)
    assert goldens[0].metadata["synthesizer_name"] == "single_hop"


def test_generate_passes_the_ragas_wrappers_and_run_config(stub_ragas):
    model, embedding = StubModel(), StubEmbedding()
    _backend(model=model, embedding=embedding, documents=["doc"], test_size=7).produce(
        GenerationConfig()
    )

    assert stub_ragas["llm"] == "ragas-llm"
    assert stub_ragas["embedding_model"] == "ragas-embeddings"
    assert stub_ragas["testset_size"] == 7
    assert stub_ragas["run_config"] is model.run_config


def test_generate_loads_documents_when_none_are_supplied(
    stub_ragas, stub_langchain_community
):
    _backend(document_dir="/docs").produce(GenerationConfig())
    assert stub_ragas["documents"] == ["doc1", "doc2"]


def test_generator_drives_the_backend(stub_ragas):
    """The preset runs the backend as its source and reports a GenerationResult.

    ``RagSynthesizer.generate()`` returned an ``EvaluationDataset`` and stashed
    it on ``.dataset``; the generator returns the result instead and keeps it
    internally for ``to_pandas``. With no stages, every produced golden survives.
    """
    generator = RagGenerator.from_documents(
        model=StubModel(),
        embedding=StubEmbedding(),
        documents=["doc"],
        test_size=2,
        config=GenerationConfig(show_progress=False),
    )
    result = generator.generate()
    assert len(result.goldens) == 2
    assert result.rejected == [] and result.errors == []
    assert generator.metadata_keys == ("synthesizer_name",)


def test_from_documents_forwards_a_custom_refine_prompt(stub_ragas):
    synth = RagGenerator.from_documents(
        model=StubModel(),
        embedding=StubEmbedding(),
        documents=["doc"],
        refine_prompt="custom {question}",
    )
    assert synth.source.refine_prompt == "custom {question}"


def test_from_documents_defaults_the_refine_prompt():
    synth = RagGenerator.from_documents(
        model=StubModel(), embedding=StubEmbedding(), documents=["doc"]
    )
    assert synth.source.refine_prompt == _DEFAULT_REFINE_PROMPT
