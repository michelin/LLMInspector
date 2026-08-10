"""The package root is a namespace, not a mirror of every subpackage.

``llminspector/__init__.py`` used to re-export 44 names — all 24 metric classes
plus the schema, model, and synthesizer types — giving every class two import
paths and making each one a compatibility commitment of the package root.

The root now holds only the evaluate entry points, the version, and the
subpackage namespaces. Each layer owns its own names.
"""

import importlib

import pytest

import llminspector

EXPECTED_ROOT_API = {
    "__version__",
    "__version_date__",
    "evaluate",
    "a_evaluate",
    "EvaluationResult",
    "config",
    "dataset",
    "metrics",
    "models",
    "reporting",
    "generation",
    "test_case",
}

#: Where each formerly-root-level name now lives.
OWNERS = {
    "llminspector.test_case": ["LLMTestCase"],
    "llminspector.dataset": [
        "EvaluationDataset",
        "Golden",
        "ColumnMapping",
        "GoldenColumnMapping",
        # Moved here from ``synthesizer``: flattening goldens to a table is a
        # dataset concern, and ``goldens_to_pandas`` is now its main caller.
        # The synthesizer layer imports it rather than re-exporting it, so the
        # name keeps exactly one import path.
        "goldens_to_dataframe",
    ],
    "llminspector.config": ["AzureSettings", "Settings"],
    "llminspector.models": [
        "BaseLLM",
        "BaseEmbeddingModel",
        "AzureOpenAIModel",
        "AzureOpenAIEmbedding",
        "StructuredOutputError",
    ],
    "llminspector.evaluate": [
        "evaluate",
        "a_evaluate",
        "EvaluationResult",
        "ResultColumns",
    ],
    "llminspector.metrics": [
        "BaseMetric",
        "DualTargetMetric",
        "RagasBackedMetric",
        "BertScoreMetric",
        "FaithfulnessMetric",
        "AnswerCorrectnessMetric",
        "AnswerRelevancyMetric",
        "ConcisenessMetric",
        "ContextPrecisionMetric",
        "ContextRecallMetric",
        "ContextUtilisationMetric",
        "ContextRelevanceMetric",
        "ContextEntityRecallMetric",
        "PIIDetectionMetric",
        "ContentModerationMetric",
        "QuestionJailbreakMetric",
        "AnswerJailbreakMetric",
        "RefusalMetric",
        "HallucinationMetric",
        "CodeDetectMetric",
        "SentimentMetric",
        "EmotionMetric",
        "LanguageDetectionMetric",
        "ReadabilityMetric",
        "TokenCountMetric",
        "PolicyComplianceMetric",
        "calculate_total_tokens",
    ],
    # Phase 3 replaced the synthesizer/engine pair of hierarchies with one
    # pipeline: a GoldenSource produces goldens, an ordered list of Stages
    # transforms them, a Generator runs it. Alignment is gone entirely.
    "llminspector.generation": [
        "Generator",
        "GenerationResult",
        "GoldenSource",
        "SyncGoldenSource",
        "Stage",
        "StageContext",
        "GenerationConfig",
        "FiltrationConfig",
        "EvolutionConfig",
        "StylingConfig",
        "ContextConfig",
        "default_stages",
        "FiltrationStage",
        "EvolutionStage",
        "StylingStage",
        "ExpectedOutputStage",
        "PerturbationStage",
        "ContextSource",
        "DocumentSource",
        "ScratchSource",
        "SeedGoldenSource",
        "AdversarialGenerator",
        "RagGenerator",
        "CuratedBankSource",
        "RagasTestsetBackend",
    ],
    "llminspector.reporting": ["summary", "errors"],
}


def test_root_api_is_exactly_the_namespaces_plus_the_entry_points():
    assert set(llminspector.__all__) == EXPECTED_ROOT_API


def test_root_api_stays_small():
    assert len(llminspector.__all__) == 12


@pytest.mark.parametrize("name", sorted(EXPECTED_ROOT_API))
def test_everything_the_root_advertises_actually_resolves(name):
    assert hasattr(llminspector, name) or name == "__version__"


def test_no_metric_class_is_reachable_from_the_root():
    from llminspector.metrics import BaseMetric

    leaked = [
        name
        for name in dir(llminspector)
        if isinstance(getattr(llminspector, name, None), type)
        and issubclass(getattr(llminspector, name), BaseMetric)
    ]
    assert leaked == [], f"metric classes leaked back to the root: {leaked}"


@pytest.mark.parametrize(
    "module, name",
    [(m, n) for m, names in OWNERS.items() for n in names],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_each_name_resolves_from_its_owning_subpackage(module, name):
    assert hasattr(importlib.import_module(module), name)


@pytest.mark.parametrize(
    "name",
    sorted(
        {
            n
            for m, names in OWNERS.items()
            for n in names
            if m != "llminspector.evaluate"
        }
    ),
)
def test_subpackage_names_are_not_mirrored_at_the_root(name):
    """One import path per name — no name has two homes."""
    assert not hasattr(llminspector, name)


@pytest.mark.parametrize("module", sorted(OWNERS))
def test_every_subpackage_declares_all(module):
    assert getattr(importlib.import_module(module), "__all__", None)


def test_evaluate_resolves_to_the_function_not_the_submodule():
    """Documented wart: the re-exported function shadows the subpackage."""
    assert callable(llminspector.evaluate)
    # the module is still reachable by path
    assert importlib.import_module("llminspector.evaluate.evaluate")
