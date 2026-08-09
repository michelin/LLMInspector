"""Phase 5 — synthesizers.

Covers the parts runnable without transformers/ragas: perturbations (incl. the
two bug fixes), the curated-bank adversarial source (pure pandas, real sample),
the alignment perturbation stage (paraphrased df injected to skip T5), the
stable BaseSynthesizer contract via fake engines, and the split construction
paths introduced in Phase 8.6.
"""

import random

import numpy as np
import pandas as pd
import pytest

from llminspector.dataset import EvaluationDataset, Golden, goldens_to_dataframe
from llminspector.synthesizer import (
    AdversarialSynthesizer,
    AlignmentSynthesizer,
    RagSynthesizer,
)
from llminspector.synthesizer import perturbations as p
from llminspector.synthesizer.engines import CuratedBankSource
from llminspector.synthesizer.engines.base import (
    AlignmentEngine,
    AttackSource,
    TestsetBackend,
)
from llminspector.synthesizer.engines.legacy_alignment import (
    LegacyTagT5Engine,
    _dataframe_to_goldens,
)
from llminspector.test_case import LLMTestCase

SAMPLE_ADVERSARIAL = "tests/test_sample/test_adversarialdata.xlsx"


# --------------------------------------------------------------------------- #
# Golden.metadata
# --------------------------------------------------------------------------- #


def test_golden_metadata_defaults_and_roundtrip():
    g = Golden(input="q")
    assert g.metadata == {}
    g2 = Golden(input="q", metadata={"capability": "toxicity"})
    assert g2.metadata["capability"] == "toxicity"


# --------------------------------------------------------------------------- #
# perturbations (with the two bug fixes)
# --------------------------------------------------------------------------- #


def test_add_contraction_actually_transforms():
    # legacy bug: returned the unmodified input
    out = p.add_contraction(["I cannot do this"], prob=1.0)
    assert out == ["I can't do this"]


def test_add_abbreviation_actually_transforms():
    out = p.add_abbreviation(["afghanistan is here"], prob=1.0)
    assert out[0].startswith("AFG")


def test_uppercase_lowercase_titlecase():
    assert p.uppercase_transform(["ab cd"]) == ["AB CD"]
    assert p.lowercase_transform(["AB CD"]) == ["ab cd"]
    assert p.titlecase_transform(["ab cd"]) == ["Ab Cd"]


def test_add_typo_changes_length_or_chars():
    random.seed(1)
    out = p.add_typo(["hello world example text"], error_rate=1.0)
    assert out != ["hello world example text"]


# --------------------------------------------------------------------------- #
# Adversarial (curated bank — pure pandas)
# --------------------------------------------------------------------------- #


def test_curated_bank_capability_filter():
    bank = pd.DataFrame(
        {
            "Capability": ["A", "A", "B"],
            "Sub Capability": ["x", "y", "z"],
            "Prompt": ["p1", "p2", "p3"],
            "Char Len": [10, 20, 30],
        }
    )
    goldens = CuratedBankSource(bank, capability="A").generate()
    assert [g.input for g in goldens] == ["p1", "p2"]
    assert goldens[0].metadata["Capability"] == "A"
    assert goldens[0].metadata["Char Len"] == 10


def test_adversarial_synthesizer_from_sample_file():
    df = pd.read_excel(SAMPLE_ADVERSARIAL)
    capability = df["Capability"].iloc[0]
    synth = AdversarialSynthesizer.from_excel(SAMPLE_ADVERSARIAL, capability=capability)
    dataset = synth.generate()
    assert isinstance(dataset, EvaluationDataset)
    assert len(dataset.goldens) > 0
    # every golden carries the adversarial metadata columns
    assert set(dataset.goldens[0].metadata) == {
        "Capability",
        "Sub Capability",
        "Char Len",
    }
    out_df = synth.to_pandas()
    assert "Capability" in out_df.columns and "input" in out_df.columns


def test_adversarial_construction_paths_are_separate():
    """8.6: the constructor takes a source; from_* build the default one."""
    # the runtime ValueError is gone — a missing source is now a TypeError
    with pytest.raises(TypeError):
        AdversarialSynthesizer()

    synth = AdversarialSynthesizer.from_excel(SAMPLE_ADVERSARIAL, capability="all")
    assert isinstance(synth.source, CuratedBankSource)

    from_df = AdversarialSynthesizer.from_dataframe(
        pd.read_excel(SAMPLE_ADVERSARIAL), capability="all"
    )
    assert isinstance(from_df.source, CuratedBankSource)


# --------------------------------------------------------------------------- #
# Alignment perturbation stage (skip T5 via injected paraphrased df)
# --------------------------------------------------------------------------- #


def test_alignment_transform_stage_applies_perturbation():
    np.random.seed(0)
    engine = LegacyTagT5Engine(
        alignment_df=pd.DataFrame({"UserInput": ["x"], "Expected_Result": ["y"]}),
        tag_keyword_dict={},
        augmentation_dict={},
        augmentations={"uppercase": ("case_change", 1.0)},
        paraphrase_count=1,
    )
    paraphrased = pd.DataFrame(
        {
            "input_prompt": ["hello world"],
            "exploded_prompt": ["hello world"],
            "paraphrased_prompt": ["hello world"],
            "Expected_Result": ["ok"],
            "augmentation_type": ["None"],
        }
    )
    result_df = engine.transform_df(paraphrased)
    assert result_df["perturbated_prompt"].iloc[0] == "HELLO WORLD"
    assert result_df["capability"].iloc[0] == "case_change"

    goldens = _dataframe_to_goldens(result_df)
    assert goldens[0].input == "HELLO WORLD"
    # legacy drops Expected_Result from the perturbation-stage columns
    assert goldens[0].expected_output is None
    assert goldens[0].metadata["subcapability"] == "uppercase"


# --------------------------------------------------------------------------- #
# Stable contract via fake engines
# --------------------------------------------------------------------------- #


class _FakeEngine(AlignmentEngine):
    def generate(self):
        return [Golden(input="p1", metadata={"augmentation_type": "T"})]


def test_alignment_synthesizer_accepts_injected_engine():
    synth = AlignmentSynthesizer(_FakeEngine())
    dataset = synth.generate()
    assert dataset.goldens[0].input == "p1"
    df = synth.to_pandas()
    assert list(df.columns) == [
        "id",
        "input",
        "expected_output",
        "context",
        "augmentation_type",
    ]


def test_goldens_to_dataframe_unions_metadata():
    goldens = [
        Golden(input="a", metadata={"x": 1}),
        Golden(input="b", metadata={"y": 2}),
    ]
    df = goldens_to_dataframe(goldens)
    assert list(df.columns) == ["id", "input", "expected_output", "context", "x", "y"]
    assert df["x"].iloc[0] == 1 and pd.isna(df["x"].iloc[1])


# --------------------------------------------------------------------------- #
# RagSynthesizer wrappers
# --------------------------------------------------------------------------- #


class _FakeBackend(TestsetBackend):
    def generate(self):
        return [
            Golden(
                input="q1",
                expected_output="gt1",
                context=["c1"],
                metadata={"synthesizer_name": "fake"},
            ),
        ]


def test_rag_generate_stores_dataset():
    synth = RagSynthesizer(_FakeBackend())
    dataset = synth.generate()
    assert synth.dataset is dataset  # never-set gap fixed
    assert dataset.goldens[0].expected_output == "gt1"


def test_rag_evaluation_wrappers_are_gone():
    """8.6: they forwarded to evaluate() / result.to_excel() and added nothing."""
    synth = RagSynthesizer(_FakeBackend())
    assert not hasattr(synth, "rag_evaluation")
    assert not hasattr(synth, "export_eval")


def test_rag_construction_paths_are_separate():
    with pytest.raises(TypeError):
        RagSynthesizer()
    assert isinstance(RagSynthesizer(_FakeBackend()).backend, _FakeBackend)


# --------------------------------------------------------------------------- #
# Declared metadata keys (Phase 8.6)
# --------------------------------------------------------------------------- #


def test_every_engine_declares_its_metadata_keys():
    from llminspector.synthesizer.engines import (
        CuratedBankSource,
        LegacyTagT5Engine,
        RagasTestsetBackend,
    )

    for cls in (CuratedBankSource, LegacyTagT5Engine, RagasTestsetBackend):
        assert cls.metadata_keys, f"{cls.__name__} declares no metadata_keys"
        assert isinstance(cls.metadata_keys, tuple)


def test_declared_keys_match_what_the_source_actually_emits():
    """The declaration is only useful if it stays true."""
    synth = AdversarialSynthesizer.from_excel(SAMPLE_ADVERSARIAL, capability="all")
    dataset = synth.generate()
    assert set(dataset.goldens[0].metadata) == set(synth.metadata_keys)


def test_output_columns_are_knowable_before_generating():
    synth = AdversarialSynthesizer.from_excel(SAMPLE_ADVERSARIAL, capability="all")
    expected = ["id", "input", "expected_output", "context", *synth.metadata_keys]
    assert list(synth.to_pandas().columns) == expected


def test_synthesizers_expose_the_engine_keys():
    assert AlignmentSynthesizer(_FakeEngine()).metadata_keys == ()
    assert RagSynthesizer(_FakeBackend()).metadata_keys == ()
