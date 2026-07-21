"""Phase 5 — synthesizers.

Covers the parts runnable without transformers/ragas: perturbations (incl. the
two bug fixes), the curated-bank adversarial source (pure pandas, real sample),
the alignment perturbation stage (paraphrased df injected to skip T5), the
stable BaseSynthesizer contract via fake engines, and the evaluate()-backed
rag_evaluation wrapper.
"""

import random

import numpy as np
import pandas as pd
import pytest

from llminspector import EvaluationDataset, Golden, LLMTestCase
from llminspector.synthesizer import (
    AdversarialSynthesizer,
    AlignmentSynthesizer,
    RagSynthesizer,
    goldens_to_dataframe,
)
from llminspector.synthesizer import perturbations as p
from llminspector.synthesizer.engines import CuratedBankSource
from llminspector.synthesizer.engines.base import AlignmentEngine, AttackSource, TestsetBackend
from llminspector.synthesizer.engines.legacy_alignment import (
    LegacyTagT5Engine,
    _dataframe_to_goldens,
)

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
    assert set(dataset.goldens[0].metadata) == {"Capability", "Sub Capability", "Char Len"}
    out_df = synth.to_pandas()
    assert "Capability" in out_df.columns and "input" in out_df.columns


def test_adversarial_requires_bank_or_source():
    with pytest.raises(ValueError):
        AdversarialSynthesizer()


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
    synth = AlignmentSynthesizer(engine=_FakeEngine())
    dataset = synth.generate()
    assert dataset.goldens[0].input == "p1"
    df = synth.to_pandas()
    assert list(df.columns) == ["input", "expected_output", "context", "augmentation_type"]


def test_goldens_to_dataframe_unions_metadata():
    goldens = [
        Golden(input="a", metadata={"x": 1}),
        Golden(input="b", metadata={"y": 2}),
    ]
    df = goldens_to_dataframe(goldens)
    assert list(df.columns) == ["input", "expected_output", "context", "x", "y"]
    assert df["x"].iloc[0] == 1 and pd.isna(df["x"].iloc[1])


# --------------------------------------------------------------------------- #
# RagSynthesizer wrappers
# --------------------------------------------------------------------------- #

class _FakeBackend(TestsetBackend):
    def generate(self):
        return [
            Golden(input="q1", expected_output="gt1", context=["c1"],
                   metadata={"synthesizer_name": "fake"}),
        ]


def test_rag_generate_stores_dataset():
    synth = RagSynthesizer(backend=_FakeBackend())
    dataset = synth.generate()
    assert synth.dataset is dataset  # never-set gap fixed
    assert dataset.goldens[0].expected_output == "gt1"


def test_rag_evaluation_delegates_to_evaluate():
    from llminspector.metrics.base_metric import BaseMetric

    class _FakeMetric(BaseMetric):
        metric_name = "answer_sentiment"
        required_inputs = {"actual_output"}

        async def a_measure(self, tc):
            self.score = "Positive"
            return self.score

    synth = RagSynthesizer(backend=_FakeBackend())
    answered = EvaluationDataset(
        test_cases=[LLMTestCase(input="q1", actual_output="a1")]
    )
    result = synth.rag_evaluation(answered, [_FakeMetric()], show_progress=False)
    assert result.rows[0]["answer_sentiment"] == "Positive"


def test_rag_requires_backend_or_models():
    with pytest.raises(ValueError):
        RagSynthesizer()
