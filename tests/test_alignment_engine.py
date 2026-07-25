"""Phase 7D — coverage for the alignment tag helper and the legacy T5 engine.

The engine's stage 2 downloads ``humarin/chatgpt_paraphraser_on_T5_base``, so
``transformers`` is stubbed at the import seam: the tests exercise the real
plumbing (tokenizer call shape, per-row fan-out, DataFrame assembly) without a
model download.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from llminspector.dataset.golden import Golden
from llminspector.synthesizer.alignment import AlignmentSynthesizer
from llminspector.synthesizer.alignment_tag import (
    KeywordNotFoundException,
    tag_replace,
)
from llminspector.synthesizer.engines.legacy_alignment import (
    LegacyTagT5Engine,
    _dataframe_to_goldens,
)

# --------------------------------------------------------------------------- #
# alignment_tag.py
# --------------------------------------------------------------------------- #


def test_keyword_not_found_exception_carries_the_tag():
    exc = KeywordNotFoundException("{greeting}")
    assert exc.tag_value == "{greeting}"
    assert "{greeting}" in str(exc)
    assert "Keyword not found" in str(exc)


def test_keyword_not_found_exception_custom_message():
    exc = KeywordNotFoundException("{x}", message="nope")
    assert str(exc) == "nope: {x}"


def test_convert_keys_to_lower():
    assert tag_replace().convert_keys_to_lower({"A": 1, "Bc": 2}) == {"a": 1, "bc": 2}


def test_replace_tag_expands_each_replacement():
    out = tag_replace().replace_tag(
        ["{greeting}, how are you?"],
        "{greeting}",
        {"{greeting}": ["hi", "hello"]},
    )
    # first element is the original sentence, then one per replacement
    assert out == [
        [
            "{greeting}, how are you?",
            "hi, how are you?",
            "hello, how are you?",
        ]
    ]


def test_replace_tag_is_case_insensitive_on_the_tag_and_the_dict_key():
    out = tag_replace().replace_tag(
        ["{GREETING} there"], "{greeting}", {"{Greeting}": ["hey"]}
    )
    assert out == [["{GREETING} there", "hey there"]]


def test_replace_tag_returns_none_when_the_tag_is_absent():
    assert (
        tag_replace().replace_tag(
            ["no tags here"], "{greeting}", {"{greeting}": ["hi"]}
        )
        is None
    )


def test_replace_tag_returns_none_when_the_keyword_is_not_configured():
    assert (
        tag_replace().replace_tag(
            ["{greeting} there"], "{greeting}", {"{other}": ["hi"]}
        )
        is None
    )


def test_replace_tag_skips_sentences_without_the_tag():
    out = tag_replace().replace_tag(
        ["{greeting} there", "unrelated"], "{greeting}", {"{greeting}": ["hi"]}
    )
    assert len(out) == 1


def test_replace_tag_handles_an_empty_replacement_list():
    out = tag_replace().replace_tag(
        ["{greeting} there"], "{greeting}", {"{greeting}": []}
    )
    assert out == [["{greeting} there"]]


# --------------------------------------------------------------------------- #
# stage 1: tag augmentation
# --------------------------------------------------------------------------- #


def _engine(**kwargs):
    defaults = dict(
        alignment_df=pd.DataFrame(
            {
                "UserInput": ["{greeting}, how are you?", "plain question"],
                "Expected_Result": ["Hey!", "An answer."],
            }
        ),
        tag_keyword_dict={"{greeting}": ["hi", "hello"]},
        augmentation_dict={"greetings": ["greeting"]},
        augmentations={},
        paraphrase_count=2,
    )
    defaults.update(kwargs)
    return LegacyTagT5Engine(**defaults)


def test_alignment_data_expands_tags_and_keeps_untagged_rows():
    df = _engine().alignment_data()
    assert list(df.columns) == [
        "input_prompt",
        "exploded_prompt",
        "Expected_Result",
        "augmentation_type",
    ]
    # the untagged row survives with augmentation_type "None"
    assert "plain question" in df["exploded_prompt"].tolist()
    assert "None" in df["augmentation_type"].tolist()
    # the tagged row was expanded into its replacements
    exploded = df["exploded_prompt"].tolist()
    assert "hi, how are you?" in exploded
    assert "hello, how are you?" in exploded
    # rows still containing an unexpanded tag are dropped
    assert not any("{" in p for p in exploded)


def test_alignment_data_labels_expansions_with_the_augmentation_key():
    df = _engine().alignment_data()
    labels = set(df["augmentation_type"])
    assert "Different greetings" in labels


def test_alignment_data_without_a_matching_augmentation_key():
    df = _engine(augmentation_dict={"other": ["nothing"]}).alignment_data()
    # no expansion happened; only the untagged row survives
    assert df["exploded_prompt"].tolist() == ["plain question"]


def test_alignment_data_with_no_tags_at_all():
    df = _engine(
        alignment_df=pd.DataFrame(
            {"UserInput": ["a", "b"], "Expected_Result": ["x", "y"]}
        )
    ).alignment_data()
    assert df["exploded_prompt"].tolist() == ["a", "b"]


def test_missing_keyword_is_logged_not_raised(caplog, monkeypatch):
    """replace_tag returning None raises KeywordNotFoundException internally,
    which the stage catches and logs rather than aborting the whole run."""
    import logging

    from llminspector.synthesizer.engines import legacy_alignment

    monkeypatch.setattr(
        legacy_alignment.tag_replace,
        "replace_tag",
        lambda self, sentences, keyword, keyword_dict: None,
    )
    engine = _engine()
    with caplog.at_level(logging.WARNING):
        engine.checklist_tagaugmentation()
    assert "Tag replacement failed" in caplog.text
    assert "{greeting}" in caplog.text
    assert engine.exploded_prompt == []  # nothing expanded


# --------------------------------------------------------------------------- #
# stage 2: paraphrase (transformers stubbed)
# --------------------------------------------------------------------------- #


class _FakeInputIds:
    input_ids = [[1, 2, 3]]


class _FakeTokenizer:
    calls: list = []

    def __call__(self, text, **kwargs):
        _FakeTokenizer.calls.append((text, kwargs))
        return _FakeInputIds()

    def batch_decode(self, outputs, skip_special_tokens=True):
        return list(outputs)

    @classmethod
    def from_pretrained(cls, name):
        return cls()


class _FakeModel:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def generate(self, input_ids, num_return_sequences=1, **kwargs):
        return [f"paraphrase {i}" for i in range(num_return_sequences)]


@pytest.fixture
def stub_transformers(monkeypatch):
    _FakeTokenizer.calls = []
    module = types.ModuleType("transformers")
    module.AutoTokenizer = _FakeTokenizer
    module.AutoModelForSeq2SeqLM = _FakeModel
    monkeypatch.setitem(sys.modules, "transformers", module)
    return module


def test_paraphrase_prompts_fans_each_row_out(stub_transformers):
    engine = _engine(paraphrase_count=3)
    input_df = pd.DataFrame(
        {
            "input_prompt": ["q1", "q2"],
            "exploded_prompt": ["q1", "q2"],
            "Expected_Result": ["a1", "a2"],
            "augmentation_type": ["None", "None"],
        }
    )
    out = engine.paraphrase_prompts(input_df)
    assert list(out.columns) == [
        "input_prompt",
        "exploded_prompt",
        "paraphrased_prompt",
        "Expected_Result",
        "augmentation_type",
    ]
    assert len(out) == 2 * 3  # one row per (input, paraphrase)
    assert set(out["paraphrased_prompt"]) == {
        "paraphrase 0",
        "paraphrase 1",
        "paraphrase 2",
    }


def test_paraphrase_prompts_prefixes_the_task_token(stub_transformers):
    engine = _engine(paraphrase_count=1)
    engine.paraphrase_prompts(
        pd.DataFrame(
            {
                "input_prompt": ["hello"],
                "exploded_prompt": ["hello"],
                "Expected_Result": ["a"],
                "augmentation_type": ["None"],
            }
        )
    )
    text, kwargs = _FakeTokenizer.calls[0]
    assert text == "paraphrase: hello"
    assert kwargs["truncation"] is True


def test_paraphrase_prompts_on_an_empty_frame(stub_transformers):
    out = _engine().paraphrase_prompts(
        pd.DataFrame(
            columns=[
                "input_prompt",
                "exploded_prompt",
                "Expected_Result",
                "augmentation_type",
            ]
        )
    )
    assert out.empty


# --------------------------------------------------------------------------- #
# stage 3: perturbation
# --------------------------------------------------------------------------- #


def _paraphrased_df(n=4):
    return pd.DataFrame(
        {
            "input_prompt": [f"q{i}" for i in range(n)],
            "exploded_prompt": [f"q{i}" for i in range(n)],
            "paraphrased_prompt": [f"para {i}" for i in range(n)],
            "Expected_Result": [f"a{i}" for i in range(n)],
            "augmentation_type": ["None"] * n,
        }
    )


def test_transform_df_applies_the_named_perturbation():
    np.random.seed(0)
    engine = _engine(augmentations={"uppercase": ("case_change", 1.0)})
    out = engine.transform_df(_paraphrased_df())
    assert list(out.columns) == [
        "capability",
        "subcapability",
        "input_prompt",
        "exploded_prompt",
        "paraphrased_prompt",
        "perturbated_prompt",
        "augmentation_type",
    ]
    assert set(out["subcapability"]) == {"uppercase"}
    assert set(out["capability"]) == {"case_change"}
    assert all(p == p.upper() for p in out["perturbated_prompt"])


def test_transform_df_accumulates_multiple_augmentations():
    np.random.seed(0)
    engine = _engine(
        augmentations={
            "uppercase": ("case_change", 1.0),
            "lowercase": ("case_change", 1.0),
        }
    )
    out = engine.transform_df(_paraphrased_df())
    assert set(out["subcapability"]) == {"uppercase", "lowercase"}
    assert len(out) == 8


def test_transform_df_warns_on_an_unknown_augmentation(caplog):
    import logging

    np.random.seed(0)
    engine = _engine(augmentations={"not_a_transform": ("x", 1.0)})
    with caplog.at_level(logging.WARNING):
        out = engine.transform_df(_paraphrased_df())
    assert out.empty
    assert "not_a_transform" in caplog.text


def test_transform_df_zero_probability_selects_no_rows(caplog):
    np.random.seed(0)
    engine = _engine(augmentations={"uppercase": ("case_change", 0.0)})
    assert engine.transform_df(_paraphrased_df()).empty


def test_transform_df_adds_the_capability_columns_when_absent():
    np.random.seed(0)
    df = _paraphrased_df()
    assert "capability" not in df.columns
    out = _engine(augmentations={"uppercase": ("c", 1.0)}).transform_df(df)
    assert "capability" in out.columns


def test_transform_df_falls_back_to_paraphrasing_when_no_frame_is_given(
    stub_transformers,
):
    np.random.seed(0)
    engine = _engine(
        alignment_df=pd.DataFrame(
            {"UserInput": ["plain question"], "Expected_Result": ["a"]}
        ),
        augmentations={"uppercase": ("case_change", 1.0)},
        paraphrase_count=1,
    )
    out = engine.transform_df()
    assert not out.empty
    assert out["perturbated_prompt"].iloc[0] == "PARAPHRASE 0"


# --------------------------------------------------------------------------- #
# output mapping
# --------------------------------------------------------------------------- #


def test_dataframe_to_goldens_maps_stage_columns_into_metadata():
    df = pd.DataFrame(
        {
            "perturbated_prompt": ["PARA 0"],
            "Expected_Result": ["a0"],
            "capability": ["case_change"],
            "subcapability": ["uppercase"],
            "input_prompt": ["q0"],
            "exploded_prompt": ["q0"],
            "paraphrased_prompt": ["para 0"],
            "augmentation_type": ["None"],
        }
    )
    goldens = _dataframe_to_goldens(df)
    assert len(goldens) == 1
    g = goldens[0]
    assert isinstance(g, Golden)
    assert g.input == "PARA 0"
    assert g.expected_output == "a0"
    assert set(g.metadata) == set(LegacyTagT5Engine.metadata_keys)
    assert g.metadata["subcapability"] == "uppercase"


def test_dataframe_to_goldens_skips_blank_prompts():
    df = pd.DataFrame(
        {
            "perturbated_prompt": ["ok", "", None, "   "],
            "Expected_Result": ["a", "b", "c", "d"],
        }
    )
    assert [g.input for g in _dataframe_to_goldens(df)] == ["ok"]


def test_dataframe_to_goldens_handles_a_missing_expected_result():
    df = pd.DataFrame({"perturbated_prompt": ["ok"], "Expected_Result": [None]})
    assert _dataframe_to_goldens(df)[0].expected_output is None


def test_engine_generate_end_to_end(stub_transformers):
    np.random.seed(0)
    engine = _engine(
        alignment_df=pd.DataFrame(
            {"UserInput": ["plain question"], "Expected_Result": ["a"]}
        ),
        augmentations={"uppercase": ("case_change", 1.0)},
        paraphrase_count=1,
    )
    goldens = engine.generate()
    assert goldens and all(isinstance(g, Golden) for g in goldens)


def test_synthesizer_drives_the_engine(stub_transformers):
    np.random.seed(0)
    synth = AlignmentSynthesizer.from_dataframe(
        pd.DataFrame({"UserInput": ["plain question"], "Expected_Result": ["a"]}),
        augmentations={"uppercase": ("case_change", 1.0)},
        paraphrase_count=1,
    )
    dataset = synth.generate()
    assert synth.dataset is dataset
    assert set(synth.metadata_keys) == set(LegacyTagT5Engine.metadata_keys)
