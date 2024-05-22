import pytest
import pandas as pd
import os
from llm_inspector.alignment import Alignment
from configparser import ConfigParser

config = ConfigParser()
config.read(".//test_sample//test_config.ini")

def test_init():
    alignment = Alignment(config)

    assert isinstance(alignment, Alignment)


def test_tagaugmentation():
    alignment = Alignment(config)
    result_df = alignment.alignment_data()

    assert isinstance(result_df, pd.DataFrame)


def test_paraphrasing():
    alignment = Alignment(config)
    result_df = alignment.alignment_data()
    paraphrased_df = alignment.paraphrase_prompts(result_df)

    assert isinstance(paraphrased_df, pd.DataFrame)


def test_perturbation():
    alignment = Alignment(config)
    perturbed_df = alignment.transform_df()

    assert isinstance(perturbed_df, pd.DataFrame)


