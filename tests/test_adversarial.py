import pytest
import pandas as pd
import os
from llm_inspector.adversarial import Adversarial
from configparser import ConfigParser

config = ConfigParser()
config.read(".//test_sample//test_config.ini")


def test_init():
    adversarial = Adversarial(config)

    assert isinstance(adversarial, Adversarial)


def test_adversarial_export():
    adversarial = Adversarial(config)

    result_df = adversarial.export_adversarial_data()
    assert isinstance(result_df, pd.DataFrame)
