import pytest
import pandas as pd
import os
from llm_inspector.rag_eval import RagEval
from configparser import ConfigParser

config = ConfigParser()
config.read(".//test_sample//test_config.ini")