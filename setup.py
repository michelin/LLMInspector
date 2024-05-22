"""Setup script for llm_inspector."""

from os.path import dirname
from os.path import join as pjoin

from setuptools import setup

try:
    from pydnx.packaging.git import write_version

    PROJECT = "llm_inspector"
    write_version(pjoin(dirname(__file__), PROJECT, "version.py"))

    from llm_inspector import __version__  # isort:skip

except ImportError:
    write_version = None
    __version__ = "0.0.0dev0"

setup(version=__version__)
