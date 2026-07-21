"""Setup script for llminspector."""

from os.path import dirname
from os.path import join as pjoin

from setuptools import setup

try:
    from pydnx.packaging.git import write_version

    PROJECT = "llminspector"
    write_version(pjoin(dirname(__file__), PROJECT, "version.py"))

    from llminspector import __version__  # isort:skip

except ImportError:
    write_version = None
    __version__ = "0.0.0dev0"

setup(version=__version__)
