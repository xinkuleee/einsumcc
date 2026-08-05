"""Compatibility shim for older pip editable installs.

Project metadata lives in setup.cfg and build metadata lives in pyproject.toml.
The Python 3.9 pip shipped with older macOS installations still needs a setup.py
entry point for `pip install -e .`.
"""

from setuptools import setup


setup()
