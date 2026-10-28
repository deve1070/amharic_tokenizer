"""
Run every module's doctests.

The examples in the docstrings are documentation a reader trusts, so a
stale one is a defect. Three in morphology.py had drifted unnoticed
because nothing ran them.
"""

import doctest
import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import amharic_morph  # noqa: E402

MODULES = sorted(m.name for m in pkgutil.iter_modules(amharic_morph.__path__))


@pytest.mark.parametrize("name", MODULES)
def test_doctests(name):
    module = importlib.import_module(f"amharic_morph.{name}")
    result = doctest.testmod(module)
    assert result.failed == 0
