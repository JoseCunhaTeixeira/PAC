"""Smoke tests: key modules import cleanly.

Guards against the kind of breakage a sigpipe path/rename can cause --
imports resolving at collection time is a cheap way to catch a stale path
before it reaches the API.
"""

import importlib

import pytest

MODULES = [
    "masw.io.inversion",
    "masw.io.acquisition",
    "masw.io.dispersion_images",
    "masw.adapters.inversion",
    "masw.adapters.windows",
    "masw.algorithms.dispersion_picking",
    "masw.runners.computing",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
