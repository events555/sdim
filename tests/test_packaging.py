"""Checks on pyproject.toml: the declared dependencies and the pytest configuration."""

import ast
import os
import re
import sys
import tomllib
import warnings

import numba.core.errors
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import names that differ from the name of the distribution providing them.
DISTRIBUTIONS = {"cirq": "cirq-core"}


def _third_party_imports(package_dir):
    names = set()
    for root, _, files in os.walk(package_dir):
        for name in files:
            if name.endswith(".py"):
                with open(os.path.join(root, name), encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names.update(alias.name.split(".")[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.level == 0:
                        names.add(node.module.split(".")[0])
    return names - set(sys.stdlib_module_names) - {"sdim"}


def test_declared_dependencies_are_the_ones_sdim_imports():
    """pyproject.toml used to require Diophantine and networkx, which nothing imports, and the cirq
    metapackage with its vendor packages, where `import cirq` only needs cirq-core. Each dependency
    has a plain floor, which the floors job in tests.yml installs."""
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as f:
        dependencies = tomllib.load(f)["project"]["dependencies"]
    for dep in dependencies:
        assert re.fullmatch(r"[A-Za-z0-9._-]+>=[0-9][0-9.]*", dep), dep
    declared = {dep.split(">=")[0].lower() for dep in dependencies}
    imported = {DISTRIBUTIONS.get(name, name) for name in _third_party_imports(os.path.join(REPO, "sdim"))}
    assert declared == imported


@pytest.mark.parametrize("module", ["sdim", "sdim.dem", "sdim.tableau.tableau_prime"])
def test_warnings_from_sdim_modules_are_errors(module):
    """A dependency's DeprecationWarning about a call that sdim makes is attributed to the sdim
    module making it; the filterwarnings setting turns it into a test failure."""
    with pytest.raises(DeprecationWarning):
        warnings.warn_explicit("deprecated", DeprecationWarning, module.replace(".", "/") + ".py", 1,
                               module=module)


@pytest.mark.parametrize("category", ["NumbaDeprecationWarning", "NumbaPendingDeprecationWarning"])
def test_numba_deprecations_are_errors(category):
    """numba re-emits a warning about a jitted function from its own files (for example a reflected
    list argument, reported from numba/core/ir_utils.py), which the sdim module filter misses."""
    category = getattr(numba.core.errors, category)
    with pytest.raises(category):
        warnings.warn_explicit("deprecated", category, numba.core.errors.__file__, 1)
