"""
Regression tests for how the numba kernels are built (`sdim._jit._kernel`): the package must
import and run when numba has nowhere to write its cache, and with the JIT switched off
(NUMBA_DISABLE_JIT=1) it must run without NumPy overflow warnings and sample exactly what the
compiled kernels sample.
"""

import importlib
import importlib.util
import os
import pkgutil
import random
import shutil
import site
import subprocess
import sys
import warnings

import numba
import numba.misc.appdirs
import numpy as np
import pytest
from numba.core.registry import CPUDispatcher

import sdim
from sdim._jit import _kernel
from sdim.circuit import Circuit
from sdim.dem import DetectorErrorModel
from sdim.program import Program
from sdim.tableau.tableau_optimized import JIT_ENABLED

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SDIM_DIR = os.path.dirname(os.path.abspath(sdim.__file__))


def _package_kernels():
    """(name, dispatcher) for every numba kernel of the package, after importing all of its modules."""
    for info in pkgutil.walk_packages(sdim.__path__, "sdim."):
        importlib.import_module(info.name)
    kernels = []
    for module_name, module in sorted(sys.modules.items()):
        if module is None or (module_name != "sdim" and not module_name.startswith("sdim.")):
            continue
        for name, obj in vars(module).items():
            if isinstance(obj, CPUDispatcher) and obj.py_func.__module__ == module_name:
                kernels.append((f"{module_name}.{name}", obj))
    return kernels


def _blocked_path(tmp_path):
    """A regular file: no directory can be created under it, even by root."""
    blocked = tmp_path / "blocked"
    blocked.write_text("")
    return blocked


# --------------------------------------------------------------------------
# No writable cache directory


@pytest.mark.skipif(not JIT_ENABLED, reason="numba's JIT is switched off")
def test_every_kernel_keeps_the_on_disk_cache():
    """Kernels compiled once are loaded from numba's cache in later processes, except the inlined helpers."""
    kernels = _package_kernels()
    assert kernels
    uncached = [name for name, kernel in kernels
                if kernel.stats.cache_path is None and kernel.targetoptions.get("inline") != "always"]
    assert not uncached, f"kernels without the on-disk cache: {uncached}"
    assert sdim.dem._sample_chunks.targetoptions.get("nogil")


@pytest.mark.skipif(not JIT_ENABLED, reason="numba's JIT is switched off")
def test_kernel_compiles_without_the_cache_when_no_cache_directory_is_writable(tmp_path, monkeypatch):
    """numba finds no cache directory: the kernel is built without one, keeps its options and still compiles lazily."""
    blocked = _blocked_path(tmp_path)
    module_dir = tmp_path / "kernels"
    module_dir.mkdir()
    (module_dir / "__pycache__").write_text("")
    module_file = module_dir / "jit_modes_kernels.py"
    module_file.write_text("from sdim._jit import _kernel\n"
                           "\n"
                           "def plain(a, b):\n"
                           "    return a + b\n"
                           "\n"
                           "@_kernel(nogil=True)\n"
                           "def add(a, b):\n"
                           "    return a + b\n")
    monkeypatch.setattr(numba.core.config, "CACHE_DIR", "")
    # numba's user-wide cache directory, which does not follow HOME on Windows.
    monkeypatch.setattr(numba.misc.appdirs, "user_cache_dir", lambda *args, **kwargs: str(blocked / "cache"))
    spec = importlib.util.spec_from_file_location("jit_modes_kernels", module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(RuntimeError, match="no locator available"):
        numba.njit(cache=True)(module.plain)
    assert isinstance(module.add, CPUDispatcher)
    assert module.add.stats.cache_path is None
    assert module.add.targetoptions.get("nogil")
    assert module.add.signatures == []
    assert module.add(2, 3) == 5
    assert len(module.add.signatures) == 1


@pytest.mark.skipif(not hasattr(numba.core.config, "CACHE_LOCATOR_CLASSES"),
                    reason="this numba has no NUMBA_CACHE_LOCATOR_CLASSES")
def test_kernel_raises_cache_errors_other_than_a_missing_cache_directory(monkeypatch):
    """Only numba's "no locator available" error falls back; a misconfigured cache still raises."""
    # numba only sets up the cache with the JIT on.
    monkeypatch.setattr(numba.core.config, "DISABLE_JIT", False)
    monkeypatch.setattr(numba.core.config, "CACHE_LOCATOR_CLASSES", "NoSuchLocator")

    def add(a, b):
        return a + b

    with pytest.raises(RuntimeError, match="NoSuchLocator"):
        _kernel(add)


def _check_kernels_without_cache(root):
    """Runs in the child process of the next test."""
    assert os.path.abspath(sdim.__file__).startswith(root), sdim.__file__
    kernels = _package_kernels()
    assert kernels
    cached = [name for name, kernel in kernels if kernel.stats.cache_path is not None]
    assert not cached, f"kernels with an on-disk cache: {cached}"
    assert sdim.dem._sample_chunks.targetoptions.get("nogil")
    out = np.empty(3, dtype=np.int64)
    sdim.program._floor_mod_int64(np.array([-7, 0, 12], dtype=np.int64), 5, out)
    assert out.tolist() == [3, 0, 2]


@pytest.mark.skipif(sys.platform == "win32", reason="numba's user-wide cache directory on Windows does not follow HOME")
def test_package_imports_and_runs_without_a_writable_cache_directory(tmp_path):
    """
    A read-only install with a read-only home directory and NUMBA_CACHE_DIR unset: `import sdim`
    used to raise "RuntimeError: cannot cache function '_rng_seed': no locator available".

    The package is copied with a regular file in place of each __pycache__ directory, and HOME
    and XDG_CACHE_HOME point below a regular file, so numba finds no cache directory even when
    the tests run as root.
    """
    root = tmp_path / "site"
    shutil.copytree(_SDIM_DIR, root / "sdim", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for directory, _, _ in os.walk(root / "sdim"):
        open(os.path.join(directory, "__pycache__"), "w").close()
    blocked = _blocked_path(tmp_path)
    env = dict(os.environ)
    env.pop("NUMBA_CACHE_DIR", None)
    env.pop("NUMBA_DISABLE_JIT", None)
    env.update({
        "HOME": str(blocked),
        "XDG_CACHE_HOME": str(blocked / "cache"),
        # Packages installed with --user stay importable without HOME.
        "PYTHONUSERBASE": site.getuserbase(),
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join([str(root), _TESTS_DIR] + [p for p in [env.get("PYTHONPATH")] if p]),
    })
    script = f"import test_jit_modes as t\nt._check_kernels_without_cache({str(root)!r})\n"
    result = subprocess.run([sys.executable, "-c", script], env=env, cwd=str(tmp_path),
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------
# JIT switched off


def test_kernel_without_jit_ignores_uint64_overflow(monkeypatch):
    """The plain Python kernels wrap uint64 arithmetic around like compiled code, without a warning."""
    monkeypatch.setattr(numba.core.config, "DISABLE_JIT", True)

    def mix(x):
        return (x + np.uint64(0x9E3779B97F4A7C15)) * np.uint64(0xBF58476D1CE4E5B9)

    x = np.uint64(0xFFFFFFFFFFFFFFF0)
    expected = np.uint64(((int(x) + 0x9E3779B97F4A7C15) * 0xBF58476D1CE4E5B9) % 2 ** 64)
    with pytest.warns(RuntimeWarning, match="overflow"):
        assert mix(x) == expected
    inner = _kernel(nogil=True)(mix)

    @_kernel
    def outer(x, fail):
        if fail:
            raise ValueError("kernel failed")
        return inner(x)

    assert not isinstance(inner, CPUDispatcher)
    errors = np.geterr()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert inner(x) == expected
        assert outer(x, False) == expected
        with pytest.raises(ValueError, match="kernel failed"):
            outer(x, True)
        assert inner(x) == expected
    assert np.geterr() == errors


def _frame_cases():
    """(name, circuit, shots) for the frame sampler: every frame gate and noise kind, d = 2 and 2**31 - 1 included."""
    dist = np.random.default_rng(0).random(5 ** 4)
    full = Circuit(4, 5)
    full.add_gate("RESET", [0, 1, 2, 3])
    full.add_gate("H", 0)
    full.add_gate("N1", [0, 1], noise_channel="d", prob=0.2)
    full.add_gate("CNOT", 0, 1)
    full.add_gate("P", 1)
    full.add_gate("N1", 2, noise_channel="f", prob=0.3)
    full.add_gate("CZ", 1, 2)
    full.add_gate("N1", 3, noise_channel="p", prob=0.3)
    full.add_gate("MUL", 2, a=2)
    full.add_gate("SWAP", 2, 3)
    full.add_gate("N2", 0, 3, prob=0.25)
    full.add_gate("N2", 1, 2, prob_dist=(dist / dist.sum()).tolist())
    full.add_gate("H_INV", 3)
    full.add_gate("CNOT_INV", 3, 0)
    full.add_gate("P_INV", 0)
    full.add_gate("CZ_INV", 0, 2)
    full.add_gate("M_X", 3)
    full.add_gate("M", [0, 1, 2])
    full.add_gate("N1", 1, noise_channel="d", prob=1.0)
    full.add_gate("H", 1)
    full.add_gate("M", 1)
    full.add_gate("DETECTOR", expr="rec[-1] - rec[-3]")
    full.add_gate("LOGICAL_OBSERVABLE", expr="rec[-5] + rec[-2]")
    cases = [("frame_5", full, 200)]
    for d in (2, 2147483647):
        c = Circuit(3, d)
        c.add_gate("H", 0)
        c.add_gate("N1", 0, noise_channel="d", prob=0.4)
        c.add_gate("CNOT", 0, 1)
        c.add_gate("N2", 1, 2, prob=0.3)
        c.add_gate("M", [0, 1])
        c.add_gate("H", 0)
        c.add_gate("M_X", 2)
        c.add_gate("M", 0)
        c.add_gate("DETECTOR", expr="rec[-1] - rec[-4]")
        c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-2]")
        cases.append((f"frame_{d}", c, 150))
    return cases


def _dem_circuit(d):
    """Flip, depolarizing, phase and two-qudit noise, and one fully mixing flip channel (a mechanism that always fires)."""
    c = Circuit(5, d)
    c.add_gate("RESET", [0, 1, 2, 3, 4])
    c.add_gate("N1", [0, 1], noise_channel="f", prob=0.05)
    c.add_gate("N1", 2, noise_channel="d", prob=0.1)
    c.add_gate("CNOT", 0, 2)
    c.add_gate("CNOT_INV", 1, 2)
    c.add_gate("N2", 1, 2, prob=0.02)
    c.add_gate("N1", 3, noise_channel="f", prob=1 - 1 / d)
    c.add_gate("CNOT", 0, 3)
    c.add_gate("H", 4)
    c.add_gate("N1", 4, noise_channel="p", prob=0.2)
    c.add_gate("M", [2, 0, 1, 3])
    c.add_gate("M_X", 4)
    c.add_gate("DETECTOR", expr="rec[-5]")
    c.add_gate("DETECTOR", expr="rec[-5] - rec[-4] + rec[-3]")
    c.add_gate("DETECTOR", expr="rec[-2] - rec[-4]")
    c.add_gate("DETECTOR", expr="rec[-1]")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-3]")
    return c


def _seeded_samples():
    """Seeded frame and DEM samples (and the models) of circuits that reach every random number path."""
    samples = {}
    for name, circuit, shots in _frame_cases():
        random.seed(5)
        np.random.seed(5)
        measurements, (detectors, observables) = Program(circuit).simulate(shots=shots, raw_detector_output=True)
        samples[f"{name}_measurements"] = np.array([m.measurement_value for q in measurements for r in q for m in r])
        samples[f"{name}_detectors"] = np.asarray(detectors)
        samples[f"{name}_observables"] = np.asarray(observables)
    for d in (2, 5, 1000003):
        dem = DetectorErrorModel.from_circuit(_dem_circuit(d))
        samples[f"dem_{d}_model"] = np.array(str(dem))
        if d < 10:
            samples[f"dem_{d}_lines"] = np.array(str(dem.to_lines()))
        samples[f"dem_{d}_detectors"], samples[f"dem_{d}_observables"] = dem.sample(3 * 256 + 17, seed=11)
    return samples


@pytest.mark.skipif(not JIT_ENABLED, reason="numba's JIT is switched off")
def test_jit_disabled_runs_without_warnings_and_matches_the_compiled_kernels(tmp_path):
    """
    With NUMBA_DISABLE_JIT=1 the xoshiro256** and splitmix64 generators of the frame sampler and
    the DEM sampler run on NumPy uint64 scalars, which used to warn "overflow encountered in
    scalar add" (an error under python -W error).  The samples must be the compiled kernels' samples.
    """
    out = tmp_path / "samples.npz"
    env = dict(os.environ)
    env["NUMBA_DISABLE_JIT"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [_TESTS_DIR, os.path.dirname(_SDIM_DIR)] + [p for p in [env.get("PYTHONPATH")] if p])
    script = ("import sys, warnings\n"
              "import numpy as np\n"
              "import test_jit_modes as t\n"
              f"assert t._SDIM_DIR == {_SDIM_DIR!r}, t._SDIM_DIR\n"
              "assert not t._package_kernels(), 'the JIT is on'\n"
              "warnings.simplefilter('error')\n"
              "np.savez(sys.argv[1], **t._seeded_samples())\n")
    result = subprocess.run([sys.executable, "-c", script, str(out)], env=env, cwd=str(tmp_path),
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr

    expected = _seeded_samples()
    with np.load(out) as uncompiled:
        assert sorted(uncompiled.files) == sorted(expected)
        for key, value in expected.items():
            np.testing.assert_array_equal(uncompiled[key], value, err_msg=key)
