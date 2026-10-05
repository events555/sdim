"""
Regression tests for the public API around circuits: how add_gate checks the qudits and the
MUL scalar it is given, where read_circuit and write_circuit look for and put files, text
encodings, the legacy DEM's output, and docstrings that name things that do not exist.
"""

import inspect
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import sdim
from sdim import generate_and_write_random_circuit
from sdim.circuit import Circuit
from sdim.circuit_io import read_circuit, write_circuit
from sdim.dem_legacy import DetectorErrorModel as LegacyDetectorErrorModel
from sdim.gatedata import GateData
from sdim.program import Program


def _names_with_arg_count(arg_count):
    """Every gate name and alias of the gates that act on arg_count qudits."""
    gate_data = GateData(3)
    primary = {name for name, gate in gate_data.gateMap.items() if gate.arg_count == arg_count}
    return sorted(primary | {alias for alias, name in gate_data.aliasMap.items() if name in primary})


TWO_QUDIT_NAMES = _names_with_arg_count(2)
ONE_QUDIT_NAMES = _names_with_arg_count(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(sdim.__file__)))


def _one_qudit_params(name):
    return {"a": 2} if GateData(3).aliasMap.get(name, name) == "MUL" else {}


# --------------------------------------------------------------------------
# Qudit counts in add_gate


def test_gate_name_lists_include_the_aliases():
    assert {"CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP", "N2", "SUM", "CX", "C", "NOISE2"} <= set(TWO_QUDIT_NAMES)
    assert {"H", "M", "RESET", "N1", "MUL", "MULT", "MR"} <= set(ONE_QUDIT_NAMES)


@pytest.mark.parametrize("name", TWO_QUDIT_NAMES)
def test_two_qudit_gate_needs_a_control_and_a_target(name):
    """
    add_gate('N2', 0) and add_gate('CNOT', 0) used to store target None, which the frame sampler
    read as the last qudit, the tableau could not apply, and sdim.dem crashed on or ignored.
    """
    primary = GateData(3).aliasMap.get(name, name)
    c = Circuit(3, 3)
    for args in [(0,), ([0, 1],), (None, 1), ()]:
        with pytest.raises(ValueError, match=f"{primary} acts on two qudits"):
            c.add_gate(name, *args)
    assert c.operations == []
    c.add_gate(name, 0, 2)
    assert [(op.qudit_index, op.target_index) for op in c.operations] == [(0, 2)]


def test_noise_missing_a_qudit_no_longer_reaches_the_last_qudit():
    """The reported case: noise from N2 0, or from N1 without a qudit, showed up on qudit 2."""
    c = Circuit(3, 3)
    with pytest.raises(ValueError, match="N2 acts on two qudits"):
        c.add_gate("N2", 0, prob=0.9)
    with pytest.raises(ValueError, match="N1 acts on one qudit"):
        c.add_gate("N1", prob=0.9, noise_channel="f")
    c.add_gate("N2", 0, 1, prob=0.9)
    c.add_gate("M", [0, 1, 2])
    np.random.seed(5)
    measurements, _ = Program(c).simulate(shots=401)
    rates = [np.mean([r.measurement_value != 0 for r in measurements[q][0]]) for q in range(3)]
    assert rates[0] > 0.3 and rates[1] > 0.3 and rates[2] == 0


@pytest.mark.parametrize("name", ONE_QUDIT_NAMES)
def test_one_qudit_gate_takes_no_target(name):
    """A target given to a one-qudit gate used to be ignored by the simulators (H 0 1 was H 0)."""
    primary = GateData(3).aliasMap.get(name, name)
    params = _one_qudit_params(name)
    c = Circuit(3, 3)
    for args in [(0, 1), (0, [1, 2]), ([0, 1], [1, 2]), (None, 1)]:
        with pytest.raises(ValueError, match=f"{primary} acts on one qudit and takes no target"):
            c.add_gate(name, *args, **params)
    assert c.operations == []
    c.add_gate(name, [0, 2], **params)
    assert [(op.qudit_index, op.target_index) for op in c.operations] == [(0, None), (2, None)]


@pytest.mark.parametrize("name", ONE_QUDIT_NAMES)
def test_one_qudit_gate_needs_its_qudit(name):
    """add_gate('N1') and add_gate('H') used to store qudit None, like a two-qudit gate without a target."""
    primary = GateData(3).aliasMap.get(name, name)
    c = Circuit(3, 3)
    with pytest.raises(ValueError, match=f"{primary} acts on one qudit: give its qudit"):
        c.add_gate(name, **_one_qudit_params(name))
    assert c.operations == []


def test_numpy_integer_qudits():
    """A single NumPy integer qudit raised TypeError, although a list of them worked."""
    c = Circuit(3, 3)
    c.add_gate("X", np.int64(0))
    c.add_gate("CNOT", np.int64(0), np.int32(1))
    c.add_gate("CZ", np.uint8(1), [2])
    c.add_gate("M", np.arange(3))
    assert [(op.name, op.qudit_index, op.target_index) for op in c.operations] == [
        ("X", 0, None), ("CNOT", 0, 1), ("CZ", 1, 2), ("M", 0, None), ("M", 1, None), ("M", 2, None)]
    assert [r.measurement_value for r in Program(c).simulate()] == [1, 1, 0]


def test_qudit_lists_still_pair_up():
    c = Circuit(4, 3)
    c.add_gate("CNOT", 0, [1, 2])
    c.add_gate("CZ", [1, 2], 3)
    c.add_gate("SWAP", [0, 1], [2, 3])
    c.add_gate("TICK")
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-1]")
    assert [(op.name, op.qudit_index, op.target_index) for op in c.operations] == [
        ("CNOT", 0, 1), ("CNOT", 0, 2), ("CZ", 1, 3), ("CZ", 2, 3), ("SWAP", 0, 2), ("SWAP", 1, 3),
        ("TICK", None, None), ("M", 0, None), ("DETECTOR", None, None)]


@pytest.mark.parametrize("line,gate", [("N2 0 prob=0.1", "N2"), ("CNOT 0", "CNOT"), ("H 0 1", "H"),
                                       ("N1 prob=0.1", "N1")])
def test_read_circuit_names_the_bad_gate_line(tmp_path, line, gate):
    path = tmp_path / "bad.chp"
    path.write_text(f"#\nd 3\nM 1\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"gate line {line!r}: {gate} acts on")):
        read_circuit(str(path))


# --------------------------------------------------------------------------
# MUL scalar


@pytest.mark.parametrize("params,match", [
    ({}, "needs its scalar"),
    ({"a": None}, "needs its scalar"),
    ({"a": None, "scalar": 2}, "needs its scalar"),  # the simulators read a first, too
    ({"a": 2.5}, "must be an integer"),
    ({"a": "2.5"}, "must be an integer"),
    ({"a": "two"}, "must be an integer"),
    ({"a": float("nan")}, "must be an integer"),
    ({"a": float("inf")}, "must be an integer"),
    ({"a": [2]}, "must be an integer"),
    ({"a": 3}, "not coprime"),
    ({"a": 0}, "not coprime"),
    ({"a": -6}, "not coprime"),
    ({"scalar": 10 ** 19 + 2}, "not coprime"),
])
def test_mul_scalar_is_checked_when_the_gate_is_added(params, match):
    """MUL without a scalar, or with a=3 at d=3, used to fail only when the circuit ran."""
    c = Circuit(1, 3)
    with pytest.raises(ValueError, match=match):
        c.add_gate("MULT", 0, **params)
    assert c.operations == []


@pytest.mark.parametrize("d", [2, 3, 4, 5, 6, 9])
def test_mul_accepts_exactly_the_scalars_the_simulators_accept(d):
    """Negative scalars and scalars of at least d are reduced mod d, as the simulators do."""
    scalars = list(range(-2 * d, 2 * d + 1)) + [10 ** 19 + 1, -(10 ** 19) - 1, 2 ** 70 + 3]
    scalars += [float(d + 1), str(d - 1), np.int64(d + 1)]
    for a in scalars:
        c = Circuit(1, d)
        c.add_gate("X", 0)
        try:
            c.add_gate("MUL", 0, a=a)
            added = True
        except ValueError:
            added = False
            # Put the gate in without add_gate, to see what the simulators make of it.
            c.add_gate("MUL", 0, a=1)
            c.operations[-1].params["a"] = a
        c.add_gate("M", 0)
        try:
            outcome = Program(c).simulate()[0].measurement_value
            Program._build_ir([c], 1)
            simulated = True
        except ValueError:
            simulated = False
        assert added == simulated, (d, a)
        if added:
            assert outcome == int(a) % d


def test_mul_scalar_alias_and_types_still_work(tmp_path):
    c = Circuit(1, 5)
    c.add_gate("X", 0)
    c.add_gate("MULTIPLY", 0, scalar=-2)
    c.add_gate("MUL", 0, a=np.int64(3))
    c.add_gate("MUL", 0, a=3.0)
    c.add_gate("MUL", 0, a="2")
    c.add_gate("M", 0)
    assert Program(c).simulate()[0].measurement_value == (-2 * 3 * 3 * 2) % 5
    assert read_circuit(write_circuit(c, "mul.chp", directory=str(tmp_path))).operations[1].params == {"scalar": -2}


def test_add_gate_docstrings_document_mul_and_the_dimension_bound():
    doc = " ".join(Circuit.add_gate.__doc__.split())
    assert "a (int): The scalar of MUL" in doc and "scalar is another name for it" in doc
    assert "2**31" in Circuit.__doc__
    with pytest.raises(ValueError, match="less than 2"):
        Circuit(1, 2 ** 31)


# --------------------------------------------------------------------------
# Where read_circuit and write_circuit look for and put files


def test_read_circuit_finds_a_relative_path_in_the_working_directory(tmp_path, monkeypatch):
    """read_circuit('my.chp') used to look next to the sdim package and raise FileNotFoundError."""
    (tmp_path / "my.chp").write_text("#\nd 3\nH 0\nM 0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    c = read_circuit("my.chp")
    assert (c.dimension, [op.name for op in c.operations]) == (3, ["H", "M"])
    assert read_circuit(str(tmp_path / "my.chp")).dimension == 3
    with pytest.raises(FileNotFoundError, match="missing.chp"):
        read_circuit("missing.chp")


def test_read_circuit_prefers_the_working_directory_and_falls_back_to_the_package(tmp_path, monkeypatch):
    if not os.path.exists(os.path.join(ROOT, "circuits", "epr.chp")):
        pytest.skip("needs the circuits/ folder of a source checkout")
    monkeypatch.chdir(tmp_path)
    # Not in the working directory: the old lookup next to the package still finds it.
    assert [op.name for op in read_circuit("circuits/epr.chp").operations] == ["H", "CNOT", "M"]
    (tmp_path / "circuits").mkdir()
    (tmp_path / "circuits" / "epr.chp").write_text("#\nd 5\nX 0\n", encoding="utf-8")
    c = read_circuit("circuits/epr.chp")
    assert (c.dimension, [op.name for op in c.operations]) == (5, ["X"])


def test_write_circuit_defaults_to_circuits_in_the_working_directory(tmp_path, monkeypatch):
    """write_circuit and generate_and_write_random_circuit used to write next to the package (site-packages)."""
    def next_to_the_package():
        # Modification times, so that a file left there by an earlier run does not count.
        paths = [os.path.join(ROOT, "circuits", name) for name in ("default_dir_test.chp", "random_dir_test.chp")]
        return {path: os.stat(path).st_mtime_ns for path in paths if os.path.exists(path)}

    before = next_to_the_package()
    monkeypatch.chdir(tmp_path)
    c = Circuit(2, 3)
    c.add_gate("CNOT", 0, 1)
    path = write_circuit(c, "default_dir_test.chp")
    assert os.path.samefile(path, tmp_path / "circuits" / "default_dir_test.chp")
    # The repository workflow: write with the default, then read back by the relative path.
    assert [op.name for op in read_circuit("circuits/default_dir_test.chp").operations] == ["CNOT"]
    generate_and_write_random_circuit(2, 5, 3, output_file="random_dir_test.chp", seed=1)
    assert (tmp_path / "circuits" / "random_dir_test.chp").exists()
    assert next_to_the_package() == before


def test_files_are_utf8_whatever_the_locale():
    """open() without an encoding uses the locale's, so .chp files read differently across platforms."""
    code = "\n".join([
        "import os, sys, tempfile, warnings",
        "import sdim",
        "from sdim import Circuit, read_circuit, write_circuit",
        "from sdim.dem_legacy import DetectorErrorModel",
        "assert sdim.__file__.startswith(sys.argv[1]), sdim.__file__",
        "warnings.simplefilter('always', EncodingWarning)",
        "with warnings.catch_warnings(record=True) as caught, tempfile.TemporaryDirectory() as tmp:",
        "    warnings.simplefilter('always', EncodingWarning)",
        "    c = Circuit(2, 3)",
        "    c.add_gate('N1', 0, prob=0.1, noise_channel='f')",
        "    c.add_gate('M', 0)",
        "    c.add_gate('DETECTOR', expr='rec[-1]', label='\\u00e9')",
        "    assert read_circuit(write_circuit(c, 'x.chp', comment='\\u00fc', directory=tmp)).operations[2].params['label'] == '\\u00e9'",
        "    DetectorErrorModel.from_circuit(c).write_to_file(tmp, 'x.sdem')",
        "    DetectorErrorModel().read_from_file(os.path.join(tmp, 'x.sdem'))",
        "print([f'{w.filename}:{w.lineno}' for w in caught",
        "       if issubclass(w.category, EncodingWarning) and w.filename.startswith(sys.argv[1])])",
    ])
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, "-X", "warn_default_encoding", "-c", code, ROOT], env=env,
                         capture_output=True, encoding="utf-8", cwd=os.path.dirname(ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[-1] == "[]"


# --------------------------------------------------------------------------
# Legacy DEM


def _legacy_circuit(n1="N1", n2="N2"):
    c = Circuit(2, 2)
    c.add_gate(n1, 0, prob=0.1, noise_channel="f")
    c.add_gate(n2, 0, 1, prob=0.05)
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="rec[-2]")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-1]")
    return c


def test_legacy_dem_prints_nothing(capsys):
    """from_circuit used to print three progress lines on every call."""
    LegacyDetectorErrorModel.from_circuit(_legacy_circuit())
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("measurements", [1, 2, 3])
def test_legacy_dem_with_fewer_than_two_noise_outcomes(noise, measurements):
    """With zero or one noise outcome (N1 'f' at d=2), from_circuit ran the tableau instead of the frame sampler."""
    c = Circuit(3, 2)
    if noise:
        c.add_gate("N1", 0, prob=0.1, noise_channel="f")
    c.add_gate("M", list(range(measurements)))
    c.add_gate("DETECTOR", expr="rec[0]")
    expected = ["DIMENSION", "2"] + (["ERROR", "prob=0.1", "D0=1"] if noise else [])
    assert str(LegacyDetectorErrorModel.from_circuit(c)).split() == expected


def test_legacy_dem_reads_noise_gate_aliases():
    """NOISE1 and NOISE2 were not counted as noise and from_circuit raised ValueError."""
    expected = str(LegacyDetectorErrorModel.from_circuit(_legacy_circuit()))
    assert str(LegacyDetectorErrorModel.from_circuit(_legacy_circuit("NOISE1", "NOISE2"))) == expected


# --------------------------------------------------------------------------
# Docstrings and notebooks


def test_package_docstring_lists_only_names_that_exist():
    """It listed generate_random_circuit, which does not exist."""
    sections = re.split(r"^## ", sdim.__doc__, flags=re.M)
    listed = {section.split("\n", 1)[0].strip(): re.findall(r"^- \*\*(\w+)\*\*", section, flags=re.M)
              for section in sections}
    assert "generate_random_clifford_circuit" in listed["Functions"]
    for name in listed["Functions"] + listed["Classes"]:
        assert hasattr(sdim, name), name
    for name in listed["Modules"]:
        __import__(f"sdim.{name}")


def test_apply_gate_docstring_matches_its_signature():
    args = Program.apply_gate.__doc__.split("Args:")[1].split("Returns:")[0]
    documented = re.findall(r"^\s+(\w+) \(", args, flags=re.M)
    assert documented == [p for p in inspect.signature(Program.apply_gate).parameters if p != "self"]


def test_generate_and_write_random_circuit_docstring_matches_its_signature():
    args = generate_and_write_random_circuit.__doc__.split("Args:")[1].split("Returns:")[0]
    documented = re.findall(r"^\s+(\w+):", args, flags=re.M)
    assert documented == list(inspect.signature(generate_and_write_random_circuit).parameters)


@pytest.mark.parametrize("name", ["repetition_code.ipynb", "steane_css.ipynb", "surface_code.ipynb"])
def test_example_notebooks_are_valid_json(name):
    """surface_code.ipynb was an empty file, which Jupyter cannot open."""
    path = os.path.join(ROOT, "examples", name)
    if not os.path.exists(path):
        pytest.skip("needs the examples/ folder of a source checkout")
    with open(path, encoding="utf-8") as file:
        notebook = json.load(file)
    assert notebook["nbformat"] == 4
    assert any(cell["cell_type"] == "code" for cell in notebook["cells"])
