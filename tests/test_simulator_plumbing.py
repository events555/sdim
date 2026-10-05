"""
Regression tests for the plumbing around the simulators in sdim.program: detector record
references, MUL scalars in the IR, circuits without measurements, programs made of several
circuits, and how the frame kernels are compiled.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
from numba.core import types
from numba.core.registry import CPUDispatcher

import sdim.program as program_module
from sdim.circuit import Circuit
from sdim.dem import DetectorErrorModel
from sdim.program import Program, _run_frame, simulate_frame


def _shifted_record_circuit(expr, label=None, gate="DETECTOR"):
    """rec 0 is qudit 0 after a random X flip, rec 1 is qudit 1 (always 0)."""
    c = Circuit(2, 3)
    c.add_gate("N1", 0, prob=1.0, noise_channel="f")
    c.add_gate("M", 0)
    c.add_gate("M", 1)
    kwargs = {"expr": expr}
    if label is not None:
        kwargs["label"] = label
    c.add_gate(gate, **kwargs)
    return c


# --------------------------------------------------------------------------
# Detector record references


@pytest.mark.parametrize("expr,arguments", [
    ("rec[-1]", [1]),
    ("rec[-2]", [0]),
    ("rec[0]", [0]),
    ("rec[1]", [1]),
    ("rec[-1] + rec[1] - rec[0]", [1, 0]),
    ("2*rec[-2] - rec[ -1 ]", [0, 1]),
    # Python reads these as integer indices too; they used to be rejected as non-integers
    ("rec[+1]", [1]),
    ("rec[ - 1]", [1]),
    ("rec[- 2] + rec[+0]", [0]),
    # One record named by its absolute and then its relative index (or the other way round)
    ("rec[1] - rec[-1] + 0*rec[0]", [1, 0]),
    ("rec[0] - rec[-2]", [0]),
    ("rec [ 1 ] + rec[0]", [1, 0]),
])
def test_record_references_resolve_like_python_indexing(expr, arguments):
    """rec[k] for k >= 0 is the k-th measurement, rec[-k] counts back from the latest one."""
    c = _shifted_record_circuit(expr)
    _, _, info = Program._build_ir([c], 1)
    assert info.detector_data[0][2] == arguments


def test_record_references_read_the_right_measurements():
    np.random.seed(5)
    c = _shifted_record_circuit("rec[-2]")
    c.add_gate("DETECTOR", expr="rec[1]")
    c.add_gate("DETECTOR", expr="2*rec[0] + rec[-1]")
    measurements, (detectors, _) = Program(c).simulate(shots=40, raw_detector_output=True)
    rec0 = np.array([m.measurement_value for m in measurements[0][0][1:]])
    assert (rec0 != 0).all()
    np.testing.assert_array_equal(detectors[0], rec0)
    np.testing.assert_array_equal(detectors[1], 0)
    np.testing.assert_array_equal(detectors[2], (2 * rec0) % 3)


def test_absolute_then_relative_reference_to_one_record():
    """
    An expression that named a record by its absolute index and then again by a relative one
    (rec[1] - rec[-1] with 2 records) used to read the wrong record for the second reference,
    which gave wrong detector and observable values (or an IndexError) without any warning.
    """
    np.random.seed(1)
    c = _shifted_record_circuit("rec[1] - rec[-1] + 0*rec[0]")  # rec 1 twice: always 0
    c.add_gate("DETECTOR", expr="rec[0] - rec[-2] + 0*rec[1]")  # rec 0 twice: always 0
    c.add_gate("DETECTOR", expr="rec[0] + rec[-2] + rec[1] - rec[-1]")
    c.add_gate("DETECTOR", expr="rec[-2] - rec[0] + rec[-1]")
    c.add_gate("LOGICAL_OBSERVABLE", expr="2*rec[0] - rec[-2] + rec[1] + rec[-1]")
    measurements, (detectors, observables) = Program(c).simulate(shots=40, raw_detector_output=True)
    rec0 = np.array([m.measurement_value for m in measurements[0][0][1:]])
    assert (rec0 != 0).all()
    np.testing.assert_array_equal(detectors[0], 0)
    np.testing.assert_array_equal(detectors[1], 0)
    np.testing.assert_array_equal(detectors[2], (2 * rec0) % 3)
    np.testing.assert_array_equal(detectors[3], 0)
    np.testing.assert_array_equal(observables[0], rec0)


def test_record_references_are_rewritten_to_shared_positions():
    resolve = program_module._resolve_record_references
    assert resolve("rec[1] - rec[-1]", 2, "D") == ("rec[0] - rec[0]", [1])
    assert resolve("rec[0] + 2*rec [ -2 ] - rec[+1]", 2, "D") == ("rec[0] + 2*rec[0] - rec[1]", [0, 1])


def test_brackets_that_do_not_index_rec_keep_their_meaning():
    """
    Every bracketed integer used to be read as a record reference, so the [1] of the list
    literal [5, 7][1] became one: the detector read a wrong value, or the range check raised
    an error about a rec[1] the expression does not contain.
    """
    np.random.seed(3)
    c = _shifted_record_circuit("rec[1] + rec[0] + [5, 7][1]")  # [1] used to read as rec[1], then [0]
    c.add_gate("DETECTOR", expr="[rec[0], rec[-1]][0] + [[0, 2]][0][1]")
    measurements, (detectors, _) = Program(c).simulate(shots=30, raw_detector_output=True)
    rec0 = np.array([m.measurement_value for m in measurements[0][0][1:]])
    assert (rec0 != 0).all()
    np.testing.assert_array_equal(detectors[0], (rec0 + 7) % 3)
    np.testing.assert_array_equal(detectors[1], (rec0 + 2) % 3)
    c = Circuit(1, 3)
    c.add_gate("M", [0, 0])
    c.add_gate("DETECTOR", expr="rec[-1] + [5, 7][1]")  # used to add 5
    _, (detectors, _) = Program(c).simulate(shots=3, raw_detector_output=True)
    np.testing.assert_array_equal(detectors[0], 7 % 3)
    c = Circuit(1, 3)
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-1] + [0, 1][1]")  # [1] would be out of range as a record
    assert Program._build_ir([c], 1)[2].detector_data[0][2] == [0]
    _, (detectors, _) = Program(c).simulate(shots=3, raw_detector_output=True)
    np.testing.assert_array_equal(detectors[0], 1)


def test_numeric_label_is_not_mistaken_for_an_index():
    """Errors named a labeled detector by its label alone, so label=5 read like the unlabeled DETECTOR 5."""
    def message(label):
        c = Circuit(1, 3)
        c.add_gate("M", 0)
        for _ in range(5):
            c.add_gate("DETECTOR", expr="rec[-1]")
        if label is not None:
            c.add_gate("DETECTOR", expr="rec[-1]")
        c.add_gate("DETECTOR", expr="rec[-9]", **({} if label is None else {"label": label}))
        with pytest.raises(ValueError) as info:
            Program(c).simulate(shots=3)
        return str(info.value)

    assert message(None).startswith("DETECTOR 5 refers to rec[-9]")
    assert message(5).startswith("DETECTOR 6 (label 5) refers to rec[-9]")
    assert message("5").startswith("DETECTOR 6 (label '5') refers to rec[-9]")


def test_record_reference_check_parses_each_expression_once(monkeypatch):
    """
    The up-front check used to parse every detector expression again (about 0.1 s for 30000
    detectors, which every shots=1 run paid); it now parses each distinct expression once.
    """
    resolved = []
    original = program_module._resolve_record_references

    def counting(*args):
        resolved.append(args)
        return original(*args)

    monkeypatch.setattr(program_module, "_resolve_record_references", counting)

    def circuit(first_round):
        c = Circuit(2, 3)
        c.add_gate("M", first_round)
        for _ in range(50):
            c.add_gate("M", [0, 1])
            c.add_gate("DETECTOR", expr="rec[-1] - rec[-3] + 0*rec[0]")
            c.add_gate("LOGICAL_OBSERVABLE", expr="rec[1]")
        return c

    program_module._check_record_references([circuit([0, 1])])
    assert resolved == []
    assert len(Program(circuit([0, 1])).simulate()) == 102
    with pytest.raises(ValueError, match=r"DETECTOR 0 refers to rec\[-3\], but only 2"):
        program_module._check_record_references([circuit([])])
    assert len(resolved) == 1


@pytest.mark.parametrize("expr,index", [("rec[-3]", -3), ("rec[2]", 2), ("rec[7]", 7), ("rec[-5]", -5),
                                        ("rec[-1] + rec[2]", 2), ("rec[+2]", 2), ("rec[ - 3]", -3)])
def test_out_of_range_record_reference_raises(expr, index):
    """These used to wrap around silently: with 2 measurements, rec[-5] and rec[7] both read rec[1]."""
    c = _shifted_record_circuit(expr, label="parity")
    message = rf"DETECTOR 0 \(label 'parity'\) refers to rec\[{index}\], but only 2 measurements"
    with pytest.raises(ValueError, match=message):
        Program._build_ir([c], 3)
    with pytest.raises(ValueError, match=message):
        Program._build_ir([c], 3, sample_noise=False)
    with pytest.raises(ValueError, match=message):
        Program(c).simulate(shots=3)
    with pytest.raises(ValueError, match=message):
        DetectorErrorModel.from_circuit(c)


def test_unlabeled_detectors_are_named_by_index():
    c = _shifted_record_circuit("rec[-1]")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[0]")
    c.add_gate("DETECTOR", expr="rec[-3]")
    with pytest.raises(ValueError, match=r"DETECTOR 1 refers to rec\[-3\]"):
        Program._build_ir([c], 1)
    c = _shifted_record_circuit("rec[-1]")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[2]")
    with pytest.raises(ValueError, match=r"LOGICAL_OBSERVABLE 0 refers to rec\[2\]"):
        Program._build_ir([c], 1)


def test_record_reference_before_any_measurement_raises_value_error():
    """This used to raise ZeroDivisionError."""
    c = Circuit(1, 3)
    c.add_gate("DETECTOR", expr="rec[-1]")
    c.add_gate("M", 0)
    with pytest.raises(ValueError, match=r"DETECTOR 0 refers to rec\[-1\], but no measurement"):
        Program(c).simulate(shots=3)


@pytest.mark.parametrize("expr", ["rec[-1 - 1]", "rec[i]", "rec[--1]", "rec[~0]"])
def test_record_index_must_be_an_integer_literal(expr):
    c = _shifted_record_circuit(expr)
    with pytest.raises(ValueError, match="integer"):
        Program._build_ir([c], 1)


@pytest.mark.parametrize("d", [3, 4])
@pytest.mark.parametrize("kwargs", [{}, {"shots": 3, "force_tableau": True}, {"shots": 2, "record_tableau": True},
                                    {"shots": 3}])
def test_every_mode_checks_record_references(d, kwargs):
    """The tableau simulation (shots=1, force_tableau, record_tableau) used to accept rec[7] after 2 measurements."""
    c = Circuit(2, d)
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="rec[-1]")
    c.add_gate("DETECTOR", expr="rec[7]")
    with pytest.raises(ValueError, match=r"DETECTOR 1 refers to rec\[7\], but only 2 measurements"):
        Program(c).simulate(**kwargs)
    c = Circuit(2, d)
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[i]", label="L")
    with pytest.raises(ValueError, match=r"LOGICAL_OBSERVABLE 0 \(label 'L'\) indexes rec with something other than an integer"):
        Program(c).simulate(**kwargs)


def test_tableau_mode_runs_valid_detectors_unchanged():
    c = Circuit(2, 3)
    c.add_gate("X", 0)
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="rec[+0] - rec[ - 1]", label="d")
    c.add_gate("M", 0)
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[2]")
    results = Program(c).simulate()
    assert [(m.qudit_index, m.measurement_value) for m in results] == [(0, 1), (0, 1), (1, 0)]


def test_detector_expression_keeps_its_other_text():
    """The parser used to delete every letter p in the expression, so pow(...) became ow(...)."""
    np.random.seed(2)
    c = _shifted_record_circuit("pow(rec[-2], 1) + 0 * rec[1]")
    measurements, (detectors, _) = Program(c).simulate(shots=20, raw_detector_output=True)
    rec0 = np.array([m.measurement_value for m in measurements[0][0][1:]])
    np.testing.assert_array_equal(detectors[0], rec0)


def test_detectors_of_appended_circuits_use_their_own_expressions():
    """Each circuit used to restart the detector function index, so the second circuit's
    detectors evaluated the first circuit's expressions."""
    first = Circuit(1, 3)
    first.add_gate("N1", 0, prob=1.0, noise_channel="f")
    first.add_gate("M", 0)
    first.add_gate("DETECTOR", expr="rec[-1]", label="first")
    second = Circuit(1, 3)
    second.add_gate("M", 0)
    second.add_gate("DETECTOR", expr="2*rec[-1]", label="second")
    second.add_gate("LOGICAL_OBSERVABLE", expr="rec[0] + rec[1]", label="both")
    program = Program(first)
    program.append_circuit(second)
    np.random.seed(0)
    measurements, det = program.simulate(shots=30)
    rec0 = np.array([m.measurement_value for m in measurements[0][0][1:]])
    rec1 = np.array([m.measurement_value for m in measurements[0][1][1:]])
    assert (rec0 != 0).all()
    assert [e["label"] for e in det["detectors"]] == ["first", "second"]
    np.testing.assert_array_equal(det["detectors"][0]["data"], rec0)
    np.testing.assert_array_equal(det["detectors"][1]["data"], (2 * rec1) % 3)
    np.testing.assert_array_equal(det["logicals"][0]["data"], (rec0 + rec1) % 3)


# --------------------------------------------------------------------------
# MUL scalars


@pytest.mark.parametrize("d,a", [(5, 10 ** 19 + 2), (5, -(10 ** 19) - 3), (4, 2 ** 70 + 3), (7, 3 - 7 * 2 ** 64)])
def test_mul_with_a_huge_scalar_works_in_frame_mode(d, a):
    """The frame sampler used to raise OverflowError storing a in the int64 IR."""
    expected = (2 * a) % d
    c = Circuit(1, d)
    c.add_gate("X", 0)
    c.add_gate("X", 0)
    c.add_gate("MUL", 0, a=a)
    c.add_gate("M", 0)
    assert Program(c).simulate()[0].measurement_value == expected
    measurements, _ = Program(c).simulate(shots=5)
    assert [r.measurement_value for r in measurements[0][0]] == [expected] * 5
    ir, _, _ = Program._build_ir([c], 1)
    assert ir["scalar"][ir["gate_id"] == 22].tolist() == [a % d]


@pytest.mark.parametrize("d,a", [(5, 10 ** 19 + 5), (6, 2 ** 70), (4, 2)])
def test_mul_scalar_must_be_coprime_in_every_mode(d, a):
    c = Circuit(1, d)
    c.add_gate("MUL", 0, a=a)
    c.add_gate("M", 0)
    for run in (lambda: Program(c).simulate(), lambda: Program(c).simulate(shots=3),
                lambda: Program._build_ir([c], 1)):
        with pytest.raises(ValueError, match="not coprime"):
            run()


# --------------------------------------------------------------------------
# Circuits without measurements


def test_frame_mode_without_measurements_returns_empty_results():
    """simulate(shots > 1) used to raise IndexError when nothing was measured."""
    c = Circuit(2, 3)
    c.add_gate("H", 0)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("N1", 0, prob=0.5)
    assert Program(c).simulate() == []
    assert Program(c).simulate(shots=4, force_tableau=True) == [[], []]
    measurements, det = Program(c).simulate(shots=5)
    assert measurements == [[], []]
    assert det == {"detectors": [], "logicals": []}
    measurements, (detectors, logicals) = Program(c).simulate(shots=5, raw_detector_output=True)
    assert measurements == [[], []]
    assert detectors.shape == (0, 4) and logicals.shape == (0, 4)


def test_frame_mode_without_measurements_and_constant_detectors():
    c = Circuit(1, 3)
    c.add_gate("H", 0)
    c.add_gate("DETECTOR", expr="0", label="constant")
    c.add_gate("LOGICAL_OBSERVABLE")
    measurements, det = Program(c).simulate(shots=4)
    assert measurements == [[]]
    assert det["detectors"][0]["label"] == "constant"
    np.testing.assert_array_equal(det["detectors"][0]["data"], np.zeros(3))
    np.testing.assert_array_equal(det["logicals"][0]["data"], np.zeros(3))


def test_frame_mode_detector_without_measurements_is_rejected():
    c = Circuit(1, 3)
    c.add_gate("H", 0)
    c.add_gate("DETECTOR", expr="rec[-1]")
    with pytest.raises(ValueError, match="no measurement"):
        Program(c).simulate(shots=4)


def test_results_to_array_without_measurements():
    ref = Program._results_to_array([[], [], []])
    assert ref.shape == (3, 0)
    ir, _, info = Program._build_ir([Circuit(3, 5)], 2)
    frame, det = simulate_frame(ir, ref, 3, 5, 2, None, info)
    assert frame.shape == (3, 0, 2)
    assert det.detection_events.shape == (0, 2)


# --------------------------------------------------------------------------
# Compilation of the frame kernels


def _noisy_circuit():
    c = Circuit(3, 5)
    c.add_gate("RESET", [0, 1, 2])
    c.add_gate("H", 0)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("N1", [0, 1, 2], prob=0.3, noise_channel="d")
    c.add_gate("N1", 1, prob=1.0, noise_channel="f")
    c.add_gate("N2", 1, 2, prob=0.2)
    c.add_gate("MUL", 2, a=2)
    c.add_gate("SWAP", 0, 2)
    c.add_gate("M_X", 0)
    c.add_gate("M", [1, 2])
    c.add_gate("DETECTOR", expr="rec[-1] - rec[-2]")
    return c


def _check_frame_kernel_signatures():
    """Runs both frame paths, then checks that each kernel was compiled once, for plain types."""
    c = _noisy_circuit()
    np.random.seed(0)
    Program(c).simulate(shots=50)  # sampled noise
    prog = Program(c)
    prog._tableau_noise_enabled = False
    prog.simulate(shots=1)
    ref = prog._results_to_array(prog.measurement_results)
    ir, noise, info = Program._build_ir([c], 7)
    _run_frame(ir, ref, c.num_qudits, c.dimension, 7, noise, info, None)  # injected noise

    signatures = program_module._frame_ops.signatures
    assert len(signatures) == 1, f"_frame_ops was compiled {len(signatures)} times: {signatures}"
    for name, obj in vars(program_module).items():
        if isinstance(obj, CPUDispatcher) and obj.py_func.__module__ == program_module.__name__:
            for signature in obj.signatures:
                literals = [str(t) for t in signature if isinstance(t, types.Literal)]
                assert not literals, f"{name} was compiled for literal arguments {literals}"


def test_frame_kernels_compile_once_without_literal_specializations(tmp_path):
    """
    Numba compiles a separate copy of a kernel for every literal argument it is called with.
    The frame kernel used to be compiled twice for the sampled-noise path alone (and a third
    time for injected noise), which made the first simulation in a fresh environment slow.

    The check runs in a fresh interpreter with an empty numba cache: kernels loaded from a
    warm cache can hide the extra compilations, and this process may have run them already.
    """
    sdim_root = os.path.dirname(os.path.dirname(os.path.abspath(program_module.__file__)))
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "numba_cache")
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(os.path.abspath(__file__)), sdim_root] + [p for p in [env.get("PYTHONPATH")] if p])
    script = ("import sdim.program, test_simulator_plumbing as t\n"
              f"assert sdim.program.__file__ == {program_module.__file__!r}, sdim.program.__file__\n"
              "t._check_frame_kernel_signatures()\n")
    result = subprocess.run([sys.executable, "-c", script], env=env, cwd=str(tmp_path),
                            capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
