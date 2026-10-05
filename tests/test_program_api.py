"""
Regression tests for the Program and Circuit API: programs that start from a custom tableau,
appended circuits on more or fewer qudits, circuit repetition, printing the stored measurements,
and NumPy integers as circuit sizes.
"""

import random

import numpy as np
import pytest

import sdim.program as program_module
from sdim.circuit import Circuit
from sdim.gatedata import GateData
from sdim.program import Program
from sdim.tableau.tableau_composite import WeylTableau
from sdim.tableau.tableau_gates import apply_X
from sdim.tableau.tableau_prime import ExtendedTableau

# (d, tableau class) pairs, as Program picks them
TABLEAUS = [(2, ExtendedTableau), (3, ExtendedTableau), (4, WeylTableau), (6, WeylTableau)]
ONE_QUDIT = ["X", "Z", "H", "H_INV", "P", "P_INV"]
TWO_QUDIT = ["CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"]


def _plus(tableau_class, d):
    """One qudit in |+> = H|0>, the 0 eigenstate of M_X."""
    t = tableau_class(1, d)
    t.hadamard(0)
    return t


def _values(rounds):
    """The outcomes of every shot of a (qudit -> round -> shot) result, round 0 of each qudit."""
    return [[r.measurement_value for r in per_qudit[0]] for per_qudit in rounds]


# --------------------------------------------------------------------------
# Frame mode with a custom initial tableau


@pytest.mark.parametrize("d,tableau_class", TABLEAUS)
def test_frame_mode_starts_from_the_custom_tableau(d, tableau_class):
    """The frame sampler assumed |0>, so M_X of |+> was random in the shots after the reference."""
    c = Circuit(1, d)
    c.add_gate("M_X", 0)
    random.seed(0)
    np.random.seed(0)
    measurements, det = Program(c, tableau=_plus(tableau_class, d)).simulate(shots=20)
    assert [(r.measurement_value, r.deterministic) for r in measurements[0][0]] == [(0, True)] * 20
    assert det == {"detectors": [], "logicals": []}


@pytest.mark.parametrize("d,tableau_class", TABLEAUS)
def test_custom_tableau_noise_and_detectors_take_the_frame_form(d, tableau_class):
    """A Z error on |+> changes its M_X outcome; qudit 1 stays in |+>."""
    t = tableau_class(2, d)
    t.hadamard(0)
    t.hadamard(1)
    c = Circuit(2, d)
    c.add_gate("N1", 0, noise_channel="p", prob=1.0)
    c.add_gate("M_X", [0, 1])
    c.add_gate("DETECTOR", expr="rec[-2]", label="flipped")
    c.add_gate("DETECTOR", expr="rec[-1]", label="quiet")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[0] + 2 * rec[1]", label="both")
    shots = 30
    random.seed(1)
    np.random.seed(1)
    measurements, det = Program(c, tableau=t).simulate(shots=shots)
    flipped, quiet = (np.array(v) for v in _values(measurements))
    assert len(flipped) == len(quiet) == shots
    assert flipped[0] == 0 and (flipped[1:] != 0).all()  # the reference shot is noiseless
    assert (quiet == 0).all()
    assert all(r.deterministic for per_qudit in measurements for r in per_qudit[0])
    assert [e["label"] for e in det["detectors"]] == ["flipped", "quiet"]
    assert [e["label"] for e in det["logicals"]] == ["both"]
    np.testing.assert_array_equal(det["detectors"][0]["data"], (flipped[1:] - flipped[0]) % d)
    np.testing.assert_array_equal(det["detectors"][1]["data"], np.zeros(shots - 1))
    np.testing.assert_array_equal(det["logicals"][0]["data"], flipped[1:] % d)

    measurements, (detectors, logicals) = Program(c, tableau=t).simulate(shots=shots, raw_detector_output=True)
    assert detectors.shape == (2, shots - 1) and logicals.shape == (1, shots - 1)
    np.testing.assert_array_equal(detectors[0], np.array(_values(measurements)[0][1:]))


@pytest.mark.parametrize("d,tableau_class", [(3, ExtendedTableau), (4, WeylTableau)])
def test_custom_tableau_detectors_skip_reset_rounds(d, tableau_class):
    """RESET adds a measurement round but no record, so rec[-2] is the M_X before it."""
    c = Circuit(1, d)
    c.add_gate("N1", 0, noise_channel="p", prob=1.0)
    c.add_gate("M_X", 0)
    c.add_gate("RESET", 0)
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-2]")
    c.add_gate("DETECTOR", expr="rec[-1]")
    random.seed(6)
    np.random.seed(6)
    measurements, (detectors, _) = Program(c, tableau=_plus(tableau_class, d)).simulate(shots=12, raw_detector_output=True)
    assert len(measurements[0]) == 3
    measured_x = np.array([r.measurement_value for r in measurements[0][0]])
    assert measured_x[0] == 0 and (measured_x[1:] != 0).all()
    np.testing.assert_array_equal(detectors[0], measured_x[1:])
    np.testing.assert_array_equal(detectors[1], np.zeros(11))


def test_custom_tableau_noise_has_the_tableau_statistics():
    """N1 with prob 1/2 on |+> gives a nonzero M_X outcome in half of the shots."""
    d = 5
    c = Circuit(1, d)
    c.add_gate("N1", 0, noise_channel="p", prob=0.5)
    c.add_gate("M_X", 0)
    random.seed(2)
    np.random.seed(2)
    measurements, _ = Program(c, tableau=_plus(ExtendedTableau, d)).simulate(shots=401)
    values = np.array(_values(measurements)[0])
    assert values[0] == 0
    assert 0.4 < np.mean(values[1:] != 0) < 0.6


def test_frame_sampler_still_runs_for_computational_basis_states(monkeypatch):
    calls = []
    run_frame = program_module._run_frame
    monkeypatch.setattr(program_module, "_run_frame", lambda *args: calls.append(1) or run_frame(*args))
    c = Circuit(2, 3)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("M", [0, 1])
    one = ExtendedTableau(2, 3)
    apply_X(one, 0)  # |10>
    assert _values(Program(c, tableau=one).simulate(shots=5)[0]) == [[1] * 5, [1] * 5]
    assert _values(Program(c).simulate(shots=5)[0]) == [[0] * 5, [0] * 5]
    assert len(calls) == 2

    # For even d the blocks are kept mod 2d, and x = d is the generator -Z, which stabilizes |d/2>.
    d = 4
    t = WeylTableau(1, d)
    t.x_block[0, 0] = d
    m = Circuit(1, d)
    m.add_gate("M", 0)
    assert _values(Program(m, tableau=t).simulate(shots=5)[0]) == [[d // 2] * 5]
    assert len(calls) == 3

    plus = ExtendedTableau(2, 3)
    plus.hadamard(0)
    Program(c, tableau=plus).simulate(shots=5)
    assert len(calls) == 3


def test_error_mechanisms_need_a_computational_basis_state():
    c = Circuit(1, 3)
    c.add_gate("N1", 0, noise_channel="f", prob=0.1)
    c.add_gate("M", 0)
    with pytest.raises(ValueError, match="computational basis state"):
        Program(c, tableau=_plus(ExtendedTableau, 3)).simulate(shots=4, building_error_mechanism=True)


# --------------------------------------------------------------------------
# Appended circuits


@pytest.mark.parametrize("d", [3, 4])
def test_appended_wider_circuit_runs_in_every_mode(d):
    """The tableau kept the first circuit's size, so every mode raised IndexError."""
    first = Circuit(1, d)
    first.add_gate("X", 0)
    second = Circuit(2, d)
    second.add_gate("CNOT", 0, 1)
    second.add_gate("M", [0, 1])
    program = Program(first)
    program.append_circuit(second)
    assert [(r.qudit_index, r.measurement_value) for r in program.simulate()] == [(0, 1), (1, 1)]
    assert _values(program.simulate(shots=3, force_tableau=True)) == [[1] * 3, [1] * 3]
    assert _values(program.simulate(shots=3)[0]) == [[1] * 3, [1] * 3]
    assert (first.num_qudits, second.num_qudits) == (1, 2)


def test_appended_narrower_and_equal_circuits_are_not_modified():
    """append_circuit used to widen the appended circuit to the program's size."""
    first = Circuit(3, 5)
    first.add_gate("X", 2)
    first.add_gate("M", 2)
    narrow = Circuit(1, 5)
    narrow.add_gate("X", 0)
    narrow.add_gate("M", 0)
    equal = Circuit(3, 5)
    equal.add_gate("CNOT", 2, 1)
    equal.add_gate("M", 1)
    program = Program(first)
    program.append_circuit(narrow)
    program.append_circuit(equal)
    assert [c.num_qudits for c in (first, narrow, equal)] == [3, 1, 3]
    assert [(r.qudit_index, r.measurement_value) for r in program.simulate()] == [(0, 1), (1, 1), (2, 1)]
    assert _values(program.simulate(shots=3, force_tableau=True)) == [[1] * 3] * 3
    assert _values(program.simulate(shots=3)[0]) == [[1] * 3] * 3


@pytest.mark.parametrize("widths", [(1, 2), (3, 2)])
def test_append_circuit_checks_the_dimension_before_changing_anything(widths):
    first, second = Circuit(widths[0], 3), Circuit(widths[1], 5)
    program = Program(first)
    with pytest.raises(ValueError, match="same dimension"):
        program.append_circuit(second)
    assert (first.num_qudits, second.num_qudits) == widths
    assert len(program.circuits) == 1 and program.initial_tableau.num_qudits == widths[0]


def test_detectors_read_records_across_a_wider_appended_circuit():
    first = Circuit(1, 3)
    first.add_gate("N1", 0, noise_channel="f", prob=1.0)
    first.add_gate("M", 0)
    second = Circuit(2, 3)
    second.add_gate("CNOT", 0, 1)
    second.add_gate("M", 1)
    second.add_gate("DETECTOR", expr="rec[-1] - rec[-2]", label="copy")
    second.add_gate("LOGICAL_OBSERVABLE", expr="rec[0]")
    program = Program(first)
    program.append_circuit(second)
    np.random.seed(3)
    measurements, det = program.simulate(shots=30)
    rec0, rec1 = (np.array(v[1:]) for v in _values(measurements))
    assert (rec0 != 0).all()
    np.testing.assert_array_equal(rec1, rec0)
    np.testing.assert_array_equal(det["detectors"][0]["data"], np.zeros(29))
    np.testing.assert_array_equal(det["logicals"][0]["data"], rec0)


def test_negative_indices_count_from_the_end_of_the_widened_program():
    """As for c1 + c2, X on qudit -1 of the first circuit lands on the appended circuit's last qudit."""
    first = Circuit(1, 3)
    first.add_gate("X", -1)
    second = Circuit(2, 3)
    second.add_gate("M", [0, 1])
    program = Program(first)
    program.append_circuit(second)
    assert [r.measurement_value for r in program.simulate()] == [0, 1]
    assert [r.measurement_value for r in Program(first + second).simulate()] == [0, 1]
    assert _values(program.simulate(shots=3, force_tableau=True)) == [[0] * 3, [1] * 3]
    assert _values(program.simulate(shots=3)[0]) == [[0] * 3, [1] * 3]


@pytest.mark.parametrize("d,tableau_class", TABLEAUS)
def test_wider_circuit_extends_a_custom_tableau_with_zero_qudits(d, tableau_class):
    t = _plus(tableau_class, d)
    second = Circuit(3, d)
    second.add_gate("M_X", 0)
    second.add_gate("X", 2)
    second.add_gate("M", [1, 2])
    program = Program(Circuit(1, d), tableau=t)
    program.append_circuit(second)
    expected = [(0, 0, True), (1, 0, True), (2, 1, True)]
    assert [(r.qudit_index, r.measurement_value, r.deterministic) for r in program.simulate()] == expected
    for measurements in (program.simulate(shots=4, force_tableau=True), program.simulate(shots=4)[0]):
        assert _values(measurements) == [[0] * 4, [0] * 4, [1] * 4]
    assert t.num_qudits == 1 and t.z_block.shape == t.x_block.shape == (1, 1)


@pytest.mark.parametrize("d", [3, 5, 4, 6])
def test_extended_tableau_is_the_tensor_product_with_zero_qudits(d):
    """
    Preparing a state on n qudits and appending a circuit on n + 2 qudits gives the same outcomes,
    seed for seed, as running both parts as one circuit on n + 2 qudits from |0...0>.
    """
    n, k = 2, 2
    rng = random.Random(d)
    for trial in range(5):
        prepare = Circuit(n, d)
        for _ in range(12):
            if rng.random() < 0.5:
                prepare.add_gate(rng.choice(TWO_QUDIT), *rng.sample(range(n), 2))
            else:
                prepare.add_gate(rng.choice(ONE_QUDIT), rng.randrange(n))
        rest = Circuit(n + k, d)
        for _ in range(12):
            if rng.random() < 0.6:
                rest.add_gate(rng.choice(TWO_QUDIT), *rng.sample(range(n + k), 2))
            else:
                rest.add_gate(rng.choice(ONE_QUDIT + ["M", "M_X"]), rng.randrange(n + k))
        rest.add_gate("M", list(range(n + k)))

        prepared = Program(prepare)
        for op in prepare.operations:
            prepared.apply_gate(op)
        program = Program(Circuit(n, d), tableau=prepared.stabilizer_tableau)
        program.append_circuit(rest)
        whole = Circuit(n + k, d)
        whole.operations = prepare.operations + rest.operations
        runs = []
        for p in (Program(whole), program):
            random.seed(trial)
            single = [(r.qudit_index, r.measurement_value, r.deterministic) for r in p.simulate()]
            random.seed(trial)
            shots = [[(r.measurement_value, r.deterministic) for r in rounds] for per_qudit in
                     p.simulate(shots=4, force_tableau=True) for rounds in per_qudit]
            runs.append((single, shots))
        assert runs[0] == runs[1]


# --------------------------------------------------------------------------
# Circuit repetition


def test_circuit_repetition_returns_a_new_circuit():
    """c * n used to extend c itself and return it."""
    c = Circuit(2, 3)
    c.add_gate("X", 0)
    c.add_gate("M", 0)
    operations = list(c.operations)
    tripled = c * 3
    assert tripled is not c and c.operations == operations
    assert tripled.operations == operations * 3
    assert (tripled.num_qudits, tripled.dimension) == (2, 3)
    assert (2 * c).operations == operations * 2
    assert (c * 0).operations == []
    assert [r.measurement_value for r in Program(tripled).simulate()] == [1, 2, 0]
    assert [r.measurement_value for r in Program(c).simulate()] == [1]

    assert (c * -1).operations == []
    same = c
    c *= 2
    assert c is same and c.operations == operations * 2


def test_circuit_repetition_keeps_the_gate_data():
    gate_data = GateData(3)
    gate_data.add_gate_alias("CNOT", ["MYCX"])
    c = Circuit(2, 3, gate_data=gate_data)
    c.add_gate("MYCX", 0, 1)
    for repeated in (c * 2, 2 * c):
        assert repeated.gate_data is gate_data
        repeated.add_gate("MYCX", 0, 1)
        assert len(repeated.operations) == 3
    assert len(c.operations) == 1


# --------------------------------------------------------------------------
# Printing the measurements


def test_show_measurement_prints_the_stored_shot_without_simulating_again(capsys):
    """print_measurements used to run the program again for one shot, replacing its results."""
    c = Circuit(1, 5)
    c.add_gate("H", 0)
    c.add_gate("M", 0)
    random.seed(4)
    quiet = Program(c).simulate(record_tableau=True)
    state = random.getstate()
    random.seed(4)
    program = Program(c)
    shown = program.simulate(record_tableau=True, show_measurement=True)
    assert random.getstate() == state
    assert [str(r) for r in shown] == [str(r) for r in quiet]
    assert shown[0].stabilizer_tableau is not None
    assert program.measurement_results[0][0][0] is shown[0]
    assert capsys.readouterr().out == f"Measurement results for shot 1:\n{shown[0]}\n"

    program.print_measurements()
    assert capsys.readouterr().out == f"{shown[0]}\n"
    assert program.measurement_results[0][0][0] is shown[0] and random.getstate() == state


def test_show_measurement_keeps_every_tableau_shot():
    c = Circuit(1, 5)
    c.add_gate("H", 0)
    c.add_gate("M", 0)
    random.seed(5)
    quiet = Program(c).simulate(shots=4, force_tableau=True)
    state = random.getstate()
    random.seed(5)
    shown = Program(c).simulate(shots=4, force_tableau=True, show_measurement=True)
    assert random.getstate() == state
    assert shown == quiet


# --------------------------------------------------------------------------
# Circuit sizes


@pytest.mark.parametrize("d", [5, 4])
def test_numpy_integer_sizes_become_python_ints(d):
    """A NumPy dimension broke MUL, which computes pow(a, -1, d)."""
    c = Circuit(np.int64(1), np.int64(d))
    assert type(c.num_qudits) is int and type(c.dimension) is int
    c.add_gate("X", 0)
    c.add_gate("MUL", 0, a=3)
    c.add_gate("M", 0)
    program = Program(c)
    tableau = program.stabilizer_tableau
    assert type(tableau.num_qudits) is int and type(tableau.dimension) is int
    expected = 3 % d
    assert program.simulate()[0].measurement_value == expected
    assert _values(program.simulate(shots=3, force_tableau=True)) == [[expected] * 3]
    assert _values(program.simulate(shots=3)[0]) == [[expected] * 3]
    assert type(Circuit(np.uint8(2), np.int32(7)).dimension) is int


@pytest.mark.parametrize("args,name", [((1.0, 3), "num_qudits"), ((2, 3.0), "dimension"),
                                       ((np.float64(2), 3), "num_qudits"), ((2, np.float32(5)), "dimension"),
                                       ((True, 3), "num_qudits"), ((np.True_, 3), "num_qudits")])
def test_float_and_bool_sizes_are_rejected(args, name):
    with pytest.raises(TypeError, match=f"^{name} must be an integer"):
        Circuit(*args)


@pytest.mark.parametrize("d,tableau_class", [(5, ExtendedTableau), (4, WeylTableau)])
def test_custom_tableau_numpy_sizes_become_python_ints(d, tableau_class):
    """A tableau built with NumPy sizes broke MUL in every mode."""
    t = tableau_class(np.int64(1), np.int64(d))
    t.hadamard(0)
    c = Circuit(1, d)
    c.add_gate("MUL", 0, a=3)
    c.add_gate("M_X", 0)
    program = Program(c, tableau=t)
    for tableau in (program.stabilizer_tableau, program.initial_tableau):
        assert type(tableau.num_qudits) is int and type(tableau.dimension) is int
    assert [(r.measurement_value, r.deterministic) for r in program.simulate()] == [(0, True)]
    assert _values(program.simulate(shots=3, force_tableau=True)) == [[0] * 3]
    assert _values(program.simulate(shots=3)[0]) == [[0] * 3]
    # The tableau already turned its NumPy sizes into Python ints when it was built.
    assert type(t.dimension) is int and type(t.num_qudits) is int
    # A tableau with Python int sizes is used as given.
    t = _plus(tableau_class, d)
    assert Program(c, tableau=t).stabilizer_tableau is t
    # Float sizes never worked (the first modulo raised UFuncTypeError); now they fail up front.
    with pytest.raises(TypeError, match="integer"):
        Program(c, tableau=tableau_class(1, float(d)))


@pytest.mark.parametrize("args", [(0, 3), (False, 3), (1, True), (1, 1), (1, 2 ** 31)])
def test_sizes_out_of_range_are_still_value_errors(args):
    with pytest.raises(ValueError):
        Circuit(*args)
