"""
Regression tests for small fixes: NumPy integers in mechanisms added to a DetectorErrorModel after
it was built, negative qudit indices in DetectorErrorModel.from_circuit and in .chp files, qudit
indices stored by Circuit.add_gate, Circuit.from_operation_list, random circuits with SWAP and N2,
and in-place circuit repetition.
"""

import math
import random

import numpy as np
import pytest

import sdim.dem as dem_module
from sdim.circuit import Circuit, CircuitInstruction
from sdim.circuit_io import read_circuit, write_circuit
from sdim.dem import DetectorErrorModel, ErrorMechanism, compile_unit_responses
from sdim.program import Program
from sdim.random_circuit import generate_random_clifford_circuit


def _plain_ints(mechanisms):
    """Whether every target and coefficient of the mechanisms is a Python int."""
    return all(type(x) is int for m in mechanisms for g in m.generators for x in (*g, *g.values()))


# --------------------------------------------------------------------------
# NumPy integers in mechanisms added after construction


def _appended(entry):
    """A model at d = 251 with 3 detectors and 1 observable, its mechanisms appended after construction."""
    dem = DetectorErrorModel(251, 3, 1)
    dem.mechanisms += [
        ErrorMechanism(0.1, [{0: 2, 1: entry(200)}], "a"),        # 200 * 126 (2**-1 mod 251) overflows a uint8
        ErrorMechanism(0.2, [{entry(0): entry(1), 1: 100}], "b"),  # the same line as a
        ErrorMechanism(0.3, [{entry(3): entry(3), 2: entry(250)}], "c"),
        ErrorMechanism(0.4, [{0: entry(1)}, {entry(3): entry(5)}], "d"),
    ]
    return dem


@pytest.mark.parametrize("entry", [np.uint8, np.int16, np.uint64, np.int64])
def test_appended_numpy_entries_are_read_as_python_ints(entry):
    """The constructor converts NumPy integers, but merge_lines read appended ones as they were: under NumPy 2 a
    uint8 after a Python int overflowed (with only a warning), and under NumPy 1.x a uint64 became a float
    (TypeError)."""
    expected = _appended(int)
    dem = _appended(entry)
    assert str(dem) == str(expected)
    assert _plain_ints(dem.mechanisms)
    for model in (dem, _appended(entry)):
        model.merge_lines()
        assert [(m.generators, m.source) for m in model.mechanisms] == [
            ([{0: 1, 1: 100}], "a+b"), ([{2: 1, 3: 248}], "c"), ([{0: 1}, {3: 5}], "d")]
        assert _plain_ints(model.mechanisms[:2])
    lines = _appended(entry).to_lines()
    assert lines == expected.to_lines() and _plain_ints(lines.mechanisms)
    for a, b in zip(_appended(entry).sample(300, seed=4), expected.sample(300, seed=4)):
        np.testing.assert_array_equal(a, b)


def test_appended_entries_are_written_as_python_ints(tmp_path):
    """Under NumPy 1.x a uint64 observable target minus num_detectors is a float, written as L0.0, and a bool
    was written as True; read_from_file refused both lines."""
    dem = DetectorErrorModel(7, 1, 1)
    dem.mechanisms += [ErrorMechanism(0.25, [{np.uint64(1): np.uint64(3), np.int8(0): np.int8(-2)}], "x"),
                       ErrorMechanism(0.5, [{True: True}], "y")]
    assert str(dem).splitlines()[-2:] == ["ERROR(0.25) D0=-2 L0=3 # x", "ERROR(0.5) L0=1 # y"]
    dem.write_to_file(tmp_path / "model.qdem")
    back = DetectorErrorModel.read_from_file(tmp_path / "model.qdem")
    assert [m.generators for m in back.mechanisms] == [[{0: 5, 1: 3}], [{1: 1}]]


def test_to_lines_reads_appended_numpy_entries_above_2_31():
    """Above 2**31 - 1, to_lines expands one dict at a time: an int8 raised OverflowError (NumPy 2) and a
    uint64 became a float (NumPy 1.x)."""
    d = 2147483659
    assert dem_module._is_prime(d)
    dem = DetectorErrorModel(d, 2, 1)
    dem.mechanisms.append(ErrorMechanism(0.1, [{np.uint64(2): np.uint64(3), 1: np.int8(-5)}], "a"))
    lines = dem.to_lines()
    assert [(m.generators, m.source) for m in lines.mechanisms] == [([{1: 1, 2: 3 * pow(-5, -1, d) % d}], "a")]
    assert _plain_ints(lines.mechanisms)


def _canonical_line_reference(gen, d):
    """_canonical_line as it was: reduce, drop zeros and sort, then scale by the inverse of the first entry."""
    items = sorted((t, v % d) for t, v in gen.items() if v % d)
    if not items or math.gcd(items[0][1], d) != 1:
        return None, dict(items)
    inv = pow(items[0][1], -1, d)
    scaled = {t: (v * inv) % d for t, v in items}
    return tuple(scaled.items()), scaled


@pytest.mark.parametrize("d", [2, 3, 4, 6, 9, 251, 2 ** 61 - 1])
def test_one_pass_line_scaling_matches_reducing_first(d):
    """merge_lines scales a line whose first entry is a unit in one pass; zeros, negative and unreduced values,
    unsorted targets and non-unit leads (composite d) give what reducing, sorting and scaling in turn gives."""
    rng = random.Random(d)
    for _ in range(2000):
        targets = rng.sample(range(12), rng.randint(0, 5))
        gen = {t: rng.choice([0, d, -d, 1, -1, 2, d - 1, d + 2, 3 * d + 1, rng.randrange(-3 * d, 3 * d)])
               for t in targets}
        key, scaled = dem_module._canonical_line(gen, d)
        expected_key, expected = _canonical_line_reference(gen, d)
        assert key == expected_key and list(scaled.items()) == list(expected.items()), (gen, d)


# --------------------------------------------------------------------------
# Negative qudit indices in from_circuit


def _negative_pair(d, seed):
    """
    The same noisy circuit on 4 qudits twice, once with indices 0 .. 3 and once with about half of them
    written as index - 4. The gates keep every measurement deterministic, so every detector is.
    """
    rng = random.Random(seed)
    plain, negative = Circuit(4, d), Circuit(4, d)

    def neg(q):
        return q - 4 if rng.random() < 0.5 else q

    def add(gate, qudits, **params):
        plain.add_gate(gate, *qudits, **params)
        negative.add_gate(gate, *[neg(q) for q in qudits], **params)

    for q in range(4):
        add("RESET", [q])
    for _ in range(40):
        r = rng.random()
        if r < 0.3:
            add(rng.choice(["X", "X_INV", "Z", "P", "P_INV"]), [rng.randrange(4)])
        elif r < 0.6:
            add(rng.choice(["CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"]), rng.sample(range(4), 2))
        elif r < 0.65:
            add("MUL", [rng.randrange(4)], a=rng.randrange(1, d))
        elif r < 0.85:
            add("N1", [rng.randrange(4)], noise_channel=rng.choice("dfp"), prob=0.1)
        else:
            add("N2", rng.sample(range(4), 2), prob=0.1)
    for q in range(4):
        add("M", [q])
    for c in (plain, negative):
        c.add_gate("DETECTOR", expr="rec[-4] - rec[-2]")
        c.add_gate("DETECTOR", expr="rec[-3]")
        c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-1] + 2*rec[-2]")
    return plain, negative


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("seed", range(4))
def test_from_circuit_counts_negative_noise_indices_from_the_end(d, seed):
    """N1 and N2 read their qudits from the circuit as they were, so N1 on -4 of 4 qudits raised IndexError."""
    plain, negative = _negative_pair(d, seed)
    assert any(op.name in ("N1", "N2") and min(op.qudit_index, op.target_index or 0) < 0 for op in negative.operations)
    dem = DetectorErrorModel.from_circuit(plain)
    assert dem.mechanisms
    assert str(DetectorErrorModel.from_circuit(negative)) == str(dem)
    assert str(DetectorErrorModel.from_circuit(negative, merge=False)) == str(DetectorErrorModel.from_circuit(
        plain, merge=False))
    assert compile_unit_responses(negative).locations == compile_unit_responses(plain).locations


@pytest.mark.parametrize("gate, qudits, error, message", [
    ("CNOT", (0, -5), IndexError, "qudit -5"),
    ("CZ", (-5, 0), IndexError, "qudit -5"),
    ("N1", (-5,), IndexError, "qudit -5"),
    ("N2", (0, -5), IndexError, "qudit -5"),
    ("N2", (-1, 3), ValueError, "twice"),
    ("N2", (0, -4), ValueError, "twice"),
    ("CNOT", (-1, 3), ValueError, "twice"),
    ("X", (-5,), IndexError, "qudit -5"),
    ("X_INV", (4,), IndexError, "qudit 4"),
    ("Z", (4,), IndexError, "qudit 4"),
    ("Z_INV", (-5,), IndexError, "qudit -5"),
    ("I", (-5,), IndexError, "qudit -5"),
    ("I", (4,), IndexError, "qudit 4"),
])
def test_from_circuit_rejects_what_the_frame_sampler_rejects(gate, qudits, error, message):
    """CNOT 0 -5 on 4 qudits compiled without the CNOT, N2 on -1 and 3 (one qudit) raised IndexError, and the
    Paulis and the identity, which change no frame, were not checked at all."""
    c = Circuit(4, 3)
    c.add_gate("N1", 0, noise_channel="f", prob=0.1)
    c.add_gate(gate, *qudits, **({"prob": 0.1} if gate.startswith("N") else {}))
    c.add_gate("M", [0, 1, 2, 3])
    c.add_gate("DETECTOR", expr="rec[-4]")
    with pytest.raises(error):
        Program(c).simulate(shots=3)
    with pytest.raises(error, match=message):
        DetectorErrorModel.from_circuit(c)


# --------------------------------------------------------------------------
# Qudit indices in add_gate, from_operation_list and .chp files


def test_add_gate_stores_python_int_indices():
    c = Circuit(4, 3)
    c.add_gate("X", np.uint64(1))
    c.add_gate("CNOT", np.int8(-1), np.array([0, 1], dtype=np.uint64))
    c.add_gate("M", [np.int16(2), np.uint8(3), True])
    c.add_gate("CZ", range(2), (np.int64(2), 3))
    assert [(op.qudit_index, op.target_index) for op in c.operations] == [
        (1, None), (-1, 0), (-1, 1), (2, None), (3, None), (1, None), (0, 2), (1, 3)]
    assert all(type(q) is int for op in c.operations for q in (op.qudit_index, op.target_index) if q is not None)


@pytest.mark.parametrize("d", [3, 4])
def test_narrow_numpy_index_on_many_qudits(d):
    """Under NumPy 2 an int8 index overflowed when the simulators counted -50 back from 200 qudits, and a uint8
    one gave outcome 0 at d = 4."""
    results = []
    for index in (150, np.int8(-50), np.uint8(150)):
        c = Circuit(200, d)
        c.add_gate("X", 150)
        c.add_gate("M", index)
        random.seed(1)
        np.random.seed(1)
        shot = [(r.qudit_index, r.measurement_value) for r in Program(c).simulate()]
        frame = Program(c).simulate(shots=3)[0][150][0]
        results.append((shot, [(r.qudit_index, r.measurement_value) for r in frame]))
    assert results[0] == results[1] == results[2] == ([(150, 1)], [(150, 1)] * 3)


@pytest.mark.parametrize("index", [np.uint64(1), [np.uint64(1)], np.array([1], dtype=np.uint64)])
def test_uint64_index_measures_a_composite_dimension(index):
    """Under NumPy 1.x the Weyl tableau turned a uint64 index into a float and raised IndexError."""
    c = Circuit(3, 6)
    c.add_gate("X", 1)
    c.add_gate("M", index)
    [result] = Program(c).simulate()
    assert (result.qudit_index, result.measurement_value) == (1, 1) and type(result.qudit_index) is int


def test_qudit_lists_hold_integers():
    """None in a list was stored, which the frame sampler read as the last qudit and the tableau failed on; text,
    floats and arrays were stored too."""
    c = Circuit(4, 3)
    with pytest.raises(ValueError, match="control list holds None"):
        c.add_gate("M", [0, None])
    with pytest.raises(ValueError, match="target list holds None"):
        c.add_gate("CNOT", 0, [1, None])
    with pytest.raises(ValueError, match="control list holds None"):
        c.add_gate("M", (q for q in [0, None]))
    for qudits in ([1.0], "1", [np.float64(1)], [np.array([1, 2])], np.array([[0, 1], [2, 3]])):
        with pytest.raises(TypeError):
            c.add_gate("X", qudits)
    assert c.operations == []


def test_negative_indices_round_trip_through_a_file(tmp_path):
    """read_circuit kept only the tokens made of digits, so CNOT -1 0 came back as CNOT 0."""
    c = Circuit(5, 3)
    c.add_gate("CNOT", -1, 0)
    c.add_gate("N2", 1, -2, prob=0.25)
    c.add_gate("MUL", -3, a=2)
    c.add_gate("M", [-5, 4])
    c.add_gate("DETECTOR", expr="rec[-1] - rec[-2]")
    back = read_circuit(write_circuit(c, "negative.chp", directory=str(tmp_path)))
    assert back.num_qudits == 5
    assert [(op.name, op.qudit_index, op.target_index, op.params) for op in back.operations] == [
        (op.name, op.qudit_index, op.target_index, op.params) for op in c.operations]
    # Without qudits=, the circuit gets as many qudits as its most negative index counts back.
    (tmp_path / "old.chp").write_text("#\nd 3\nX -3\nM 1\n", encoding="utf-8")
    old = read_circuit(str(tmp_path / "old.chp"))
    assert old.num_qudits == 3 and [(op.name, op.qudit_index) for op in old.operations] == [("X", -3), ("M", 1)]


def test_negative_index_outside_the_declared_qudits_adds_no_qudits(tmp_path):
    """read_circuit grew the declared count to 5 for X -5, which moved X -1 to qudit 4 and made the circuit
    valid."""
    c = Circuit(3, 3)
    c.add_gate("X", -1)
    c.add_gate("X", -5)
    c.add_gate("M", [0, 1, 2])
    back = read_circuit(write_circuit(c, "outside.chp", directory=str(tmp_path)))
    assert back.num_qudits == 3
    assert [(op.name, op.qudit_index) for op in back.operations] == [(op.name, op.qudit_index) for op in c.operations]
    for circuit in (c, back):
        with pytest.raises(IndexError, match="qudit -5"):
            Program(circuit).simulate()


def test_from_operation_list_keeps_parameters():
    """Instructions lost their parameters: MUL raised for its missing scalar, and noise gates took prob=0.01."""
    c = Circuit(3, 5)
    c.add_gate("X", 0)
    c.add_gate("MUL", 0, a=2)
    c.add_gate("N1", 1, noise_channel="f", prob=0.3)
    c.add_gate("N2", 1, 2, prob=0.2)
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="rec[-1]", label="d")
    copied = Circuit.from_operation_list(c.operations, 3, 5)
    assert [(op.name, op.qudit_index, op.target_index, op.params) for op in copied.operations] == [
        (op.name, op.qudit_index, op.target_index, op.params) for op in c.operations]
    assert Program(copied).simulate()[0].measurement_value == 2
    bare = CircuitInstruction(c.gate_data, "H", 0)
    assert [(op.name, op.params) for op in Circuit.from_operation_list([bare], 1, 5).operations] == [("H", {})]


# --------------------------------------------------------------------------
# Random circuits and circuit repetition


def test_random_circuits_place_swap_and_n2_on_two_qudits():
    """SWAP and N2 in gate_set were given one qudit, which add_gate refuses."""
    c = generate_random_clifford_circuit(4, 200, 3, seed=2, gate_set=["SWAP", "N2", "cx", "H", "N1"])
    arity = {op.name: op.target_index is not None for op in c.operations}
    assert arity == {"SWAP": True, "N2": True, "CNOT": True, "H": False, "N1": False}
    assert all(op.qudit_index != op.target_index for op in c.operations)
    # The default gate set draws the same circuit as before.
    random.seed(9)
    expected = []
    for _ in range(50):
        gate = random.choice(["H", "P", "CNOT", "X", "Z", "H_INV", "P_INV", "CNOT_INV", "X_INV", "Z_INV", "CZ",
                              "CZ_INV"])
        if gate in ("CNOT", "CNOT_INV", "CZ", "CZ_INV"):
            expected.append((gate, *random.sample(range(4), 2)))
        else:
            expected.append((gate, random.randint(0, 3), None))
    c = generate_random_clifford_circuit(4, 50, 3, seed=9)
    assert [(op.name, op.qudit_index, op.target_index) for op in c.operations] == expected


@pytest.mark.parametrize("repetitions", [0, -2])
def test_repeating_in_place_at_most_zero_times_empties_the_circuit(repetitions):
    c = Circuit(2, 3)
    c.add_gate("X", 0)
    c.add_gate("M", [0, 1])
    same, operations = c, c.operations
    c *= repetitions
    assert c is same and c.operations is operations and c.operations == []
    assert (c.num_qudits, c.dimension) == (2, 3)
    assert Program(c).simulate() == []
    c.add_gate("M", 1)
    assert [r.measurement_value for r in Program(c).simulate()] == [0]
