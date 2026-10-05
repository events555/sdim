"""
Regression tests for negative qudit indices, which count from the end of the program's qudits.

Every simulation mode has to agree with the same circuit written with non-negative indices, for
prime and composite dimensions, and indices outside the program raise a clear error.  Also checks
that show_measurement prints each tableau shot once.
"""

import random

import numpy as np
import pytest
from sympy import isprime

from sdim.circuit import Circuit
from sdim.dem import DetectorErrorModel
from sdim.program import Program
from sdim.tableau.tableau_composite import WeylTableau
from sdim.tableau.tableau_prime import ExtendedTableau

DIMENSIONS = [2, 3, 5, 4, 6, 9]
ONE_QUDIT = ["I", "X", "X_INV", "Z", "Z_INV", "H", "H_INV", "P", "P_INV", "M", "M_X", "RESET"]
TWO_QUDIT = ["CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"]


def _tableau(n, d):
    """|0...0> in the tableau class Program picks for d."""
    return ExtendedTableau(n, d) if isprime(d) else WeylTableau(n, d)


def _plus_tableau(n, d):
    """Qudit 0 in |+>, so the program does not start in a computational basis state."""
    t = _tableau(n, d)
    t.hadamard(0)
    return t


def _circuit_pair(d, n, seed, width=None):
    """
    The same random circuit on n qudits twice: once with indices 0 .. width - 1, and once with
    about half of them written as index - width.  width defaults to n.
    """
    width = width or n
    rng = random.Random(seed)
    plain, negative = Circuit(n, d), Circuit(n, d)

    def neg(q):
        return q - width if rng.random() < 0.5 else q

    units = [a for a in range(1, d) if np.gcd(a, d) == 1]
    measured = 0
    for _ in range(30):
        r = rng.random()
        if r < 0.45:
            gate, q = rng.choice(ONE_QUDIT), rng.randrange(width)
            plain.add_gate(gate, q)
            negative.add_gate(gate, neg(q))
            measured += gate in ("M", "M_X")
        elif r < 0.75:
            gate, (a, b) = rng.choice(TWO_QUDIT), rng.sample(range(width), 2)
            plain.add_gate(gate, a, b)
            negative.add_gate(gate, neg(a), neg(b))
        elif r < 0.82:
            a, q = rng.choice(units), rng.randrange(width)
            plain.add_gate("MUL", q, a=a)
            negative.add_gate("MUL", neg(q), a=a)
        elif r < 0.9:
            channel, q = rng.choice("dfp"), rng.randrange(width)
            plain.add_gate("N1", q, noise_channel=channel, prob=0.3)
            negative.add_gate("N1", neg(q), noise_channel=channel, prob=0.3)
        elif r < 0.95:
            a, b = rng.sample(range(width), 2)
            plain.add_gate("N2", a, b, prob=0.3)
            negative.add_gate("N2", neg(a), neg(b), prob=0.3)
        elif measured:
            for c in (plain, negative):
                c.add_gate("DETECTOR", expr=f"rec[-{rng.randrange(1, measured + 1)}]")
                c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-1] + 2*rec[0]")
    plain.add_gate("M", list(range(width)))
    negative.add_gate("M", [q - width for q in range(width)])
    return plain, negative


def _seeded(seed, run):
    random.seed(seed)
    np.random.seed(seed)
    return run()


def _flat(results):
    return [(r.qudit_index, r.measurement_value, r.deterministic) for r in results]


def _grid(results):
    return [[[(r.qudit_index, r.measurement_value, r.deterministic) for r in shots] for shots in rounds]
            for rounds in results]


def _detectors(det):
    return [(entry["label"], np.asarray(entry["data"]).tolist()) for key in ("detectors", "logicals")
            for entry in det[key]]


def _every_mode(make_program, seed):
    """The results of shots=1, force_tableau and frame mode (or the custom-tableau sampling)."""
    measurements, det = _seeded(seed, lambda: make_program().simulate(shots=8))
    return (_flat(_seeded(seed, lambda: make_program().simulate())),
            _grid(_seeded(seed, lambda: make_program().simulate(shots=3, force_tableau=True))),
            _grid(measurements), _detectors(det))


# --------------------------------------------------------------------------
# Negative indices give the same results as non-negative ones


@pytest.mark.parametrize("d", DIMENSIONS)
def test_measuring_a_negative_index_measures_that_qudit(d):
    """For composite d, M on qudit -2 measured qudit 2 in the X basis: random outcomes instead of 0."""
    c = Circuit(4, d)
    c.add_gate("H_INV", 0)
    c.add_gate("M", -2)
    expected = (2, 0, True)
    assert _flat(Program(c).simulate()) == [expected]
    assert _grid(Program(c).simulate(shots=6, force_tableau=True)) == [[], [], [[expected] * 6], []]
    assert _grid(Program(c).simulate(shots=6)[0]) == [[], [], [[expected] * 6], []]
    custom = Program(c, tableau=_plus_tableau(4, d)).simulate(shots=6)[0]
    assert _grid(custom) == [[], [], [[expected] * 6], []]


@pytest.mark.parametrize("d", DIMENSIONS)
@pytest.mark.parametrize("seed", range(3))
def test_every_mode_matches_the_circuit_with_non_negative_indices(d, seed):
    """Every gate, noise, MUL, M, M_X, RESET and detectors, seed for seed."""
    plain, negative = _circuit_pair(d, 4, seed)
    assert _every_mode(lambda: Program(negative), seed) == _every_mode(lambda: Program(plain), seed)
    custom = [_every_mode(lambda: Program(c, tableau=_plus_tableau(4, d)), seed) for c in (negative, plain)]
    assert custom[0] == custom[1]


@pytest.mark.parametrize("d", DIMENSIONS)
@pytest.mark.parametrize("seed", range(2))
def test_negative_indices_count_from_the_end_of_a_widened_program(d, seed):
    """In a circuit followed by a wider one, -1 is the last qudit of the wider circuit."""
    plain, negative = _circuit_pair(d, 2, seed, width=4)
    second = Circuit(4, d)
    second.add_gate("CNOT", 0, 3)
    second.add_gate("M", [0, 1, 2, 3])
    second.add_gate("DETECTOR", expr="rec[-1] - rec[-4]")

    def program(first):
        p = Program(first)
        p.append_circuit(second)
        return p

    assert _every_mode(lambda: program(negative), seed) == _every_mode(lambda: program(plain), seed)


@pytest.mark.parametrize("d", DIMENSIONS)
def test_negative_indices_count_from_the_end_of_a_wider_custom_tableau(d):
    """The frame sampler's IR has to count back from the tableau's qudits, as the tableau does."""
    results = []
    for a, b in ((3, 1), (-1, -3)):
        c = Circuit(2, d)
        c.add_gate("X", a)
        c.add_gate("N1", b, noise_channel="f", prob=0.5)
        c.add_gate("CNOT", a, b)
        c.add_gate("M", [b, a])
        c.add_gate("DETECTOR", expr="rec[-2] - 1")
        results.append(_every_mode(lambda: Program(c, tableau=_tableau(4, d)), 7))
    assert results[0] == results[1]
    frame = results[0][2]
    assert [len(rounds) for rounds in frame] == [0, 1, 0, 1] and len(frame[1][0]) == len(frame[3][0]) == 8


@pytest.mark.parametrize("d", [4, 6, 9])
def test_weyl_measure_z_takes_negative_indices_like_the_gates(d):
    t = WeylTableau(4, d)
    t.hadamard_inv(0)
    for _ in range(5):
        result = t.measure_z(-2)
        assert (result.measurement_value, result.deterministic) == (0, True)
    with pytest.raises(IndexError, match="out of range"):
        t.measure_z(-5)


def test_dem_reads_a_negative_two_qudit_target():
    """The DEM took the -1 of CNOT 0 -1 for a missing target and left the CNOT out."""
    models = []
    for target in (2, -1):
        c = Circuit(3, 3)
        c.add_gate("N1", 0, noise_channel="f", prob=0.1)
        c.add_gate("CNOT", 0, target)
        c.add_gate("M", [0, 1, 2])
        c.add_gate("DETECTOR", expr="rec[-1]")
        c.add_gate("DETECTOR", expr="rec[-3]")
        models.append(str(DetectorErrorModel.from_circuit(c)))
    assert models[0] == models[1]


def test_build_ir_counts_negative_indices_back_from_the_program():
    c = Circuit(3, 5)
    c.add_gate("CNOT", -1, 0)
    c.add_gate("M", -3)
    c.add_gate("X", -4)
    c.add_gate("DETECTOR", expr="rec[-1]")
    # -4 is outside 3 qudits, so it is left as it is.
    ir, _, _ = Program._build_ir([c], 1)
    assert ir[["qudit_index", "target_index"]].tolist() == [(2, 0), (0, -1), (-4, -1), (-1, -1)]
    ir, _, _ = Program._build_ir([c], 1, num_qudits=5)
    assert ir[["qudit_index", "target_index"]].tolist() == [(4, 0), (2, -1), (1, -1), (-1, -1)]


# --------------------------------------------------------------------------
# Indices outside the program


@pytest.mark.parametrize("d", [3, 4])
@pytest.mark.parametrize("gate,qudits,params", [
    ("M", (4,), {}),
    ("M", (-5,), {}),
    ("RESET", (-5,), {}),
    ("M_X", (-8,), {}),
    ("I", (-5,), {}),
    ("CNOT", (0, -5), {}),
    ("CZ", (4, 0), {}),
    ("N1", (-5,), {"prob": 0.0}),
    ("N2", (0, -5), {"prob": 0.0}),
])
def test_index_outside_the_program_raises_in_every_mode(d, gate, qudits, params):
    """They raised numpy's or list.remove's errors, or nothing while noise did not fire."""
    c = Circuit(4, d)
    c.add_gate(gate, *qudits, **params)
    c.add_gate("M", 0)
    for options in ({}, {"shots": 3, "force_tableau": True}, {"shots": 3}):
        with pytest.raises(IndexError, match=rf"{gate} acts on qudit -?\d, but the program has 4 qudits"):
            Program(c).simulate(**options)


@pytest.mark.parametrize("d", [3, 4])
@pytest.mark.parametrize("gate", TWO_QUDIT + ["N2"])
def test_two_qudit_gate_on_one_qudit_raises_in_every_mode(d, gate):
    """add_gate cannot see that -1 and 3 are one qudit; the composite tableau became inconsistent."""
    c = Circuit(4, d)
    c.add_gate(gate, -1, 3)
    c.add_gate("M", 0)
    for options in ({}, {"shots": 3, "force_tableau": True}, {"shots": 3}):
        with pytest.raises(ValueError, match=f"{gate} needs two different qudits"):
            Program(c).simulate(**options)


# --------------------------------------------------------------------------
# Printing the measurements


def test_show_measurement_prints_each_tableau_shot_once(capsys):
    """Each shot reprinted every shot before it."""
    c = Circuit(2, 5)
    c.add_gate("H", 0)
    c.add_gate("M", [1, 0, 0])
    random.seed(2)
    results = Program(c).simulate(shots=3, force_tableau=True, show_measurement=True)
    expected = "".join(
        f"Measurement results for shot {shot + 1}:\n"
        + "".join(f"{shots[shot]}\n" for rounds in results for shots in rounds)
        for shot in range(3)
    )
    assert capsys.readouterr().out == expected
