"""Regression tests for the tableau and frame simulators at large qudit dimensions."""

import copy
import random
import time

import numpy as np
import pytest

from sdim.circuit import Circuit
from sdim.program import Program
from sdim.tableau.tableau_gates import apply_X, apply_Z, _apply_pauli_powers


ONE_QUDIT = ["H", "H_INV", "P", "P_INV", "X", "Z"]
TWO_QUDIT = ["CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"]
TABLEAU_ARRAYS = ("x_block", "z_block", "phase_vector", "destab_x_block", "destab_z_block", "destab_phase_vector")


def _random_circuit(d, n, depth, seed, measure_and_reset=True):
    rng = random.Random(seed)
    c = Circuit(n, d)
    for _ in range(depth):
        r = rng.random()
        if r < 0.5:
            a, b = rng.sample(range(n), 2)
            c.add_gate(rng.choice(TWO_QUDIT), a, b)
        elif r < 0.8 or not measure_and_reset:
            c.add_gate(rng.choice(ONE_QUDIT), rng.randrange(n))
        elif r < 0.85:
            c.add_gate("MUL", rng.randrange(n), a=rng.randrange(1, d))
        elif r < 0.95:
            c.add_gate("M", rng.randrange(n))
        else:
            c.add_gate("RESET", rng.randrange(n))
    c.add_gate("M", list(range(n)))
    return c


def _measure(circuit, seed, exact_integers=False):
    random.seed(seed)
    np.random.seed(seed)
    program = Program(circuit)
    if exact_integers:
        # Python integers can't overflow, so this run is the reference answer.
        for name in TABLEAU_ARRAYS:
            setattr(program.initial_tableau, name, getattr(program.initial_tableau, name).astype(object))
    return [int(r.measurement_value) for r in program.simulate(shots=1)]


@pytest.mark.parametrize("d", [3, 1000003, 2147483647])
def test_tableau_matches_exact_integer_reference(d):
    """Chained CNOTs used to push tableau entries past int64 at large d, which silently changed outcomes."""
    for seed in range(8):
        circuit = _random_circuit(d, 5, 70, seed)
        assert _measure(circuit, seed) == _measure(circuit, seed, exact_integers=True), seed


@pytest.mark.parametrize("d", [2, 3, 5, 7, 4, 6])
def test_pauli_powers_match_repeated_gates(d):
    """The constant-time X^a Z^b gives exactly the tableau of applying X a times and Z b times."""
    rng = random.Random(d)
    for trial in range(4):
        circuit = _random_circuit(d, 3, 30, 100 * d + trial, measure_and_reset=False)
        program = Program(circuit)
        for op in circuit.operations[:-3]:   # skip the final measurements
            program.apply_gate(op)
        start = program.stabilizer_tableau
        for x_exp in range(d):
            for z_exp in range(d):
                q = rng.randrange(3)
                slow, fast = copy.deepcopy(start), copy.deepcopy(start)
                for _ in range(x_exp):
                    apply_X(slow, q)
                for _ in range(z_exp):
                    apply_Z(slow, q)
                _apply_pauli_powers(fast, q, x_exp, z_exp)
                for name in TABLEAU_ARRAYS:
                    if getattr(slow, name, None) is not None:
                        np.testing.assert_array_equal(getattr(slow, name) % slow.order,
                                                      getattr(fast, name) % fast.order)


def test_tableau_noise_and_reset_are_fast_at_large_dimension():
    """Pauli powers and RESET corrections used to cost one gate call per unit of the exponent."""
    d = 1000003
    c = Circuit(2, d)
    c.add_gate("H", 0)
    c.add_gate("RESET", 0)
    c.add_gate("N1", 0, prob=1.0, noise_channel="d")
    c.add_gate("N2", 0, 1, prob=1.0)
    c.add_gate("M", [0, 1])
    start = time.time()
    Program(c).simulate(shots=20, force_tableau=True)
    assert time.time() - start < 5.0


@pytest.mark.parametrize("d,a", [(5, -1), (5, 7), (7, -3), (4, -1), (4, 7)])
def test_multiplication_scalar_is_reduced(d, a):
    """MUL with a negative scalar or one of at least d is the same gate as a mod d."""
    expected = (2 * a) % d
    c = Circuit(1, d)
    c.add_gate("X", 0)
    c.add_gate("X", 0)
    c.add_gate("MUL", 0, a=a)
    c.add_gate("M", 0)
    assert Program(c).simulate()[0].measurement_value == expected
    measurements, _ = Program(c).simulate(shots=4)
    assert all(r.measurement_value == expected for r in measurements[0][0])


def test_malformed_prob_dist_is_rejected():
    """A wrong-length N2 prob_dist used to act as identity in the frame sampler."""
    c = Circuit(2, 3)
    with pytest.raises(ValueError, match="prob_dist"):
        c.add_gate("N2", 0, 1, prob_dist=[1.0, 0.0])
    with pytest.raises(ValueError, match="sum to 1"):
        c.add_gate("N2", 0, 1, prob_dist=np.full(81, 0.5))
