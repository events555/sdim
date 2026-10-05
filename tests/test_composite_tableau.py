"""
Regression tests for the composite-dimension tableau (WeylTableau).

The reference is an exact statevector simulation built from the gate matrices in sdim.unitary.
A WeylTableau generator (p, z, x) is the operator omega^-p tau^(x.z) X^x Z^z, with
tau = exp(i pi (d^2 + 1) / d), and the state is its +1 eigenstate.
"""

import copy
import glob
import math
import os
import random
import time
import warnings

import numpy as np
import pytest

from sdim.circuit import Circuit
from sdim.program import Program
from sdim.tableau.dataclasses import Tableau
from sdim.tableau import tableau_composite
from sdim.tableau.tableau_composite import WeylTableau
from sdim.unitary import (generate_cnot_matrix, generate_h_matrix, generate_m_matrix, generate_p_matrix,
                          generate_tau, generate_x_matrix, generate_z_matrix)

ONE_QUDIT = ["X", "X_INV", "Z", "Z_INV", "H", "H_INV", "P", "P_INV", "MUL"]
TWO_QUDIT = ["CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"]
MEASUREMENTS = ("M", "M_X", "RESET")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Exact statevector reference

def _one_qudit_matrix(name, d, a=None):
    if name in ("X", "X_INV"):
        U = generate_x_matrix(d)
    elif name in ("Z", "Z_INV"):
        U = generate_z_matrix(d)
    elif name in ("H", "H_INV"):
        U = generate_h_matrix(d)
    elif name in ("P", "P_INV"):
        U = generate_p_matrix(d)
    elif name == "MUL":
        return generate_m_matrix(d, a % d)
    elif name == "I":
        return np.eye(d)
    else:
        raise KeyError(name)
    return U.conj().T if name.endswith("_INV") else U


def _two_qudit_matrix(name, d):
    if name in ("CNOT", "CNOT_INV"):
        U = generate_cnot_matrix(d)
        U = U.conj().T if name == "CNOT_INV" else U
    elif name in ("CZ", "CZ_INV"):
        sign = 1 if name == "CZ" else -1
        U = np.diag([np.exp(2j * np.pi * sign * i * j / d) for i in range(d) for j in range(d)])
    elif name == "SWAP":
        U = np.zeros((d * d, d * d), dtype=complex)
        for i in range(d):
            for j in range(d):
                U[j * d + i, i * d + j] = 1
    else:
        raise KeyError(name)
    return U.reshape(d, d, d, d)


def _apply_one(psi, U, q):
    return np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)


def _apply_gate(psi, op, d):
    if op.name in TWO_QUDIT:
        U = _two_qudit_matrix(op.name, d)
        psi = np.tensordot(U, psi, axes=([2, 3], [op.qudit_index, op.target_index]))
        return np.moveaxis(psi, [0, 1], [op.qudit_index, op.target_index])
    a = int(op.params.get("a", op.params.get("scalar"))) if op.name == "MUL" else None
    return _apply_one(psi, _one_qudit_matrix(op.name, d, a), op.qudit_index)


def _replay(circuit, outcomes):
    """
    Replays the circuit exactly, forcing the given outcomes (in circuit order).

    Returns (the final state, a list of (outcome, support) per measurement).  An outcome with
    probability 0 is reported with an empty support and stops the replay.
    """
    d, n = circuit.dimension, circuit.num_qudits
    psi = np.zeros((d,) * n, dtype=complex)
    psi[(0,) * n] = 1
    seen = []
    k = 0
    for op in circuit.operations:
        if op.name not in MEASUREMENTS:
            psi = _apply_gate(psi, op, d)
            continue
        q, m = op.qudit_index, outcomes[k]
        k += 1
        phi = _apply_one(psi, _one_qudit_matrix("H_INV", d), q) if op.name == "M_X" else psi
        probs = np.array([np.sum(np.abs(np.take(phi, j, axis=q)) ** 2) for j in range(d)])
        support = np.flatnonzero(probs > 1e-9).tolist()
        if m not in support:
            seen.append((m, []))
            return None, seen
        seen.append((m, support))
        block = np.take(phi, m, axis=q)
        psi = np.zeros_like(phi)
        index = [slice(None)] * n
        index[q] = 0 if op.name == "RESET" else m
        psi[tuple(index)] = block / np.linalg.norm(block)
        if op.name == "M_X":
            psi = _apply_one(psi, _one_qudit_matrix("H", d), q)
    return psi, seen


def _outcomes_in_circuit_order(circuit, results):
    per_qudit = {}
    for r in results:
        per_qudit.setdefault(r.qudit_index, []).append(int(r.measurement_value))
    counts = {}
    out = []
    for op in circuit.operations:
        if op.name in MEASUREMENTS:
            q = op.qudit_index
            out.append(per_qudit[q][counts.get(q, 0)])
            counts[q] = counts.get(q, 0) + 1
    return out


def _generator_operator(tableau, j):
    d, n = tableau.dimension, tableau.num_qudits
    X, Z = generate_x_matrix(d), generate_z_matrix(d)
    zs = [int(v) for v in tableau.z_block[:, j]]
    xs = [int(v) for v in tableau.x_block[:, j]]
    op = np.array([[1.0 + 0j]])
    for q in range(n):
        op = np.kron(op, np.linalg.matrix_power(X, xs[q] % d) @ np.linalg.matrix_power(Z, zs[q] % d))
    exponent = (-2 * int(tableau.phase_vector[j]) + sum(a * b for a, b in zip(xs, zs))) % (2 * d)
    return generate_tau(d) ** exponent * op


def _assert_stabilizes(tableau, psi):
    v = psi.reshape(-1)
    for j in range(tableau.z_block.shape[1]):
        assert np.linalg.norm(_generator_operator(tableau, j) @ v - v) < 1e-6, j


def _check_against_statevector(circuit, seed):
    """One tableau shot: every outcome must be possible, and the final tableau must stabilize the state."""
    random.seed(seed)
    np.random.seed(seed)
    program = Program(circuit)
    outcomes = _outcomes_in_circuit_order(circuit, program.simulate(shots=1))
    psi, seen = _replay(circuit, outcomes)
    assert psi is not None, f"zero-probability outcome {seen[-1][0]} (outcomes so far {seen})"
    _assert_stabilizes(program.stabilizer_tableau, psi)
    return seen


def _unit_representative(d, rng):
    a = rng.randrange(1, d)
    while math.gcd(a, d) != 1:
        a = rng.randrange(1, d)
    return rng.choice([a, -a, a + d, a + 2 * d, a - 3 * d])


def _random_circuit(d, n, depth, rng, mid_measurements=3):
    c = Circuit(n, d)
    mids = 0
    for _ in range(depth):
        r = rng.random()
        if r < 0.04 and mids < mid_measurements:
            c.add_gate(rng.choice(MEASUREMENTS), rng.randrange(n))
            mids += 1
        elif r < 0.45:
            a, b = rng.sample(range(n), 2)
            c.add_gate(rng.choice(TWO_QUDIT), a, b)
        else:
            g = rng.choice(ONE_QUDIT)
            if g == "MUL":
                c.add_gate("MUL", rng.randrange(n), a=_unit_representative(d, rng))
            else:
                c.add_gate(g, rng.randrange(n))
    for q in range(n):
        c.add_gate(rng.choice(["M", "M_X"]), q)
    return c


# 1. modulo() reduced the blocks mod d instead of mod the order

@pytest.mark.parametrize("d", [4, 6, 8])
def test_periodic_modulo_keeps_even_dimension_signs(d):
    """
    H; P_INV leaves an X entry of 2d - 1.  Program reduces the tableau every 64 gates, and reducing
    that entry mod d flipped the generator's sign, so the final M gave d/2 instead of 0.
    """
    c = Circuit(1, d)
    c.add_gate("H", 0)
    c.add_gate("P_INV", 0)
    for _ in range(62):
        c.add_gate("I", 0)
    c.add_gate("P", 0)
    c.add_gate("H_INV", 0)
    c.add_gate("M", 0)
    assert [r.measurement_value for r in Program(c).simulate()] == [0]
    tableau_shots = Program(c).simulate(shots=10, force_tableau=True)
    assert {r.measurement_value for r in tableau_shots[0][0]} == {0}
    frame_shots, _ = Program(c).simulate(shots=10)    # the reference shot comes from the tableau
    assert {r.measurement_value for r in frame_shots[0][0]} == {0}


@pytest.mark.parametrize("d", [4, 6])
def test_appended_circuit_keeps_even_dimension_signs(d):
    """Each appended circuit starts with a modulo() call, which had the same sign error."""
    first = Circuit(1, d)
    first.add_gate("H", 0)
    first.add_gate("P_INV", 0)
    second = Circuit(1, d)
    second.add_gate("P", 0)
    second.add_gate("H_INV", 0)
    second.add_gate("M", 0)
    program = Program(first)
    program.append_circuit(second)
    assert {r.measurement_value for r in program.simulate(shots=10, force_tableau=True)[0][0]} == {0}


def test_weyl_modulo_reduces_blocks_mod_order():
    t = WeylTableau(1, 6)
    t.hadamard(0)
    t.phase_inv(0)
    t.x(0)
    t.z_block += 3 * t.order
    t.x_block -= t.order
    t.phase_vector += 5 * t.dimension
    before = (t.z_block % t.order, t.x_block % t.order, t.phase_vector % t.dimension)
    t.modulo()
    np.testing.assert_array_equal(t.z_block, before[0])
    np.testing.assert_array_equal(t.x_block, before[1])
    np.testing.assert_array_equal(t.phase_vector, before[2])
    assert t.x_block[0, 0] == 11    # 2d - 1, not d - 1


# 2. measure_z gave zero-probability outcomes for some representatives mod 2d

# A circuit found by the review: the outcome support of qudit 0 is {0, 2, 4}.  With MUL 5 and MUL 1
# the old measurement shifted the X entry of a generator by d and corrected its phase by d / 2, which
# is only right when the generator's Z entry on that qudit is odd; it was even, so every outcome was odd.
_REVIEW_OPS = [
    ('CNOT_INV', 1, 0, None), ('H', 0, None, None), ('CZ', 0, 1, None), ('SWAP', 1, 0, None),
    ('H', 0, None, None), ('H_INV', 1, None, None), ('H', 0, None, None), ('H_INV', 1, None, None),
    ('CZ', 0, 1, None), ('CNOT', 1, 0, None), ('CNOT', 0, 1, None), ('SWAP', 0, 1, None),
    ('CNOT_INV', 1, 0, None), ('P', 0, None, None), ('P', 0, None, None), ('H_INV', 0, None, None),
    ('H', 1, None, None), ('MUL', 0, None, 'first'), ('CNOT', 1, 0, None), ('SWAP', 1, 0, None),
    ('CNOT', 0, 1, None), ('H_INV', 1, None, None), ('CNOT_INV', 0, 1, None), ('H', 1, None, None),
    ('CZ_INV', 0, 1, None), ('CNOT', 1, 0, None), ('CZ', 0, 1, None), ('CNOT_INV', 0, 1, None),
    ('CZ', 1, 0, None), ('H', 0, None, None), ('H', 0, None, None), ('P', 1, None, None),
    ('P', 1, None, None), ('CZ', 0, 1, None), ('CNOT', 1, 0, None), ('CZ', 0, 1, None),
    ('CZ', 1, 0, None), ('CNOT_INV', 0, 1, None), ('CNOT_INV', 0, 1, None), ('P', 1, None, None),
    ('H', 0, None, None), ('P', 0, None, None), ('CZ_INV', 1, 0, None), ('H', 1, None, None),
    ('SWAP', 1, 0, None), ('H', 1, None, None), ('P', 1, None, None), ('CNOT_INV', 1, 0, None),
    ('H', 1, None, None), ('MUL', 0, None, 'second'), ('P_INV', 0, None, None), ('CZ_INV', 0, 1, None),
    ('P_INV', 0, None, None), ('H', 0, None, None), ('CNOT_INV', 0, 1, None),
]


@pytest.mark.parametrize("first,second", [(5, 1), (11, 19), (5, 7), (11, 1), (-1, 1), (-7, 13)])
def test_measurement_does_not_depend_on_representative(first, second):
    d = 6
    c = Circuit(2, d)
    for name, a, b, p in _REVIEW_OPS:
        if name == "MUL":
            c.add_gate("MUL", a, a=first if p == "first" else second)
        elif b is None:
            c.add_gate(name, a)
        else:
            c.add_gate(name, a, b)
    c.add_gate("M", 0)
    random.seed(0)
    outcomes = {r.measurement_value for r in Program(c).simulate(shots=60, force_tableau=True)[0][0]}
    assert outcomes == {0, 2, 4}


def test_measurement_with_even_z_entry_on_the_measured_qudit():
    """
    Before M, generator 0 has X entry 4 and Z entry 0 on qudit 0 (d = 6, entries mod 12).  The old
    code moved the X entry to 10 and added d / 2 to the phase, but W(z, x + d e_0) = (-1)^(z_0) W(z, x)
    is unchanged for even z_0, so it returned 1, 3 or 5 where the support is {0, 2, 4}.
    """
    d = 6
    c = Circuit(2, d)
    for gate in [("H", 1), ("CZ", 0, 1), ("CNOT", 1, 0), ("CNOT", 1, 0), ("P_INV", 0),
                 ("CNOT", 1, 0), ("CNOT", 0, 1), ("H_INV", 0)]:
        c.add_gate(*gate)
    program = Program(c)
    for op in c.operations:
        program.apply_gate(op)
    tableau = program.stabilizer_tableau
    assert tableau.x_block[0, 0] == 4 and tableau.z_block[0, 0] % 2 == 0
    psi, _ = _replay(c, [])
    _assert_stabilizes(tableau, psi)

    random.seed(3)
    outcomes = set()
    for _ in range(40):
        measured = copy.deepcopy(tableau)
        result = measured.measure_z(0)
        assert not result.deterministic
        outcomes.add(result.measurement_value)
        after, _ = _replay(c + _single_measurement(d, 0), [result.measurement_value])
        _assert_stabilizes(measured, after)
    assert outcomes == {0, 2, 4}


def _single_measurement(d, q):
    c = Circuit(2, d)
    c.add_gate("M", q)
    return c


@pytest.mark.parametrize("d", [4, 6, 8, 9])
def test_mul_representatives_give_the_same_tableau(d):
    """
    MUL a and MUL a + k d are the same gate, and now produce the same tableau.  The inverse of a
    must be taken mod the order (2d for even d): mod d alone gives wrong tableaus at d = 8.
    """
    rng = random.Random(d)
    for _ in range(5):
        base = _random_circuit(d, 2, 25, rng, mid_measurements=0)
        a = rng.choice([x for x in range(1, d) if math.gcd(x, d) == 1])
        tableaus = []
        for rep in (a, a + d, a - d, a + 2 * d, a + 7 * d):
            c = copy.deepcopy(base)
            c.operations = c.operations[:-2]
            c.add_gate("MUL", 0, a=rep)
            program = Program(c)
            for op in c.operations:
                program.apply_gate(op)
            t = program.stabilizer_tableau
            tableaus.append((t.z_block.tolist(), t.x_block.tolist(), (t.phase_vector % d).tolist()))
            psi, _ = _replay(c, [])
            _assert_stabilizes(t, psi)
        assert all(t == tableaus[0] for t in tableaus)


@pytest.mark.parametrize("d", [4, 6, 8, 9, 10, 12])
def test_random_circuits_match_statevector(d):
    """Random circuits with every gate, assorted MUL representatives, mid-circuit M, M_X and RESET."""
    rng = random.Random(100 + d)
    random_outcomes = 0
    for trial in range(12):
        n = 3 if d <= 6 else 2
        c = _random_circuit(d, n, rng.randint(1, 130), rng)
        seen = _check_against_statevector(c, seed=1000 * d + trial)
        random_outcomes += sum(1 for _, support in seen if len(support) > 1)
    assert random_outcomes > 0


@pytest.mark.parametrize("d", [4, 6])
def test_random_appended_circuits_match_statevector(d):
    rng = random.Random(200 + d)
    for trial in range(6):
        c = _random_circuit(d, 2, rng.randint(70, 150), rng)
        cut = rng.randrange(1, len(c.operations))
        first, second = Circuit(2, d), Circuit(2, d)
        first.operations = c.operations[:cut]
        second.operations = c.operations[cut:]
        random.seed(trial)
        program = Program(first)
        program.append_circuit(second)
        outcomes = _outcomes_in_circuit_order(c, program.simulate(shots=1))
        psi, seen = _replay(c, outcomes)
        assert psi is not None, seen
        _assert_stabilizes(program.stabilizer_tableau, psi)


@pytest.mark.parametrize("d", [4, 6, 12])
def test_partially_random_outcomes_are_uniform(d):
    """An outcome with s possible values is uniform over them."""
    shots = 1200
    for s in [m for m in range(2, d) if d % m == 0]:
        # Qudit 1 is H|0> = sum_j |j>; d / s CNOTs onto qudit 0 give sum_j |(d / s) j>|j>, and X
        # shifts it, so M on qudit 0 is uniform over the s values 1 + (d / s) j.
        c = Circuit(2, d)
        c.add_gate("H", 1)
        c.add_gate("CNOT", 1, 0)
        for _ in range(d // s - 1):
            c.add_gate("CNOT", 1, 0)
        c.add_gate("X", 0)
        c.add_gate("M", 0)
        random.seed(s)
        values = [r.measurement_value for r in Program(c).simulate(shots=shots, force_tableau=True)[0][0]]
        counts = np.bincount(values, minlength=d)
        support = [(1 + (d // s) * j) % d for j in range(s)]
        assert set(np.flatnonzero(counts)) == set(support)
        assert counts[support].min() > shots / s * 0.75


# 3. coprime sets of size d and int64 overflow at large composite d

def test_bell_pair_measures_quickly_at_large_even_dimension():
    d = 2147483646
    c = Circuit(2, d)
    c.add_gate("H", 0)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("MUL", 1, a=d - 1)
    c.add_gate("M", [0, 1])
    start = time.time()
    for _ in range(5):
        m0, m1 = (r.measurement_value for r in Program(c).simulate())
        assert (m0 + m1) % d == 0
    assert time.time() - start < 5.0


def test_measurement_does_not_build_coprime_sets():
    t = WeylTableau(2, 2147483646)
    t.hadamard(0)
    t.cnot(0, 1)
    t.measure_z(0)
    t.measure_z(1)
    assert "coprime_order" not in t.__dict__ and "coprime_dimension" not in t.__dict__


@pytest.mark.parametrize("d", [2147483646, 2147483644])
def test_multiplication_does_not_overflow_at_large_dimension(d):
    """MUL multiplied entries mod 2d (close to 2**32) by scalars of the same size in int64."""
    rng = random.Random(d)
    t = WeylTableau(2, d)
    reference = WeylTableau(2, d)
    for name in ("x_block", "z_block", "phase_vector"):
        setattr(reference, name, getattr(reference, name).astype(object))
    for tableau in (t, reference):
        tableau.hadamard(0)
        tableau.phase(0)
        tableau.cnot(0, 1)
    for _ in range(20):
        a = _unit_representative(d, rng)
        q = rng.randrange(2)
        t.multiply(q, a)
        reference.multiply(q, a)
        assert t.z_block.tolist() == reference.z_block.tolist()
        assert t.x_block.tolist() == reference.x_block.tolist()


def test_prime_check_is_fast():
    start = time.time()
    assert Tableau(1, 2147483647).prime
    assert not Tableau(1, 2147483646).prime
    assert time.time() - start < 1.0


_INVERSE = {"X": "X_INV", "X_INV": "X", "Z": "Z_INV", "Z_INV": "Z", "H": "H_INV", "H_INV": "H",
            "P": "P_INV", "P_INV": "P", "CNOT": "CNOT_INV", "CNOT_INV": "CNOT", "CZ": "CZ_INV",
            "CZ_INV": "CZ", "SWAP": "SWAP"}


@pytest.mark.parametrize("d", [2147483646, 2147483644, 999999999, 4 * 1000003])
def test_circuit_and_inverse_return_to_zero_at_large_dimension(d):
    """MUL by a scalar near 2**31 multiplied entries near 2**32, which overflowed int64."""
    rng = random.Random(d % 1000)
    for _ in range(4):
        n = 3
        c = _random_circuit(d, n, 60, rng, mid_measurements=0)
        ops = c.operations[:-n]
        c.operations = list(ops)
        for op in reversed(ops):
            if op.name == "MUL":
                c.add_gate("MUL", op.qudit_index, a=pow(int(op.params["a"]) % d, -1, d))
            elif op.name in TWO_QUDIT:
                c.add_gate(_INVERSE[op.name], op.qudit_index, op.target_index)
            else:
                c.add_gate(_INVERSE[op.name], op.qudit_index)
        c.add_gate("M", list(range(n)))
        results = Program(c).simulate()
        assert [(r.measurement_value, r.deterministic) for r in results] == [(0, True)] * n


# numpy 1.x (the version in poetry.lock) reduces an array of Python integers (dtype=object) with
# `a or b`, so np.any returns the integers themselves rather than booleans; numpy 2 returns booleans.
# The measurement took `~np.any(block % d, axis=0)` as a mask, which on numpy 1.x is a bitwise NOT of
# integers and raised IndexError whenever the Python-integer path ran (every large composite d).
# Tests that take `numpy1_any` run the measurement with np.any behaving as in numpy 1.x, whatever
# numpy is installed.

_REAL_ANY = np.any


def _numpy1_any(a, axis=None, out=None, keepdims=False, **kwargs):
    a = np.asarray(a)
    if a.dtype == object and out is None and not kwargs:
        return np.logical_or.reduce(a, axis=axis, keepdims=keepdims)
    return _REAL_ANY(a, axis=axis, out=out, keepdims=keepdims, **kwargs)


class _Numpy1:
    """The numpy module, except that `any` behaves as in numpy 1.x."""
    any = staticmethod(_numpy1_any)

    def __getattr__(self, name):
        return getattr(np, name)


@pytest.fixture
def numpy1_any(monkeypatch):
    monkeypatch.setattr(tableau_composite, "np", _Numpy1())


def test_numpy1_any_emulation():
    values = np.array([[0, 3, 0], [0, 0, 5]], dtype=object)
    assert _numpy1_any(values, axis=0).dtype == object
    assert _numpy1_any(values, axis=0).tolist() == [0, 3, 5]
    assert _numpy1_any(values) == 3
    assert _numpy1_any(values != 0, axis=0).tolist() == [False, True, True]
    assert _numpy1_any(np.array([0, 2]), axis=0)


@pytest.mark.parametrize("d", [2147483646, 1073741824])
def test_large_composite_dimension_measures_with_numpy1_any(d, numpy1_any):
    """The minimal failing case: any measurement at this d takes the Python-integer path."""
    assert WeylTableau(2, d)._work_dtype() is object
    c = Circuit(2, d)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("M", [0, 1])
    assert [(r.measurement_value, r.deterministic) for r in Program(c).simulate()] == [(0, True)] * 2
    c = Circuit(2, d)
    c.add_gate("H", 0)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("M", [0, 1, 0, 1])
    m0, m1, m2, m3 = (r.measurement_value for r in Program(c).simulate())
    assert m0 == m1 == m2 == m3


@pytest.mark.parametrize("d", [4, 6, 8, 9, 10, 12])
def test_python_integer_path_matches_statevector(d, numpy1_any, monkeypatch):
    """Forces the Python-integer path at small d, where the exact statevector is available."""
    monkeypatch.setattr(WeylTableau, "_work_dtype", lambda self: object)
    rng = random.Random(300 + d)
    random_outcomes = 0
    for trial in range(16):
        n = 3 if d <= 6 else 2
        c = _random_circuit(d, n, rng.randint(1, 130), rng)
        seen = _check_against_statevector(c, seed=3000 * d + trial)
        random_outcomes += sum(1 for _, support in seen if len(support) > 1)
    assert random_outcomes > 0


@pytest.mark.parametrize("numpy1", [False, True])
@pytest.mark.parametrize("d", [2147483646, 1073741824, 4 * 1000003])
def test_large_composite_dimension_matches_exact_integers(d, numpy1, monkeypatch):
    """The int64 tableau gives exactly what a tableau of Python integers gives."""
    if numpy1:
        monkeypatch.setattr(tableau_composite, "np", _Numpy1())
    rng = random.Random(d % 997)
    for trial in range(4):
        c = _random_circuit(d, 3, 80, rng, mid_measurements=4)
        c.operations = c.operations[:-3]
        c.add_gate("M", [0, 1, 2])
        c.add_gate("M", [0, 1, 2])
        runs = []
        for exact_integers in (False, True):
            random.seed(trial)
            np.random.seed(trial)
            program = Program(c)
            if exact_integers:
                for name in ("x_block", "z_block", "phase_vector"):
                    setattr(program.initial_tableau, name, getattr(program.initial_tableau, name).astype(object))
            values = [r.measurement_value for r in program.simulate(shots=1)]
            t = program.stabilizer_tableau
            runs.append((values, t.z_block.tolist(), t.x_block.tolist(), t.phase_vector.tolist()))
        assert runs[0] == runs[1]
        # The last two rounds measure every qudit twice in a row, so they agree.
        per_qudit = {}
        for r in program.simulate(shots=1):
            per_qudit.setdefault(r.qudit_index, []).append(r.measurement_value)
        assert all(v[-1] == v[-2] for v in per_qudit.values())


# 5. Notes on the measurement

def test_exact_flag_has_no_effect():
    """`exact=True` used to select a Diophantine solver; the measurement is always exact now."""
    rng = random.Random(7)
    for d in (4, 6, 12):
        c = _random_circuit(d, 3, 60, rng)
        runs = []
        for exact in (False, True):
            random.seed(d)
            np.random.seed(d)
            program = Program(c)
            values = [r.measurement_value for r in program.simulate(shots=1, exact=exact)]
            t = program.stabilizer_tableau
            runs.append((values, t.z_block.tolist(), t.x_block.tolist(), t.phase_vector.tolist()))
        assert runs[0] == runs[1]


def test_composite_notes_describe_the_current_measurement():
    with open(os.path.join(REPO, "sdim", "tableau", "COMPOSITE.md")) as f:
        notes = f.read()
    assert "fails under certain edge cases" not in notes
    assert "no longer uses the Diophantine solver" in notes
    assert "exact=True)` is still accepted for compatibility, but it has no effect" in notes


# 4. Invalid escape sequences broke imports under -W error

def test_sources_compile_with_warnings_as_errors():
    paths = sorted(glob.glob(os.path.join(REPO, "sdim", "**", "*.py"), recursive=True))
    paths += sorted(glob.glob(os.path.join(REPO, "tests", "*.py")))
    assert paths
    for path in paths:
        with open(path) as f:
            source = f.read()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compile(source, path, "exec")
