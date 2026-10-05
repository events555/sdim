"""
Tests for the Cirq gates in sdim.unitary (powers, inverses, agreement with the tableau) and for
tableaus built with NumPy-integer sizes.
"""

import math
import random

import cirq
import numpy as np
import pytest

import sdim.unitary as unitary
from sdim.circuit import Circuit
from sdim.circuit_io import circuit_to_cirq_circuit, cirq_statevector_from_circuit
from sdim.program import Program
from sdim.tableau.tableau_composite import WeylTableau
from sdim.tableau.tableau_gates import apply_CNOT, apply_H, apply_measure, apply_multiplication
from sdim.tableau.tableau_prime import ExtendedTableau

DIMENSIONS = [2, 3, 4, 5, 6]
ONE_QUDIT = ["X", "X_INV", "Z", "Z_INV", "H", "H_INV", "P", "P_INV", "MUL"]
TWO_QUDIT = ["CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"]


def _units(d):
    return [a for a in range(1, d) if math.gcd(a, d) == 1]


def _gates(d):
    """One instance of every gate class in sdim.unitary, with every multiplier a coprime to d."""
    gates = []
    for cls in vars(unitary).values():
        if not (isinstance(cls, type) and issubclass(cls, cirq.Gate) and cls.__module__ == unitary.__name__):
            continue
        if cls is unitary.GeneralizedMultiplicationGate:
            gates += [cls(d, a) for a in _units(d)]
        else:
            gates.append(cls(d))
    return gates


@pytest.mark.parametrize("d", DIMENSIONS)
def test_powers_match_matrix_powers(d):
    """
    Z**-1 had the unitary of Z, and CNOT, CZ and their inverses returned None for exponents they
    do not handle, so cirq.pow ignored its default and gate**2 was None instead of a TypeError.
    """
    unhandled = object()
    for gate in _gates(d):
        u = cirq.unitary(gate)
        for k in (-1, 0, 1, 2, -2, 3):
            power = cirq.pow(gate, k, unhandled)
            if power is unhandled:
                assert k not in (-1, 1), gate
                with pytest.raises(TypeError):
                    gate ** k
                continue
            assert power is not None, f"{type(gate).__name__}**{k}"
            assert cirq.qid_shape(power) == cirq.qid_shape(gate)
            np.testing.assert_allclose(cirq.unitary(power), np.linalg.matrix_power(u, k), atol=1e-9,
                                       err_msg=f"{type(gate).__name__}**{k}")
        if not isinstance(gate, unitary.IdentityGate):
            assert cirq.pow(gate, 0.5, unhandled) is unhandled


def test_multiplication_gate_inverse_takes_numpy_integers():
    """MUL**-1 raised TypeError for NumPy-integer a or d, which pow(a, -1, d) rejects."""
    for d, a in [(np.int64(5), 2), (5, np.int64(2)), (np.int32(5), np.int64(7))]:
        gate = unitary.GeneralizedMultiplicationGate(d, a)
        np.testing.assert_allclose(cirq.unitary(cirq.inverse(gate)), cirq.unitary(gate).T, atol=1e-9)
    # A float a has no unitary, so its inverse is no gate either.
    for d, a in [(5, 2.0), (5, np.float64(2)), (5.0, 2)]:
        with pytest.raises(TypeError):
            unitary.GeneralizedMultiplicationGate(d, a) ** -1


@pytest.mark.parametrize("d", DIMENSIONS)
def test_gates_pass_cirq_protocol_checks(d):
    """
    No gate defined _has_unitary_, so cirq.has_unitary built the matrix (and raised for a MUL
    without one) and cirq.testing rejected every gate. Cirq's repr check is left out, since the
    gates have no evaluable repr or value equality.
    """
    for gate in _gates(d):
        for k in (-1, 0, 1):
            power = cirq.pow(gate, k, None)
            if power is None:
                continue
            assert cirq.has_unitary(power) is True
            cirq.testing.assert_specifies_has_unitary_if_unitary(power)
            cirq.testing.assert_has_consistent_qid_shape(power)
            cirq.testing.assert_has_consistent_apply_unitary(power)
            cirq.testing.assert_all_implemented_act_on_effects_match_unitary(power)
            cirq.testing.assert_decompose_is_consistent_with_unitary(power)
            cirq.testing.assert_unitary_is_consistent(power)
            cirq.testing.assert_controlled_and_controlled_by_identical(power)
            cirq.testing.assert_controlled_unitary_consistent(power)
    assert cirq.has_unitary(unitary.GeneralizedMultiplicationGate(d, 0)) is False


def _every_gate_circuit(d):
    c = Circuit(2, d)
    c.add_gate("H", 0)
    c.add_gate("P", 1)
    for name in ONE_QUDIT:
        if name == "MUL":
            c.add_gate(name, 0, a=d - 1)
        else:
            c.add_gate(name, 0)
        c.add_gate("H", 1)
        for two in TWO_QUDIT:
            c.add_gate(two, 0, 1)
    return c


@pytest.mark.parametrize("d", DIMENSIONS)
def test_inverse_of_a_converted_circuit_undoes_it(d):
    """cirq.inverse(circuit_to_cirq_circuit(c)) applied Z instead of Z^-1."""
    cirq_circuit = circuit_to_cirq_circuit(_every_gate_circuit(d))
    u = cirq.unitary(cirq_circuit)
    np.testing.assert_allclose(cirq.unitary(cirq.inverse(cirq_circuit)) @ u, np.eye(len(u)), atol=1e-9)


def _apply_generator(tableau, j, psi):
    """
    Applies generator j of a tableau to a state of shape (d,) * n.

    An ExtendedTableau generator (p, z, x) is omega^(p / phase_order) X^x Z^z; a WeylTableau one
    is omega^-p tau^(x.z) X^x Z^z (see sdim.tableau.tableau_composite).
    """
    d = tableau.dimension
    xs = [int(v) for v in tableau.x_block[:, j]]
    zs = [int(v) for v in tableau.z_block[:, j]]
    X, Z = unitary.generate_x_matrix(d), unitary.generate_z_matrix(d)
    for q, (x, z) in enumerate(zip(xs, zs)):
        factor = np.linalg.matrix_power(X, x % d) @ np.linalg.matrix_power(Z, z % d)
        psi = np.moveaxis(np.tensordot(factor, psi, axes=([1], [q])), 0, q)
    p = int(tableau.phase_vector[j])
    if isinstance(tableau, ExtendedTableau):
        return np.exp(2j * np.pi * p / (tableau.phase_order * d)) * psi
    return unitary.generate_tau(d) ** ((-2 * p + sum(x * z for x, z in zip(xs, zs))) % (2 * d)) * psi


@pytest.mark.parametrize("d", DIMENSIONS)
@pytest.mark.parametrize("name", ONE_QUDIT + TWO_QUDIT)
def test_cirq_gates_match_the_tableau(d, name):
    """
    Each Cirq gate is the gate the tableau applies, up to a global phase.

    The gate acts on halves of maximally entangled pairs, which fixes it up to a global phase, and
    the tableau after the same circuit must stabilize the state Cirq computes.
    """
    width = 1 if name in ONE_QUDIT else 2
    for a in (_units(d) if name == "MUL" else [None]):
        c = Circuit(2 * width, d)
        for q in range(width):
            c.add_gate("H", q)
            c.add_gate("CNOT", q, q + width)
        if a is None:
            c.add_gate(name, *range(width))
        else:
            c.add_gate(name, 0, a=a)
        program = Program(c)
        program.simulate()
        tableau = program.stabilizer_tableau
        psi = cirq_statevector_from_circuit(c).reshape((d,) * 2 * width)
        for j in range(tableau.z_block.shape[1]):
            np.testing.assert_allclose(_apply_generator(tableau, j, psi), psi, atol=1e-6, err_msg=f"{name} a={a}")


@pytest.mark.parametrize("cls,d", [(ExtendedTableau, 2), (ExtendedTableau, 5), (WeylTableau, 4), (WeylTableau, 6)])
def test_tableau_takes_numpy_integer_sizes(cls, d):
    """
    A NumPy-integer dimension reached pow(x, -1, dimension), a TypeError in measurement and MUL.
    A float or bool size raises TypeError; a bool dimension used to build a tableau.
    """
    tableaus = [cls(2, d), cls(np.int64(2), np.int64(d)), cls(np.int32(2), np.uint16(d)), cls(2, np.int64(d))]
    results = []
    for tableau in tableaus:
        assert all(array.dtype == np.int64 for array in vars(tableau).values() if isinstance(array, np.ndarray))
        random.seed(1)
        apply_H(tableau, 0)
        apply_multiplication(tableau, 0, None, {"a": d - 1})
        apply_CNOT(tableau, 0, 1)
        apply_multiplication(tableau, 1, None, {"a": d - 1})
        results.append([apply_measure(tableau, q).measurement_value for q in (0, 1, 0)])
    assert all(result == results[0] for result in results)
    for tableau in tableaus[1:]:
        assert type(tableau.num_qudits) is int and type(tableau.dimension) is int
        for name, array in vars(tableaus[0]).items():
            if isinstance(array, np.ndarray):
                assert np.array_equal(vars(tableau)[name], array) and vars(tableau)[name].dtype == np.int64
    for sizes, name in [((2, float(d)), "dimension"), ((2.0, d), "num_qudits"), ((2, np.float64(d)), "dimension"),
                        ((True, d), "num_qudits"), ((np.True_, d), "num_qudits"), ((2, True), "dimension"),
                        ((2, np.True_), "dimension")]:
        with pytest.raises(TypeError, match=f"^{name} must be an integer"):
            cls(*sizes)
