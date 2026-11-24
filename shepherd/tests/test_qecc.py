"""Tests of PauliString arithmetic and QECC's logical_paulis / logical_cliffords."""

from __future__ import annotations

import numpy as np
import pytest
from sdim import Circuit

from shepherd.qecc import QECC, PauliString


def matrix(p: PauliString) -> np.ndarray:
    """ω^phase ⊗_i X^{x_i} Z^{z_i}, with X|k> = |k+1> and Z|k> = ω^k |k>, so Z X = ω X Z."""
    d = p.dimension
    w = np.exp(2j * np.pi / d)
    X = np.roll(np.eye(d), 1, axis=0)
    Z = np.diag(w ** np.arange(d))
    out = np.array([[w ** p.overall_phase]])
    for x, z in zip(p.x_powers, p.z_powers):
        out = np.kron(out, np.linalg.matrix_power(X, x) @ np.linalg.matrix_power(Z, z))
    return out


def random_pauli(rng, d: int, n: int) -> PauliString:
    return PauliString(d, rng.integers(d), list(rng.integers(d, size=n)), list(rng.integers(d, size=n)))


# -- toy codes -----------------------------------------------------------------------------------

class RepetitionCode(QECC):
    """3-qudit repetition code against X errors: checks Z0 Z1^-1, Z1 Z2^-1; X̄ = X0 X1 X2, Z̄ = Z0."""

    def stabilizers(self):
        d = self.dimension
        return [PauliString(d, 0, [0, 0, 0], [1, -1, 0]), PauliString(d, 0, [0, 0, 0], [0, 1, -1])]

    def logical_operators(self):
        d = self.dimension
        return [PauliString(d, 0, [1, 1, 1], [0, 0, 0])], [PauliString(d, 0, [0, 0, 0], [1, 0, 0])]


class BareQudits(QECC):
    """Two unencoded qudits: X̄_i = X_i, Z̄_i = Z_i, and every logical gate is the physical gate."""

    def stabilizers(self):
        return []

    def logical_operators(self):
        d = self.dimension
        return ([PauliString(d, 0, [1, 0], [0, 0]), PauliString(d, 0, [0, 1], [0, 0])],
                [PauliString(d, 0, [0, 0], [1, 0]), PauliString(d, 0, [0, 0], [0, 1])])

    def _logical_clifford_circuit(self, gate, qudits):
        self.last_call = (gate, qudits)
        c = Circuit(2, self.dimension)
        c.add_gate(gate, *qudits)
        return c


# -- PauliString arithmetic ----------------------------------------------------------------------

@pytest.mark.parametrize("d", [2, 3, 5])
def test_product_matches_matrices(d):
    rng = np.random.default_rng(d)
    for _ in range(20):
        p, q = random_pauli(rng, d, 2), random_pauli(rng, d, 2)
        assert np.allclose(matrix(p * q), matrix(p) @ matrix(q))


@pytest.mark.parametrize("d", [2, 3, 5])
def test_power_matches_matrices(d):
    rng = np.random.default_rng(10 + d)
    for _ in range(5):
        p = random_pauli(rng, d, 2)
        for k in range(-3, 2 * d + 2):
            m = matrix(p)
            expected = np.linalg.matrix_power(np.linalg.inv(m) if k < 0 else m, abs(k))
            assert np.allclose(matrix(p ** k), expected)


def test_identity_and_mismatched_spaces():
    assert np.allclose(matrix(PauliString.identity(3, 2)), np.eye(9))
    with pytest.raises(ValueError):
        PauliString(3, 0, [1], [0]) * PauliString(3, 0, [1, 0], [0, 0])


# -- logical_paulis ------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [2, 3, 5])
def test_logical_paulis_are_logical(d):
    code = RepetitionCode(d)
    (xbar,), (zbar,) = code.logical_operators()
    for k in range(d):
        for j in range(d):
            L = code.logical_paulis(k, j)
            assert all(PauliString.symplectic_inner_product(L, s) == 0 for s in code.stabilizers())
            assert PauliString.symplectic_inner_product(L, zbar) == k % d
            assert PauliString.symplectic_inner_product(L, xbar) == (-j) % d
            assert np.allclose(matrix(L), matrix(xbar ** k) @ matrix(zbar ** j))


def test_logical_paulis_inverse_and_qudit_index():
    code = BareQudits(3)
    identity = PauliString.identity(3, 2)
    assert code.logical_paulis(-1, 0, qudit=1) * code.logical_paulis(1, 0, qudit=1) == identity
    assert code.logical_paulis(0, -2, qudit=1) * code.logical_paulis(0, 2, qudit=1) == identity
    assert code.logical_paulis(1, 0, qudit=1).x_powers == [0, 1]
    with pytest.raises(IndexError):
        code.logical_paulis(1, 0, qudit=2)


# -- logical_cliffords ---------------------------------------------------------------------------

@pytest.mark.parametrize("given, sdim_name, qudits", [
    ("H", "H", ()), ("f", "H", (1,)), ("F_INV", "H_INV", (0,)), ("p_inv", "P_INV", (1,)),
    ("SUM", "CNOT", (0, 1)), ("sum_inv", "CNOT_INV", (1, 0)), ("CNOT", "CNOT", (1, 0)),
])
def test_cliffords_are_checked_and_named_for_sdim(given, sdim_name, qudits):
    code = BareQudits(3)
    circuit = code.logical_cliffords(given, *qudits)
    assert code.last_call == (sdim_name, qudits or (0,))
    assert circuit.dimension == 3


@pytest.mark.parametrize("gate, qudits, error", [
    ("T", (0,), ValueError),             # not an accepted Clifford
    ("CZ", (0, 1), ValueError),          # nor this one
    ("H", (0, 1), ValueError),           # one-qudit gate, two qudits
    ("CNOT", (0,), ValueError),          # two-qudit gate, one qudit
    ("CNOT", (1, 1), ValueError),        # control == target
    ("P", (2,), IndexError),             # no logical qudit 2
])
def test_bad_clifford_requests_are_rejected(gate, qudits, error):
    with pytest.raises(error):
        BareQudits(3).logical_cliffords(gate, *qudits)


def test_code_without_clifford_circuits_says_so():
    with pytest.raises(NotImplementedError, match="RepetitionCode"):
        RepetitionCode(3).logical_cliffords("H")


def test_circuit_of_wrong_dimension_is_rejected():
    class WrongDimension(BareQudits):
        def _logical_clifford_circuit(self, gate, qudits):
            return Circuit(2, self.dimension + 2)
    with pytest.raises(ValueError, match="dimension"):
        WrongDimension(3).logical_cliffords("H")
