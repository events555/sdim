"""Qudit stabilizer tableau over the Weyl-Heisenberg group.

Reference: de Beaudrap, QIC 13.1-2 (2013), arXiv:1102.3354v4.
See docs/markdown/algorithm.md for the full protocol.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..gatedata import gate_id_to_name, is_gate_noisy


def _snf_mod(matrix: list[list[int]], d: int):
    """Lazy-import wrapper around modularsnf."""
    from modularsnf import smith_normal_form_mod

    return smith_normal_form_mod(matrix, d)


class TableauSimulator:
    """Stabilizer tableau for qudits of dimension d.

    Generators are stored in rows ``0 .. l-1``.  Arrays are
    pre-allocated to ``2n`` rows (the maximum after composite-d
    measurements); only the first ``l`` rows are active.

    Attributes:
        n: Number of qudits.
        d: Local dimension.
        l: Current number of generators (<= 2n).
        X: ``(2n, n)`` array, entries mod d.
        Z: ``(2n, n)`` array, entries mod d.
        tau_exp: ``(2n,)`` array, entries mod 2d.
    """

    __slots__ = ("n", "d", "l", "X", "Z", "tau_exp")

    def __init__(self, n: int = 1, d: int = 2) -> None:
        self.n = n
        self.d = d
        self.l = n
        self.X = np.zeros((2 * n, n), dtype=np.int64)
        self.Z = np.zeros((2 * n, n), dtype=np.int64)
        self.tau_exp = np.zeros(2 * n, dtype=np.int64)
        np.fill_diagonal(self.Z[:n], 1)

    @property
    def even(self) -> bool:
        return self.d % 2 == 0

    def modulo(self) -> None:
        self.X %= self.d
        self.Z %= self.d
        self.tau_exp %= 2 * self.d

    def _product(
        self,
        X1: np.ndarray,
        Z1: np.ndarray,
        t1: int,
        X2: np.ndarray,
        Z2: np.ndarray,
        t2: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Ordered product of two Weyl vectors."""
        d, D = self.d, 2 * self.d
        return (
            (X1 + X2) % d,
            (Z1 + Z2) % d,
            (t1 + t2 + 2 * int(np.sum(Z1 * X2))) % D,
        )

    def _power(
        self,
        base_X: np.ndarray,
        base_Z: np.ndarray,
        base_t: int,
        c: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Closed-form power rule: ``(tau^{-t} Z^z X^x)^c``."""
        d, D = self.d, 2 * self.d
        return (
            (c * base_X) % d,
            (c * base_Z) % d,
            (c * base_t + c * (c - 1) * int(np.sum(base_Z * base_X))) % D,
        )

    def _row_add_mult(self, k: int, p: int, m: int) -> None:
        """Row k <- row k * (row p)^m."""
        if m == 0:
            return
        pX, pZ, pt = self._power(
            self.X[p],
            self.Z[p],
            int(self.tau_exp[p]),
            m,
        )
        self.X[k], self.Z[k], self.tau_exp[k] = self._product(
            self.X[k],
            self.Z[k],
            int(self.tau_exp[k]),
            pX,
            pZ,
            pt,
        )

    def _apply_column_transform(self, V: np.ndarray) -> None:
        """Apply unimodular V to generator rows ``0 .. l-1``."""
        d, D = self.d, 2 * self.d
        l = self.l
        V = np.asarray(V, dtype=np.int64)

        Z_old = self.Z[:l].copy()
        X_old = self.X[:l].copy()
        tau_old = self.tau_exp[:l].copy()

        G = Z_old @ X_old.T

        self.Z[:l] = (V.T @ Z_old) % d
        self.X[:l] = (V.T @ X_old) % d

        t1 = V.T @ tau_old
        t2 = (V * (V - 1)).T @ np.diag(G)
        t3 = 2 * np.diag(V.T @ np.triu(G, 1) @ V)
        self.tau_exp[:l] = (t1 + t2 + t3) % D

    def _rows(self) -> slice:
        return slice(0, self.l)

    def hadamard(self, q: int, dagger: bool = False) -> None:
        r = self._rows()
        dir_ = -1 if dagger else 1
        x_old = self.X[r, q].copy()
        z_old = self.Z[r, q].copy()
        self.tau_exp[r] = (self.tau_exp[r] - 2 * z_old * x_old) % (2 * self.d)
        self.X[r, q] = (-dir_ * z_old) % self.d
        self.Z[r, q] = (dir_ * x_old) % self.d

    def phase_gate(self, q: int, dagger: bool = False) -> None:
        r = self._rows()
        dir_ = -1 if dagger else 1
        xj = self.X[r, q]
        self.Z[r, q] = (self.Z[r, q] + dir_ * xj) % self.d
        if self.even:
            self.tau_exp[r] = (self.tau_exp[r] + dir_ * (xj * xj)) % (2 * self.d)
        else:
            self.tau_exp[r] = (self.tau_exp[r] + dir_ * (xj * (xj + 1))) % (2 * self.d)

    def pauli_x(self, q: int, power: int = 1, dagger: bool = False) -> None:
        r = self._rows()
        dir_ = -1 if dagger else 1
        self.tau_exp[r] = (self.tau_exp[r] + dir_ * 2 * power * self.Z[r, q]) % (
            2 * self.d
        )

    def pauli_z(self, q: int, power: int = 1, dagger: bool = False) -> None:
        r = self._rows()
        dir_ = -1 if dagger else 1
        self.tau_exp[r] = (self.tau_exp[r] - dir_ * 2 * power * self.X[r, q]) % (
            2 * self.d
        )

    def cnot(self, c: int, t: int, dagger: bool = False) -> None:
        r = self._rows()
        dir_ = -1 if dagger else 1
        self.X[r, t] = (self.X[r, t] + dir_ * self.X[r, c]) % self.d
        self.Z[r, c] = (self.Z[r, c] - dir_ * self.Z[r, t]) % self.d

    def cz(self, q1: int, q2: int, dagger: bool = False) -> None:
        r = self._rows()
        dir_ = -1 if dagger else 1
        x1 = self.X[r, q1].copy()
        x2 = self.X[r, q2].copy()
        self.Z[r, q1] = (self.Z[r, q1] + dir_ * x2) % self.d
        self.Z[r, q2] = (self.Z[r, q2] + dir_ * x1) % self.d
        self.tau_exp[r] = (self.tau_exp[r] + 2 * dir_ * (x1 * x2)) % (2 * self.d)

    def swap(self, q1: int, q2: int) -> None:
        if q1 != q2:
            r = self._rows()
            self.X[r, q1], self.X[r, q2] = (
                self.X[r, q2].copy(),
                self.X[r, q1].copy(),
            )
            self.Z[r, q1], self.Z[r, q2] = (
                self.Z[r, q2].copy(),
                self.Z[r, q1].copy(),
            )

    def multiply(self, q: int, a: int, dagger: bool = False) -> None:
        """Multiplier gate M_a on qudit q."""
        if math.gcd(a, self.d) != 1:
            raise ValueError("gcd(a, d) must be 1")
        a_inv = pow(a, -1, self.d)
        if dagger:
            a, a_inv = a_inv, a
        r = self._rows()
        self.X[r, q] = (a * self.X[r, q]) % self.d
        self.Z[r, q] = (a_inv * self.Z[r, q]) % self.d

    def apply_gate(
        self,
        gate_id: int,
        qudit_idx: int,
        target_idx: int,
        arg0: Optional[int | float] = None,
    ) -> None:
        if is_gate_noisy(gate_id):
            return
        if arg0 is None or (isinstance(arg0, float) and math.isnan(arg0)):
            power = 1
        else:
            power = int(arg0) % self.d

        name = gate_id_to_name(gate_id)
        match name:
            case "H":
                self.hadamard(qudit_idx)
            case "H_INV":
                self.hadamard(qudit_idx, dagger=True)
            case "P":
                self.phase_gate(qudit_idx)
            case "P_INV":
                self.phase_gate(qudit_idx, dagger=True)
            case "X":
                self.pauli_x(qudit_idx, power)
            case "X_INV":
                self.pauli_x(qudit_idx, power, dagger=True)
            case "Z":
                self.pauli_z(qudit_idx, power)
            case "Z_INV":
                self.pauli_z(qudit_idx, power, dagger=True)
            case "CNOT":
                self.cnot(qudit_idx, target_idx)
            case "CNOT_INV":
                self.cnot(qudit_idx, target_idx, dagger=True)
            case "CZ":
                self.cz(qudit_idx, target_idx)
            case "CZ_INV":
                self.cz(qudit_idx, target_idx, dagger=True)
            case "SWAP":
                self.swap(qudit_idx, target_idx)
            case "MULTIPLY":
                self.multiply(qudit_idx, power)
            case "MULTIPLY_INV":
                self.multiply(qudit_idx, power, dagger=True)
            case _:
                raise ValueError(f"Unknown gate: {name}")

    def measure(self, q: int) -> int:
        """Measure qudit q in the Z basis. Returns outcome in [0, d)."""
        a = np.zeros(self.n, dtype=np.int64)
        b = np.zeros(self.n, dtype=np.int64)
        a[q] = 1
        return self.measure_pauli(a, b, 0)

    def measure_pauli(
        self,
        a: np.ndarray,
        b: np.ndarray,
        delta: int,
    ) -> int:
        """Measure Pauli ``P = tau^{-delta} Z^a X^b``.

        Returns outcome h in ``[0, d)``.  Follows the unified
        Steps 1-5 from algorithm.md (deterministic when eta = d).
        """
        n, d, D = self.n, self.d, 2 * self.d
        l = self.l

        c = (a @ self.X[:l].T - b @ self.Z[:l].T) % d
        eta = math.gcd(d, *c.tolist())
        s = d // eta

        if eta < d:
            _S, _U, V = _snf_mod(c.reshape(1, -1).tolist(), d)
            self._apply_column_transform(np.array(V, dtype=np.int64))

        t, f1 = self._eigenvalue_Ps(a, b, delta, s)
        h = self._sample_outcome(t, s, eta)

        if eta < d:
            self._collapse(a, b, delta, h, s, f1)

        return h

    def _eigenvalue_Ps(
        self,
        a: np.ndarray,
        b: np.ndarray,
        delta: int,
        s: int,
    ) -> tuple[int, int]:
        """Step 3: find the eigenvalue of P^s by decomposing it over
        the current generators via direct linear solve.

        Returns ``(t, f1)`` where ``2 s h ≡ t (mod D)`` gives the
        measurement outcome, and ``f1`` is the first invariant factor
        of the generator Weyl block (used in the collapse decision).
        """
        n, d, D = self.n, self.d, 2 * self.d
        l = self.l

        ps_Z = (s * a) % d
        ps_X = (s * b) % d
        ps_t = (s * delta + s * (s - 1) * int(np.sum(a * b))) % D

        gen_Z = self.Z[:l].T
        gen_X = self.X[:l].T
        G = np.vstack([gen_Z, gen_X])
        ps = np.concatenate([ps_Z, ps_X])

        S_list, U_list, V_list = _snf_mod(G.tolist(), d)
        S_np = np.array(S_list, dtype=np.int64)
        U_np = np.array(U_list, dtype=np.int64)
        V_np = np.array(V_list, dtype=np.int64)

        f1 = int(S_np[0, 0]) if S_np.size else 1

        rhs = (U_np @ ps) % d
        y = np.zeros(l, dtype=np.int64)
        for i in range(min(S_np.shape)):
            dii = int(S_np[i, i]) % d
            bi = int(rhs[i]) % d
            if dii == 0:
                continue
            g = math.gcd(dii, d)
            y[i] = (bi // g * pow(dii // g, -1, d // g)) % (d // g)

        coeffs = (V_np @ y) % d

        acc_X = np.zeros(n, dtype=np.int64)
        acc_Z = np.zeros(n, dtype=np.int64)
        acc_t = 0
        for k in range(l):
            ck = int(coeffs[k]) % d
            if ck == 0:
                continue
            pX, pZ, pt = self._power(
                self.X[k],
                self.Z[k],
                int(self.tau_exp[k]),
                ck,
            )
            acc_X, acc_Z, acc_t = self._product(
                acc_X,
                acc_Z,
                acc_t,
                pX,
                pZ,
                pt,
            )

        t = (acc_t - ps_t) % D
        return t, f1

    def _sample_outcome(self, t: int, s: int, eta: int) -> int:
        """Step 4: solve ``2 s h ≡ t (mod D)``, sample uniformly."""
        d, D = self.d, 2 * self.d
        g = math.gcd(2 * s, D)
        h0 = (t // g * pow(2 * s // g, -1, D // g)) % (D // g)
        k = int(np.random.randint(0, d // eta))
        return int((h0 + eta * k) % d)

    def _collapse(
        self,
        a: np.ndarray,
        b: np.ndarray,
        delta: int,
        h: int,
        s: int,
        f1: int,
    ) -> None:
        """Step 5: scale generator 0, insert measurement result R."""
        d, D = self.d, 2 * self.d
        l = self.l

        scaled_X, scaled_Z, scaled_t = self._power(
            self.X[0],
            self.Z[0],
            int(self.tau_exp[0]),
            s,
        )

        trivial = (
            np.all(scaled_X % d == 0)
            and np.all(scaled_Z % d == 0)
            and scaled_t % D == 0
        )
        keep = not trivial and f1 % s != 0

        if keep:
            self.X[l] = scaled_X % d
            self.Z[l] = scaled_Z % d
            self.tau_exp[l] = scaled_t % D
            self.l = l + 1

        self.X[0] = b % d
        self.Z[0] = a % d
        self.tau_exp[0] = (delta + 2 * h) % D

    def print_tableau(self) -> None:
        l = self.l
        print(f"l={l}, d={self.d}")
        print(f"X:\n{self.X[:l]}")
        print(f"Z:\n{self.Z[:l]}")
        print(f"tau_exp: {self.tau_exp[:l]}")
