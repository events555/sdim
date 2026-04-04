from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import cached_property
from math import gcd
from typing import Optional

import numpy as np

from ..gatedata import gate_id_to_name, is_gate_noisy


@dataclass
class TableauSimulator:
    """
    Represents a stabilizer tableau simulator for quantum circuit simulation.

    This class combines stabilizer and destabilizer information for efficient
    simulation of Clifford operations on qudits of prime dimension.

    This follows as a generalization to prime dimensions from
    "Improved Simulation of Stabilizer Circuits" by Aaronson and Gottesman.

    **Note that we are assuming conjugation of Z all the way to the front of the Pauli string**.

    Attributes:
        num_qudits (int): The number of qudits in the system.
        dimension (int): The dimension of each qudit (default is 2 for qubits).
        phase_vector (np.ndarray): The phase vector of the stabilizer tableau.
        z_block (np.ndarray): The Z block of the stabilizer tableau.
        x_block (np.ndarray): The X block of the stabilizer tableau.
        destab_phase_vector (np.ndarray): The phase vector for destabilizers.
        destab_z_block (np.ndarray): The Z block for destabilizers.
        destab_x_block (np.ndarray): The X block for destabilizers.
    """

    num_qudits: int = 1
    dimension: int = 2
    phase_vector: Optional[np.ndarray] = None
    z_block: Optional[np.ndarray] = None
    x_block: Optional[np.ndarray] = None
    destab_phase_vector: Optional[np.ndarray] = None
    destab_z_block: Optional[np.ndarray] = None
    destab_x_block: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.phase_vector is None:
            self.phase_vector = np.zeros(self.num_qudits, dtype=np.int64)
        if self.z_block is None:
            self.z_block = np.eye(self.num_qudits, dtype=np.int64)
        if self.x_block is None:
            self.x_block = np.zeros((self.num_qudits, self.num_qudits), dtype=np.int64)
        if self.destab_phase_vector is None:
            self.destab_phase_vector = np.zeros(self.num_qudits, dtype=np.int64)
        if self.destab_z_block is None:
            self.destab_z_block = np.zeros((self.num_qudits, self.num_qudits), dtype=np.int64)
        if self.destab_x_block is None:
            self.destab_x_block = np.eye(self.num_qudits, dtype=np.int64)

    # ── Properties ──────────────────────────────────────────────────────

    @cached_property
    def coprime_order(self) -> set:
        return {i for i in range(1, self.order) if gcd(i, self.order) == 1}

    @cached_property
    def coprime_dimension(self) -> set:
        return {i for i in range(1, self.dimension) if gcd(i, self.dimension) == 1}

    @cached_property
    def prime(self) -> bool:
        return not any(self.dimension % i == 0 for i in range(2, self.dimension))

    @property
    def even(self) -> bool:
        return self.dimension % 2 == 0

    @property
    def order(self) -> int:
        return self.dimension * 2 if self.even else self.dimension

    @property
    def phase_order(self) -> int:
        return 2 if self.even else 1

    @property
    def pauli_size(self) -> int:
        return 2 * self.num_qudits + 1

    @property
    def num_generators(self) -> int:
        return self.z_block.shape[1]

    @property
    def stab_tableau(self) -> np.ndarray:
        return np.vstack((self.phase_vector, self.z_block, self.x_block))

    @property
    def destab_tableau(self) -> np.ndarray:
        return np.vstack((self.destab_phase_vector, self.destab_z_block, self.destab_x_block))

    @property
    def tableau(self) -> np.ndarray:
        return np.hstack((self.stab_tableau, self.destab_tableau))

    # ── Tableau utilities ───────────────────────────────────────────────

    def modulo(self):
        self.z_block %= self.dimension
        self.x_block %= self.dimension
        self.phase_vector %= self.order
        self.destab_z_block %= self.dimension
        self.destab_x_block %= self.dimension
        self.destab_phase_vector %= self.order

    def _print_labeled_matrix(self, label: str, matrix: np.ndarray):
        print(f"{label}:")
        print(matrix)

    def print_phase_vector(self):
        self._print_labeled_matrix("Phase Vector", self.phase_vector)

    def print_z_block(self):
        self._print_labeled_matrix("Z Block", self.z_block)

    def print_x_block(self):
        self._print_labeled_matrix("X Block", self.x_block)

    def print_destab_phase_vector(self):
        self._print_labeled_matrix("Destabilizer Phase Vector", self.destab_phase_vector)

    def print_destab_z_block(self):
        self._print_labeled_matrix("Destabilizer Z Block", self.destab_z_block)

    def print_destab_x_block(self):
        self._print_labeled_matrix("Destabilizer X Block", self.destab_x_block)

    def print_tableau(self):
        self.print_phase_vector()
        self.print_z_block()
        self.print_x_block()
        self.print_destab_phase_vector()
        self.print_destab_z_block()
        self.print_destab_x_block()

    # ── Gate operations ─────────────────────────────────────────────────

    def hadamard(self, qudit_index: int):
        new_x = -self.z_block[qudit_index, :].copy()
        new_z = self.x_block[qudit_index, :].copy()
        self.x_block[qudit_index, :] = new_x
        self.z_block[qudit_index, :] = new_z
        self.phase_vector += self.phase_order * (self.x_block[qudit_index, :] * self.z_block[qudit_index, :])

        new_destab_x = -self.destab_z_block[qudit_index, :].copy()
        new_destab_z = self.destab_x_block[qudit_index, :].copy()
        self.destab_x_block[qudit_index, :] = new_destab_x
        self.destab_z_block[qudit_index, :] = new_destab_z
        self.destab_phase_vector += self.phase_order * (self.destab_x_block[qudit_index, :] * self.destab_z_block[qudit_index, :])

    def hadamard_inv(self, qudit_index: int):
        new_x = self.z_block[qudit_index, :].copy()
        new_z = -self.x_block[qudit_index, :].copy()
        self.z_block[qudit_index, :] = new_z
        self.x_block[qudit_index, :] = new_x
        self.phase_vector += self.phase_order * (self.x_block[qudit_index, :] * self.z_block[qudit_index, :])

        new_destab_x = self.destab_z_block[qudit_index, :].copy()
        new_destab_z = -self.destab_x_block[qudit_index, :].copy()
        self.destab_z_block[qudit_index, :] = new_destab_z
        self.destab_x_block[qudit_index, :] = new_destab_x
        self.destab_phase_vector += self.phase_order * (self.destab_x_block[qudit_index, :] * self.destab_z_block[qudit_index, :])

    def phase(self, qudit_index: int):
        if self.even:
            self.phase_vector += self.x_block[qudit_index, :] ** 2
            self.destab_phase_vector += self.destab_x_block[qudit_index, :] ** 2
        else:
            self.phase_vector += self.x_block[qudit_index, :] * (self.x_block[qudit_index, :] - 1) // 2
            self.destab_phase_vector += self.destab_x_block[qudit_index, :] * (self.destab_x_block[qudit_index, :] - 1) // 2
        self.z_block[qudit_index, :] += self.x_block[qudit_index, :]
        self.destab_z_block[qudit_index, :] += self.destab_x_block[qudit_index, :]

    def phase_inv(self, qudit_index: int):
        if self.even:
            self.phase_vector -= self.x_block[qudit_index, :] ** 2
            self.destab_phase_vector -= self.destab_x_block[qudit_index, :] ** 2
        else:
            self.phase_vector -= self.x_block[qudit_index, :] * (self.x_block[qudit_index, :] - 1) // 2
            self.destab_phase_vector -= self.destab_x_block[qudit_index, :] * (self.destab_x_block[qudit_index, :] - 1) // 2
        self.z_block[qudit_index, :] -= self.x_block[qudit_index, :]
        self.destab_z_block[qudit_index, :] -= self.destab_x_block[qudit_index, :]

    def x(self, qudit_index: int, multiplier: int = 1):
        self.phase_vector -= self.z_block[qudit_index, :] * self.phase_order * multiplier
        self.destab_phase_vector -= self.destab_z_block[qudit_index, :] * self.phase_order * multiplier

    def x_inv(self, qudit_index: int, multiplier: int = 1):
        self.phase_vector += self.z_block[qudit_index, :] * self.phase_order * multiplier
        self.destab_phase_vector += self.destab_z_block[qudit_index, :] * self.phase_order * multiplier

    def z(self, qudit_index: int, multiplier: int = 1):
        self.phase_vector += self.x_block[qudit_index, :] * self.phase_order * multiplier
        self.destab_phase_vector += self.destab_x_block[qudit_index, :] * self.phase_order * multiplier

    def z_inv(self, qudit_index: int, multiplier: int = 1):
        self.phase_vector -= self.x_block[qudit_index, :] * self.phase_order * multiplier
        self.destab_phase_vector -= self.destab_x_block[qudit_index, :] * self.phase_order * multiplier

    def multiply(self, q: int, a: int):
        """Apply M_a on qudit q."""
        if math.gcd(a, self.dimension) != 1:
            raise ValueError("gcd(a,d) must be 1")
        a_inv = pow(a, -1, self.dimension)
        self._multiply_internal(q, a, a_inv)

    def multiply_inv(self, q: int, a: int):
        """Apply M_a† ≡ M_{a^{-1}} on qudit q."""
        if math.gcd(a, self.dimension) != 1:
            raise ValueError("gcd(a,d) must be 1")
        a_inv = pow(a, -1, self.dimension)
        self._multiply_internal(q, a_inv, a)

    def _multiply_internal(self, q: int, a: int, a_inv: int):
        d = self.dimension
        self.x_block[q]        = (a     * self.x_block[q])        % d
        self.destab_x_block[q] = (a     * self.destab_x_block[q]) % d
        self.z_block[q]        = (a_inv * self.z_block[q])        % d
        self.destab_z_block[q] = (a_inv * self.destab_z_block[q]) % d

        if self.even:
            kappa = ((a + a_inv) % d) // 2
            delta = kappa * (self.x_block[q] * self.z_block[q])
            self.phase_vector        = (self.phase_vector        + delta) % self.order
            self.destab_phase_vector = (self.destab_phase_vector + delta) % self.order

        self.modulo()

    def cnot(self, control: int, target: int):
        self.x_block[target, :] += self.x_block[control, :]
        self.z_block[control, :] -= self.z_block[target, :]
        self.destab_x_block[target, :] += self.destab_x_block[control, :]
        self.destab_z_block[control, :] -= self.destab_z_block[target, :]

    def cnot_inv(self, control: int, target: int):
        self.x_block[target, :] -= self.x_block[control, :]
        self.z_block[control, :] += self.z_block[target, :]
        self.destab_x_block[target, :] -= self.destab_x_block[control, :]
        self.destab_z_block[control, :] += self.destab_z_block[target, :]

    def cz(self, qudit1: int, qudit2: int):
        self.z_block[qudit1, :] += self.x_block[qudit2, :]
        self.z_block[qudit2, :] += self.x_block[qudit1, :]
        self.destab_z_block[qudit1, :] += self.destab_x_block[qudit2, :]
        self.destab_z_block[qudit2, :] += self.destab_x_block[qudit1, :]
        self.phase_vector += self.phase_order * (self.x_block[qudit1, :] * self.x_block[qudit2, :])
        self.destab_phase_vector += self.phase_order * (self.destab_x_block[qudit1, :] * self.destab_x_block[qudit2, :])

    def cz_inv(self, qudit1: int, qudit2: int):
        self.z_block[qudit1, :] -= self.x_block[qudit2, :]
        self.z_block[qudit2, :] -= self.x_block[qudit1, :]
        self.destab_z_block[qudit1, :] -= self.destab_x_block[qudit2, :]
        self.destab_z_block[qudit2, :] -= self.destab_x_block[qudit1, :]
        self.phase_vector -= self.phase_order * (self.x_block[qudit1, :] * self.x_block[qudit2, :])
        self.destab_phase_vector -= self.phase_order * (self.destab_x_block[qudit1, :] * self.destab_x_block[qudit2, :])

    def swap(self, qudit1: int, qudit2: int):
        self.x_block[[qudit1, qudit2], :] = self.x_block[[qudit2, qudit1], :].copy()
        self.z_block[[qudit1, qudit2], :] = self.z_block[[qudit2, qudit1], :].copy()
        self.destab_x_block[[qudit1, qudit2], :] = self.destab_x_block[[qudit2, qudit1], :].copy()
        self.destab_z_block[[qudit1, qudit2], :] = self.destab_z_block[[qudit2, qudit1], :].copy()

    def apply_gate(self, gate_id: int, qudit_idx: int, target_idx: int, arg0: Optional[int | float] = None):
        if is_gate_noisy(gate_id):
            return
        if arg0 is None or (isinstance(arg0, float) and math.isnan(arg0)):
            power = 1
        else:
            power = int(arg0) % self.dimension

        gate_name = gate_id_to_name(gate_id)
        gate_operations = {
            "H": lambda: self.hadamard(qudit_idx),
            "H_INV": lambda: self.hadamard_inv(qudit_idx),
            "P": lambda: self.phase(qudit_idx),
            "P_INV": lambda: self.phase_inv(qudit_idx),
            "X": lambda: self.x(qudit_idx, power),
            "X_INV": lambda: self.x_inv(qudit_idx, power),
            "Z": lambda: self.z(qudit_idx, power),
            "Z_INV": lambda: self.z_inv(qudit_idx, power),
            "CNOT": lambda: self.cnot(qudit_idx, target_idx),
            "CNOT_INV": lambda: self.cnot_inv(qudit_idx, target_idx),
            "CZ": lambda: self.cz(qudit_idx, target_idx),
            "CZ_INV": lambda: self.cz_inv(qudit_idx, target_idx),
            "SWAP": lambda: self.swap(qudit_idx, target_idx),
            "MULTIPLY": lambda: self.multiply(qudit_idx, power),
            "MULTIPLY_INV": lambda: self.multiply_inv(qudit_idx, power),
        }
        try:
            gate_operations[gate_name]()
        except KeyError:
            raise ValueError(f"Unknown gate: {gate_name}")

    # ── Measurement ─────────────────────────────────────────────────────

    def measure(self, qudit_index: int) -> int:
        first_xpow = None
        for i in range(self.num_qudits):
            xpow = self.x_block[qudit_index, i] % self.dimension
            if xpow > 0:
                first_xpow = i
                if xpow != 1:
                    inverse = pow(int(xpow), -1, self.dimension)
                    self.exponentiate(first_xpow, inverse)
                break
        self.x_block %= self.dimension
        self.z_block %= self.dimension
        self.phase_vector %= self.order
        self.destab_x_block %= self.dimension
        self.destab_z_block %= self.dimension
        self.destab_phase_vector %= self.order
        if first_xpow is not None:
            return self._random_measurement(qudit_index, first_xpow)
        return self._det_measurement(qudit_index)

    def _random_measurement(self, qudit_index: int, first_xpow: int) -> int:
        for i in range(self.num_qudits):
            if self.destab_x_block[qudit_index, i] != 0:
                destab_factor = -self.destab_x_block[qudit_index, i] % self.dimension
                commute_phase = np.dot(self.destab_z_block[:, i], self.x_block[:, first_xpow]*destab_factor)
                commute_phase += np.dot(self.x_block[:, first_xpow], self.z_block[:, first_xpow]) * destab_factor*(destab_factor-1)//2 * self.phase_order
                self.destab_x_block[:, i] = (self.destab_x_block[:, i] + self.x_block[:, first_xpow] * destab_factor) % self.dimension
                self.destab_z_block[:, i] = (self.destab_z_block[:, i] + self.z_block[:, first_xpow] * destab_factor) % self.dimension
                self.destab_phase_vector[i] = (self.destab_phase_vector[i] + self.phase_vector[first_xpow]*destab_factor + self.phase_order * commute_phase) % self.order
            if self.x_block[qudit_index, i] != 0 and i != first_xpow:
                stab_factor = -self.x_block[qudit_index, i] % self.dimension
                commute_phase = np.dot(self.z_block[:, i], self.x_block[:, first_xpow]*stab_factor)
                commute_phase += np.dot(self.x_block[:, first_xpow], self.z_block[:, first_xpow]) * stab_factor*(stab_factor-1)//2 * self.phase_order
                self.x_block[:, i] = (self.x_block[:, i] + self.x_block[:, first_xpow] * stab_factor) % self.dimension
                self.z_block[:, i] = (self.z_block[:, i] + self.z_block[:, first_xpow] * stab_factor) % self.dimension
                self.phase_vector[i] = (self.phase_vector[i] + self.phase_vector[first_xpow]*stab_factor + self.phase_order * commute_phase) % self.order

        self.destab_x_block[:, first_xpow] = self.x_block[:, first_xpow]
        self.destab_z_block[:, first_xpow] = self.z_block[:, first_xpow]
        self.destab_phase_vector[first_xpow] = self.phase_vector[first_xpow]
        self.z_block[:, first_xpow] = 0
        self.z_block[qudit_index, first_xpow] = 1
        self.x_block[:, first_xpow] = 0
        measurement_outcome = random.choice(range(self.dimension))
        self.phase_vector[first_xpow] = (-measurement_outcome * self.phase_order) % self.order
        return measurement_outcome

    def _det_measurement(self, qudit_index: int) -> int:
        ancilla_x = np.zeros(self.num_qudits, dtype=np.int64)
        ancilla_z = np.zeros(self.num_qudits, dtype=np.int64)
        ancilla_phase = 0
        for i in range(self.num_qudits):
            factor = self.destab_x_block[qudit_index, i] % self.dimension
            if factor != 0:
                commute_phase = np.dot(ancilla_z, factor * self.x_block[:, i])
                commute_phase += np.dot(self.x_block[:, i], self.z_block[:, i]) * factor*(factor-1)//2 * self.phase_order
                ancilla_x += self.x_block[:, i] * factor
                ancilla_z += self.z_block[:, i] * factor
                ancilla_phase += (factor * self.phase_vector[i] + self.phase_order * commute_phase)
        ancilla_x %= self.dimension
        ancilla_z %= self.dimension
        ancilla_phase %= self.order
        measurement_outcome = (-ancilla_phase // self.phase_order) % self.dimension
        return measurement_outcome

    def exponentiate(self, col: int, exponent: int):
        self.phase_vector[col] *= exponent
        self.phase_vector[col] += np.dot(self.x_block[:, col], self.z_block[:, col]) * exponent*(exponent-1)//2 * self.phase_order
        self.x_block[:, col] *= exponent
        self.z_block[:, col] *= exponent
        self.phase_vector[col] %= self.order
