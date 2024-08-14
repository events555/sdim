import numpy as np
from dataclasses import dataclass
from functools import cached_property
from typing import Optional
from math import gcd
from .diophantine import solve
import diophantine as dp
from sympy import Matrix

@dataclass
class MeasurementResult:
    qudit_index: int
    deterministic: bool
    measurement_value: int

    def __str__(self):
        measurement_type_str = "deterministic" if self.deterministic else "random"
        return f"Measured qudit ({self.qudit_index}) as ({self.measurement_value}) and was {measurement_type_str}"

    def __repr__(self):
        return str(self)

@dataclass
class Tableau:
    num_qudits: int = 1
    dimension: int = 2
    phase_vector: Optional[np.ndarray] = None
    z_block: Optional[np.ndarray] = None
    x_block: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.phase_vector is None:
            self.phase_vector = np.zeros(self.num_qudits, dtype=np.int64)
        if self.z_block is None:
            self.z_block = np.eye(self.num_qudits, dtype=np.int64)
        if self.x_block is None:
            self.x_block = np.zeros((self.num_qudits, self.num_qudits), dtype=np.int64)

    @cached_property
    def coprime(self) -> set:
        return {i for i in range(1, self.order) if gcd(i, self.order) == 1}
    
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
    def weyl_size(self) -> int:
        return 2 * self.num_qudits
    
    @property
    def pauli_size(self) -> int:
        return 2 * self.num_qudits + 1
    
    @property
    def weyl_block(self) -> np.ndarray:
        """Return the Z and X blocks as a vertically stacked matrix, known as the Weyl block."""
        return np.vstack((self.z_block, self.x_block))
    
    @property
    def tableau(self) -> np.ndarray:
        """Return the phase vector and the Weyl blocks as a vertically stacked matrix."""
        return np.vstack((self.phase_vector, self.weyl_block))

    def _print_labeled_matrix(self, label, matrix):
        print(f"{label}:")
        print(matrix)

    def print_phase_vector(self):
        self._print_labeled_matrix("Phase Vector", self.phase_vector)

    def print_z_block(self):
        self._print_labeled_matrix("Z Block", self.z_block)

    def print_x_block(self):
        self._print_labeled_matrix("X Block", self.x_block)

    def print_tableau(self):
        self.print_phase_vector()
        self.print_z_block()
        self.print_x_block()

    def print_weyl_block(self):
        self._print_labeled_matrix("Weyl Block", self.weyl_block)

    @staticmethod
    def _generate_measurement_outcome(kappa: int, eta: int, dimension: int) -> int:
        """Given distribution parameters generate a random measurement result.
        .. math::
            \kappa + \eta \mathbb{Z}_d 
        """
        import random
        if eta == 0 or eta == dimension:
            return kappa % dimension
        else:
            distribution = (eta * random.randint(0, dimension - 1)) % dimension
            return (kappa + distribution) % dimension
        
    @staticmethod
    def _symplectic_product(row1: np.ndarray, row2: np.ndarray, num_qudits: int) -> int:
        """Compute the symplectic product of two rows."""
        return np.dot(row1[:num_qudits], row2[num_qudits:]) - np.dot(row1[num_qudits:], row2[:num_qudits])

    def symplectic_product(self, index1: int, index2: int) -> int:
        """Compute the symplectic product of two generators."""
        return np.dot(self.z_block[:, index1], self.x_block[:, index2]) - np.dot(self.z_block[:, index2], self.x_block[:, index1])

    def append(self, pauli_vector: np.ndarray) -> None:
        if pauli_vector.size != self.pauli_size:
            raise ValueError(f"Pauli vector dimensions do not match. Expected {2*self.num_qudits + 1} rows, got {pauli_vector.shape[0]}")
        new_phase = pauli_vector[0]
        new_z = np.c_[pauli_vector[1:self.num_qudits+1]]
        new_x = np.c_[pauli_vector[self.num_qudits+1:]]
        self.phase_vector = np.hstack((self.phase_vector, new_phase))
        self.z_block = np.hstack((self.z_block, new_z))
        self.x_block = np.hstack((self.x_block, new_x))

    def update(self, pauli_vector: np.ndarray, index: int) -> None:
        if pauli_vector.size != self.pauli_size:
            raise ValueError(f"Pauli vector dimensions do not match. Expected {2*self.num_qudits + 1} rows, got {pauli_vector.shape[0]}")
        
        if index < 0 or index >= self.z_block.shape[1]:
            raise ValueError(f"Invalid generator index. Must be between 0 and {self.z_block.shape[1] - 1}")
        self.phase_vector[index] = pauli_vector[0]
        self.z_block[:, index] = pauli_vector[1:self.num_qudits+1]
        self.x_block[:, index] = pauli_vector[self.num_qudits+1:]

    def add_generators(self, index1: int, index2: int, scalar: int = 1):
        """Add the generators at column index2 to the generators at column index1."""
        self.z_block[:, index1] += self.z_block[:, index2] * scalar
        self.x_block[:, index1] += self.x_block[:, index2] * scalar
        self.phase_vector[index1] += (scalar*(self.phase_vector[index2] + (self.symplectic_product(index1, index2)//2 if self.even else 0))) % self.order
        self.z_block[:, index1] %= self.order
        self.x_block[:, index1] %= self.order

    def multiply_generator(self, index: int, scalar: int, allow_non_coprime: bool = False):
        """Multiply the generators at column index by a scalar."""
        if scalar not in self.coprime and not allow_non_coprime:
            raise ValueError(f"Scalar {scalar} is not coprime with the order {self.order}.")
        self.z_block[:, index] = (self.z_block[:, index] * scalar) % self.order
        self.x_block[:, index] = (self.x_block[:, index] * scalar) % self.order
        self.phase_vector[index] = (self.phase_vector[index] * scalar) % self.order

    def swap_generators(self, index1: int, index2: int):
        """Swap generators at index1 to index2."""
        if index1 < 0 or index2 < 0 or index1 >= self.z_block.shape[1] or index2 >= self.z_block.shape[1]:
            raise ValueError(f"Invalid indices. Must be between 0 and {self.z_block.shape[1] - 1}")
        self.z_block[:, [index1, index2]] = self.z_block[:, [index2, index1]]
        self.x_block[:, [index1, index2]] = self.x_block[:, [index2, index1]]
        self.phase_vector[[index1, index2]] = self.phase_vector[[index2, index1]]
        
    def _generate_auxiliary_column(self, weyl_vector: np.ndarray) -> np.ndarray:
        if weyl_vector.size != 2*self.num_qudits:
            raise ValueError("Pauli vector dimensions do not match expected number from Tableau.")
        u0 = np.zeros(self.pauli_size, dtype=np.int64)
        u0[0] = self.dimension
        aux_matrix = u0.T
        for i in range(1, self.pauli_size):
            uj = np.zeros(self.pauli_size, dtype=np.int64)
            uj[i] = self.dimension
            uj[0] = self._symplectic_product(uj[1:], weyl_vector, self.num_qudits) // 2
            aux_matrix = np.vstack((aux_matrix, uj.T))
        return aux_matrix.T
    
    def _get_single_eta(self, qudit_index: int):
        """Get the eta value assuming a Z measurement at specified index"""
        row = self.x_block[qudit_index, :]
        
        if np.all(row == 0):
            return self.dimension
        
        if self.x_block.shape[1] > 1:
            row = self._eliminate_columns(qudit_index, row)
        return self._get_eta_as_divisor(row)

    def _eliminate_columns(self, qudit_index: int, row: np.ndarray) -> np.ndarray:
        left, right = 0, 1
        while right < self.x_block.shape[1]:
            if row[left] != 0 and row[right] != 0:
                self._eliminate_non_zero_pair(left, right, row)
            elif row[left] != 0 and row[right] == 0:
                self.swap_generators(left, right)
            left, right = left + 1, right + 1
            row = self.x_block[qudit_index, :]
        return row
        
    def _eliminate_non_zero_pair(self, left: int, right: int, row: np.ndarray):
        a, b = int(row[left]) % self.order, int(row[right]) % self.order
        
        # Try to solve: left + x * right ≡ 0 (mod self.order)
        g = gcd(b, self.order)
        if a % g == 0:
            x = ((-a // g) * pow(b // g, -1, self.order // g)) % (self.order // g)
            self.add_generators(left, right, x)
            return
        
        # If that doesn't work, try: right + x * left ≡ 0 (mod self.order)
        g = gcd(a, self.order)
        if b % g == 0:
            x = ((-b // g) * pow(a // g, -1, self.order // g)) % (self.order // g)
            self.add_generators(right, left, x)
            self.swap_generators(left, right)
            return
        
    def _get_eta_as_divisor(self, row: np.ndarray):
        for i in range(row.shape[0] - 1, -1, -1):
            if row[i] != 0:
                if self.dimension % row[i] == 0:
                    return row[i]
                else:
                    for alpha in self.coprime:
                        value = (row[i] * alpha) % self.order
                        if self.dimension % value == 0:
                            self.multiply_generator(i, alpha)
                            return value
        return None
    
    def _create_weyl_vector(self, qudit_index: int) -> np.ndarray:
        weyl_vector = np.zeros(2*self.num_qudits, dtype=np.int64)
        weyl_vector[qudit_index] = 1
        return weyl_vector
    
    def _prepare_tableau_matrix(self, weyl_vector: np.ndarray, s: int) -> np.ndarray:
        if self.prime and self.even:
            auxiliary_column = self._generate_auxiliary_column(s * weyl_vector)
            return np.hstack((auxiliary_column, self.tableau))
        if not self.prime and self.even:
            ones_column = self.order * np.ones((self.tableau.shape[0], 1), dtype=np.int64)
            return np.hstack((auxiliary_column, self.tableau, ones_column))
        else:
            return self.tableau
    
    def _prepare_excluding_commuting_matrix(self, new_stabilizer: np.ndarray) -> np.ndarray:
        excluding_commuting = self.tableau[:, :-1]
        excluding_commuting = np.hstack((excluding_commuting, np.c_[new_stabilizer]))
        return np.hstack((excluding_commuting, self.dimension * np.ones((self.pauli_size, 1), dtype=np.int64)))

    def _is_last_column_identity(self, last_column: np.ndarray) -> bool:
        return np.array_equal(last_column[1:] % self.dimension, np.zeros(2*self.num_qudits, dtype=np.int64))
    
    def _handle_non_deterministic_case(self, new_stabilizer: np.ndarray, s: int, qudit_index: int, measurement_value: int) -> MeasurementResult:
        last_column = self.tableau[:, -1] * s
        last_column_index = self.tableau.shape[1] - 1

        if self._is_last_column_identity(last_column):
            self.update(new_stabilizer, last_column_index)
        else:
            excluding_commuting = self._prepare_excluding_commuting_matrix(new_stabilizer)
            try:
                result = solve(excluding_commuting, last_column)
            except Exception as e:
                print(f"Exception occurred with numpy solve")
                # Fall back to sympy
                try:
                    result = dp.solve(Matrix(excluding_commuting), Matrix(last_column))
                except Exception as e:
                    print(f"Exception occurred with sympy solve")
                    result = []            
            if not result:
                self.multiply_generator(last_column_index, s, allow_non_coprime=True)
                self.append(new_stabilizer)
            else:
                self.update(new_stabilizer, last_column_index)

        return MeasurementResult(qudit_index=qudit_index, deterministic=False, measurement_value=measurement_value)

    def _add_column_matrix(self, matrix: np.ndarray, src_col: int, dest_col: int, factor: int):
        if self.even:
            phase_correction = self._symplectic_product(factor * matrix[1:, src_col], matrix[1:, dest_col], self.num_qudits) // 2
        else:
            phase_correction = 0
        matrix[:, dest_col] += factor * matrix[:, src_col]
        matrix[0, dest_col] += phase_correction
        matrix[:, dest_col] %= self.order

    def column_reduction(self, tableau_matrix: np.ndarray, weyl_vector: np.ndarray, s: int) -> Optional[int]:
        pauli_vector = np.hstack((0, weyl_vector)).reshape(-1, 1) 
        full_tableau = np.hstack((tableau_matrix, -s*pauli_vector)) % self.order
        n = full_tableau.shape[0]

        for i in range(1, n):
            # Skip row if all columns in the row are zero
            if np.all(full_tableau[i:] == 0):
                continue
            pivot_col = None
            for j in range(1, full_tableau.shape[1]):
                if full_tableau[i, j] in self.coprime:
                    pivot_col = j
                    full_tableau[:, [i, pivot_col]] = full_tableau[:, [pivot_col, i]]
                    pivot_col = i
                    break
            if pivot_col is not None:
                pivot = int(full_tableau[i, i])
                for j in range(1, full_tableau.shape[1]):
                    if full_tableau[i, j] != 0 and j != i:
                        factor = (-int(full_tableau[i, j]) * pow(pivot, -1, self.order))
                        self._add_column_matrix(full_tableau, i, j, factor)
            else:
                # If no coprime element is found, check whether there is more than one element equal to self.dimension and use one of them to eliminate the other
                dimension_cols = [j for j in range(1, full_tableau.shape[1]) if full_tableau[i, j] == self.dimension]
                if len(dimension_cols) > 1:
                    for j in dimension_cols:
                        if j != i:
                            self._add_column_matrix(full_tableau, i, j, 1)
        solution = full_tableau[:, -1]
        return solution[0]
            
    def _find_valid_t(self, tableau_matrix: np.ndarray, weyl_vector: np.ndarray, s: int) -> Optional[int]:
        if self.prime:
            return self.column_reduction(tableau_matrix, weyl_vector, s)
        else:
            for t in range(self.dimension):
                solution = np.hstack((t, s * weyl_vector))
                try:
                    if solve(tableau_matrix, solution):
                        return t
                except Exception as e:
                    print(f"Exception occurred with numpy solve")
                    # Fall back to sympy
                    try:
                        if dp.solve(Matrix(tableau_matrix), Matrix(solution)):
                            return t
                    except Exception as e:
                        print(f"Exception occurred with sympy solve")
                        # If both methods fail, continue to the next t
                        continue
            return None

    def _create_measurement_result(self, t: int, eta: int, s: int, qudit_index: int, weyl_vector: np.ndarray) -> MeasurementResult:
        kappa = (t * eta) // self.dimension
        measurement_value = self._generate_measurement_outcome(kappa, eta, self.dimension)
        
        if s == 1:
            return MeasurementResult(qudit_index=qudit_index, deterministic=True, measurement_value=measurement_value)
        
        new_stabilizer = np.hstack((measurement_value, weyl_vector))
        return self._handle_non_deterministic_case(new_stabilizer, s, qudit_index, measurement_value)

    def measure_z(self, qudit_index: int) -> Optional[MeasurementResult]:
        weyl_vector = self._create_weyl_vector(qudit_index)
        eta = self._get_single_eta(qudit_index)
        s = self.dimension // eta
        tableau_matrix = self._prepare_tableau_matrix(weyl_vector, s)

        t = self._find_valid_t(tableau_matrix, weyl_vector, s)
        
        if t is not None:
            return self._create_measurement_result(t, eta, s, qudit_index, weyl_vector)
        
        return None
    
    def multiply(self, qudit_index: int, scalar: int):
        """Apply multiplication gate to qudit at index 
        given a value in the multiplicative group of units modulo d"""
        if scalar not in self.coprime:
            raise ValueError(f"Scalar {scalar} is not coprime with the order {self.order}.")
        self.z_block[qudit_index, :] = (self.z_block[qudit_index, :] * pow(scalar, -1, self.order)) % self.order
        self.x_block[qudit_index, :] = (self.x_block[qudit_index, :] * scalar) % self.order

    def hadamard(self, qudit_index: int):
        """Apply generalized Hadamard gate to qudit at index."""
        self.z_block[qudit_index, :], self.x_block[qudit_index, :] = self.x_block[qudit_index, :], -self.z_block[qudit_index, :]
        self.z_block[qudit_index, :] %= self.order
        self.x_block[qudit_index, :] %= self.order

    def phase(self, qudit_index: int):
        """Apply phase gate to qudit at index."""
        self.z_block[qudit_index, :] += self.x_block[qudit_index, :]
        self.z_block[qudit_index, :] %= self.order

    def cnot(self, control_index: int, target_index: int):
        """Apply CNOT gate to control and target qudits."""
        self.z_block[control_index, :] -= self.z_block[target_index, :]
        self.x_block[target_index, :] += self.x_block[control_index, :]
        self.z_block[control_index, :] %= self.order
        self.x_block[target_index, :] %= self.order

        



