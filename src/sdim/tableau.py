import numpy as np
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Tuple
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
                try:
                    result = dp.solve(Matrix(excluding_commuting), Matrix(last_column))
                except Exception as e:
                    result = True 
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

    def _extended_euclidean(self, a: int, b: int) -> Tuple[int, int, int]:
        """
        Compute the extended Euclidean algorithm for a and b.
        Returns x, y, and gcd(a, b) such that a*x + b*y = gcd(a, b), with x > 0, y > 0, and one of x or y is coprime to self.order.
        """
        x, y, u, v = 1, 0, 0, 1
        while b != 0:
            q, r = a // b, a % b
            m, n = x - u * q, y - v * q
            a, b, x, y, u, v = b, r, u, v, m, n

        # Ensure x and y are positive
        if a < 0:
            x, y = -x, -y

        # Check if one of x or y is coprime to self.order
        if x in self.coprime or y in self.coprime:
            return x, y, a
        else:
            # Swap x and y to ensure one is coprime to self.order
            return y, x, a

    def _swap_columns_matrix(self, matrix: np.ndarray, col1: int, col2: int):
        matrix[:, [col1, col2]] = matrix[:, [col2, col1]]

    def column_reduction(self, tableau_matrix: np.ndarray, weyl_vector: np.ndarray, s: int) -> Optional[int]:
        pauli_vector = np.hstack((0, weyl_vector)).reshape(-1, 1) 
        full_tableau = np.hstack((tableau_matrix, -s*pauli_vector)) % self.order
        rows = full_tableau.shape[0] 
        cols = full_tableau.shape[1]
        pivot_row = 1
        for col in range(cols):
            if pivot_row >= rows:
                break
            # Find pivot in the current column that is coprime with self.order
            coprime = False
            gcd_col = False
            pivot_col = col
            for row in range(pivot_row, rows):
                row_gcd = np.gcd.reduce(full_tableau[row, col:])
                if np.all(full_tableau[row, col+1:] == 0):
                    if np.all(full_tableau[row+1:, col] == 0):
                        break
                    else:
                        continue
                for i in range(col, cols):
                    if full_tableau[row, i] in self.coprime:
                        pivot_row = row
                        coprime = True
                        if pivot_col != i:
                            self._swap_columns_matrix(full_tableau, col, i)
                        break
                    if full_tableau[row, i] == row_gcd:
                            pivot_row = row
                            gcd_col = True
                            if pivot_col != i:
                                self._swap_columns_matrix(full_tableau, col, i)
                            break
                if not coprime and not gcd_col:
                    for i in range(col, cols):
                        for j in range(i + 1, cols):
                            if np.gcd(full_tableau[row, i], full_tableau[row, j]) == row_gcd:
                                # Solve Bezout's identity to get the column with row_gcd
                                x, y, _ = self._extended_euclidean(full_tableau[row, i], full_tableau[row, j])
                                x, y = x % self.order, y % self.order
                                if x in self.coprime:
                                    full_tableau[:, i] *= x
                                    full_tableau[:, i] %= self.order
                                    self._add_column_matrix(full_tableau, j, i, y)
                                    
                                elif y in self.coprime:
                                    full_tableau[:, j] *= y
                                    full_tableau[:, j] %= self.order
                                    self._add_column_matrix(full_tableau, i, j, x)
                                    self._swap_columns_matrix(full_tableau, i, j)
                                else:
                                    raise ValueError("Could not find suitable column to swap.")
                                pivot_col = i
                                pivot_row = row
                                gcd_col = True
                                break
                        if coprime or gcd_col:
                            break
                if coprime or gcd_col:
                    break
            if coprime:
                # Calculate the multiplicative inverse of the pivot element modulo self.order
                pivot = int(full_tableau[pivot_row, pivot_col])
                inv_pivot = pow(pivot, -1, self.order)
                # Eliminate other columns using the pivot column
                for i in range(pivot_col, cols):
                    if i != pivot_col and full_tableau[pivot_row, i] != 0:
                        factor = (-int(full_tableau[pivot_row, i]) * inv_pivot) % self.order
                        self._add_column_matrix(full_tableau, pivot_col, i, factor)
            if gcd_col:
                # eliminate other columns using the gcd_col
                pivot = int(full_tableau[pivot_row, pivot_col])
                for i in range(pivot_col, cols):
                    target = int(full_tableau[pivot_row, i])
                    if i != pivot_col and target != 0:
                        g = gcd(pivot, self.order)
                        if full_tableau[pivot_row, i] % g == 0:
                            factor = ((-target // g) * pow(pivot // g, -1, self.order // g)) % (self.order // g)
                            self._add_column_matrix(full_tableau, pivot_col, i, factor)
            pivot_row += 1
        solution = full_tableau[:, -1]
        return solution[0]
            
    def _create_measurement_result(self, t: int, eta: int, s: int, qudit_index: int, weyl_vector: np.ndarray) -> MeasurementResult:
        kappa = (t * eta) // self.dimension
        measurement_value = self._generate_measurement_outcome(kappa, eta, self.dimension)
        
        if s == 1:
            return MeasurementResult(qudit_index=qudit_index, deterministic=True, measurement_value=measurement_value)
        
        new_stabilizer = np.hstack((measurement_value, weyl_vector))
        return self._handle_non_deterministic_case(new_stabilizer, s, qudit_index, measurement_value)

    def measure_z(self, qudit_index: int) -> Optional[MeasurementResult]:
        weyl_vector = np.zeros(2*self.num_qudits, dtype=np.int64)
        weyl_vector[qudit_index] = 1
        eta = self._get_single_eta(qudit_index)
        s = self.dimension // eta
        if self.even:
            aux_matrix = np.zeros((self.pauli_size, self.pauli_size-1), dtype=np.int64)
            for i in range(self.pauli_size-1):
                aux_matrix[i+1, i] = self.dimension
            tableau_matrix = np.hstack((aux_matrix, self.tableau))
        else:
            tableau_matrix = self.tableau
        t = 0 if s == self.dimension else self.column_reduction(tableau_matrix, weyl_vector, s)
        return self._create_measurement_result(t, eta, s, qudit_index, weyl_vector)
        
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

    def hadamard_inv(self, qudit_index: int):
        """Apply inverse generalized Hadamard gate to qudit at index."""
        # Swap and negate the values
        new_z_block = -self.x_block[qudit_index, :].copy()
        new_x_block = self.z_block[qudit_index, :].copy()

        # Apply the modulus operation
        self.z_block[qudit_index, :] = new_z_block % self.order
        self.x_block[qudit_index, :] = new_x_block % self.order

    def phase(self, qudit_index: int):
        """Apply phase gate to qudit at index."""
        self.z_block[qudit_index, :] += self.x_block[qudit_index, :]
        self.z_block[qudit_index, :] %= self.order

    def phase_inv(self, qudit_index: int):
        """Apply inverse phase gate to qudit at index."""
        self.z_block[qudit_index, :] -= self.x_block[qudit_index, :]
        self.z_block[qudit_index, :] %= self.order

    def cnot(self, control_index: int, target_index: int):
        """Apply CNOT gate to control and target qudits."""
        self.z_block[control_index, :] -= self.z_block[target_index, :]
        self.x_block[target_index, :] += self.x_block[control_index, :]
        self.z_block[control_index, :] %= self.order
        self.x_block[target_index, :] %= self.order

    def cnot_inv(self, control_index: int, target_index: int):
        """Apply inverse CNOT gate to control and target qudits."""
        self.z_block[control_index, :] += self.z_block[target_index, :]
        self.x_block[target_index, :] -= self.x_block[control_index, :]
        self.z_block[control_index, :] %= self.order
        self.x_block[target_index, :] %= self.order

    def x(self, qudit_index: int):
        """Apply Pauli X gate to qudit at index."""
        factor = self.z_block[qudit_index, :]
        self.phase_vector += factor
        self.phase_vector %= self.dimension

    def x_inv(self, qudit_index: int):
        """Apply Pauli X inverse gate to qudit at index."""
        factor = self.z_block[qudit_index, :]
        self.phase_vector -= factor
        self.phase_vector %= self.dimension
    
    def z(self, qudit_index: int):
        """Apply Pauli Z gate to qudit at index."""
        factor = self.x_block[qudit_index, :]
        self.phase_vector -= factor
        self.phase_vector %= self.dimension
    
    def z_inv(self, qudit_index: int):
        """Apply Pauli Z inverse gate to qudit at index."""
        factor = self.x_block[qudit_index, :]
        self.phase_vector += factor
        self.phase_vector %= self.dimension
    


