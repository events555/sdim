import numpy as np
import diophantine as dp
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Tuple
from math import gcd
from sympy import Matrix
from sdim.tableau.dataclasses import MeasurementResult, Tableau
from sdim.diophantine import solve

@dataclass
class WeylTableau(Tableau):
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
        if scalar not in self.coprime_order and not allow_non_coprime:
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
        return self._get_eta_as_divisor(qudit_index, row)

    def _eliminate_columns(self, qudit_index: int, row: np.ndarray) -> np.ndarray:
        row_gcd = np.gcd.reduce(row)
        cols = row.shape[0]
        gcd_col = False
        coprime_col = False
        pivot_col = None
        for col in range(cols):
            if row[col] in self.coprime_order:
                coprime_col = True
                pivot_col = col
                break
            if row[col] == row_gcd:
                gcd_col = True
                pivot_col = col
                break
        if not coprime_col and not gcd_col:
            # Find a pair of columns with gcd equal to row_gcd
            for i in range(cols):
                for j in range(i + 1, cols):
                    if np.gcd(row[i], row[j]) == row_gcd:
                        x, y, _ = self._extended_euclidean(row[i], row[j])
                        x, y = x % self.order, y % self.order
                        if x in self.coprime_order:
                            self.multiply_generator(i, x)
                            self.add_generators(i, j, y)
                            pivot_col = i
                            gcd_col = True
                        elif y in self.coprime_order:
                            self.multiply_generator(j, y)
                            self.add_generators(j, i, x)
                            pivot_col = j
                            gcd_col = True
                        else:
                            raise ValueError("Could not find suitable column to swap.")
                        break
                if gcd_col:
                    break
        if coprime_col:
            # Calculate the multiplicative inverse of the pivot element modulo self.order
            pivot = int(row[pivot_col])
            inv_pivot = pow(pivot, -1, self.order)
            # Eliminate other columns using the pivot column
            for i in range(cols):
                if i != pivot_col and row[i] != 0:
                    factor = (-int(row[i]) * inv_pivot) % self.order
                    self.add_generators(i, pivot_col, factor)
        if gcd_col:
            # eliminate other columns using the gcd_col
            pivot = int(row[pivot_col])
            for i in range(cols):
                target = int(row[i])
                if i != pivot_col and target != 0:
                    g = gcd(pivot, self.order)
                    if row[i] % g == 0:
                        factor = ((-target // g) * pow(pivot // g, -1, self.order // g)) % (self.order // g)
                        self.add_generators(i, pivot_col, factor)
        last_col = cols - 1
        self.swap_generators(pivot_col, last_col)
        return row
        
    def _get_eta_as_divisor(self, qudit_index: int, row: np.ndarray):
        last_col = row.shape[0] - 1
        if row[-1] != 0:
            if self.dimension % row[-1]== 0:
                return row[-1]
            else:
                for alpha in self.coprime_order:
                    value = (row[-1] * alpha) % self.order
                    if self.dimension % value == 0:
                        self.multiply_generator(last_col, alpha)
                        return value
            # reduce the row modulo the dimension
            if row[-1] > self.dimension:
                self.x_block[qudit_index, -1] -= self.dimension
                self.phase_vector[-1] -= self.dimension//2
            else:
                self.x_block[qudit_index, -1] += self.dimension
                self.phase_vector[-1] += self.dimension//2
            for alpha in self.coprime_order:
                    value = (row[-1] * alpha) % self.order
                    if self.dimension % value == 0:
                        self.multiply_generator(last_col, alpha)
                        return value
        return None
      
    def _prepare_excluding_commuting_matrix(self, new_stabilizer: np.ndarray) -> np.ndarray:
        excluding_commuting = self.stab_tableau[:, :-1]
        excluding_commuting = np.hstack((excluding_commuting, np.c_[new_stabilizer]))
        return np.hstack((excluding_commuting, self.dimension * np.ones((self.pauli_size, 1), dtype=np.int64)))

    def _is_last_column_identity(self, last_column: np.ndarray) -> bool:
        return np.array_equal(last_column[1:] % self.dimension, np.zeros(2*self.num_qudits, dtype=np.int64))
    
    def _handle_non_deterministic_case(self, new_stabilizer: np.ndarray, s: int, qudit_index: int, measurement_value: int) -> MeasurementResult:
        last_column = self.stab_tableau[:, -1] * s
        last_column_index = self.stab_tableau.shape[1] - 1

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
        if x % self.order in self.coprime_order or y % self.order in self.coprime_order:
            return x, y, a
        raise ValueError("Could not find suitable x and y.")

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
            coprime = False
            gcd_col = False
            pivot_col = col
            for row in range(pivot_row, rows):
                if np.all(full_tableau[row, col:] == 0): # if everything to the right including pivot is zero, check next row
                    continue
                elif np.all(full_tableau[row, col+1:] == 0): # if everything to the right of the pivot is zero, go to next column if everything below is zero
                    if row < rows - 1 and np.all(full_tableau[row+1:, col] == 0):
                        break
                    continue
                if full_tableau[row, -1] != 0: # if the last column is non-zero, check if the element above is non zero and that everything between pivot and last column is non zero 
                    if self.num_qudits > 1 and row > 1:
                        if np.any(full_tableau[row, col+1:-1]) and full_tableau[row-1, col] != 0:
                            break
                row_gcd = np.gcd.reduce(full_tableau[row, col:])
                for i in range(col, cols-1):
                    if full_tableau[row, i] in self.coprime_order:
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
                    for i in range(col, cols-1):
                        for j in range(i + 1, cols-1):
                            if np.gcd(full_tableau[row, i], full_tableau[row, j]) == row_gcd:
                                # Solve Bezout's identity to get the column with row_gcd
                                x, y, _ = self._extended_euclidean(full_tableau[row, i], full_tableau[row, j])
                                x, y = x % self.order, y % self.order
                                if x in self.coprime_order:
                                    full_tableau[:, i] *= x
                                    full_tableau[:, i] %= self.order
                                    self._add_column_matrix(full_tableau, j, i, y)
                                    
                                elif y in self.coprime_order:
                                    full_tableau[:, j] *= y
                                    full_tableau[:, j] %= self.order
                                    self._add_column_matrix(full_tableau, i, j, x)
                                    self._swap_columns_matrix(full_tableau, i, j)
                                else:
                                    raise ValueError("Could not find suitable column to swap.")
                                if pivot_col != i:
                                    self._swap_columns_matrix(full_tableau, col, i)
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
        return full_tableau[0, -1]
            
    def _create_measurement_result(self, t: int, eta: int, s: int, qudit_index: int, weyl_vector: np.ndarray) -> MeasurementResult:
        kappa = (t * eta) // self.dimension
        measurement_value = self._generate_measurement_outcome(kappa, eta, self.dimension)
        
        if s == 1:
            return MeasurementResult(qudit_index=qudit_index, deterministic=True, measurement_value=measurement_value)
        
        new_stabilizer = np.hstack((measurement_value, weyl_vector))
        return self._handle_non_deterministic_case(new_stabilizer, s, qudit_index, measurement_value)

    def t_diophantine(self, tableau_matrix: np.ndarray, weyl_vector: np.ndarray, qudit_index: int, s: int) -> Optional[int]:
        modulo_constraint = np.ones((self.pauli_size, 1), dtype=np.int64) * self.order
        tableau_matrix = np.hstack((tableau_matrix, modulo_constraint))
        if self.even:
            tableau_matrix[0, qudit_index+self.num_qudits] = (-self.dimension * s // 2) % self.order
            identity = np.zeros((self.pauli_size, 1), dtype=np.int64)
            identity[0] = self.dimension
            tableau_matrix = np.hstack((identity, tableau_matrix))
        for t in range(self.dimension):
            solution = np.hstack((s*t, s*weyl_vector)) % self.order
            try:
                if solve(tableau_matrix, solution):
                    return t
            except Exception:
                sympy_tableau = Matrix(tableau_matrix)
                sympy_solution = Matrix(solution)
                try:
                    if dp.solve(sympy_tableau, sympy_solution):
                        return t
                except NotImplementedError:
                    return t

    def measure_z(self, qudit_index: int, exact: bool = False) -> Optional[MeasurementResult]:
        weyl_vector = np.zeros(2*self.num_qudits, dtype=np.int64)
        weyl_vector[qudit_index] = 1
        eta = self._get_single_eta(qudit_index)
        s = self.dimension // eta
        if s == self.dimension:
            return self._create_measurement_result(0, eta, s, qudit_index, weyl_vector)

        if self.even:
            aux_matrix = np.zeros((self.pauli_size, self.pauli_size-1), dtype=np.int64)
            for i in range(self.pauli_size-1):
                aux_matrix[i+1, i] = self.dimension
            tableau_matrix = np.hstack((aux_matrix, self.stab_tableau))
        else:
            tableau_matrix = self.stab_tableau

        if not exact:
            t = self.column_reduction(tableau_matrix, weyl_vector, s)
            # print("eta, t, s", eta, t, s)
            return self._create_measurement_result(t, eta, s, qudit_index, weyl_vector)
        else:
            t = self.t_diophantine(tableau_matrix, weyl_vector, qudit_index, s)
            # print("eta, t, s", eta, t, s)
            return self._create_measurement_result(t, eta, s, qudit_index, weyl_vector)
        
    def multiply(self, qudit_index: int, scalar: int):
        """Apply multiplication gate to qudit at index 
        given a value in the multiplicative group of units modulo d"""
        if scalar not in self.coprime_order:
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
    


