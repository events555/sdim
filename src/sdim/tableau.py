from dataclasses import dataclass
from sympy import Matrix, eye, zeros, gcd, pprint
from typing import Optional
from diophantine import solve
from functools import cached_property

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
    phase_vector: Optional[Matrix] = None
    z_block: Optional[Matrix] = None
    x_block: Optional[Matrix] = None

    def __post_init__(self):
        if self.phase_vector is None:
            self.phase_vector = zeros(1, self.num_qudits)
        if self.z_block is None:
            self.z_block = eye(self.num_qudits)
        if self.x_block is None:
            self.x_block = zeros(self.num_qudits)


    @cached_property
    def coprime(self) -> set:
        return {i for i in range(1, self.order) if gcd(i, self.order) == 1}
    
    @cached_property
    def even(self) -> bool:
        return self.dimension % 2 == 0
    
    @cached_property
    def order(self) -> int:
        return self.dimension * 2 if self.even else self.dimension
    
    @property
    def weyl_block(self) -> Matrix:
        """Return the Z and X blocks as a vertically stacked matrix, known as the Weyl block."""
        return Matrix.vstack(self.z_block, self.x_block)
    
    @property
    def tableau(self) -> Matrix:
        """Return the phase vector and the Weyl blocks as a vertically stacked matrix."""
        return Matrix.vstack(self.phase_vector, self.weyl_block)

    def _print_labeled_matrix(self, label, matrix):
            print(f"{label}:")
            pprint(matrix)

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
    def _symplectic_product(row1: Matrix, row2: Matrix, num_qudits: int) -> int:
        """Compute the symplectic product of two rows."""
        return row1[:num_qudits, :].dot(row2[num_qudits:, :]) - row1[num_qudits:, :].dot(row2[:num_qudits, :])

    def symplectic_product(self, index1: int, index2: int) -> int:
        """Compute the symplectic product of two generators."""
        return self.z_block.col(index1).dot(self.x_block.col(index2)) - self.z_block.col(index2).dot(self.x_block.col(index1))

    def append(self, pauli_vector) -> Matrix:
        if pauli_vector.rows != (2*self.num_qudits + 1) or pauli_vector.cols != 1:
            raise ValueError(f"Pauli vector dimensions do not match. Expected {2*self.num_qudits + 1} rows, got {pauli_vector.rows}")
        new_phase = pauli_vector[0, 0]
        new_z = pauli_vector[1:self.num_qudits+1, 0]
        new_x = pauli_vector[self.num_qudits+1:, 0]
        self.phase_vector = Matrix.hstack(self.phase_vector, Matrix([[new_phase]]))
        self.z_block = Matrix.hstack(self.z_block, new_z)
        self.x_block = Matrix.hstack(self.x_block, new_x)
        
    def update(self, pauli_vector: Matrix, generator_index: int) -> None:
        if pauli_vector.rows != (2*self.num_qudits + 1) or pauli_vector.cols != 1:
            raise ValueError(f"Pauli vector dimensions do not match. Expected {2*self.num_qudits + 1} rows, got {pauli_vector.rows}")
        
        if generator_index < 0 or generator_index >= self.z_block.cols:
            raise ValueError(f"Invalid generator index. Must be between 0 and {self.z_block.cols - 1}")

        new_phase = pauli_vector[0, 0]
        new_z = pauli_vector[1:self.num_qudits+1, 0]
        new_x = pauli_vector[self.num_qudits+1:, 0]

        self.phase_vector[0, generator_index] = new_phase
        self.z_block[:, generator_index] = new_z
        self.x_block[:, generator_index] = new_x

    def add_generators(self, index1: int, index2: int, scalar: int = 1):
        """Add the generators at index2 to the generators at index1."""
        self.z_block[:, index1] += self.z_block[:, index2] * scalar
        self.x_block[:, index1] += self.x_block[:, index2] * scalar
        self.phase_vector[index1] += (scalar*(self.phase_vector[index2] + (self.symplectic_product(index1, index2)//2))) % self.order
    
    def multiply_generator(self, index: int, scalar: int, allow_non_coprime: bool = False):
        """Multiply the generators at index by a scalar."""
        if scalar not in self.coprime and not allow_non_coprime:
            raise ValueError(f"Scalar {scalar} is not coprime with the order {self.order}.")
        self.z_block[:, index] *= scalar
        self.z_block[:, index] %= self.order
        self.x_block[:, index] *= scalar
        self.x_block[:, index] %= self.order
        self.phase_vector[index] *= scalar
        self.phase_vector[index] %= self.order

    def swap_generators(self, index1: int, index2: int):
        """Swap generators at index1 to index2."""
        if index1 < 0 or index2 < 0 or index1 >= self.z_block.cols or index2 >= self.z_block.cols:
            raise ValueError(f"Invalid indices. Must be between 0 and {self.z_block.cols - 1}")
        self.z_block = self.z_block.elementary_col_op(op='n<->m', col1=index1, col2=index2)
        self.x_block = self.x_block.elementary_col_op(op='n<->m', col1=index1, col2=index2)
        self.phase_vector = self.phase_vector.elementary_col_op(op='n<->m', col1=index1, col2=index2)
        
    def _generate_auxiliary_column(self, weyl_vector: Matrix) -> Matrix:
        if weyl_vector.rows != (2*self.num_qudits) and weyl_vector.cols != 1:
            raise ValueError("Pauli vector dimensions do not match expected number from Tableau.")
        u0 = Matrix.zeros(self.tableau.rows, 1)
        u0[0, 0] = self.dimension
        aux_matrix = u0
        for i in range(1, self.tableau.rows):
            uj = Matrix.zeros(self.tableau.rows, 1)
            uj[i, 0] = self.dimension
            uj[0, 0] = self._symplectic_product(uj[1:, :], weyl_vector, self.num_qudits)//2
            aux_matrix = Matrix.hstack(aux_matrix, uj)
        return aux_matrix
    
    def _get_single_eta(self, qudit_index: int):
        """Get the eta value assuming a Z measurement at specified index"""
        row = self.x_block.row(qudit_index)
        
        if all(x == 0 for x in row):
            return self.dimension
        
        if self.x_block.cols > 1:
            row = self._eliminate_columns(qudit_index, row)
        
        return self._get_eta_as_divisor(row)

    def _eliminate_columns(self, qudit_index: int, row: Matrix) -> Matrix:
        left, right = 0, 1
        while right < self.x_block.cols:
            if row[left] != 0 and row[right] != 0:
                self._eliminate_non_zero_pair(left, right, row)
            elif row[left] != 0 and row[right] == 0:
                self.swap_generators(left, right)
            left, right = left + 1, right + 1
            row = self.x_block.row(qudit_index)
        return row

    def _eliminate_non_zero_pair(self, left: int, right: int, row: Matrix):
        g = gcd(row[right], self.order)
        if row[left] % g == 0:
            k = (-row[left] * pow(row[right], -1, self.order)) % self.order
            self.add_generators(left, right, k)
        else:
            k = (-row[right] * pow(row[left], -1, self.order)) % self.order
            self.add_generators(right, left, k)
            self.swap_generators(left, right)

    def _get_eta_as_divisor(self, row):
        for i in range(row.cols - 1, -1, -1):
            if row[i] != 0:
                for alpha in self.coprime:
                    value = (row[i] * alpha) % self.order
                    if self.dimension % value == 0:
                        self.multiply_generator(i, alpha)
                        return value
        return None
    
    def _create_weyl_vector(self, qudit_index: int) -> Matrix:
        weyl_vector = Matrix.zeros(2*self.num_qudits, 1)
        weyl_vector[qudit_index, 0] = 1
        return weyl_vector
    
    def _prepare_tableau_matrix(self, weyl_vector: Matrix, s: int) -> Matrix:
        auxiliary_column = self._generate_auxiliary_column(s * weyl_vector)
        ones_column = self.order * Matrix.ones(self.tableau.rows, 1)
        return Matrix.hstack(auxiliary_column, self.tableau, ones_column)
    
    def _prepare_excluding_commuting_matrix(self, new_stabilizer: Matrix) -> Matrix:
        excluding_commuting = self.tableau[:, :-1]
        excluding_commuting = Matrix.hstack(excluding_commuting, new_stabilizer)
        return Matrix.hstack(excluding_commuting, self.dimension * Matrix.ones(self.tableau.rows, 1))

    def _is_last_column_identity(self, last_column: Matrix) -> bool:
        return (last_column[1:, 0] % self.dimension == Matrix.zeros(2*self.num_qudits, 1))
    
    def _handle_non_deterministic_case(self, new_stabilizer: Matrix, s: int, qudit_index: int, measurement_value: int) -> MeasurementResult:
        last_column = self.tableau[:, -1] * s
        last_column_index = self.tableau.cols - 1

        if self._is_last_column_identity(last_column):
            self.update(new_stabilizer, last_column_index)
        else:
            excluding_commuting = self._prepare_excluding_commuting_matrix(new_stabilizer)
            result = solve(excluding_commuting, last_column)
            
            if not result:
                self.multiply_generator(last_column_index, s, allow_non_coprime=True)
                self.append(new_stabilizer)
            else:
                self.update(new_stabilizer, last_column_index)

        return MeasurementResult(qudit_index=qudit_index, deterministic=False, measurement_value=measurement_value)

    def measure_z(self, qudit_index: int) -> MeasurementResult:
        weyl_vector = self._create_weyl_vector(qudit_index)
        eta = self._get_single_eta(qudit_index)
        s = self.dimension // eta
        tableau_matrix = self._prepare_tableau_matrix(weyl_vector, s)
        for t in range(self.dimension):
            solution = Matrix.vstack(Matrix([t]), s*weyl_vector)
            result = solve(tableau_matrix, solution)
            if result:
                kappa = (t*eta)//self.dimension
                measurement_value = self._generate_measurement_outcome(kappa, eta, self.dimension)
                if s == 1:
                    return MeasurementResult(qudit_index=qudit_index, deterministic=True, measurement_value=measurement_value)
                
                new_stabilizer = Matrix.vstack(Matrix([measurement_value]), weyl_vector)
                return self._handle_non_deterministic_case(new_stabilizer, s, qudit_index, measurement_value)
            
        return None
    
    def multiply(self, qudit_index: int, scalar: int):
        """Apply multiplication gate to qudit at index 
        given a value in the multiplicative group of units modulo d"""
        if scalar not in self.coprime:
            raise ValueError(f"Scalar {scalar} is not coprime with the order {self.order}.")
        self.z_block[qudit_index, :] *= pow(scalar, -1, self.order)
        self.z_block[qudit_index, :] %= self.order
        self.x_block[qudit_index, :] *= scalar
        self.x_block[qudit_index, :] %= self.order

    def hadamard(self, qudit_index: int):
        """Apply generalized Hadamard gate to qudit at index."""
        # Create temporary copies of the rows
        temp_z = self.z_block.row(qudit_index)
        temp_x = self.x_block.row(qudit_index)
        
        # Assign the swapped rows back to the matrices
        self.z_block[qudit_index, :] = temp_x
        self.x_block[qudit_index, :] = -temp_z

    def phase(self, qudit_index: int):
        """Apply phase gate to qudit at index."""
        self.z_block[qudit_index, :] += self.x_block[qudit_index, :]

    def cnot(self, control_index: int, target_index: int):
        """Apply CNOT gate to control and target qudits."""
        self.z_block[control_index, :] -= self.z_block[target_index, :]
        self.x_block[target_index, :] += self.x_block[control_index, :]


        



