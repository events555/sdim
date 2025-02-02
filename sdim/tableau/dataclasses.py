from dataclasses import dataclass
from typing import Optional, Tuple
from functools import cached_property
import numpy as np
from math import gcd

@dataclass
class Tableau:
    """
    Represents a stabilizer tableau for a quantum circuit simulation.

    This class encapsulates the phase vector, Z block, and X block that 
    describe the state of a quantum system in the stabilizer formalism.
    The stabilizers are stored as columns in the Z and X blocks.

    Attributes:
        num_qudits (int): The number of qudits in the system.
        dimension (int): The dimension of each qudit (default is 2 for qubits).
        phase_vector (np.ndarray): The phase vector of the tableau.
        z_block (np.ndarray): The Z block of the tableau.
        x_block (np.ndarray): The X block of the tableau.
    """

    num_qudits: int = 1
    dimension: int = 2
    phase_vector: Optional[np.ndarray] = None
    z_block: Optional[np.ndarray] = None
    x_block: Optional[np.ndarray] = None

    def __post_init__(self):
        """
        Initializes the tableau with default values if not provided.
        """
        if self.phase_vector is None:
            self.phase_vector = np.zeros(self.num_qudits, dtype=np.int64)
        if self.z_block is None:
            self.z_block = np.eye(self.num_qudits, dtype=np.int64)
        if self.x_block is None:
            self.x_block = np.zeros((self.num_qudits, self.num_qudits), dtype=np.int64)

    def modulo(self):
        """
        Applies the modulo operators to the phase vector and stabilizers according to the order and dimension
        """
        self.phase_vector %= self.phase_order
        self.z_block %= self.dimension
        self.x_block %= self.dimension

    @cached_property
    def coprime_order(self) -> set:
        """
        Returns a set of integers coprime to the order.

        Returns:
            set: Integers coprime to the order.
        """
        return {i for i in range(1, self.order) if gcd(i, self.order) == 1}
    
    @cached_property
    def coprime_dimension(self) -> set:
        """
        Returns a set of integers coprime to the dimension.

        Returns:
            set: Integers coprime to the dimension.
        """
        return {i for i in range(1, self.dimension) if gcd(i, self.dimension) == 1}
    
    @cached_property
    def prime(self) -> bool:
        """
        Checks if the dimension is prime.

        Returns:
            bool: True if the dimension is prime, False otherwise.
        """
        return not any(self.dimension % i == 0 for i in range(2, self.dimension))
    
    @property
    def even(self) -> bool:
        """
        Checks if the dimension is even.

        Returns:
            bool: True if the dimension is even, False otherwise.
        """
        return self.dimension % 2 == 0
    
    @property
    def order(self) -> int:
        """
        Calculates the order of the Weyl-Heisenberg group.

        Returns:
            int: The order of the Weyl-Heisenberg group.
        """
        return self.dimension * 2 if self.even else self.dimension
    
    @property
    def phase_order(self) -> int:
        """
        Calculates the order of the phase.

        Returns:
            int: The order of the phase.
        """
        return 2 if self.even else 1
    
    @property
    def pauli_size(self) -> int:
        """
        Calculates the size of the Pauli group.

        Returns:
            int: The size of the Pauli group.
        """
        return 2 * self.num_qudits + 1
    
    @property
    def stab_tableau(self) -> np.ndarray:
        """
        Returns the full stabilizer tableau.

        Returns:
            np.ndarray: The stabilizer tableau as a vertically stacked matrix.
        """
        return np.vstack((self.phase_vector, self.z_block, self.x_block))

    def _print_labeled_matrix(self, label: str, matrix: np.ndarray):
        """
        Prints a labeled matrix.

        Args:
            label (str): The label for the matrix.
            matrix (np.ndarray): The matrix to print.
        """
        print(f"{label}:")
        print(matrix)

    def print_phase_vector(self):
        """
        Prints the phase vector of the tableau.
        """
        self._print_labeled_matrix("Phase Vector", self.phase_vector)

    def print_z_block(self):
        """
        Prints the Z block of the tableau.
        """
        self._print_labeled_matrix("Z Block", self.z_block)

    def print_x_block(self):
        """
        Prints the X block of the tableau.
        """
        self._print_labeled_matrix("X Block", self.x_block)

    def print_tableau(self):
        """
        Prints the full tableau, including phase vector, Z block, and X block.
        """
        self.print_phase_vector()
        self.print_z_block()
        self.print_x_block()


@dataclass
class MeasurementResult:
    """
    Represents the result of a qudit measurement in a quantum circuit.

    Attributes:
        qudit_index (int): The index of the measured qudit.
        deterministic (bool): Whether the measurement result was deterministic.
        measurement_value (int): The measured value of the qudit.
    """

    qudit_index: int
    deterministic: bool
    measurement_value: int
    stabilizer_tableau: Optional[Tableau] = None
    
    def __str__(self) -> str:
        """
        Returns a string representation of the measurement result.

        Returns:
            str: A human-readable description of the measurement result.
        """
        measurement_type_str = "deterministic" if self.deterministic else "random"
        return f"Measured qudit ({self.qudit_index}) as ({self.measurement_value}) and was {measurement_type_str}"

    def __repr__(self) -> str:
        """
        Returns a string representation of the measurement result.

        Returns:
            str: Same as __str__ method.
        """
        return str(self)
    
    def get_tableau(self):
        """
        Returns the stabilizer tableau.

        Returns:
            Tableau: The stabilizer tableau.
        """
        if self.stabilizer_tableau is None:
            raise ValueError("Stabilizer tableau not recorded during measurement")
        return self.stabilizer_tableau