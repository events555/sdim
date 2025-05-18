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
        self.z_block %= self.dimension
        self.x_block %= self.dimension
        self.phase_vector %= self.order

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
        Calculates the order of the Weyl-Heisenberg group (2d for even and d for odd).

        Returns:
            int: The order of the Weyl-Heisenberg group.
        """
        return self.dimension * 2 if self.even else self.dimension
    
    @property
    def phase_order(self) -> int:
        """
        Calculates the order of the phase (2 for even 1 for odd).

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