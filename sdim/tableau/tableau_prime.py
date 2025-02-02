import numpy as np
import random
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Tuple
from math import gcd
from sdim.tableau.dataclasses import MeasurementResult, Tableau
from sdim.tableau.tableau_optimized import hadamard_optimized, phase_optimized
from numba import njit, prange

@dataclass
class ExtendedTableau(Tableau):
    """
    Represents an extended stabilizer tableau for quantum circuit simulation.

    This class extends the Tableau class by including destabilizer information,
    which allows for more efficient simulation of certain quantum operations.

    This follows as a generalization to prime dimensions from
    "Improved Simulation of Stabilizer Circuits" by Aaronson and Gottesman.

    Attributes:
        destab_phase_vector (np.ndarray): The phase vector for destabilizers.
        destab_z_block (np.ndarray): The Z block for destabilizers.
        destab_x_block (np.ndarray): The X block for destabilizers.
    """
    destab_phase_vector: Optional[np.ndarray] = None
    destab_z_block: Optional[np.ndarray] = None
    destab_x_block: Optional[np.ndarray] = None

    def print_destab_phase_vector(self):
        """
        Prints the phase vector of the destabilizer tableau.
        """
        self._print_labeled_matrix("Destabilizer Phase Vector", self.destab_phase_vector)

    def print_destab_z_block(self):
        """
        Prints the Z block of the destabilizer tableau.
        """
        self._print_labeled_matrix("Destabilizer Z Block", self.destab_z_block)

    def print_destab_x_block(self):
        """
        Prints the X block of the destabilizer tableau.
        """
        self._print_labeled_matrix("Destabilizer X Block", self.destab_x_block)

    def print_tableau(self):
        """
        Prints the full tableau, including phase vector, Z block, X block,
        and the destabilizer components.
        """
        super().print_tableau()
        self.print_destab_phase_vector()
        self.print_destab_z_block()
        self.print_destab_x_block()

    @property
    def destab_tableau(self) -> np.ndarray:
        """
        Returns the destabilizer tableau as a vertically stacked matrix.

        Returns:
            np.ndarray: The destabilizer tableau.
        """
        return np.vstack((self.destab_phase_vector, self.destab_z_block, self.destab_x_block))
    
    @property
    def tableau(self) -> np.ndarray:
        """
        Returns the full tableau with stabilizers and destabilizers.

        Returns:
            np.ndarray: The full tableau.
        """
        return np.hstack((self.stab_tableau, self.destab_tableau))

    def __post_init__(self):
        """
        Initializes the extended tableau with default values if not provided.
        """
        super().__post_init__()
        if self.destab_z_block is None:
            self.destab_z_block = np.zeros((self.num_qudits, self.num_qudits), dtype=np.int64)
        if self.destab_x_block is None:
            self.destab_x_block = np.eye(self.num_qudits, dtype=np.int64)
        if self.destab_phase_vector is None:
            self.destab_phase_vector = np.zeros(self.num_qudits, dtype=np.int64)

    def modulo(self):
        """
        Reduces the tableau modulo the qudit dimension.
        """
        super().modulo()
        self.destab_x_block %= self.dimension
        self.destab_z_block %= self.dimension
        self.destab_phase_vector %= self.order
    
    def hadamard(self, qudit_index: int):
        """
        Applies the Hadamard gate to the qudit at the specified index.

        The Hadamard gate performs the following transformations:

        | Input | Output   |
        |-------|----------|
        | $X$   | $Z$      |
        | $Z$   | $X^{-1}$ |

        The phase transformation is given by:
        
        $$ H\cdot X\cdot Z\ket{\psi} = Z\cdot X^{-1}\ket{\psi} = \omega^{d-1} XZ\ket{\psi} $$

        where $\omega = e^{2\pi i / d}$ and $d$ is the qudit dimension.

        Args:
            qudit_index (int): The index of the qudit to apply the Hadamard gate to.
        """
        hadamard_optimized(
            self.x_block, self.z_block, self.phase_vector,
            self.destab_x_block, self.destab_z_block, self.destab_phase_vector,
            qudit_index, self.num_qudits, self.phase_order
        )
    def hadamard_inv(self, qudit_index: int):
        """
        Applies the inverse Hadamard gate to the qudit at the specified index.

        The inverse Hadamard gate performs the following transformations:

        | Input | Output   |
        |-------|----------|
        | $X$   | $Z^{-1}$ |
        | $Z$   | $X$      |

        Args:
            qudit_index (int): The index of the qudit to apply the inverse Hadamard gate to.
        """
        hadamard_optimized(
            self.x_block, self.z_block, self.phase_vector,
            self.destab_x_block, self.destab_z_block, self.destab_phase_vector,
            qudit_index, self.num_qudits, self.phase_order
        )
    def phase(self, qudit_index: int):
        """
        Applies the Phase gate to the qudit at the specified index.

        The Phase gate transformations depend on whether the qudit dimension is odd or even:

        For odd dimensions:

        | Input | Output |
        |-------|--------|
        | $X$   | $XZ$   |
        | $Z$   | $Z$    |

        For even dimensions:

        | Input           | Output               |
        |-----------------|----------------------|
        | $X$             | $\omega^{1/2} XZ$    |
        | $Z$             | $Z$                  |

        Where $\omega = e^{2\pi i / d}$ and $d$ is the qudit dimension.

        The phase accumulation for even dimensions is given by:

        $$\\text{phase} += x^2 $$

        where $x$ is the X-power in the Pauli string.

        Args:
            qudit_index (int): The index of the qudit to apply the Phase gate to.
        """
        phase_optimized(self.x_block, self.z_block, self.phase_vector,
                       self.destab_x_block, self.destab_z_block,
                       self.destab_phase_vector, 
                       qudit_index,
                       self.num_qudits,
                       self.phase_order,
                       self.even)

    def phase_inv(self, qudit_index: int):
        """
        Applies the inverse Phase gate to the qudit at the specified index.

        The inverse Phase gate transformations depend on whether the qudit dimension is odd or even:

        For odd dimensions:

        | Input | Output    |
        |-------|-----------|
        | $X$   | $XZ^{-1}$ |
        | $Z$   | $Z$       |

        For even dimensions:

        | Input        | Output                  |
        |--------------|-------------------------|
        | $X$          | $\omega^{-1/2} XZ^{-1}$ |
        | $Z$          | $Z$                     |

        Args:
            qudit_index (int): The index of the qudit to apply the inverse Phase gate to.
        """
        phase_optimized(self.x_block, self.z_block, self.phase_vector,
                       self.destab_x_block, self.destab_z_block,
                       self.destab_phase_vector, 
                       qudit_index,
                       self.num_qudits,
                       self.phase_order,
                       self.even)
    def cnot(self, control: int, target: int):
        """
        Applies the CNOT gate with the specified control and target qudits.

        The CNOT gate performs the following transformations:

        | Input         | Output        |
        |---------------|---------------|
        | $X \otimes I$ | $X \otimes X$ |
        | $I \otimes X$ | $I \otimes X$ |
        | $Z \otimes I$ | $Z \otimes I$ |
        | $I \otimes Z$ | $Z^{-1}Z$     |


        Args:
            control (int): The index of the control qudit.
            target (int): The index of the target qudit.
        """
        for i in range(self.num_qudits):
            self.x_block[target, i] = (self.x_block[target, i] + self.x_block[control, i]) % self.dimension
            self.z_block[control, i] = (self.z_block[control, i] + (self.z_block[target, i] * (self.dimension - 1))) % self.dimension
            self.destab_x_block[target, i] = (self.destab_x_block[target, i] + self.destab_x_block[control, i]) % self.dimension
            self.destab_z_block[control, i] = (self.destab_z_block[control, i] + (self.destab_z_block[target, i] * (self.dimension - 1))) % self.dimension
    
    def cnot_inv(self, control: int, target: int):
        """
        Applies the inverse CNOT gate with the specified control and target qudits.

        The inverse CNOT gate transformations depend on whether the qudit dimension is odd or even:

        | Input         | Output        |
        |---------------|---------------|
        | $X \otimes X$ | $X \otimes I$ |
        | $I \otimes X$ | $I \otimes X$ |
        | $Z \otimes I$ | $Z \otimes I$ |
        | $Z^{-1}Z$     | $I \otimes Z$ |

        For even dimensions:

        | Input         | Output        |
        |---------------|---------------|
        | $X \otimes X$ | $X \otimes I$ |
        | $I \otimes X$ | $I \otimes X$ |
        | $Z \otimes I$ | $Z \otimes I$ |
        | $Z^{-1}Z$     | $I \otimes Z$ |
        Args:
            control (int): The index of the control qudit.
            target (int): The index of the target qudit.
        """
        for i in range(self.num_qudits):
            self.x_block[target, i] = (self.x_block[target, i] - self.x_block[control, i]) % self.dimension
            self.z_block[control, i] = (self.z_block[control, i] - (self.z_block[target, i] * (self.dimension - 1))) % self.dimension
            self.destab_x_block[target, i] = (self.destab_x_block[target, i] - self.destab_x_block[control, i]) % self.dimension
            self.destab_z_block[control, i] = (self.destab_z_block[control, i] - (self.destab_z_block[target, i] * (self.dimension - 1))) % self.dimension
            
    def measure(self, qudit_index: int) -> MeasurementResult:
        """
        Measures the qudit at the specified index in the Z basis.

        Args:
            qudit_index (int): The index of the qudit to measure.

        Returns:
            MeasurementResult: The result of the measurement, including whether it was
                               deterministic and the measured value.
        """
        first_xpow = None
        # Find the first non-zero X in the tableau zlogical
        for i in range(self.num_qudits):
            xpow = self.x_block[qudit_index, i] % self.dimension
            if xpow > 0:
                first_xpow = i
                if xpow != 1:
                    # Calculate multiplicative inverse
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
    
    def _random_measurement(self, qudit_index: int, first_xpow: int) -> MeasurementResult:
        """
        Make Tableau commute with Z measurement operator at qudit_index using the generator at first_xpow
        
        This method is called when the measurement outcome is not deterministic.

        Args:
            qudit_index (int): The index of the qudit to measure.
            first_xpow (int): The index of the first stabilizer with a non-zero X power.

        Returns:
            MeasurementResult: The result of the random measurement.
        """
        # First make Tableau commute with Z measurement operator
        for i in range(self.num_qudits):
            if self.destab_x_block[qudit_index, i] != 0:
                destab_factor = -self.destab_x_block[qudit_index, i] % self.dimension
                commute_phase = np.dot(self.destab_z_block[:, i], self.x_block[:, first_xpow]*destab_factor) # phase factor from commuting
                commute_phase += np.dot(self.x_block[:, first_xpow], self.z_block[:, first_xpow]) * destab_factor*(destab_factor-1)//2 * self.phase_order # phase factor from exponentiation
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
            
        # Set destabilizer equal to first_xpow
        self.destab_x_block[:, first_xpow] = self.x_block[:, first_xpow]
        self.destab_z_block[:, first_xpow] = self.z_block[:, first_xpow]
        self.destab_phase_vector[first_xpow] = self.phase_vector[first_xpow]
        # Generate measurement outcome
        self.z_block[:, first_xpow] = 0
        self.z_block[qudit_index, first_xpow] = 1
        self.x_block[:, first_xpow] = 0
        measurement_outcome = random.choice(range(self.dimension))
        self.phase_vector[first_xpow] = (-measurement_outcome * self.phase_order) % self.order
        return MeasurementResult(qudit_index, False, measurement_outcome)

    def _det_measurement(self, qudit_index: int) -> MeasurementResult:
        """
        Use ancilla to obtain the right phase value for the measurement outcome
        
        This method is called when the measurement outcome is deterministic.

        Args:
            qudit_index (int): The index of the qudit to measure.

        Returns:
            MeasurementResult: The result of the deterministic measurement.
        """
        ancilla_x = np.zeros(self.num_qudits, dtype=np.int64)
        ancilla_z = np.zeros(self.num_qudits, dtype=np.int64)
        ancilla_phase = 0
        for i in range(self.num_qudits):
            factor = self.destab_x_block[qudit_index, i] % self.dimension
            if factor != 0:
                commute_phase = np.dot(ancilla_z, factor * self.x_block[:, i]) # phase factor from commuting
                commute_phase += np.dot(self.x_block[:, i], self.z_block[:, i]) * factor*(factor-1)//2 * self.phase_order # phase factor from exponentiation
                ancilla_x += self.x_block[:, i] * factor
                ancilla_z += self.z_block[:, i] * factor
                ancilla_phase += (factor * self.phase_vector[i] + self.phase_order * commute_phase)
        ancilla_x %= self.dimension
        ancilla_z %= self.dimension
        ancilla_phase %= self.order
        measurement_outcome = (-ancilla_phase // self.phase_order) % self.dimension
        return MeasurementResult(qudit_index, True, measurement_outcome)

    def exponentiate(self, col: int, exponent: int):
        """
        Exponentiates a Pauli string by the given exponent.

        This operation performs the following transformation:
        $$(X^a Z^b)^n = \omega^{(ab\cdot n(n-1)/2)} X^{(na)} Z^{(nb)}$$

        Args:
            col (int): The column index of the Pauli string to exponentiate.
            exponent (int): The exponent to raise the Pauli string to.
        """
        self.phase_vector[col] *= exponent
        self.phase_vector[col] += np.dot(self.x_block[:, col], self.z_block[:, col]) * exponent*(exponent-1)//2 * self.phase_order
        self.x_block[:, col] *=  exponent 
        self.z_block[:, col] *= exponent
        self.phase_vector[col] %= self.order
