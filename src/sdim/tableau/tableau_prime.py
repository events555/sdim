import numpy as np
import random
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Tuple
from math import gcd
from sdim.tableau.dataclasses import MeasurementResult, Tableau


@dataclass
class ExtendedTableau(Tableau):
    destab_phase_vector: Optional[np.ndarray] = None
    destab_z_block: Optional[np.ndarray] = None
    destab_x_block: Optional[np.ndarray] = None

    @property
    def destab_tableau(self) -> np.ndarray:
        """Return the phase vector and the Weyl blocks as a vertically stacked matrix."""
        return np.vstack((self.destab_phase_vector, self.destab_z_block, self.destab_x_block))
    
    @property
    def tableau(self) -> np.ndarray:
        """Return the full tableau with stabilizers and destabilizers"""
        return np.hstack((self.stab_tableau, self.destab_tableau))

    def __post_init__(self):
        super().__post_init__()
        if self.destab_z_block is None:
            self.destab_z_block = np.zeros((self.num_qudits, self.num_qudits), dtype=np.int64)
        if self.destab_x_block is None:
            self.destab_x_block = np.eye(self.num_qudits, dtype=np.int64)
        if self.destab_phase_vector is None:
            self.destab_phase_vector = np.zeros(self.num_qudits, dtype=np.int64)
            
    
    def hadamard(self, qudit_index: int):
        """
        Apply H gate to qudit at qudit_index
        X -> Z
        Z -> X!
        """
        for i in range(self.num_qudits):
            # We gain a phase from commuting XZ that depends on the product of xpow and zpow but multiply by 2 because we are tracking omega 1/2
            # ie. HXZP' = ZX! = w^d-1 XZ
            self.x_block[qudit_index, i], self.z_block[qudit_index, i] = self.z_block[qudit_index, i] * (self.dimension - 1), self.x_block[qudit_index, i]
            self.phase_vector[i] += self.phase_order * self.x_block[qudit_index, i] * self.z_block[qudit_index, i]
            self.phase_vector[i] %= self.order

            self.destab_x_block[qudit_index, i], self.destab_z_block[qudit_index, i] = self.destab_z_block[qudit_index, i] * (self.dimension - 1), self.destab_x_block[qudit_index, i]
            self.destab_phase_vector[i] += self.phase_order * self.destab_x_block[qudit_index, i] * self.destab_z_block[qudit_index, i]
            self.destab_phase_vector[i] %= self.order

    def phase(self, qudit_index: int):
        """
        Apply P gate to qudit at qudit_index
        d is odd:
        X -> XZ
        Z -> Z

        d is even:
        XZ -> w^(1/2) XZ
        Z -> Z
        """
        for i in range(self.num_qudits):
            if self.even:
                # Original commutation was xpow*(xpow-1)/2, but we are tracking number of omega 1/2 so we multiply by 2
                # We also gain an omega 1/2 for every xpow so we get 2*xpow*(xpow-1)/2 + xpow
                # This simplifies to xpow^2
                self.phase_vector[i] += self.x_block[qudit_index, i] ** 2
                self.phase_vector[i] %= self.order
                self.destab_phase_vector[i] += self.destab_x_block[qudit_index, i] ** 2
                self.destab_phase_vector[i] %= self.order
            else:
                # We gain a phase from commuting XZ depending on the number of X from PXP' = XZ
                # ie. PXXXP' = XZXZXZ = w^3 XXXZZZ
                # This followed from (XZ)^r = w^(r(r-1)/2)X^r Z^r
                self.phase_vector[i] += self.x_block[qudit_index, i] * (self.x_block[qudit_index, i]-1) // 2
                self.phase_vector[i] %= self.order
                self.destab_phase_vector[i] += self.destab_x_block[qudit_index, i] * (self.destab_x_block[qudit_index, i]-1) // 2
                self.destab_phase_vector[i] %= self.order
            self.z_block[qudit_index, i] = (self.z_block[qudit_index, i] + self.x_block[qudit_index, i]) % self.dimension
            self.destab_z_block[qudit_index, i] = (self.destab_z_block[qudit_index, i] + self.destab_x_block[qudit_index, i]) % self.dimension
    
    def cnot(self, control: int, target: int):
        """
        Apply CNOT gate to control and target qudits
        XI -> XX
        IX -> IX
        ZI -> ZI
        IZ -> Z!Z

        Include w^(1/2) phase for all conjugations if d is even
        """
        for i in range(self.num_qudits):
            self.x_block[target, i] = (self.x_block[target, i] + self.x_block[control, i]) % self.dimension
            self.z_block[control, i] = (self.z_block[control, i] + (self.z_block[target, i] * (self.dimension - 1))) % self.dimension
            self.destab_x_block[target, i] = (self.destab_x_block[target, i] + self.destab_x_block[control, i]) % self.dimension
            self.destab_z_block[control, i] = (self.destab_z_block[control, i] + (self.destab_z_block[target, i] * (self.dimension - 1))) % self.dimension
            
    def measure(self, qudit_index: int) -> MeasurementResult:
        """
        Measure in Z basis qudit at qudit_index
        """
        first_xpow = None
        # Find the first non-zero X in the tableau zlogical
        for i in range(self.num_qudits):
            xpow = self.x_block[qudit_index, i]
            if xpow > 0:
                first_xpow = i
                if xpow != 1:
                    # Calculate multiplicative inverse
                    inverse = pow(xpow, -1, self.dimension) 
                    self.exponentiate(first_xpow, inverse)   
                break
        if first_xpow is not None:
            return self._random_measurement(qudit_index, first_xpow)
        return self._det_measurement(qudit_index)
    
    def _random_measurement(self, qudit_index: int, first_xpow: int) -> MeasurementResult:
        """
        Make Tableau commute with Z measurement operator at qudit_index using the generator at first_xpow
        
        Args:
            qudit_index: int - The index of the qudit to measure
            first_xpow: int - The index of the first stabilizer with a non-zero X power
        Returns:
            MeasurementResult - The measurement result with random outcome
        """
        # First make Tableau commute with Z measurement operator
        for i in range(self.num_qudits):
            if self.destab_x_block[qudit_index, i] != 0:
                destab_factor = pow(int(self.destab_x_block[qudit_index, i]), -1, self.dimension)
                commute_phase = np.dot(self.destab_z_block[:, i], self.x_block[:, first_xpow]) % self.dimension
                self.destab_x_block[:, i] += self.x_block[:, first_xpow] * destab_factor
                self.destab_z_block[:, i] += self.z_block[:, first_xpow] * destab_factor
                self.destab_x_block[:, i] %= self.dimension
                self.destab_z_block[:, i] %= self.dimension
                self.destab_phase_vector[i] += (self.phase_vector[first_xpow] + self.phase_order * commute_phase)*destab_factor
                self.destab_phase_vector[i] %= self.order
            if self.x_block[qudit_index, i] != 0 and i != first_xpow:
                stab_factor = pow(int(self.x_block[qudit_index, i]), -1, self.dimension)
                commute_phase = np.dot(self.z_block[:, i], self.x_block[:, first_xpow]) % self.dimension
                self.x_block[:, i] += self.x_block[:, first_xpow] * stab_factor
                self.z_block[:, i] += self.z_block[:, first_xpow] * stab_factor
                self.x_block[:, i] %= self.dimension
                self.z_block[:, i] %= self.dimension
                self.phase_vector[i] += (self.phase_vector[first_xpow] + self.phase_order * commute_phase)*stab_factor
                self.phase_vector[i] %= self.order
            
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
        Args:
            qudit_index: int - The index of the qudit to measure
        Returns:
            MeasurementResult - The measurement result with deterministic outcome
        """
        ancilla_x = np.zeros(self.num_qudits, dtype=np.int64)
        ancilla_z = np.zeros(self.num_qudits, dtype=np.int64)
        ancilla_phase = 0
        for i in range(self.num_qudits):
            factor = self.destab_x_block[qudit_index, i]
            if factor != 0:
                commute_phase = np.dot(ancilla_z, self.x_block[:, i]) % self.dimension
                ancilla_x += self.x_block[:, i] * factor
                ancilla_z += self.z_block[:, i] * factor
                ancilla_x %= self.dimension
                ancilla_z %= self.dimension
                ancilla_phase += (self.phase_vector[i] + self.phase_order * commute_phase) * factor
        measurement_outcome = (-ancilla_phase // self.phase_order) % self.dimension
        return MeasurementResult(qudit_index, True, measurement_outcome)

    def exponentiate(self, col: int, exponent: int):
        """
        Exponentiate a Pauli string by n
        Example:
        (X^a Z^b)^n = w^(ab*n(n-1)/2) X^(na) Z^(nb) 
        Reminder that we multiply the global phase by n because it is outside the parentheses
        Args:
            row: Pauli string
            n: int
            phase_order: int
        Returns:
            The Pauli string after exponentiation
        """
        self.phase_vector[col] *= exponent
        self.phase_vector[col] += np.dot(self.x_block[:, col], self.z_block[:, col]) * exponent*(exponent-1)//2 * self.phase_order
        self.x_block[:, col] *=  exponent 
        self.z_block[:, col] *= exponent
        self.phase_vector[col] %= self.order
