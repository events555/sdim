from .paulistring import PauliString
from sympy import *
from dataclasses import dataclass, field
from functools import cached_property
from typing import List
import numpy as np

@dataclass
class Tableau:
    num_qudits: int
    dimension: int = 2
    generator: List[PauliString] = field(init=False)
    phase_correction: np.ndarray = field(init=False)

    def __post_init__(self):
        self.generator = [PauliString(self.num_qudits, dimension=self.dimension) for _ in range(2*self.num_qudits)]
        self.phase_correction = np.zeros((2*self.num_qudits, 2*self.num_qudits), dtype=int)
        for i in range(self.num_qudits):
            self.generator[i][i] = "Z"
    @cached_property
    def order(self):
        return 2 * self.dimension if self.dimension % 2 == 0 else self.dimension
    
    @cached_property
    def even(self):
        return self.dimension % 2 == 0
    
    @cached_property
    def coprime(self):
        return {i for i in range(1, self.order) if gcd(i, self.order) == 1}
    
    def simplify_row_to_single_column(self, target_row):
        """
        Simplifies the target row to have at most one non-zero entry in the rightmost column.
        
        Args:
        target_row (int): The index of the row to simplify.
        """
        num_columns = len(self.generator)
        
        # Find the rightmost non-zero entry
        for col in range(num_columns - 1, -1, -1):
            entry = self.generator[col].zpow[target_row] 
            if entry != 0:
                pivot_col = col
                break
        else:
            # If no non-zero entry found, the row is already simplified
            return
        
        # Use the rightmost non-zero entry to eliminate others
        for col in range(pivot_col):
            current_entry = self.generator[col].zpow[target_row]
            if current_entry != 0:
                factor = current_entry * pow(entry, -1, self.dimension) % self.dimension
                self.add_generators(pivot_col, col, -factor)
        
        # Ensure the final non-zero entry is a divisor of the dimension
        if entry != 0:
            divisor = gcd(entry, self.dimension)
            if divisor != entry:
                factor = entry // divisor
                self.multiply_generator(pivot_col, pow(factor, -1, self.dimension))
        return pivot_col

    def add_generators(self, h, j, factor):
        """
        Adds factor * h-th generator to the j-th generator.
        
        Args:
        h (int): Index of the source generator
        j (int): Index of the target generator
        factor (int): The scalar factor to multiply the h-th generator by before adding
        """
        self.generator[j] = self.generator[j] + factor * self.generator[h]
        
        self.phase_correction[:, j] = (self.phase_correction[:, j] + factor * self.phase_correction[:, h]) % self.order
        self.phase_correction[j, :] = (self.phase_correction[j, :] - factor * self.phase_correction[h, :]) % self.order
        
        self.generator[j].phase = (self.generator[j].phase + factor * self.phase_correction[h, j]) % self.order

    def multiply_generator(self, col, factor):
        """
        Multiplies a generator by a scalar factor.
        """
        self.generator[col] = factor * self.generator[col]
        self.phase_correction[:, col] = (factor * self.phase_correction[:, col]) % self.order
        self.phase_correction[col, :] = (factor * self.phase_correction[col, :]) % self.order
    def row_echelon_form(self):
        """
        Perform Gaussian elimination on the tableau using coprime elements as pivots.
        """
        n = len(self.generator)
        m = self.num_qudits * 2  # Total number of X and Z components

        for i in range(n):
            # Find a suitable pivot
            pivot_row = self.find_pivot(i, i, m)
            if pivot_row is None:
                continue  # No suitable pivot found, move to next column

            # Swap rows if necessary
            if pivot_row != i:
                self.swap_rows(i, pivot_row)

            # Eliminate entries below the pivot
            for j in range(i + 1, n):
                self.eliminate(i, j, i, m)

    def find_pivot(self, start_row, col, m):
        """
        Find a suitable pivot element that is coprime with the order.
        """
        for row in range(start_row, len(self.generator)):
            for k in range(col, m):
                if k < self.num_qudits:
                    value = self.generator[row].xpow[k]
                else:
                    value = self.generator[row].zpow[k - self.num_qudits]
                
                if value in self.coprime:
                    return row
        return None

    def swap_rows(self, i, j):
        """
        Swap two rows in the tableau.
        """
        self.generator[i], self.generator[j] = self.generator[j], self.generator[i]
        self.phase_correction[i, :], self.phase_correction[j, :] = self.phase_correction[j, :], self.phase_correction[i, :].copy()

    def eliminate(self, pivot_row, target_row, col, m):
        """
        Eliminate the entry in the target row using the pivot row.
        """
        if col < self.num_qudits:
            multiplier = self.generator[target_row].xpow[col]
            pivot_value = self.generator[pivot_row].xpow[col]
        else:
            multiplier = self.generator[target_row].zpow[col - self.num_qudits]
            pivot_value = self.generator[pivot_row].zpow[col - self.num_qudits]

        if multiplier == 0:
            return

        inverse = next(x for x in range(1, self.order) if (x * pivot_value) % self.order == 1)
        factor = (multiplier * inverse) % self.order

        # Subtract factor * pivot_row from target_row
        self.generator[target_row] = self.generator[target_row] - factor * self.generator[pivot_row]
        self.phase_correction[target_row, :] = (self.phase_correction[target_row, :] - factor * self.phase_correction[pivot_row, :]) % self.order
        self.phase_correction[:, target_row] = (self.phase_correction[:, target_row] + factor * self.phase_correction[:, pivot_row]) % self.order

    def __str__(self):
        zlogical_str = ", ".join(str(ps) for ps in self.generator)
        return f"Z-Logical: [{zlogical_str}]"

    def print_matrix(self):
        num_generators = len(self.generator)
        num_qudits = self.num_qudits
        phase_row = ["Phase"] + [f"{self.generator[j].phase}" for j in range(num_generators)]
        print("\t".join(phase_row))
        # Print a separator line
        print("-" * (8 * num_generators))
        for i in range(num_qudits):
            row = [f"Z{i}"] + [f"{self.generator[j].zpow[i]}" for j in range(num_generators)]
            print("\t".join(row))

        # Print a separator line
        print("-" * (8 * num_generators))
        for i in range(num_qudits):
            row = [f"X{i}"] + [f"{self.generator[j].xpow[i]}" for j in range(num_generators)]
            print("\t".join(row))
    
    def print_phase_correction(self):
        print("Phase Correction Matrix:")
        print(self.phase_correction)

    def gate1(self, xmap, zmap):
        """
        Creates a Tableau representing a single qudit gate.
        Args:
            xmap: The output-side observable assuming the input-side is the logical X operator
            zmap: The output-side observable assuming the input-side is the logical Z operator
        """
        self.generator[0] = str(zmap)
        return self

    def gate2(self, xmap, zmap):
        """
        Creates a Tableau representing a two-qudit gate.
        Args:
            xmap: The output-side observable assuming the input-side is the logical X operator
            zmap: The output-side observable assuming the input-side is the logical Z operator
        """
        self.generator = [PauliString(self.num_qudits, z) for z in zmap]