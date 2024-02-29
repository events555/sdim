from dataclasses import dataclass
import numpy as np
from .tableau import Tableau



@dataclass
class Gate:
    name: str
    arg_count: int
    gate_id: int
    tableau: Tableau
    unitary_matrix: np.ndarray

    def __str__(self):
        return f"{self.name} {self.gate_id}"
