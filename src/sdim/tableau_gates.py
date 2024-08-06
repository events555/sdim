import random
import math
import copy
from dataclasses import dataclass
from typing import Tuple, Optional, List
from .tableau import Tableau
import numpy as np

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
    
# Gate application functions
def apply_I(tableau: Tableau, qudit_index: int, _) -> None:
    """Apply identity gate (do nothing)"""
    return None

def apply_X(tableau: Tableau, qudit_index: int, _) -> None:
    """Apply Pauli X gate"""
    if tableau.dimension == 2:
        tableau.hadamard(qudit_index)
        tableau.phase(qudit_index)
        tableau.phase(qudit_index)
        tableau.hadamard(qudit_index)
    else:
        raise ValueError("X gate is only defined for qudits of dimension 2")
    return None

def apply_Z(tableau: Tableau, qudit_index: int, _) -> None:
    """Apply Pauli Z gate"""
    pass
    return None

def apply_H(tableau: Tableau, qudit_index: int, _) -> None:
    """Apply Hadamard gate"""
    tableau.hadamard(qudit_index)
    return None

def apply_P(tableau: Tableau, qudit_index: int, _) -> None:
    """Apply Phase gate"""
    tableau.phase(qudit_index)
    return None

def apply_CNOT(tableau: Tableau, control: int, target: int) -> None:
    """Apply CNOT gate"""
    tableau.cnot(control, target)
    return None

def apply_measure(tableau: Tableau, qudit_index: int, _) -> Optional[MeasurementResult]:
    """Apply measurement"""
    return tableau.measure_z(qudit_index)

