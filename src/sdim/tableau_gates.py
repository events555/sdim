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
def apply_I(tableau: Tableau, qudit_index: int, _, exact: bool = False) -> None:
    """Apply identity gate (do nothing)"""
    return None

def apply_X(tableau: Tableau, qudit_index: int, _, exact: bool = False) -> None:
    """Apply Pauli X gate"""
    tableau.x(qudit_index)
    return None

def apply_X_inv(tableau: Tableau, qudit_index: int, _, exact: bool = False) -> None:
    """Apply Pauli X inverse gate"""
    tableau.x_inv(qudit_index)  
    return None

def apply_Z(tableau: Tableau, qudit_index: int, _, exact: bool = False) -> None:
    """Apply Pauli Z gate"""
    tableau.z(qudit_index)
    return None

def apply_Z_inv(tableau: Tableau, qudit_index: int, _, exact: bool = False) -> None:
    """Apply Pauli Z inverse gate"""
    tableau.z_inv(qudit_index)
    return None

def apply_H(tableau: Tableau, qudit_index: int, _, exact: bool = False,) -> None:
    """Apply Hadamard gate"""
    tableau.hadamard(qudit_index)
    return None

def apply_H_inv(tableau: Tableau, qudit_index: int, _, exact: bool = False) -> None:
    """Apply Hadamard inverse gate"""
    tableau.hadamard_inv(qudit_index)
    return None

def apply_P(tableau: Tableau, qudit_index: int, _, exact: bool = False) -> None:
    """Apply Phase gate"""
    tableau.phase(qudit_index)
    return None

def apply_P_inv(tableau: Tableau, qudit_index: int, _, exact: bool = False) -> None:
    """Apply Phase inverse gate"""
    tableau.phase_inv(qudit_index)
    return None

def apply_CNOT(tableau: Tableau, control: int, target: int, exact: bool = False) -> None:
    """Apply CNOT gate"""
    tableau.cnot(control, target)
    return None

def apply_CNOT_inv(tableau: Tableau, control: int, target: int, exact: bool = False) -> None:
    """Apply CNOT inverse gate"""
    tableau.cnot_inv(control, target)
    return None

def apply_measure(tableau: Tableau, qudit_index: int, _, exact: bool = False) -> Optional[MeasurementResult]:
    """Apply measurement"""
    return tableau.measure_z(qudit_index, exact)

def apply_SWAP(tableau: Tableau, qudit_index: int, target_index: int, exact: bool = False) -> None:
    """Apply SWAP gate
    Taken from Beaudrap Lemma 6 (eq 19)
    """
    tableau.cnot(qudit_index, target_index)
    tableau.cnot_inv(target_index, qudit_index)
    tableau.cnot(qudit_index, target_index)
    tableau.hadamard(qudit_index)
    tableau.hadamard(qudit_index)
    return None

