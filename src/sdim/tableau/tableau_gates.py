import random
import math
import copy
from dataclasses import dataclass
from typing import Tuple, Optional, List
from .tableau_composite import WeylTableau
from .tableau_prime import ExtendedTableau
from .dataclasses import MeasurementResult, Tableau
import numpy as np
    
# Gate application functions
def apply_I(tableau: Tableau, qudit_index: int, _) -> None:
    """
    Apply identity gate (do nothing).

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the qudit to apply the gate to.
        _ : Unused target index.

    Returns:
        None
    """
    return None

def apply_X(tableau: Tableau, qudit_index: int, _) -> None:
    """
    Apply Pauli X gate.

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the qudit to apply the gate to.
        _ : Unused target index.

    Returns:
        None
    """
    if isinstance(tableau, WeylTableau):
        tableau.x(qudit_index)
    else:
        if tableau.even:
            tableau.hadamard(qudit_index)
            tableau.phase(qudit_index)
            tableau.phase(qudit_index)
            tableau.hadamard(qudit_index)
        else:
            tableau.hadamard(qudit_index)
            tableau.phase_inv(qudit_index)
            tableau.hadamard(qudit_index)
            tableau.hadamard(qudit_index)
            tableau.phase(qudit_index)
            tableau.hadamard(qudit_index)
    return None

def apply_X_inv(tableau: Tableau, qudit_index: int, _) -> None:
    """
    Apply Pauli X inverse gate.

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the qudit to apply the gate to.
        _ : Unused target index.

    Returns:
        None
    """
    if isinstance(tableau, WeylTableau):
        tableau.x_inv(qudit_index)
    else:
        if tableau.even:
            tableau.hadamard(qudit_index)
            tableau.phase(qudit_index)
            tableau.phase(qudit_index)
            tableau.hadamard(qudit_index)
        else:
            tableau.hadamard(qudit_index)
            tableau.phase(qudit_index)
            tableau.hadamard(qudit_index)
            tableau.hadamard(qudit_index)
            tableau.phase_inv(qudit_index)
            tableau.hadamard(qudit_index)
    return None

def apply_Z(tableau: Tableau, qudit_index: int, _) -> None:
    """
    Apply Pauli Z gate.

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the qudit to apply the gate to.
        _ : Unused target index.

    Returns:
        None
    """
    if isinstance(tableau, WeylTableau):
        tableau.z(qudit_index)
    else:
        if tableau.even:
            tableau.phase(qudit_index)
            tableau.phase(qudit_index)
        else:
            tableau.phase_inv(qudit_index)
            tableau.hadamard(qudit_index)
            tableau.hadamard(qudit_index)
            tableau.phase(qudit_index)
            tableau.hadamard(qudit_index)
            tableau.hadamard(qudit_index)
    return None

def apply_Z_inv(tableau: Tableau, qudit_index: int, _) -> None:
    """
    Apply Pauli Z inverse gate.

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the qudit to apply the gate to.
        _ : Unused target index.

    Returns:
        None
    """
    if isinstance(tableau, WeylTableau):
        tableau.z_inv(qudit_index)
    else:
        if tableau.even:
            tableau.phase_inv(qudit_index)
            tableau.phase_inv(qudit_index)
        else:
            tableau.phase(qudit_index)
            tableau.hadamard_inv(qudit_index)
            tableau.hadamard_inv(qudit_index)
            tableau.phase_inv(qudit_index)
            tableau.hadamard_inv(qudit_index)
            tableau.hadamard_inv(qudit_index)
    return None

def apply_H(tableau: Tableau, qudit_index: int, _) -> None:
    """
    Apply Hadamard gate.

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the qudit to apply the gate to.
        _ : Unused target index.

    Returns:
        None
    """
    tableau.hadamard(qudit_index)
    return None

def apply_H_inv(tableau: Tableau, qudit_index: int, _) -> None:
    """
    Apply Hadamard inverse gate.

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the qudit to apply the gate to.
        _ : Unused target index.

    Returns:
        None
    """
    tableau.hadamard_inv(qudit_index)
    return None

def apply_P(tableau: Tableau, qudit_index: int, _) -> None:
    """
    Apply Phase gate.

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the qudit to apply the gate to.
        _ : Unused target index.

    Returns:
        None
    """
    tableau.phase(qudit_index)
    return None

def apply_P_inv(tableau: Tableau, qudit_index: int, _) -> None:
    """
    Apply Phase inverse gate.

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the qudit to apply the gate to.
        _ : Unused target index.

    Returns:
        None
    """
    tableau.phase_inv(qudit_index)
    return None

def apply_CNOT(tableau: Tableau, control: int, target: int) -> None:
    """
    Apply CNOT gate.

    Args:
        tableau (Tableau): The quantum tableau.
        control (int): The index of the control qudit.
        target (int): The index of the target qudit.

    Returns:
        None
    """
    tableau.cnot(control, target)
    return None

def apply_CNOT_inv(tableau: Tableau, control: int, target: int) -> None:
    """
    Apply CNOT inverse gate.

    Args:
        tableau (Tableau): The quantum tableau.
        control (int): The index of the control qudit.
        target (int): The index of the target qudit.

    Returns:
        None
    """
    tableau.cnot_inv(control, target)
    return None

def apply_CZ(tableau: Tableau, control: int, target: int) -> None:
    """
    Apply CZ gate.

    Args:
        tableau (Tableau): The quantum tableau.
        control (int): The index of the control qudit.
        target (int): The index of the target qudit.

    Returns:
        None
    """
    tableau.hadamard(target)
    tableau.cnot(control, target)
    tableau.hadamard_inv(target)
    return None

def apply_CZ_inv(tableau: Tableau, control: int, target: int) -> None:
    """
    Apply CZ inverse gate.

    Args:
        tableau (Tableau): The quantum tableau.
        control (int): The index of the control qudit.
        target (int): The index of the target qudit.

    Returns:
        None
    """
    tableau.hadamard(target)
    tableau.cnot_inv(control, target)
    tableau.hadamard_inv(target)
    return None

def apply_measure(tableau: Tableau, qudit_index: int, _) -> Optional[MeasurementResult]:
    """
    Apply measurement.

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the qudit to measure.
        _ : Unused target index.

    Returns:
        Optional[MeasurementResult]: The result of the measurement, if applicable.
    """
    if isinstance(tableau, WeylTableau):
        return tableau.measure_z(qudit_index)
    else:
        return tableau.measure(qudit_index)

def apply_measure_x(tableau: Tableau, qudit_index: int, _) -> Optional[MeasurementResult]:
    """
    Apply X-basis measurement.

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the qudit to measure.
        _ : Unused target index.

    Returns:
        Optional[MeasurementResult]: The result of the measurement, if applicable.
    """
    tableau.hadamard(qudit_index)
    if isinstance(tableau, WeylTableau):
        return tableau.measure_z(qudit_index)
    else:
        return tableau.measure(qudit_index)

def apply_SWAP(tableau: Tableau, qudit_index: int, target_index: int) -> None:
    """
    Apply SWAP gate.

    Taken from Beaudrap Lemma 6 (eq 19)

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the first qudit to swap.
        target_index (int): The index of the second qudit to swap.

    Returns:
        None
    """
    tableau.cnot(qudit_index, target_index)
    tableau.cnot_inv(target_index, qudit_index)
    tableau.cnot(qudit_index, target_index)
    tableau.hadamard(qudit_index)
    tableau.hadamard(qudit_index)
    return None

def apply_reset(tableau: Tableau, qudit_index: int, _) -> Optional[MeasurementResult]:
    """
    Apply reset gate. IF the qudit is measured to be in the $\ket{1}$ state, reset it to the $\ket{0}$ state.

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the qudit to reset.
        _ : Unused target index.

    Returns:
        Optional[MeasurementResult]: The result of the measurement, if applicable.
    """
    if isinstance(tableau, WeylTableau):
        return tableau.measure_z(qudit_index)
    else:
        return tableau.measure(qudit_index)

