import random
import math
import copy
from dataclasses import dataclass
from typing import Tuple, Optional, List
from .tableau_composite import WeylTableau
from .tableau_prime import ExtendedTableau
from .dataclasses import MeasurementResult, Tableau
import numpy as np
import random
    
# Gate application functions
def apply_I(tableau: Tableau, qudit_index: int, *_) -> None:
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

def apply_X(tableau: Tableau, qudit_index: int, *_) -> None:
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

def apply_X_inv(tableau: Tableau, qudit_index: int, *_) -> None:
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

def apply_Z(tableau: Tableau, qudit_index: int, *_) -> None:
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

def apply_Z_inv(tableau: Tableau, qudit_index: int, *_) -> None:
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

def apply_H(tableau: Tableau, qudit_index: int, *_) -> None:
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

def apply_H_inv(tableau: Tableau, qudit_index: int, *_) -> None:
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

def apply_P(tableau: Tableau, qudit_index: int, *_) -> None:
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

def apply_P_inv(tableau: Tableau, qudit_index: int, *_) -> None:
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

def apply_CNOT(tableau: Tableau, control: int, target: int, *_) -> None:
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

def apply_CNOT_inv(tableau: Tableau, control: int, target: int, *_) -> None:
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

def apply_CZ(tableau: Tableau, control: int, target: int, *_) -> None:
    """
    Apply CZ gate.

    Args:
        tableau (Tableau): The quantum tableau.
        control (int): The index of the control qudit.
        target (int): The index of the target qudit.

    Returns:
        None
    """
    tableau.hadamard_inv(target)
    tableau.cnot(control, target)
    tableau.hadamard(target)
    return None

def apply_CZ_inv(tableau: Tableau, control: int, target: int, *_) -> None:
    """
    Apply CZ inverse gate.

    Args:
        tableau (Tableau): The quantum tableau.
        control (int): The index of the control qudit.
        target (int): The index of the target qudit.

    Returns:
        None
    """
    tableau.hadamard_inv(target)
    tableau.cnot_inv(control, target)
    tableau.hadamard(target)
    return None

def apply_measure(tableau: Tableau, qudit_index: int, *_) -> Optional[MeasurementResult]:
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

def apply_measure_x(tableau: Tableau, qudit_index: int, *_) -> Optional[MeasurementResult]:
    """
    Apply X-basis measurement.

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the qudit to measure.
        _ : Unused target index.

    Returns:
        Optional[MeasurementResult]: The result of the measurement, if applicable.
    """
    tableau.hadamard_inv(qudit_index)
    if isinstance(tableau, WeylTableau):
        return tableau.measure_z(qudit_index)
    else:
        return tableau.measure(qudit_index)

def apply_SWAP(tableau: Tableau, qudit_index: int, target_index: int, *_) -> None:
    """
    Apply SWAP gate.

    Taken from Beaudrap Lemma 6 (eq 19) for composite case
    Taken from Farinholt Figure 1 for prime case.

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the first qudit to swap.
        target_index (int): The index of the second qudit to swap.

    Returns:
        None
    """
    if isinstance(tableau, WeylTableau):
        tableau.cnot(qudit_index, target_index)
        tableau.cnot_inv(target_index, qudit_index)
        tableau.cnot(qudit_index, target_index)
        tableau.hadamard(qudit_index)
        tableau.hadamard(qudit_index)
    else:
        tableau.cnot(qudit_index, target_index)
        tableau.hadamard(qudit_index)
        tableau.hadamard(target_index)
        tableau.cnot(qudit_index, target_index)
        tableau.hadamard(qudit_index)
        tableau.hadamard(target_index)
        tableau.cnot(qudit_index, target_index)
        tableau.hadamard(target_index)
        tableau.hadamard(target_index)
    return None

def apply_reset(tableau: Tableau, qudit_index: int, *_) -> Optional[MeasurementResult]:
    """
    Apply reset gate. IF the qudit is measured to be in the $/ket{1}$ state, reset it to the $/ket{0}$ state.

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
    

def apply_single_qudit_noise(tableau : Tableau, qudit_index: int, _, params: dict) -> None:
    """
    Given a probability and noise channel type, probabilistically applies a qudit Pauli to a target qudit.  Supports channels for flips, phase dampening, and depolarizing noise.  
    Defaults to applying depolarizing noise with probability 0.5 if no parameters are passed.  
    For a tableau of dimension d and with a probability argument p, the channels behave in the following way:

    Flip: Applies a qudit Pauli of the form X^a on the target qudit with probability p, where a is randomly sampled from {0, 1, ..., d - 1}.

    Phase dampening: Applies a qudit Pauli of the form Z^a on the target qudit with probability p, where a is randomly sampled from {0, 1, ..., d - 1}.

    Depolarizing: Applies a qudit Pauli of the form of X^a Z^b with probability p, where a and b are elements of {0, 1, ..., d - 1}, and the tuple (a, b) is randomly sampled from the set {(a, b) | (a != 0) OR (b != 0)}.

    Args:
        tableau (Tableau): The quantum tableau.
        qudit_index (int): The index of the qudit on which to apply noise.
        _ : Unused target index.
        noise_channel ((named) str): The type of noise applied to the target qudit.  Flip, phase, and depolarizing errors are "f", "p", and "d", respectively.
        prob ((named) float): The probability that a non-identity Pauli gate is applied to the target qudit.

    Returns:
        None  
    """

    # Extract parameters
    # Default behavior is p=0.5, noise_channel='d'
    if params is None:
        prob = 0.5
        noise_channel = 'd'
    else:
        prob = float(params.get('prob', 0.5))
        noise_channel = params.get('noise_channel', 'd')

    # TODO: Sanity check parameters

    if noise_channel == 'd':
        num = random.uniform(0.0, 1.0)
        # Determine whether to apply I or not
        if num < 1.0 - prob:
            return None
        # Sample error from all non-identity Pauli gates 
        z_exp, x_exp = 0, 0
        while z_exp == 0 and x_exp == 0:
            z_exp, x_exp = random.randint(0, tableau.dimension - 1), random.randint(0, tableau.dimension - 1)
        # Apply the Pauli gates.  Order does not matter up to phase since measurement statistics are identical.
        for _ in range(x_exp):
            apply_X(tableau=tableau, qudit_index=qudit_index)
        for _ in range(z_exp):
            apply_Z(tableau=tableau, qudit_index=qudit_index)

        return None
    
    elif noise_channel == 'f' or noise_channel == 'p':
        num = random.uniform(0.0, 1.0)
        flip = (noise_channel == 'f')
        # Determine whether to apply I or not
        if num < 1.0 - prob:
            return None
        # Sample non-identity Pauli exponent
        op_exp = random.randint(1, tableau.dimension - 1)
        # Apply appropriate error
        if flip:
            for _ in range(op_exp):
                apply_X(tableau=tableau, qudit_index=qudit_index)
        else:
            for _ in range(op_exp):
                apply_Z(tableau=tableau, qudit_index=qudit_index)
        
        return None
    

    raise ValueError("Must specify a valid noise channel.")
    

