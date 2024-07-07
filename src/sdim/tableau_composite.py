import random
import math
from typing import Tuple, Optional, List
from .paulistring import PauliString
from .tableau import Tableau
from .tableau_simulator import MeasurementResult
import numpy as np


def apply_H(tableau: Tableau, qudit_index: int, _) -> Tuple[Tableau, Optional[MeasurementResult]]:
    """
    Apply H gate to qudit at qudit_index
    X -> Z
    Z -> X!
    """
    for pauli in tableau.generator:
        pauli.xpow[qudit_index], pauli.zpow[qudit_index] = (pauli.zpow[qudit_index]) * (tableau.dimension - 1), pauli.xpow[qudit_index] # swap and set xpow to (d-1)*zpow, no need to track phases
    return tableau, None

def apply_P(tableau: Tableau, qudit_index: int, _) -> Tuple[Tableau, Optional[MeasurementResult]]:
    """
    Apply P gate to qudit at qudit_index

    d is odd:
    X -> XZ
    Z -> Z

    d is even:
    XZ -> w^(1/2) XZ
    Z -> Z
    """
    for pauli in tableau.generator:
        pauli.zpow[qudit_index] = (pauli.xpow[qudit_index] + pauli.zpow[qudit_index]) % tableau.order
    return tableau, None

def apply_CNOT(tableau: Tableau, control: int, target: int) -> Tuple[Tableau, Optional[MeasurementResult]]:
    """
    Apply CNOT gate to control and target qudits
    XI -> XX
    IX -> IX
    ZI -> ZI
    IZ -> Z!Z
    """
    for pauli in tableau.generator:
        pauli.xpow[target] = (pauli.xpow[target] + pauli.xpow[control]) % tableau.order
        pauli.zpow[control] = (pauli.zpow[control] - pauli.zpow[target]) % tableau.order
    return tableau, None

def measure(tableau: Tableau, qudit_index: int, _) -> Tuple[Tableau, Optional[MeasurementResult]]:
    """
    Measure a qudit in the tableau
    Args:
        tableau: The tableau to measure
        qudit_index: The index of the qudit to measure
        _: Placeholder for the simulator interface
    Returns:
        The tableau after measurement and the measurement result
    """
    # Compute phase vector, assuming Z measurement
    pauli_vector = PauliString(tableau.num_qudits, dimension=tableau.dimension)
    pauli_vector.zpow[qudit_index] = 1
    phase_vector = [symplectic_inner_product(pauli_vector, pauli) for pauli in tableau.generator]
    for i, pauli in enumerate(tableau.generator):
        pauli.zpow[qudit_index] = 0
        pauli.xpow[qudit_index] = phase_vector[i]
    commuting_col = tableau.simplify_row_to_single_column(qudit_index)
    print(commuting_col)
    eta = tableau.generator[commuting_col].xpow[qudit_index] % tableau.order
    s = tableau.dimension//eta
    print(s)
    # s = tableau.dimension//eta
    # temp = tableau
    # if tableau.even:
    #     # Add auxiliary columns to the tableau
    #     u0 = PauliString(tableau.num_qudits, dimension=tableau.dimension) 
    #     u0.phase = tableau.dimension
    #     temp.generator.append(u0)
    #     for i in range(1, 2 * tableau.num_qudits + 1):
    #         ui = PauliString(tableau.num_qudits, dimension=tableau.dimension)
    #         if i < tableau.num_qudits + 1:
    #             ui.zpow[i] = tableau.dimension
    #             ui.phase = (tableau.dimension//2) * symplectic_inner_product(ui/tableau.dimension, s * pauli_vector)
    #         else:
    #             ui.xpow[i - tableau.num_qudits - 1] = tableau.dimension
    #             ui.phase = (tableau.dimension//2) * symplectic_inner_product(ui/tableau.dimension, s * pauli_vector)
    #         temp.generator.append(ui)
    # else:
    #     temp.generator.append(-s*pauli_vector)
    return tableau, None

def symplectic_inner_product(row1: PauliString, row2: PauliString) -> int:
    """
    Computes the symplectic inner product of two Pauli strings
    Args:
        row1: Pauli string
        row2: Pauli string
    Returns:
        The symplectic inner product
    """
    return np.sum(row1.xpow * row2.zpow - row1.zpow * row2.xpow)
