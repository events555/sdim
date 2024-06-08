import random
import math
from typing import Tuple, Optional
from dataclasses import dataclass
from .paulistring import PauliString
from .tableau import Tableau

@dataclass
class MeasurementResult:
    qudit_index: int
    deterministic: bool
    measurement_value: int

    def __str__(self):
        measurement_type_str = "deterministic" if self.deterministic else "random"
        return f"Measured qudit ({self.qudit_index}) as ({self.measurement_value}) and was {measurement_type_str}."

    def __repr__(self):
        return str(self)

def apply_H(tableau: Tableau, qudit_index: int, _) -> Tuple[Tableau, Optional[MeasurementResult]]:
    """
    Apply H gate to qudit at qudit_index
    X -> Z
    Z -> X!
    """
    for pauli in tableau.xlogical+tableau.zlogical:
        pauli.xpow[qudit_index], pauli.zpow[qudit_index] = (pauli.zpow[qudit_index]) * (tableau.dimension - 1), pauli.xpow[qudit_index] # swap and set xpow to (d-1)*zpow
        # We gain a phase from commuting XZ that depends on the product of xpow and zpow but multiply by 2 because we are tracking omega 1/2
        # ie. HXZP' = ZX! = w^d-1 XZ
        pauli.phase = (pauli.phase + tableau.phase_order*(pauli.xpow[qudit_index] * pauli.zpow[qudit_index])) % (tableau.phase_order*tableau.dimension)
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
    for pauli in tableau.xlogical+tableau.zlogical:
        if tableau.dimension % 2 == 0:
            # Original commutation was xpow*(xpow-1)/2, but we are tracking number of omega 1/2 so we multiply by 2
            # We also gain an omega 1/2 for every xpow so we get 2*xpow*(xpow-1)/2 + xpow
            # This simplifies to xpow^2
            pauli.phase = (pauli.phase + pauli.xpow[qudit_index]**2) % (tableau.phase_order*tableau.dimension)
        else:
            # We gain a phase from commuting XZ depending on the number of X from PXP' = XZ
            # ie. PXXXP' = XZXZXZ = w^3 XXXZZZ
            # This followed from (XZ)^r = w^(r(r-1)/2)X^r Z^r
            pauli.phase = (pauli.phase + pauli.xpow[qudit_index]*(pauli.xpow[qudit_index]-1)//2) % tableau.dimension
        pauli.zpow[qudit_index] = (pauli.xpow[qudit_index] + pauli.zpow[qudit_index]) % tableau.dimension
    return tableau, None


def apply_CNOT(tableau: Tableau, control: int, target: int) -> Tuple[Tableau, Optional[MeasurementResult]]:
    """
    Apply CNOT gate to control and target qudits
    XI -> XX
    IX -> IX
    ZI -> ZI
    IZ -> Z!Z

    Include w^(1/2) phase for all conjugations if d is even
    """
    for pauli in tableau.xlogical+tableau.zlogical:
        # if tableau.dimension % 2 == 0:
        #     pauli.phase = (pauli.phase + ((pauli.xpow[control] + pauli.xpow[target]) + (pauli.zpow[control] + pauli.zpow[target]))) % (tableau.phase_order * tableau.dimension)
        pauli.xpow[target] = (pauli.xpow[target] + pauli.xpow[control]) % tableau.dimension
        pauli.zpow[control] = (pauli.zpow[control]+((pauli.zpow[target])  * (tableau.dimension - 1))) % tableau.dimension
    return tableau, None

def measure(tableau: Tableau, qudit_index: int, _) -> Tuple[Tableau, Optional[MeasurementResult]]:
    """
    Measure in Z basis qudit at qudit_index
    """
    first_xpow = None
    result = MeasurementResult(qudit_index, False, 0)
    # Find the first non-zero X in the tableau zlogical
    for row, pauli in enumerate(tableau.zlogical):
        xpow = pauli.xpow[qudit_index]
        if xpow > 0:
            if math.gcd(xpow, tableau.dimension) == 1:
                first_xpow = row
                # guarantees that the pauli found anti commutes by a single omega
                if xpow != 1:
                    exponent = pow(xpow, -1, tableau.dimension)
                    pauli = exponentiate(pauli, exponent, tableau.dimension, phase_order=tableau.phase_order)
                break
    if first_xpow is not None:
        result = _random_measurement(tableau, qudit_index, first_xpow)
    else:
        result = _det_measurement(tableau, qudit_index)
    return tableau, result

def _random_measurement(tableau: Tableau, qudit_index: int, first_xpow: int) -> MeasurementResult:
    # call rowsum(i, p) for all paulis in tableau such that i =/= p and pauli has a non-zero X on qudit_index
    # this effectively makes all other paulis commute with the first non-zero X we found
    # the following is guaranteed to converge within d-1 iterations because the first_xpow pauli has xpow == 1
    iden_pauli = PauliString(tableau.num_qudits, dimension=tableau.dimension)
    for row, pauli in enumerate(tableau.xlogical):
        while pauli.xpow[qudit_index] != 0:
            rowsum(tableau, pauli, tableau.zlogical[first_xpow])
    for row, pauli in enumerate(tableau.zlogical):
        while row != first_xpow and pauli.xpow[qudit_index] != 0:
            rowsum(tableau, pauli, tableau.zlogical[first_xpow])
    tableau.xlogical[first_xpow] = tableau.zlogical[first_xpow]
    measurement_outcome = random.choice(range(tableau.dimension))
    iden_pauli.zpow[qudit_index] = 1
    iden_pauli.phase = (-measurement_outcome*tableau.phase_order) % (tableau.phase_order*tableau.dimension)
    tableau.zlogical[first_xpow] = iden_pauli
    return MeasurementResult(qudit_index, False, measurement_outcome)

def _det_measurement(tableau: Tableau, qudit_index: int) -> MeasurementResult:
    iden_pauli = PauliString(tableau.num_qudits, dimension=tableau.dimension)
    for row, pauli in enumerate(tableau.xlogical):
        for _ in range(pauli.xpow[qudit_index]):
            rowsum(tableau, iden_pauli, tableau.zlogical[row])
    measurement_outcome = (-iden_pauli.phase // tableau.phase_order) % tableau.dimension
    return MeasurementResult(qudit_index, True, measurement_outcome)

def rowsum(tableau: Tableau, hrow: PauliString, irow: PauliString):
    hrow.phase = (hrow.phase + irow.phase + tableau.phase_order * commute_phase(hrow, irow)) % (tableau.phase_order * tableau.dimension)
    for i in range(tableau.num_qudits):
        hrow.xpow[i] = (hrow.xpow[i] + irow.xpow[i]) % tableau.dimension
        hrow.zpow[i] = (hrow.zpow[i] + irow.zpow[i]) % tableau.dimension

def commute_phase(row1: PauliString, row2: PauliString) -> int:
    """
    Computes the phase after multiplying two Pauli strings and commuting them into (XX..X)(ZZ..Z) form
    Args:
        row1: Pauli string
        row2: Pauli string
    Returns:
        The phase of the commutator
    """
    total_phase = 0
    for i in range(len(row1.xpow)):
        total_phase += row1.zpow[i] * row2.xpow[i]
    return total_phase

def exponentiate(row: PauliString, n: int, dimension: int, phase_order:int = 1) -> PauliString:
    """
    Exponentiate a Pauli string by n
    Example:
    (X^a Z^b)^n = w^(abn(n-1)/2)) X^(na) Z^(nb) 
    Reminder that we multiply the global phase by n because it is outside the parentheses
    Args:
        row: Pauli string
        n: int
        phase_order: int
    Returns:
        The Pauli string after exponentiation
    """
    row.phase *= n
    for i in range(len(row.xpow)):
        row.phase += row.xpow[i] * row.zpow[i] * n*(n-1)//2 * phase_order
        row.xpow[i] = (row.xpow[i] * n) % dimension
        row.zpow[i] = (row.zpow[i] * n) % dimension
    row.phase = row.phase % (dimension * phase_order)
    return row
