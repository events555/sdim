import random
from paulistring import PauliString
def apply_H(tableau, qudit_index, _):
    """
    Apply H gate to qudit at qudit_index
    """
    ind = qudit_index
    for pauli in tableau.xlogical+tableau.zlogical:
        pauli.phase = (pauli.phase + pauli.xpow[ind] * pauli.zpow[ind]) % tableau.dimension
        pauli.xpow[ind], pauli.zpow[ind] = pauli.zpow[ind] * (tableau.dimension - 1), pauli.xpow[ind] 
    return tableau, None


def apply_P(tableau, qudit_index, _):
    """
    Apply P gate to qudit at qudit_index
    """
    ind = qudit_index
    for pauli in tableau.xlogical+tableau.zlogical:
        pauli.phase = (pauli.phase + pauli.xpow[ind] * pauli.zpow[ind]) % tableau.dimension
        pauli.zpow[ind] = (pauli.xpow[ind] + pauli.zpow[ind]) % tableau.dimension
    return tableau, None


def apply_CNOT(tableau, control, target):
    """
    Apply CNOT gate to control and target qudits
    """
    for pauli in tableau.xlogical+tableau.zlogical:
        pauli.phase = (pauli.phase + (pauli.xpow[control] * pauli.zpow[target])+(pauli.xpow[target]+pauli.zpow[control]+1)) % tableau.dimension
        pauli.xpow[target] = (pauli.xpow[target] + pauli.xpow[control]) % tableau.dimension
        pauli.zpow[control] = (pauli.zpow[target]  * (tableau.dimension - 1) + pauli.zpow[control]) % tableau.dimension
    return tableau, None

def measure(tableau, qudit_index, _):
    """
    Measure in Z basis qudit at qudit_index
    """
    p = None
    temp = PauliString(tableau.num_qudits, dimension=tableau.dimension)
    det = False
    for row, pauli in enumerate(tableau.zlogical):
        if pauli.xpow[qudit_index] != 0:
            p = row
            break
    if p is not None:
        for row, pauli in enumerate(tableau.xlogical):
            if pauli.xpow[qudit_index] != 0:
                rowsum(tableau, pauli, tableau.zlogical[p])
        for row, pauli in enumerate(tableau.zlogical):
            if row != p and pauli.xpow[qudit_index] != 0:
                rowsum(tableau, pauli, tableau.zlogical[p])
        tableau.xlogical[p] = tableau.zlogical[p]
        tableau.zlogical[p] = temp
        tableau.zlogical[p].zpow[qudit_index] = 1
        tableau.zlogical[p].phase = random.choice(range(tableau.dimension))
    else:
        det = True
        for row, pauli in enumerate(tableau.xlogical):
            if pauli.xpow[qudit_index] != 0:
                rowsum(tableau, temp, tableau.zlogical[row])
    return tableau, det

def rowsum(tableau, hrow, irow):
    if tableau.dimension%2 == 0:
        phase = 2*(hrow.phase + irow.phase + commute_phase(hrow, irow))
        if phase % 4 == 0:
            hrow.phase = 0
        else:
            hrow.phase = 1
    else:
        hrow.phase = (hrow.phase + irow.phase + commute_phase(hrow, irow)) % tableau.dimension
    for i in range(tableau.num_qudits):
        hrow.xpow[i] = (hrow.xpow[i] + irow.xpow[i]) % tableau.dimension
        hrow.zpow[i] = (hrow.zpow[i] + irow.zpow[i]) % tableau.dimension

def commute_phase(row1, row2):
    """
    Computes the phase after multiplying two Pauli strings
    Args:
    Returns:
        The phase of the commutator
    """
    sum = 0
    for i in range(len(row1.xpow)):
            sum += row1.xpow[i] * row2.zpow[i] - row1.zpow[i] * row2.xpow[i]
    return sum