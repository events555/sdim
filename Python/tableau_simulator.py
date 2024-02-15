def apply_gate(tableau, instruc):
    """
    Apply a gate to the tableau
    """
    gate_id = instruc.gate_id
    qudit_index = instruc.qudit_index
    target_index = instruc.target_index
    if gate_id == 0:  # I gate
        pass
    elif gate_id == 1:  # X gate
        pass
    elif gate_id == 2:  # Z gate
        pass
    elif gate_id == 3:  # H gate
        apply_H(tableau, qudit_index)
    elif gate_id == 4:  # P gate
        apply_P(tableau, qudit_index)
    elif gate_id == 5:  # CNOT gate
        apply_CNOT(tableau, qudit_index, target_index)
    else:
        raise ValueError("Invalid gate value")
    return tableau


def apply_H(tableau, qudit_index):
    """
    Apply H gate to qudit at qudit_index
    """
    ind = qudit_index

    for pauli in tableau.xlogical:
        pauli.phase = (pauli.phase + pauli.xpow[ind] * pauli.zpow[ind]) % tableau.dimension
        pauli.xpow[ind], pauli.zpow[ind] = pauli.zpow[ind] * (tableau.dimension - 1), pauli.xpow[ind] 

    for pauli in tableau.zlogical:
        pauli.phase = (pauli.phase + pauli.xpow[ind] * pauli.zpow[ind]) % tableau.dimension
        pauli.xpow[ind], pauli.zpow[ind] = pauli.zpow[ind] * (tableau.dimension - 1), pauli.xpow[ind]

    return tableau


def apply_P(tableau, qudit_index):
    """
    Apply P gate to qudit at qudit_index
    """
    ind = qudit_index
    for pauli in tableau.xlogical:
        pauli.phase = (pauli.phase + pauli.xpow[ind] * pauli.zpow[ind]) % tableau.dimension
        pauli.zpow[ind] = (pauli.xpow[ind] + pauli.zpow[ind]) % tableau.dimension
    for pauli in tableau.zlogical:
        pauli.phase = (pauli.phase + pauli.xpow[ind] * pauli.zpow[ind]) % tableau.dimension
        pauli.zpow[ind] = (pauli.xpow[ind] + pauli.zpow[ind]) % tableau.dimension
    return tableau


def apply_CNOT(tableau, control, target):
    """
    Apply CNOT gate to control and target qudits
    """
    for pauli in tableau.xlogical:
        pauli.xpow[target] = (pauli.xpow[target] + pauli.xpow[control]) % tableau.dimension
        pauli.zpow[control] = (pauli.zpow[target] * (tableau.dimension - 1) + pauli.zpow[control]) % tableau.dimension
    for pauli in tableau.zlogical:
        pauli.xpow[target] = (pauli.xpow[target] + pauli.xpow[control]) % tableau.dimension
        pauli.zpow[control] = (pauli.zpow[target]* (tableau.dimension - 1) + pauli.zpow[control]) % tableau.dimension