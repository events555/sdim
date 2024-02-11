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
    for i in range(tableau.num_qudits):
        tempx = tableau.xlogical[i].xpow[qudit_index]
        tableau.xlogical[i].xpow[qudit_index] = tableau.xlogical[i].zpow[qudit_index]
        tableau.xlogical[i].zpow[qudit_index] = tempx
        tempz = tableau.zlogical[i].xpow[qudit_index]
        tableau.zlogical[i].xpow[qudit_index] = tableau.zlogical[i].zpow[qudit_index]
        tableau.zlogical[i].zpow[qudit_index] = tempz
    return tableau


def apply_P(tableau, qudit_index):
    """
    Apply P gate to qudit at qudit_index
    """
    for i in range(tableau.num_qudits):
        tableau.xlogical[i].zpow[qudit_index] = (
            tableau.xlogical[i].zpow[qudit_index]
            + tableau.xlogical[i].xpow[qudit_index]
        ) % tableau.dimension
        tableau.zlogical[i].zpow[qudit_index] = (
            tableau.zlogical[i].zpow[qudit_index]
            + tableau.zlogical[i].xpow[qudit_index]
        ) % tableau.dimension
    return tableau


def apply_CNOT(tableau, control, target):
    """
    Apply CNOT gate to control and target qudits
    """
    for i in range(tableau.num_qudits):
        tableau.xlogical[i].xpow[target] = (
            tableau.xlogical[i].xpow[target] + tableau.xlogical[i].xpow[control]
        ) % tableau.dimension
        tableau.xlogical[i].zpow[control] = (
            tableau.xlogical[i].zpow[target] + tableau.xlogical[i].zpow[control]
        ) % tableau.dimension
        tableau.zlogical[i].xpow[target] = (
            tableau.zlogical[i].xpow[target] + tableau.zlogical[i].xpow[control]
        ) % tableau.dimension
        tableau.zlogical[i].zpow[control] = (
            tableau.zlogical[i].zpow[target] + tableau.zlogical[i].zpow[control]
        ) % tableau.dimension
