import pytest
from sdim.gatedata import GATE_DATA, gate_name_to_id, gate_id_to_name, is_gate_noisy, is_gate_two_qubit, is_gate_pauli


def test_round_trip_canonical_names():
    # For every canonical name, convert to id and back.
    for canonical_name in GATE_DATA.keys():
        gate_id = gate_name_to_id(canonical_name)
        # The reverse should give us the canonical name.
        assert gate_id_to_name(gate_id) == canonical_name

def test_alias_resolution():
    # Test a few aliases and check they map to the same canonical id.
    # "NOT" is an alias for "X".
    x_id = gate_name_to_id("X")
    not_id = gate_name_to_id("NOT")
    assert x_id == not_id

    # "DFT" is an alias for "H".
    h_id = gate_name_to_id("H")
    dft_id = gate_name_to_id("DFT")
    assert h_id == dft_id

    # "CX" is an alias for "CNOT".
    cnot_id = gate_name_to_id("CNOT")
    cx_id = gate_name_to_id("CX")
    assert cnot_id == cx_id

def test_invalid_gate_name():
    # A non-existent gate name should raise a ValueError.
    with pytest.raises(ValueError):
        gate_name_to_id("INVALID_GATE")

def test_invalid_gate_id():
    # Use an id that is outside the valid range.
    invalid_id = len(GATE_DATA)  # valid ids are 0 to len(GATE_DATA)-1.
    with pytest.raises(ValueError):
        gate_id_to_name(invalid_id)

def test_is_gate_noisy():
    # Test gates that are expected to be noisy.
    noisy_gates = ["X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2", "M", "M_X", "N1"]
    for gate in noisy_gates:
        gate_id = gate_name_to_id(gate)
        assert is_gate_noisy(gate_id) is True

    # A non-noisy gate should return False.
    non_noisy = ["H", "CNOT", "RESET"]
    for gate in non_noisy:
        gate_id = gate_name_to_id(gate)
        assert is_gate_noisy(gate_id) is False

def test_is_gate_two_qubit():
    # Test gates that are expected to be two-qubit gates.
    two_qubit_gates = ["CNOT", "CZ", "SWAP", "CNOT_INV", "CZ_INV", "DEPOLARIZE2"]
    for gate in two_qubit_gates:
        gate_id = gate_name_to_id(gate)
        assert is_gate_two_qubit(gate_id) is True

    # Single-qubit gates should not be two-qubit.
    single_qubit = ["H", "X", "P", "RESET"]
    for gate in single_qubit:
        gate_id = gate_name_to_id(gate)
        assert is_gate_two_qubit(gate_id) is False

def test_is_gate_pauli():
    # Only "X" and "Z" should return True.
    pauli_true = ["X", "Z", "NOT"]  # "NOT" is an alias for "X"
    for gate in pauli_true:
        gate_id = gate_name_to_id(gate)
        assert is_gate_pauli(gate_id) is True

    # Other gates should return False.
    pauli_false = ["X_INV", "H", "CNOT", "P"]
    for gate in pauli_false:
        gate_id = gate_name_to_id(gate)
        assert is_gate_pauli(gate_id) is False