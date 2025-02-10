import os
import random
import numpy as np
from .circuit_io import write_circuit

import random
from sdim.circuit import Circuit, GateData

def generate_random_clifford_circuit(num_qudits, num_gates, dimension, measurement_rounds=0, seed=None, gate_set=None):
    """
    Generates a random quantum circuit with gates sampled uniformly from the implemented Clifford gates.
    
    The available gates are:
        "H", "P", "CNOT", "X", "Z",
        "H_INV", "P_INV", "CNOT_INV", "X_INV", "Z_INV",
        "CZ", "CZ_INV"
    
    Note:
        - Two-qudit gates are: "CNOT", "CNOT_INV", "CZ", "CZ_INV".
        - All other gates are assumed to be single-qudit gates.
        - Measurement gates (with label "m") are added as extra rounds at the end.
    
    Args:
        num_qudits (int): Number of qudits in the circuit.
        num_gates (int): Total number of gates (excluding measurement rounds).
        dimension (int): The Hilbert space dimension of each qudit.
        measurement_rounds (int, optional): Number of measurement rounds to add at the end.
            In each round, every qudit is measured. Defaults to 0.
        seed (int, optional): Seed for reproducibility. Defaults to None.
    
    Returns:
        Circuit: A randomly generated Circuit object.
    
    Raises:
        Any exceptions that may be raised by the Circuit class.
    """
    # Set the random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Define the list of implemented Clifford gates.
    available_gates = gate_set or ["H", "P", "CNOT", "X", "Z", "H_INV", "P_INV", "CNOT_INV", "X_INV", "Z_INV", "CZ", "CZ_INV"]

    
    # Define the two-qudit gates.
    two_qudit_gates = {"CNOT", "CNOT_INV", "CZ", "CZ_INV"}
    
    # Create the Circuit object (assumes Circuit(num_qudits, dimension) is defined).
    circuit = Circuit(num_qudits, dimension)
    
    # Uniformly select num_gates gates.
    for _ in range(num_gates):
        gate = random.choice(available_gates)
        if gate in two_qudit_gates:
            # For two-qudit gates, choose two distinct qudits.
            qudits = random.sample(range(num_qudits), 2)
            circuit.add_gate(gate, qudits[0], qudits[1])
        else:
            # For single-qudit gates, choose one qudit.
            qudit = random.randint(0, num_qudits - 1)
            circuit.add_gate(gate, qudit)
    
    # Add measurement rounds (if any)
    for _ in range(measurement_rounds):
        for qudit in range(num_qudits):
            circuit.add_gate("M", qudit)
    
    return circuit

def generate_and_write_random_circuit(num_qudits, num_gates, dimension, measurement_rounds=0, output_file="random_circuit.chp", seed=None):
    """
    Generates a random quantum circuit and writes it to a file.

    This function creates a random circuit with specified gate type percentages
    and writes the resulting circuit to a file.

    Args:
        c_percentage: Percentage of CNOT gates.
        h_percentage: Percentage of Hadamard gates.
        p_percentage: Percentage of Phase gates.
        m_percentage: Percentage of Measurement gates.
        num_qudits: Number of qudits in the circuit.
        num_gates: Total number of gates in the circuit.
        dimension: Dimension of the qudits.
        measurement_rounds: Number of measurement rounds. Defaults to 0.
        output_file: Name of the output file. Defaults to "random_circuit.chp".
        seed: Random seed for reproducibility. Defaults to None.

    Returns:
        Circuit: The randomly generated Circuit object.

    Note:
        The percentages should sum to 100 (not 1).
        The circuit is written to the specified output file.
    """
    circuit = generate_random_clifford_circuit(num_qudits, num_gates, dimension, measurement_rounds, seed)
    write_circuit(circuit, output_file)
    return circuit

