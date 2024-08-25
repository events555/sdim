import os
import random
import numpy as np
from .circuit_io import write_circuit

import random
from sdim.circuit import Circuit, GateData

def generate_random_circuit(c_percentage, h_percentage, p_percentage, m_percentage, num_qudits, num_gates, dimension, measurement_rounds=0, seed=None):
    """
    Generates a random quantum circuit with specified gate type percentages.

    This function creates a circuit with a given distribution of gate types. 
    By default, it generates no measurement gates unless specified.

    Args:
        c_percentage: Percentage of CNOT gates.
        h_percentage: Percentage of Hadamard gates.
        p_percentage: Percentage of Phase gates.
        m_percentage: Percentage of Measurement gates.
        num_qudits: Number of qudits in the circuit.
        num_gates: Total number of gates in the circuit.
        dimension: Dimension of the qudits.
        measurement_rounds: Number of measurement rounds. Defaults to 0.
        seed: Random seed for reproducibility. Defaults to None.

    Returns:
        Circuit: A randomly generated Circuit object.

    Raises:
        ValueError: If the sum of all percentages is not equal to 100.

    Note:
        The percentages should sum to 100 (not 1).
    """
    # Check that the percentages sum to 100
    if c_percentage + h_percentage + p_percentage + m_percentage != 100:
        raise ValueError("The percentages do not sum to 100")

    # Set the seed for random sampling
    if seed is not None:
        random.seed(seed)

    # Calculate the number of each gate
    num_c = int((c_percentage / 100) * num_gates)
    num_h = int((h_percentage / 100) * num_gates)
    num_p = int((p_percentage / 100) * num_gates)
    num_m = int((m_percentage / 100) * num_gates)

    # Generate the gates
    gates = ['c'] * num_c + ['h'] * num_h + ['p'] * num_p + ['m'] * num_m
    random.shuffle(gates)

    # Create the Circuit object
    circuit = Circuit(num_qudits, dimension)

    # Generate the circuit
    for gate in gates:
        if gate == 'c':
            qudits = random.sample(range(num_qudits), 2)
            circuit.add_gate(gate, qudits[0], qudits[1])
        else:
            qudit = random.randint(0, num_qudits - 1)
            circuit.add_gate(gate, qudit)

    # Add measurements
    for _ in range(measurement_rounds):
        for qubit in range(num_qudits):
            circuit.add_gate('m', qubit)

    return circuit

def generate_and_write_random_circuit(c_percentage, h_percentage, p_percentage, m_percentage, num_qudits, num_gates, dimension, measurement_rounds=0, output_file="random_circuit.chp", seed=None):
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
    circuit = generate_random_circuit(c_percentage, h_percentage, p_percentage, m_percentage, num_qudits, num_gates, dimension, measurement_rounds, seed)
    write_circuit(circuit, output_file)

