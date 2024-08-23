import os
import random
import cirq
import numpy as np
from .unitary import GeneralizedHadamardGate, GeneralizedPhaseShiftGate, GeneralizedCNOTGate, GeneralizedXPauliGate, IdentityGate
from .chp_parser import write_circuit

import random
from sdim.circuit import Circuit, GateData

def generate_random_circuit(c_percentage, h_percentage, p_percentage, m_percentage, num_qudits, num_gates, dimension, measurement_rounds=0, seed=None):
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
    circuit = generate_random_circuit(c_percentage, h_percentage, p_percentage, m_percentage, num_qudits, num_gates, dimension, measurement_rounds, seed)
    write_circuit(circuit, output_file)

def circuit_to_cirq_circuit(circuit, measurement=False):
    # Create a list of qudits.
    qudits = [cirq.LineQid(i, dimension=circuit.dimension) for i in range(circuit.num_qudits)]

    # Create the generalized gates.
    gate_map = {
        "I": IdentityGate(circuit.dimension),
        "H": GeneralizedHadamardGate(circuit.dimension),
        "P": GeneralizedPhaseShiftGate(circuit.dimension),
        "CNOT": GeneralizedCNOTGate(circuit.dimension),
        "X": GeneralizedXPauliGate(circuit.dimension),
    }

    # Create a Cirq circuit.
    cirq_circuit = cirq.Circuit()

    # Apply each gate in the circuit.
    for op in circuit.operations:
        # Choose the appropriate gate.
        if op.name in gate_map:
            gate = gate_map[op.name]
            # Add the gate to the Cirq circuit.
            if op.target_index is None:
                # Single-qudit gate.
                cirq_circuit.append(gate.on(qudits[op.qudit_index]))
            else:
                # Two-qudit gate.
                cirq_circuit.append(gate.on(qudits[op.qudit_index], qudits[op.target_index]))
        elif op.name == "M":
            if measurement:
                cirq_circuit.append(cirq.measure(qudits[op.qudit_index], key=f'm_{op.qudit_index}'))
            continue
        else:
            raise ValueError(f"Gate {op.name} not found")
    for qudit in qudits:
        if not any(op.qubits[0] == qudit for op in cirq_circuit.all_operations()):
            # Append identity to qudits with no gates
            cirq_circuit.append(IdentityGate(circuit.dimension).on(qudit))
    return cirq_circuit

def cirq_statevector_from_circuit(circuit):
    # Start with an initial state. For a quantum computer, this is usually the state |0...0>.
    cirq_circuit = circuit_to_cirq_circuit(circuit)
    # print(cirq_circuit)
    # Simulate the Cirq circuit.
    simulator = cirq.Simulator()
    result = simulator.simulate(cirq_circuit)
    
    # Return the final state vector.
    return result.final_state_vector