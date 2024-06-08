import os
import random
import cirq
import numpy as np
from .unitary import GeneralizedHadamardGate, GeneralizedPhaseShiftGate, GeneralizedCNOTGate
def generate_chp_file(c_percentage, h_percentage, p_percentage, m_percentage, num_qudits, num_gates, dimension, measurement_rounds = 0, output_file="random_circuit.chp", seed=None):
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

    # Generate the .chp file content
    chp_content = "Randomly-generated Clifford group quantum circuit\n"
    chp_content += f"# \nd {dimension}\n"
    for gate in gates:
        qudits = random.sample(range(num_qudits), 1 if gate != 'c' else 2)
        # if gate == 'c':
        #     start_qudit = random.randint(0, num_qudits - 2)  # -2 to ensure we have room for a consecutive qudit
        #     qudits = [start_qudit, start_qudit + 1]
        # else:
        #     qudits = random.sample(range(num_qudits), 1)
        chp_content += f"{gate} {' '.join(map(str, qudits))}\n"

    # Append measurements across every qubit based on the number of measurement rounds
    for _ in range(measurement_rounds):
        for qubit in range(num_qudits):
            chp_content += f"m {qubit}\n"

    # Define the directory where the file will be saved
    script_dir = os.path.dirname(os.path.realpath(__file__))
    directory = os.path.join(script_dir, '../circuits/')

    # Join the directory with the output file name
    output_path = os.path.join(directory, output_file)

    # Write the content to the .chp file
    with open(output_path, "w") as file:
        file.write(chp_content)

def circuit_to_cirq_circuit(circuit, measurement=False):
    # Create a list of qudits.
    qudits = [cirq.LineQid(i, dimension=circuit.dimension) for i in range(circuit.num_qudits)]

    # Create the generalized gates.
    gate_map = {
        "H": GeneralizedHadamardGate(circuit.dimension),
        "P": GeneralizedPhaseShiftGate(circuit.dimension),
        "CNOT": GeneralizedCNOTGate(circuit.dimension)
    }

    # Create a Cirq circuit.
    cirq_circuit = cirq.Circuit()

    # Apply each gate in the circuit.
    for op in circuit.operations:
        # Choose the appropriate gate.
        if op.name in gate_map:
            gate = gate_map[op.name]
        elif op.name == "M":
            if measurement:
                cirq_circuit.append(cirq.measure(qudits[op.qudit_index], key=f'm_{op.qudit_index}'))
            continue
        else:
            raise ValueError(f"Gate {op.name} not found")

        # Add the gate to the Cirq circuit.
        if op.target_index is None:
            # Single-qudit gate.
            cirq_circuit.append(gate.on(qudits[op.qudit_index]))
        else:
            # Two-qudit gate.
            cirq_circuit.append(gate.on(qudits[op.qudit_index], qudits[op.target_index]))

    return cirq_circuit

def cirq_statevector_from_circuit(circuit):
    # Start with an initial state. For a quantum computer, this is usually the state |0...0>.
    state = np.zeros((circuit.dimension**circuit.num_qudits,), dtype=np.complex128)
    state[0] = 1

    cirq_circuit = circuit_to_cirq_circuit(circuit)

    # Simulate the Cirq circuit.
    simulator = cirq.Simulator()
    result = simulator.simulate(cirq_circuit, initial_state=state)
    # print(cirq_circuit)
    # Return the final state vector.
    return result.final_state_vector