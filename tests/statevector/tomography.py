import cirq
import numpy as np

def state_vector(circuit):
    # Start with an initial state. For a quantum computer, this is usually the state |0...0>.
    state = np.zeros((circuit.dimension**circuit.num_qudits,), dtype=np.complex128)
    state[0] = 1

    # Create a list of qubits.
    qubits = [cirq.LineQid(i, dimension=circuit.dimension) for i in range(circuit.num_qudits)]

    # Create a Cirq circuit.
    cirq_circuit = cirq.Circuit()

    # Apply each gate in the circuit.
    for op in circuit.operations:
        # Get the unitary for this gate.
        U = op.gate.unitary_matrix

        # Create a Cirq gate from the unitary.
        gate = cirq.MatrixGate(U)

        # Add the gate to the Cirq circuit.
        if op.target_index is None:
            # Single-qubit gate.
            cirq_circuit.append(gate.on(qubits[op.qudit_index]))
        else:
            # Two-qubit gate.
            cirq_circuit.append(gate.on(qubits[op.qudit_index], qubits[op.target_index]))

    # Simulate the Cirq circuit.
    simulator = cirq.Simulator()
    result = simulator.simulate(cirq_circuit, initial_state=state)

    # Return the final state vector.
    return result.final_state_vector


def calculate_amplitudes(state_vector):
    # Calculate the amplitudes.
    amplitudes = np.abs(state_vector)

    # Return the amplitudes.
    return amplitudes

def check_closeness(amplitudes, expected_amplitudes):
    # Check that the amplitudes are close to the expected amplitudes.
    assert np.allclose(amplitudes, expected_amplitudes, atol=1e-3)
