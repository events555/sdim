import cirq
from sdim.unitary import GeneralizedHadamardGate, GeneralizedPhaseShiftGate, GeneralizedCNOTGate, IdentityGate
from sdim.random_circuit import circuit_to_cirq_circuit
import numpy as np
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

def numpy_statevector_from_circuit(circuit):
    # Start with an initial state. For a quantum computer, this is usually the state |0...0>.
    state = np.zeros((circuit.dimension**circuit.num_qudits,), dtype=np.complex128)
    state[0] = 1

    # Create the generalized gates.
    gate_map = {
        "H": GeneralizedHadamardGate(circuit.dimension),
        "P": GeneralizedPhaseShiftGate(circuit.dimension),
        "CNOT": GeneralizedCNOTGate(circuit.dimension),
        "M": IdentityGate(circuit.dimension)
    }

    # Apply each gate in the circuit.
    for op in circuit.operations:
        if op.name in gate_map:
            gate = gate_map[op.name]
        else:
            raise ValueError(f"Gate {op.name} not found")

        # Create a list of identities with the actual gate operation at the correct qudit index.
        gate_list = [np.eye(circuit.dimension, dtype=np.complex128) for _ in range(circuit.num_qudits)]
        if op.target_index is None:
            # Single-qudit gate.
            gate_list[op.qudit_index] = gate
        else:
            # Two-qudit gate.
            control_index = min(op.qudit_index, op.target_index)
            target_index = max(op.qudit_index, op.target_index)
            gate_list[control_index] = gate
            gate_list.pop(target_index)

        # Combine the gate list elements using Kronecker products.
        gate_matrix = np.eye(1, dtype=np.complex128)
        for gate in gate_list:
            gate_matrix = np.kron(gate_matrix, gate)

        # Apply the gate to the state.
        print(gate_matrix)
        state = np.dot(gate_matrix, state)

    # Construct full size unitary matrix
    return state