import cirq
from cirq_gates import GeneralizedHadamardGate, GeneralizedPhaseShiftGate, GeneralizedCNOTGate, IdentityGate
import numpy as np
def cirq_statevector_from_circuit(circuit):
    # Start with an initial state. For a quantum computer, this is usually the state |0...0>.
    state = np.zeros((circuit.dimension**circuit.num_qudits,), dtype=np.complex128)
    state[0] = 1

    # Create a list of qubits.
    qubits = [cirq.LineQid(i, dimension=circuit.dimension) for i in range(circuit.num_qudits)]

    # Create the generalized gates.
    gate_map = {
        "H": GeneralizedHadamardGate(circuit.dimension),
        "P": GeneralizedPhaseShiftGate(circuit.dimension),
        "CNOT": GeneralizedCNOTGate(circuit.dimension),
        "M": IdentityGate(circuit.dimension)
    }

    # Create a Cirq circuit.
    cirq_circuit = cirq.Circuit()

    # Apply each gate in the circuit.
    for op in circuit.operations:
        # Choose the appropriate gate.
        if op.name in gate_map:
            gate = gate_map[op.name]
        else:
            raise ValueError(f"Gate {op.name} not found")

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