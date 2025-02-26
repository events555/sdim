from .gatedata import gate_id_to_name, is_gate_two_qubit
from .circuit import Circuit
from .unitary import *
import cirq
import os

def read_circuit(filename):
    """
    Reads a circuit from a file and creates a Circuit object.

    Args:
        filename (str): The name of the file containing the circuit description.

    Returns:
        Circuit: A Circuit object representing the circuit described in the file.

    Raises:
        ValueError: If an unexpected number of arguments is found for a gate.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.join(script_dir, '..')

    # Construct the absolute path to the file
    abs_file_path = os.path.join(parent_dir, filename)

    with open(abs_file_path, 'r') as file:
        lines = file.readlines()

    # Find the line with only '#'
    start_index = next(i for i, line in enumerate(lines) if line.strip() == '#')

    # Extract the lines after '#'
    gate_lines = lines[start_index + 1:]

    # Default dimension
    dimension = 2

    # Check the line immediately after '#'
    parts = gate_lines[0].split()
    if parts[0].upper() == 'D':
        dimension = int(parts[1])
        gate_lines = gate_lines[1:] 

    # Find the max integer in the gate lines
    max_int = max(int(s) for line in gate_lines for s in line.split() if s.isdigit())

    # Create a circuit with max_int - 1 qubits and the specified dimension
    circuit = Circuit(max_int+1, dimension)

    # Append the gates to the circuit
    for line in gate_lines:
        if not line.strip():
            continue
        parts = line.split()
        gate_name = parts[0].upper()

        # Extract all purely numerical arguments as qubit indices
        gate_qubits = [int(qubit) for qubit in parts[1:] if qubit.isdigit()]

        # Extract all extra parameters
        extra_params = [text for text in parts[1:] if '=' in text]

        params_dict = None if len(extra_params) == 0 else dict()

        # Check for extra parameters
        for param in extra_params:
            param_parts = param.split('=')
            if len(param_parts) != 2:
                raise ValueError("Extra parameter doesn't have the correct format.")
            params_dict[param_parts[0]] = param_parts[1]

        num_indices = len(gate_qubits)

        if num_indices == 1:
            # Single-qubit gate
            if params_dict is None:
                circuit.append(gate_name, gate_qubits[0])
            else:    
                circuit.append(gate_name, gate_qubits[0], **params_dict)
        elif num_indices == 2:
            # Two-qubit gate
            if params_dict is None:
                circuit.append(gate_name, gate_qubits[0], gate_qubits[1])
            else:
                circuit.append(gate_name, gate_qubits[0], gate_qubits[1], **params_dict)
        else:
            raise ValueError(f"Unexpected number of arguments for gate {gate_name}")

    return circuit

def write_circuit(circuit: Circuit, output_file: str = "random_circuit.chp", comment: str = "", directory: str = None):
    """
    Writes a Circuit object to a file in the .chp format.

    Args:
        circuit (Circuit): The Circuit object to write.
        output_file (str): The name of the output file. Defaults to "random_circuit.chp".
        comment (str): An optional comment to include at the beginning of the file.
        directory (str): Optional directory to save the file. If None, uses the default '../circuits/' relative to the script.

    Returns:
        str: The path to the written file.
    """
    chp_content = ""
    if comment:
        chp_content += f"{comment}\n#\n"
    else:
        chp_content += "Randomly-generated Clifford group quantum circuit\n#\n"
    chp_content += f"d {circuit.dimension}\n"

    for op in circuit.operations:
        gate_str = gate_id_to_name(op.gate_type)
        if op.args is not None:
            for arg in op.args:
                gate_str += f"({arg})"
        for target in op.targets:
            gate_str += f" {target._value}"

        chp_content += f"{gate_str}\n"

    if directory is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        directory = os.path.join(script_dir, '../circuits/')

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Join the directory with the output file name
    output_path = os.path.join(directory, output_file)

    # Write the content to the .chp file
    with open(output_path, "w") as file:
        file.write(chp_content)

    return output_path


def circuit_to_cirq_circuit(circuit, measurement=False, print_circuit=False):
    """
    Converts a Circuit object to a Cirq Circuit object.

    Args:
        circuit (Circuit): The Circuit object to convert.
        measurement (bool): Whether to include measurement gates. Defaults to False.
        print_circuit (bool): Whether to print the Cirq circuit. Defaults to False.

    Returns:
        cirq.Circuit: The equivalent Cirq Circuit object.
    """
    # Create a list of qudits.
    qudits = [cirq.LineQid(i, dimension=circuit.dimension) for i in range(circuit.num_qudits)]

    # Create the generalized gates.
    gate_map = {
        "I": IdentityGate(circuit.dimension),
        "H": GeneralizedHadamardGate(circuit.dimension),
        "P": GeneralizedPhaseShiftGate(circuit.dimension),
        "CNOT": GeneralizedCNOTGate(circuit.dimension),
        "X": GeneralizedXPauliGate(circuit.dimension),
        "Z": GeneralizedZPauliGate(circuit.dimension),
        "H_INV": GeneralizedHadamardGateInverse(circuit.dimension),
        "P_INV": GeneralizedPhaseShiftGateInverse(circuit.dimension),
        "CNOT_INV": GeneralizedCNOTGateInverse(circuit.dimension),
        "X_INV": GeneralizedXPauliGateInverse(circuit.dimension),
        "Z_INV": GeneralizedZPauliGateInverse(circuit.dimension),
        "CZ": GeneralizedCZGate(circuit.dimension),
        "CZ_INV": GeneralizedCZGateInverse(circuit.dimension),
        "N1" : IdentityGate(circuit.dimension) # TODO: Implement for Cirq circuit in unitary.py.  Need to figure out how to pass probability and noise_channel parameters.
    }

    # Create a Cirq circuit.
    cirq_circuit = cirq.Circuit()

    # Apply each gate in the circuit.
    for op in circuit.operations:
        # Choose the appropriate gate.
        name = gate_id_to_name(op.gate_type)


        if name in gate_map:
            gate = gate_map[name]
            if is_gate_two_qubit(op.gate_type):
                for i in range(0, len(op.targets), 2):
                    cirq_circuit.append(gate.on(qudits[op.targets[i]._value], 
                                                qudits[op.targets[i+1]._value]))
            else:
                for target in op.targets:
                    cirq_circuit.append(gate.on(qudits[target._value]))
        elif name == "M":
            if measurement:
                for target in op.targets:
                    cirq_circuit.append(cirq.measure(qudits[target._value], key=f'm_{target._value}'))
            continue
        else:
            raise NotImplementedError(f"Gate {name} not implemented")
    for qudit in qudits:
        if not any(op.qubits[0] == qudit for op in cirq_circuit.all_operations()):
            # Append identity to qudits with no gates
            cirq_circuit.append(IdentityGate(circuit.dimension).on(qudit))
    if print_circuit:
        print(cirq_circuit)
    return cirq_circuit

def cirq_statevector_from_circuit(circuit, print_circuit=False):
    """
    Simulates a Circuit object using Cirq and returns the final state vector.

    Args:
        circuit (Circuit): The Circuit object to simulate.
        print_circuit (bool): Whether to print the Cirq circuit. Defaults to False.

    Returns:
        np.ndarray: The final state vector given by the Cirq simulator.
    """
    # Start with an initial state. For a quantum computer, this is usually the state |0...0>.
    cirq_circuit = circuit_to_cirq_circuit(circuit, print_circuit=print_circuit)
    # Simulate the Cirq circuit.
    simulator = cirq.Simulator()
    result = simulator.simulate(cirq_circuit)
    
    # Return the final state vector.
    return result.final_state_vector
