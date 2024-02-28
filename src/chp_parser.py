from circuit import Circuit
import os

def read_circuit(filename):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the absolute path to the file
    abs_file_path = os.path.join(script_dir, filename)

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
        parts = line.split()
        gate_name = parts[0].upper()
        gate_qubits = [int(qubit) for qubit in parts[1:]]

        # The number of arguments is the number of parts minus 1 (for the gate name)
        num_args = len(parts) - 1

        if num_args == 1:
            # Single-qubit gate
            circuit.add_gate(gate_name, gate_qubits[0])
        elif num_args == 2:
            # Two-qubit gate
            circuit.add_gate(gate_name, gate_qubits[0], gate_qubits[1])
        else:
            raise ValueError(f"Unexpected number of arguments for gate {gate_name}")

    return circuit