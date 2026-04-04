"""CHP format reader/writer."""

import os

from ..circuit import Circuit
from ..gates.registry import gate_id_to_name


def read_circuit(filename: str) -> Circuit:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.join(script_dir, "../..")

    abs_file_path = os.path.join(parent_dir, filename)

    with open(abs_file_path, "r") as file:
        lines = file.readlines()

    start_index = next(i for i, line in enumerate(lines) if line.strip() == "#")
    gate_lines = lines[start_index + 1 :]

    dimension = 2
    parts = gate_lines[0].split()
    if parts[0].upper() == "D":
        dimension = int(parts[1])
        gate_lines = gate_lines[1:]

    max_int = max(
        int(s) for line in gate_lines for s in line.split() if s.isdigit()
    )
    circuit = Circuit(max_int + 1, dimension)

    for line in gate_lines:
        if not line.strip():
            continue
        parts = line.split()
        gate_name = parts[0].upper()

        gate_qubits = [int(qubit) for qubit in parts[1:] if qubit.isdigit()]
        extra_params = [text for text in parts[1:] if "=" in text]
        params_dict: dict[str, str] = {}
        for param in extra_params:
            param_parts = param.split("=")
            if len(param_parts) != 2:
                raise ValueError(
                    "Extra parameter doesn't have the correct format."
                )
            params_dict[param_parts[0]] = param_parts[1]

        num_indices = len(gate_qubits)
        if num_indices == 1:
            circuit.append(gate_name, gate_qubits[0])
        elif num_indices == 2:
            circuit.append(gate_name, gate_qubits[0], gate_qubits[1])
        else:
            raise ValueError(
                f"Unexpected number of arguments for gate {gate_name}"
            )

    return circuit


def write_circuit(
    circuit: Circuit,
    output_file: str = "random_circuit.chp",
    comment: str = "",
    directory: str | None = None,
) -> str:
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
        directory = os.path.join(script_dir, "../../circuits/")

    os.makedirs(directory, exist_ok=True)
    output_path = os.path.join(directory, output_file)

    with open(output_path, "w") as file:
        file.write(chp_content)

    return output_path
