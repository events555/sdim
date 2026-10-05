from .circuit import Circuit
from .unitary import *
import cirq
import numpy as np
import os
import re
import shlex


# How write_circuit writes a gate parameter, and how read_circuit reads it back:
#   - text:                          key="text", with backslashes and double quotes escaped;
#   - None, booleans and numbers:    key=value, unquoted, the repr of the plain Python value (also for
#                                    NumPy scalars and subclasses of int and float);
#   - an N2 prob_dist (any sequence): key="[p0, p1, ...]", read back as a float64 array;
#   - anything else:                 key="str(value)", read back as text.
# Older versions wrote every value as key="str(value)".  For those files (and quoted values in
# general), prob, a and scalar are read as numbers and prob_dist as an array; the rest stays text.
_NUMBER_PARAMS = {"prob", "a", "scalar"}
_FLOAT_LIST_PARAMS = {"prob_dist"}
_LITERALS = {"None": None, "True": True, "False": False}

# One whitespace-separated token of a gate line as write_circuit writes it: a word (gate name or
# qudit index), key="text" (only \\ and \" escaped, as shlex reads them) or key=text.
_TOKEN = re.compile(r'([^\s"\'\\=]+)(?:=(?:"((?:[^"\\]|\\.)*)"|([^\s"\'\\]*)))?(?=\s|$)')


def _format_param(key, value) -> str:
    """
    Formats one gate parameter for a .chp line (see the table above).

    Floats are written with repr, which reads back as the same float.  A prob_dist is written as
    a bracketed, comma-separated list on one line, like str() of a list of floats.  Subclasses of
    int and float (an IntEnum, ...) are written as the plain int or float, since their own repr
    need not read back as a number.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if key in _FLOAT_LIST_PARAMS and value is not None and not isinstance(value, str):
        text = "[" + ", ".join(repr(float(v)) for v in np.asarray(value, dtype=float).reshape(-1)) + "]"
    elif value is None or isinstance(value, bool):  # bool cannot be subclassed
        return f"{key}={value!r}"
    elif isinstance(value, int):
        return f"{key}={int(value)!r}"
    elif isinstance(value, float):
        return f"{key}={float(value)!r}"
    else:
        text = str(value)
    if "\n" in text or "\r" in text:
        raise ValueError(f"Cannot write gate parameter {key}={text!r}: a .chp line cannot hold a line break.")
    return key + '="' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _split_gate_line(line: str) -> list:
    """
    Splits a gate line into its tokens, like shlex.split, and tells which values were quoted.

    Returns a list of (word, value, quoted): value is None for a token without '=', otherwise
    word is the key.  A line that write_circuit would not write (single quotes, a backslash or
    a quote outside a value, ...) is split by shlex.split, with every value taken as quoted text,
    which is how older versions read every line.
    """
    tokens = []
    position = 0
    while True:
        while position < len(line) and line[position].isspace():
            position += 1
        if position == len(line):
            return tokens
        match = _TOKEN.match(line, position)
        if match is None:
            break
        word, quoted_text, bare_text = match.groups()
        if quoted_text is not None:
            tokens.append((word, re.sub(r'\\(["\\])', r"\1", quoted_text), True))
        else:
            tokens.append((word, bare_text, False))
        position = match.end()
    tokens = []
    for part in shlex.split(line):
        if "=" in part:
            key, value = part.split("=", 1)
            tokens.append((key, value, True))
        else:
            tokens.append((part, None, True))
    return tokens


def _parse_number(text: str):
    """text as an int, or else as a float; None if it is neither."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return None


def _parse_float_list(text: str):
    """Reads a prob_dist written by _format_param, or by older versions (str() of a list or a short array)."""
    body = text.strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    items = [item for item in re.split(r"[\s,]+", body) if item]
    return np.array([float(item) for item in items], dtype=np.float64)


def _parse_param(key: str, text: str, quoted: bool, cache: dict):
    """
    Converts the text of a gate parameter back to its value (see the table above).

    An unquoted value is None, True, False, an int or a float when it reads as one, and text
    otherwise.  A quoted value is text, except that prob, a and scalar become numbers and
    prob_dist a float array (None for "None").  Text that does not convert is returned as it
    is, so the gate reports it the same way it did before.  Equal prob_dist texts share one array.
    """
    if key in _FLOAT_LIST_PARAMS:
        if text == "None":
            return None
        if text not in cache:
            try:
                cache[text] = _parse_float_list(text)
            except ValueError:
                return text
        return cache[text]
    if not quoted:
        if text in _LITERALS:
            return _LITERALS[text]
        number = _parse_number(text)
        return text if number is None else number
    if key in _NUMBER_PARAMS:
        number = _parse_number(text)
        return text if number is None else number
    return text


def read_circuit(filename):
    """
    Reads a circuit from a file and creates a Circuit object.

    The file holds an optional comment, a line with only '#', an optional dimension line
    "d <dimension>" (with "qudits=<n>" in files written by write_circuit), then one gate per line:
    the gate name, its qudit indices, and its parameters as key="text" or key=value (see
    write_circuit for which values come back with which type).  Files written by older versions
    hold only key="text": prob, a / scalar and prob_dist are read from them as numbers, and the
    other parameters as text.  The file is read as UTF-8.

    Args:
        filename (str): Path to the file, absolute or relative to the current working directory.
            A relative path that does not exist there is looked up relative to the folder above
            the sdim package (the repository root in a source checkout), as older versions did.

    Returns:
        Circuit: A Circuit object representing the circuit described in the file.

    Raises:
        FileNotFoundError: If the file exists in neither place.
        ValueError: If the file has no line with only '#', a gate has more than two qudit indices,
            or a gate line is not a valid gate (Circuit.add_gate rejects it).
    """
    path = filename
    if not os.path.exists(path):
        # Older versions resolved every name against the folder above the package.
        script_dir = os.path.dirname(os.path.realpath(__file__))
        fallback = os.path.join(script_dir, '..', filename)
        if os.path.exists(fallback):
            path = fallback

    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Find the line with only '#'
    start_index = next((i for i, line in enumerate(lines) if line.strip() == '#'), None)
    if start_index is None:
        raise ValueError(f"{filename} has no line with only '#' before the gates.")

    # Extract the non-blank lines after '#'
    gate_lines = [line for line in lines[start_index + 1:] if line.strip()]

    # Default dimension and qudit count (the qudit count is otherwise the largest index + 1)
    dimension = 2
    declared_qudits = 0

    # Check the line immediately after '#'
    if gate_lines:
        parts = gate_lines[0].split()
        if parts[0].upper() == 'D':
            dimension = int(parts[1])
            for part in parts[2:]:
                if part.lower().startswith('qudits='):
                    declared_qudits = int(part.split('=', 1)[1])
            gate_lines = gate_lines[1:]

    # Parse every gate line: qudit indices are the purely numerical arguments, parameters are key=value.
    gates = []
    prob_dist_cache = {}
    for line in gate_lines:
        tokens = _split_gate_line(line)
        gate_name = tokens[0][0].upper()
        gate_qubits = [int(word) for word, value, _ in tokens[1:] if value is None and word.isdigit()]
        if len(gate_qubits) > 2:
            raise ValueError(f"Unexpected number of arguments for gate {gate_name}")
        params_dict = None
        for key, value, quoted in tokens[1:]:
            if value is None:
                continue
            if not key:
                raise ValueError("Extra parameter doesn't have the correct format.")
            if params_dict is None:
                params_dict = dict()
            params_dict[key] = _parse_param(key, value, quoted, prob_dist_cache)
        gates.append((gate_name, gate_qubits, params_dict, line.strip()))

    num_qudits = max([declared_qudits, 1] + [q + 1 for _, qubits, _, _ in gates for q in qubits])
    circuit = Circuit(num_qudits, dimension)

    # Append the gates to the circuit
    for gate_name, gate_qubits, params_dict, line in gates:
        try:
            if params_dict is not None:
                circuit.add_gate(gate_name, *gate_qubits, **params_dict)
            else:
                circuit.add_gate(gate_name, *gate_qubits)
        except ValueError as error:
            raise ValueError(f"{filename}: gate line {line!r}: {error}") from error

    return circuit

def write_circuit(circuit: Circuit, output_file: str = "random_circuit.chp", comment: str = "", directory: str = None):
    """
    Writes a Circuit object to a file in the .chp format.

    Every gate parameter is written, and read_circuit reads the circuit back with the same
    gates and qudit count.  Parameters that are numbers, booleans or None come back with the
    same value and type (a NumPy scalar, or a subclass of int or float such as an IntEnum, as
    the matching Python int, float or bool), and an N2 prob_dist as a float64 array with the same
    entries.  Text comes back as the same text, except under the keys prob, a and scalar, where
    text that reads as a number comes back as that number (prob="0.1" as the float 0.1, a="2"
    as the int 2), the way files of older versions are read; the simulators convert these
    parameters to numbers anyway.  A value of any other type is written as its str() and comes
    back as that text (under prob, a and scalar, as a number when the text reads as one).
    The file is written as UTF-8.

    Args:
        circuit (Circuit): The Circuit object to write.
        output_file (str): The name of the output file. Defaults to "random_circuit.chp".
        comment (str): An optional comment to include at the beginning of the file.
        directory (str): Optional directory to save the file in, created if needed. If None, uses
            circuits/ in the current working directory, which is the repository's circuits/ folder
            when run from the root of a source checkout. (Older versions wrote to the circuits/
            folder next to the sdim package, which for an installed package is in site-packages.)

    Returns:
        str: The path to the written file.
    """
    chp_content = ""
    if comment:
        chp_content += f"{comment}\n#\n"
    else:
        chp_content += "Randomly-generated Clifford group quantum circuit\n#\n"
    # Older readers take the dimension from the second token and ignore the rest of the line.
    chp_content += f"d {circuit.dimension} qudits={circuit.num_qudits}\n"

    for gate in circuit.operations:
        gate_str = gate.gate_name
        if gate.target_index is not None:
            gate_str += f" {gate.qudit_index} {gate.target_index}"
        elif gate.qudit_index is not None:
            gate_str += f" {gate.qudit_index}"
        # Writing extra parameters
        if not (gate.params is None):
            for key, value in gate.params.items():
                gate_str += " " + _format_param(key, value)

        chp_content += f"{gate_str}\n"

    if directory is None:
        # Not next to the package, which for an installed sdim is inside site-packages.
        directory = 'circuits'

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Join the directory with the output file name
    output_path = os.path.join(directory, output_file)

    # Write the content to the .chp file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(chp_content)

    return output_path


class GeneralizedSwapGate(cirq.Gate):
    """Swaps two qudits of dimension d: |i, j> -> |j, i>."""

    def __init__(self, d):
        super(GeneralizedSwapGate, self).__init__()
        self.d = d

    def _qid_shape_(self):
        return (self.d, self.d)

    def _unitary_(self):
        d = self.d
        swap = np.zeros((d * d, d * d), dtype=np.complex128)
        for i in range(d):
            for j in range(d):
                swap[j * d + i, i * d + j] = 1
        return swap

    def __pow__(self, exponent):
        if exponent in (1, -1):
            return self
        return NotImplemented

    def _circuit_diagram_info_(self, args):
        return (f"SWAP_{self.d}", f"SWAP_{self.d}")


def circuit_to_cirq_circuit(circuit, measurement=False, print_circuit=False, ignore_noise=True):
    """
    Converts a Circuit object to a Cirq Circuit object.

    With measurement=True, every measurement record of sdim becomes a Cirq measurement with key
    f"m_{qudit}": M, M_X (H_INV, measure, H, which leaves the qudit in the matching X
    eigenstate) and RESET (the pre-reset outcome, then cirq.ResetChannel).  The k-th instance of
    key f"m_{q}" in Cirq's records is then measurement round k of qudit q in sdim's results.
    With measurement=False, M and M_X are left out and RESET is only the reset channel.
    TICK, DETECTOR and LOGICAL_OBSERVABLE do not act on the state and are skipped.
    The noise gates N1 and N2 are left out as well, so the result is the noiseless circuit,
    unless ignore_noise=False.

    Args:
        circuit (Circuit): The Circuit object to convert.
        measurement (bool): Whether to include measurement gates. Defaults to False.
        print_circuit (bool): Whether to print the Cirq circuit. Defaults to False.
        ignore_noise (bool): Leave out the noise gates N1 and N2. Defaults to True; with False,
            a noise gate raises NotImplementedError.

    Returns:
        cirq.Circuit: The equivalent Cirq Circuit object.

    Raises:
        NotImplementedError: For a noise gate (N1, N2) when ignore_noise=False.
    """
    d = circuit.dimension
    # Create a list of qudits.
    qudits = [cirq.LineQid(i, dimension=d) for i in range(circuit.num_qudits)]

    # Create the generalized gates.
    gate_map = {
        "I": IdentityGate(d),
        "H": GeneralizedHadamardGate(d),
        "P": GeneralizedPhaseShiftGate(d),
        "CNOT": GeneralizedCNOTGate(d),
        "X": GeneralizedXPauliGate(d),
        "Z": GeneralizedZPauliGate(d),
        "H_INV": GeneralizedHadamardGateInverse(d),
        "P_INV": GeneralizedPhaseShiftGateInverse(d),
        "CNOT_INV": GeneralizedCNOTGateInverse(d),
        "X_INV": GeneralizedXPauliGateInverse(d),
        "Z_INV": GeneralizedZPauliGateInverse(d),
        "CZ": GeneralizedCZGate(d),
        "CZ_INV": GeneralizedCZGateInverse(d),
        "SWAP": GeneralizedSwapGate(d),
    }

    # Create a Cirq circuit.
    cirq_circuit = cirq.Circuit()

    # Apply each gate in the circuit.
    for op in circuit.operations:
        if op.name in ("TICK", "DETECTOR", "LOGICAL_OBSERVABLE"):
            continue
        if op.name in ("N1", "N2"):
            if ignore_noise:
                continue
            raise NotImplementedError(
                f"Noise gate {op.name} has no Cirq equivalent here; with ignore_noise=True (the default) "
                "it is left out and the circuit is converted without noise."
            )
        if op.name in ("M", "M_X", "RESET"):
            qudit = qudits[op.qudit_index]
            if measurement:
                if op.name == "M_X":
                    cirq_circuit.append(gate_map["H_INV"].on(qudit))
                cirq_circuit.append(cirq.measure(qudit, key=f'm_{op.qudit_index}'))
                if op.name == "M_X":
                    cirq_circuit.append(gate_map["H"].on(qudit))
            if op.name == "RESET":
                cirq_circuit.append(cirq.ResetChannel(dimension=d).on(qudit))
            continue

        # Choose the appropriate gate.
        if op.name in gate_map:
            gate = gate_map[op.name]
        elif op.name == "MUL":
            if op.params is None:
                raise ValueError("Multiplication gate requires an 'a' parameter.")
            scalar = op.params.get("a", op.params.get("scalar"))
            if scalar is None:
                raise ValueError("Multiplication gate requires an 'a' parameter.")
            # Only a mod d matters (the simulators reduce it too).
            gate = GeneralizedMultiplicationGate(d, int(scalar) % d)
        else:
            raise NotImplementedError(f"Gate {op.name} not implemented")

        # Add the gate to the Cirq circuit.
        if op.target_index is None:
            # Single-qudit gate.
            cirq_circuit.append(gate.on(qudits[op.qudit_index]))
        else:
            # Two-qudit gate.
            cirq_circuit.append(gate.on(qudits[op.qudit_index], qudits[op.target_index]))
    for qudit in qudits:
        if not any(op.qubits[0] == qudit for op in cirq_circuit.all_operations()):
            # Append identity to qudits with no gates
            cirq_circuit.append(IdentityGate(d).on(qudit))
    if print_circuit:
        print(cirq_circuit)
    return cirq_circuit

def cirq_statevector_from_circuit(circuit, print_circuit=False, ignore_noise=True):
    """
    Simulates a Circuit object using Cirq and returns the final state vector.

    Measurements and noise are left out (see circuit_to_cirq_circuit), and RESET is applied as a
    reset channel.

    Args:
        circuit (Circuit): The Circuit object to simulate.
        print_circuit (bool): Whether to print the Cirq circuit. Defaults to False.
        ignore_noise (bool): Leave out the noise gates N1 and N2. Defaults to True; with False,
            a noise gate raises NotImplementedError.

    Returns:
        np.ndarray: The final state vector given by the Cirq simulator.
    """
    # Start with an initial state. For a quantum computer, this is usually the state |0...0>.
    cirq_circuit = circuit_to_cirq_circuit(circuit, print_circuit=print_circuit, ignore_noise=ignore_noise)
    # Simulate the Cirq circuit.
    simulator = cirq.Simulator()
    result = simulator.simulate(cirq_circuit)
    
    # Return the final state vector.
    return result.final_state_vector
