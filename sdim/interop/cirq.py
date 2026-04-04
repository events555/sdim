"""Cirq circuit conversion and statevector simulation."""

import numpy as np

from ..gates.registry import gate_id_to_name, is_gate_two_qubit
from .unitary import (
    generate_cnot_matrix,
    generate_h_matrix,
    generate_identity_matrix,
    generate_multiply_matrix,
    generate_p_matrix,
    generate_x_matrix,
    generate_z_matrix,
)


def _gate_classes():
    """Lazy-build cirq.Gate subclasses (avoids top-level cirq import)."""
    import cirq

    class IdentityGate(cirq.Gate):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def _qid_shape_(self):
            return (self.d,)

        def _unitary_(self):
            return generate_identity_matrix(self.d)

        def __pow__(self, power):
            return IdentityGate(self.d)

        def _circuit_diagram_info_(self, args):
            return f"I_{self.d}"

    class GeneralizedHadamardGate(cirq.Gate):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def _qid_shape_(self):
            return (self.d,)

        def _unitary_(self):
            return generate_h_matrix(self.d)

        def __pow__(self, power):
            if power == 0:
                return IdentityGate(self.d)
            if power == 1:
                return self
            if power == -1:
                return GeneralizedHadamardGateInverse(self.d)
            return NotImplemented

        def _circuit_diagram_info_(self, args):
            return f"H_{self.d}"

    class GeneralizedHadamardGateInverse(cirq.Gate):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def _qid_shape_(self):
            return (self.d,)

        def _unitary_(self):
            return np.conj(generate_h_matrix(self.d)).T

        def __pow__(self, power):
            if power == 0:
                return IdentityGate(self.d)
            if power == 1:
                return self
            if power == -1:
                return GeneralizedHadamardGate(self.d)
            return NotImplemented

        def _circuit_diagram_info_(self, args):
            return f"H_{self.d}†"

    class GeneralizedPhaseShiftGate(cirq.Gate):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def _qid_shape_(self):
            return (self.d,)

        def _unitary_(self):
            return generate_p_matrix(self.d)

        def __pow__(self, power):
            if power == 0:
                return IdentityGate(self.d)
            if power == 1:
                return self
            if power == -1:
                return GeneralizedPhaseShiftGateInverse(self.d)
            return NotImplemented

        def _circuit_diagram_info_(self, args):
            return f"P_{self.d}"

    class GeneralizedPhaseShiftGateInverse(cirq.Gate):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def _qid_shape_(self):
            return (self.d,)

        def _unitary_(self):
            return np.conj(generate_p_matrix(self.d)).T

        def __pow__(self, power):
            if power == 0:
                return IdentityGate(self.d)
            if power == 1:
                return self
            if power == -1:
                return GeneralizedPhaseShiftGate(self.d)
            return NotImplemented

        def _circuit_diagram_info_(self, args):
            return f"P_{self.d}†"

    class GeneralizedXPauliGate(cirq.Gate):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def _qid_shape_(self):
            return (self.d,)

        def _unitary_(self):
            return generate_x_matrix(self.d)

        def __pow__(self, power):
            if power == 0:
                return IdentityGate(self.d)
            if power == 1:
                return self
            if power == -1:
                return GeneralizedXPauliGateInverse(self.d)
            return NotImplemented

        def _circuit_diagram_info_(self, args):
            return f"X_{self.d}"

    class GeneralizedXPauliGateInverse(cirq.Gate):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def _qid_shape_(self):
            return (self.d,)

        def _unitary_(self):
            return np.conj(generate_x_matrix(self.d)).T

        def __pow__(self, power):
            if power == 0:
                return IdentityGate(self.d)
            if power == 1:
                return self
            if power == -1:
                return GeneralizedXPauliGate(self.d)
            return NotImplemented

        def _circuit_diagram_info_(self, args):
            return f"X_{self.d}†"

    class GeneralizedZPauliGate(cirq.Gate):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def _qid_shape_(self):
            return (self.d,)

        def _unitary_(self):
            return generate_z_matrix(self.d)

        def __pow__(self, power):
            if power == 0:
                return IdentityGate(self.d)
            if power == 1:
                return self
            if power == -1:
                return GeneralizedZPauliGateInverse(self.d)
            return NotImplemented

        def _circuit_diagram_info_(self, args):
            return f"Z_{self.d}"

    class GeneralizedZPauliGateInverse(cirq.Gate):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def _qid_shape_(self):
            return (self.d,)

        def _unitary_(self):
            return np.conj(generate_z_matrix(self.d)).T

        def __pow__(self, power):
            if power == 0:
                return IdentityGate(self.d)
            if power == 1:
                return self
            if power == -1:
                return GeneralizedZPauliGate(self.d)
            return NotImplemented

        def _circuit_diagram_info_(self, args):
            return f"Z_{self.d}†"

    class GeneralizedMultiplyGate(cirq.Gate):
        def __init__(self, d: int, a: int):
            super().__init__()
            if np.gcd(a, d) != 1:
                raise ValueError("a and d must be coprime")
            self.d = d
            self.a = a % d

        def _qid_shape_(self):
            return (self.d,)

        def _unitary_(self):
            return generate_multiply_matrix(self.d, self.a)

        def __pow__(self, power):
            if power == 1:
                return self
            if power == -1:
                return GeneralizedMultiplyGate(self.d, pow(self.a, -1, self.d))
            return NotImplemented

        def _circuit_diagram_info_(self, args):
            return f"MUL_{self.a}"

    class GeneralizedMultiplyGateInverse(GeneralizedMultiplyGate):
        def __init__(self, d: int, a: int):
            a_inv = pow(a, -1, d)
            super().__init__(d, a_inv)

        def _circuit_diagram_info_(self, args):
            return f"MUL_{pow(self.a, -1, self.d)}†"

    class GeneralizedCNOTGate(cirq.Gate):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def _qid_shape_(self):
            return (self.d, self.d)

        def _unitary_(self):
            return generate_cnot_matrix(self.d)

        def __pow__(self, power):
            if power == 1:
                return self
            if power == -1:
                return GeneralizedCNOTGateInverse(self.d)

        def _circuit_diagram_info_(self, args):
            return (f"CNOT_{self.d}_control", f"CNOT_{self.d}_target")

    class GeneralizedCNOTGateInverse(cirq.Gate):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def _qid_shape_(self):
            return (self.d, self.d)

        def _unitary_(self):
            return np.conj(generate_cnot_matrix(self.d)).T

        def __pow__(self, power):
            if power == 1:
                return self
            if power == -1:
                return GeneralizedCNOTGate(self.d)

        def _circuit_diagram_info_(self, args):
            return (f"CNOT_{self.d}_control†", f"CNOT_{self.d}_target†")

    class GeneralizedCZGate(cirq.Gate):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def _qid_shape_(self):
            return (self.d, self.d)

        def _unitary_(self):
            I = generate_identity_matrix(self.d)
            H = generate_h_matrix(self.d)
            CNOT = generate_cnot_matrix(self.d)
            return np.kron(I, H) @ CNOT @ np.kron(I, np.conj(H).T)

        def __pow__(self, power):
            if power == 1:
                return self
            if power == -1:
                return GeneralizedCZGateInverse(self.d)

        def _circuit_diagram_info_(self, args):
            return (f"CZ_{self.d}_control", f"CZ_{self.d}_target")

    class GeneralizedCZGateInverse(cirq.Gate):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def _qid_shape_(self):
            return (self.d, self.d)

        def _unitary_(self):
            I = generate_identity_matrix(self.d)
            H = generate_h_matrix(self.d)
            CNOT = generate_cnot_matrix(self.d)
            return np.conj(np.kron(I, H) @ CNOT @ np.kron(I, np.conj(H).T)).T

        def __pow__(self, power):
            if power == 1:
                return self
            if power == -1:
                return GeneralizedCZGate(self.d)

        def _circuit_diagram_info_(self, args):
            return (f"CZ_{self.d}_control†", f"CZ_{self.d}_target†")

    return {
        "IdentityGate": IdentityGate,
        "GeneralizedHadamardGate": GeneralizedHadamardGate,
        "GeneralizedHadamardGateInverse": GeneralizedHadamardGateInverse,
        "GeneralizedPhaseShiftGate": GeneralizedPhaseShiftGate,
        "GeneralizedPhaseShiftGateInverse": GeneralizedPhaseShiftGateInverse,
        "GeneralizedXPauliGate": GeneralizedXPauliGate,
        "GeneralizedXPauliGateInverse": GeneralizedXPauliGateInverse,
        "GeneralizedZPauliGate": GeneralizedZPauliGate,
        "GeneralizedZPauliGateInverse": GeneralizedZPauliGateInverse,
        "GeneralizedMultiplyGate": GeneralizedMultiplyGate,
        "GeneralizedMultiplyGateInverse": GeneralizedMultiplyGateInverse,
        "GeneralizedCNOTGate": GeneralizedCNOTGate,
        "GeneralizedCNOTGateInverse": GeneralizedCNOTGateInverse,
        "GeneralizedCZGate": GeneralizedCZGate,
        "GeneralizedCZGateInverse": GeneralizedCZGateInverse,
    }


def circuit_to_cirq_circuit(circuit, measurement=False, print_circuit=False):
    import cirq

    gates = _gate_classes()
    IdentityGate = gates["IdentityGate"]

    qudits = [
        cirq.LineQid(i, dimension=circuit.dimension)
        for i in range(circuit.num_qudits)
    ]

    CONST = {
        "I": gates["IdentityGate"](circuit.dimension),
        "H": gates["GeneralizedHadamardGate"](circuit.dimension),
        "P": gates["GeneralizedPhaseShiftGate"](circuit.dimension),
        "CNOT": gates["GeneralizedCNOTGate"](circuit.dimension),
        "X": gates["GeneralizedXPauliGate"](circuit.dimension),
        "Z": gates["GeneralizedZPauliGate"](circuit.dimension),
        "H_INV": gates["GeneralizedHadamardGateInverse"](circuit.dimension),
        "P_INV": gates["GeneralizedPhaseShiftGateInverse"](circuit.dimension),
        "CNOT_INV": gates["GeneralizedCNOTGateInverse"](circuit.dimension),
        "X_INV": gates["GeneralizedXPauliGateInverse"](circuit.dimension),
        "Z_INV": gates["GeneralizedZPauliGateInverse"](circuit.dimension),
        "CZ": gates["GeneralizedCZGate"](circuit.dimension),
        "CZ_INV": gates["GeneralizedCZGateInverse"](circuit.dimension),
    }

    cirq_circuit = cirq.Circuit()

    for inst in circuit.operations:
        name = gate_id_to_name(inst.gate_type)

        if name in ("MULTIPLY", "MULTIPLY_INV"):
            if not inst.args:
                raise ValueError(f"{name} requires a multiplier argument")
            a = int(inst.args[0]) % circuit.dimension
            gate = (
                gates["GeneralizedMultiplyGate"](circuit.dimension, a)
                if name == "MULTIPLY"
                else gates["GeneralizedMultiplyGateInverse"](
                    circuit.dimension, a
                )
            )
            for t in inst.targets:
                cirq_circuit.append(gate.on(qudits[t._value]))
            continue

        if name in CONST:
            gate = CONST[name]
            if is_gate_two_qubit(inst.gate_type):
                for c, t in zip(inst.targets[::2], inst.targets[1::2]):
                    cirq_circuit.append(
                        gate.on(qudits[c._value], qudits[t._value])
                    )
            else:
                for t in inst.targets:
                    cirq_circuit.append(gate.on(qudits[t._value]))
        elif name == "M":
            if measurement:
                for t in inst.targets:
                    cirq_circuit.append(
                        cirq.measure(qudits[t._value], key=f"m_{t._value}")
                    )
        else:
            raise NotImplementedError(f"Gate {name} not implemented")

    for q in qudits:
        if not any(op.qubits[0] == q for op in cirq_circuit.all_operations()):
            cirq_circuit.append(IdentityGate(circuit.dimension).on(q))

    if print_circuit:
        print(cirq_circuit)
    return cirq_circuit


def cirq_statevector_from_circuit(circuit, print_circuit=False):
    import cirq

    cirq_circuit = circuit_to_cirq_circuit(circuit, print_circuit=print_circuit)
    simulator = cirq.Simulator()
    result = simulator.simulate(cirq_circuit)
    return result.final_state_vector
