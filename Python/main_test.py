from circuit import Circuit
from program import Program


def main():
    circuit = Circuit(2, dimension=3)
    # circuit.add_gate("H", 0)
    # circuit.add_gate("P", 0)
    # circuit.add_gate("H", 0)
    # circuit.add_gate("H", 0)
    # circuit.add_gate("P", 0)
    # circuit.add_gate("P", 0)
    # circuit.add_gate("H", 0)

    # circuit.add_gate("H", 0)
    # circuit.add_gate("P", 0)
    # circuit.add_gate("H", 0)
    # circuit.add_gate("H", 0)
    # circuit.add_gate("P", 0)
    # circuit.add_gate("P", 0)
    # circuit.add_gate("H", 0)

    # circuit.add_gate("CNOT", 0, 1)
    # circuit.add_gate("H", 0)
    # circuit.add_gate("H", 1)
    # circuit.add_gate("CNOT", 0, 1)
    # circuit.add_gate("H", 0)
    # circuit.add_gate("H", 1)
    # circuit.add_gate("CNOT", 0, 1)
    # circuit.add_gate("H", 1)
    # circuit.add_gate("H", 1)

    # circuit.add_gate("H", 0)
    # circuit.add_gate("P", 0)
    # circuit.add_gate("P", 0)
    # circuit.add_gate("H", 0)

    # circuit.add_gate("CNOT", 0, 1)
    # circuit.add_gate("CNOT", 1, 0)
    # circuit.add_gate("CNOT", 0, 1)

    circuit.add_gate("H", 1)
    circuit.add_gate("M", 1)

    program = Program(circuit)
    program.simulate()


if __name__ == "__main__":
    main()
