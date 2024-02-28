from chp_parser import read_circuit
from program import Program


def main():
    circuit = read_circuit("circuits/rand2.chp")
    # circuit = Circuit(2, 2)
    # circuit.add_gate("R", 0)
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

    program = Program(circuit)
    program.simulate()


if __name__ == "__main__":
    main()
