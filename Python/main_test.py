from circuit import Circuit
from tableau import Program


def main():
    circuit = Circuit(1, dimension=3)
    circuit.add_gate("H", 0)
    circuit.add_gate("P", 0)
    # circuit.add_gate("CNOT", 0, 1)

    program = Program(circuit)
    program.simulate()


if __name__ == "__main__":
    main()
