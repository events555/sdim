import cProfile
import pstats
import matplotlib
import matplotlib.pyplot as plt
import cirq
import numpy as np
from sdim.chp_parser import read_circuit
from sdim.program import Program
from sdim.random_circuit import generate_chp_file, circuit_to_cirq_circuit
from sdim.circuit import Circuit

def main():
    circuit = read_circuit("circuits/rand.chp")
    cirq_circuit = circuit_to_cirq_circuit(circuit, measurement=False)
    print(cirq_circuit)
    program = Program(circuit)
    cirq_simulator = cirq.Simulator()
    cirq_result = cirq_simulator.simulate(cirq_circuit)
    print(cirq_result)
    program.simulate(show_measurement=True, verbose=True, show_gate=True)

if __name__ == "__main__":
    main()