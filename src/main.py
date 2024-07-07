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
from sympy import *
from sdim.tableau import Tableau
def main():
    circuit = Circuit(1, 2)
    circuit.add_gate("H", 0)
    circuit.add_gate("M", 0)
    program = Program(circuit)
    program.simulate(show_measurement=True, verbose=True, show_gate=True)

if __name__ == "__main__":
    main()