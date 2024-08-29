"""
Stabilizer Dimension (sdim) Package

This package provides tools for working with qudit stabilizer circuits, particularly
focusing on error correction and applications for fault-tolerant quantum computing.

Modules:
    circuit_io: Functions for reading, writing, and converting circuits.
    program: Contains the Program class for managing quantum programs.
    random_circuit: Functions for generating random quantum circuits.
    circuit: Defines the Circuit class for representing quantum circuits.
    tableau: Submodule for working with tableau representations of quantum states.
    diophantine: NumPy-based implementation of the Diophantine solver.
    unitary: Functions for generating and working with unitary matrices.

Classes:
    Program: Manages quantum programs (collection of Circuits and Tableaus) and their execution.
    Circuit: Represents a quantum circuit (series of operations).
    WeylTableau: Represents a Weyl tableau for a composite dimension stabilizer state.
    MeasurementResult: Represents the result of a quantum measurement.
    Tableau: Base class for tableau representations.
    ExtendedTableau: Extended tableau representation for a prime dimension stabilizer state.

Functions:
    read_circuit: Reads a quantum circuit from a file.
    write_circuit: Writes a quantum circuit to a file.
    circuit_to_cirq_circuit: Converts a Circuit to a Cirq circuit.
    cirq_statevector_from_circuit: Generates a Cirq statevector from a Circuit.
    generate_random_circuit: Generates a random quantum circuit.
    generate_and_write_random_circuit: Generates and writes a random circuit to a file.
"""

from .circuit_io import read_circuit, write_circuit, circuit_to_cirq_circuit, cirq_statevector_from_circuit
from .program import Program
from .random_circuit import generate_random_circuit, generate_and_write_random_circuit
from .circuit import Circuit
from .tableau.tableau_composite import WeylTableau
from .tableau.dataclasses import MeasurementResult, Tableau
from .tableau.tableau_prime import ExtendedTableau

