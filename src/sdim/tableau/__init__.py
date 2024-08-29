"""
Tableau Submodule

This submodule provides classes and functions for working with tableau
representations of quantum states, particularly for Pauli and Weyl operators.

Classes:
    WeylTableau: Represents a Weyl tableau for qudit operations. Works with qudits of any dimension.
    MeasurementResult: Represents the result of a quantum measurement.
    Tableau: Base class for tableau representations.
    ExtendedTableau: Extended tableau representation for quantum states. Works with qudits of prime dimension.

The tableau representations allow for efficient simulation of certain
quantum operations, particularly those involving stabilizer states and
Clifford group operations.
"""

from .tableau_composite import WeylTableau
from .dataclasses import MeasurementResult, Tableau
from .tableau_prime import ExtendedTableau
