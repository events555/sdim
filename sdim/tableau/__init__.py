"""
# Tableau Submodule

This submodule provides classes and functions for working with tableau representations of quantum states, particularly for Pauli and Weyl operators.

## Key Concepts

- **Generators**: Every tableau should have a list of generators. The coefficients of these generators are stored in the `z_block` and `x_block` attributes.
- **Phases**: The phases of the generators are tracked in the `phase_vector` attribute.
- **Qudits**: A tableau must know the number of qudits it represents (`num_qudits`) and the dimension of each qudit (`dimension`).

### Weyl Tableau Specifics

- The number of generators for a Weyl tableau is bound by `2 * num_qudits`.
- Initially, only `num_qudits` generators are required.
- Every Weyl operator has an inherent phase tracked by the symplectic inner product. Refer to [de Beaudrap (2013)](#1) for more details.

### Extended Tableau Specifics

- The number of generators is guaranteed to be exactly `num_qudits`.
- The phase vector tracks `2*dimension` for even dimensions and `dimension` for odd dimensions. This is equivalent to tracking powers of $i$ for the phase.

## Classes

- **WeylTableau**: Represents a Weyl tableau for qudit operations. It works with qudits of any dimension.
- **MeasurementResult**: Represents the result of a quantum measurement.
- **Tableau**: The base class for tableau representations.
- **ExtendedTableau**: An extended tableau representation for quantum states. It works with qudits of prime dimension.

## Purpose

The tableau representations allow for efficient simulation of certain quantum operations, particularly those involving stabilizer states and Clifford group operations.

This submodule is essential for working with quantum states in a structured and efficient manner, making it easier to simulate and analyze quantum operations.

## References
<a id="1">[1]
</a>de Beaudrap, Niel. “A Linearized Stabilizer Formalism for Systems of Finite Dimension.” Quantum Information and Computation, vol. 13, no. 1 & 2, Jan. 2013, pp. 73–115. arXiv.org, https://doi.org/10.26421/QIC13.1-2-6.


"""

from .tableau_composite import WeylTableau
from .dataclasses import MeasurementResult, Tableau
from .tableau_prime import ExtendedTableau
from .tableau_optimized import hadamard_optimized, phase_optimized, phase_inv_optimized, hadamard_inv_optimized
