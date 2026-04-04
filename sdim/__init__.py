"""
This package provides tools for working with qudit stabilizer circuits, 
particularly focusing on error correction simulations and applications for 
fault-tolerant quantum computing.

It supports circuit construction, noise modeling, and efficient sampling of 
measurement outcomes and detector events.

## Getting Started

Example usage:

```python
import sdim
import numpy as np

# Create a new quantum circuit for 2 qubits (dimension 2)
circuit = sdim.Circuit(2, 2)

# Add gates
circuit.append("H", 0)
circuit.append("CNOT", 0, 1)
circuit.append("DEPOLARIZE1", 0, args=0.01) # Depolarizing noise on qubit 0 with p=0.01
circuit.append("M", [0, 1])

# Compile a measurement sampler
sampler = circuit.compile_sampler()
measurement_samples = sampler.sample(shots=10)
print("Measurement Samples (shots, num_measurements):")
print(measurement_samples)
# Expected output shape: (10, 2)

# For error correction codes, define detectors and compile a detector sampler
# (Assuming circuit has appropriate DETECTOR and OBSERVABLE_INCLUDE instructions)
detector_circuit = circuit.append("DETECTOR", [sdim.target_rec(-1), sdim.target_rec(-2)])
detector_sampler = detector_circuit.compile_detector_sampler()
detection_events = detector_sampler.sample(shots=100) 
```


## Key Components

- **`sdim.Circuit`**: The primary class for building quantum circuits.
    - Supports various qudit gates, noise channels, and annotations.
    - Methods like `append()`, `compile_sampler()`, `compile_detector_sampler()`.
- **Samplers**:
    - **`sdim.CompiledMeasurementSampler`**: For sampling measurement outcomes from noisy circuits.
    - **`sdim.CompiledDetectorSampler`**: For sampling detection events and logical observable flips, crucial for quantum error correction simulations.
- **Targeting**:
    - **`sdim.GateTarget`**: Class representing different types of gate targets.
    - Module-level functions like `sdim.target_qudit()`, `sdim.target_rec()`, 
      `sdim.target_x()`, etc., for convenient target creation, mimicking Stim's API.
- **Circuit I/O**:
    - Functions like `sdim.read_circuit()` and `sdim.write_circuit()`.
- **Tableau Simulator**:
    - `sdim.ExtendedTableauSimulator` for simulating stabilizer circuits (primarily for reference sample generation and internal use).

## Modules

Internal modules providing core functionality:

- **circuit_io**: Functions for reading, writing, and converting circuits.
- **random_circuit**: Functions for generating random quantum circuits.
- **circuit**: Defines the `Circuit` and `CircuitInstruction` classes.
- **sampler**: Defines the sampler classes.
- **gatedata**: Defines gate properties, `GateTarget`, and related utilities.
- **tableau**: Submodule for working with tableau representations.
- **unitary**: Functions for generating and working with unitary matrices (mainly for Cirq conversion and reference).
"""

# Core classes
from .circuit import Circuit
from .sampler import CompiledMeasurementSampler, CompiledDetectorSampler

# Circuit I/O (cirq interop is lazy — requires sdim[interop])
from .circuit_io import read_circuit, write_circuit

def circuit_to_cirq_circuit(*args, **kwargs):
    from .circuit_io import circuit_to_cirq_circuit as _f
    return _f(*args, **kwargs)

def cirq_statevector_from_circuit(*args, **kwargs):
    from .circuit_io import cirq_statevector_from_circuit as _f
    return _f(*args, **kwargs)
from .random_circuit import (
    generate_random_clifford_circuit,
    generate_and_write_random_circuit
)

# Gate and Target related imports
from .gatedata import (
    GateTarget,  # The class itself
    GATE_DATA,
    gate_name_to_id,
    gate_id_to_name,
    is_gate_noisy,
    is_gate_two_qubit,
    is_gate_pauli,
    is_gate_records,
    is_gate_collapsing,
    is_gate_collapsing_and_records,
    is_not_a_gate
)

from .simulators.tableau_simulator import TableauSimulator


# --- Module-level target creation functions (like stim.target_xxx) ---

def target_qudit(index: int, *, invert: bool = False) -> GateTarget:
    """
    Creates a qubit/qudit target.

    Args:
        index: The 0-based index of the qudit.
        invert: If True, creates an inverted target (e.g., `!5` in Stim).

    Returns:
        A GateTarget instance.
    """
    return GateTarget.qudit(index, invert=invert)

def target_rec(lookback: int) -> GateTarget:
    """
    Creates a measurement record target.

    Args:
        lookback: A negative integer specifying which previous measurement
                  result to target (e.g., -1 for the most recent).

    Returns:
        A GateTarget instance.
    """
    return GateTarget.rec(lookback)

def target_x(index: int, *, invert: bool = False) -> GateTarget:
    """
    Creates an X Pauli target (e.g., for `MPP X0*Z1`).

    Args:
        index: The 0-based index of the qudit.
        invert: If True, inverts the Pauli target (e.g., `!X5`).

    Returns:
        A GateTarget instance.
    """
    return GateTarget.x(index, invert=invert)

def target_y(index: int, *, invert: bool = False) -> GateTarget:
    """Creates a Y Pauli target."""
    return GateTarget.y(index, invert=invert)

def target_z(index: int, *, invert: bool = False) -> GateTarget:
    """Creates a Z Pauli target."""
    return GateTarget.z(index, invert=invert)

def target_sweep_bit(index: int) -> GateTarget:
    """
    Creates a sweep bit target (for parametric circuits).

    Args:
        index: The 0-based index of the sweep bit.

    Returns:
        A GateTarget instance.
    """
    return GateTarget.sweep_bit(index)

def target_combiner() -> GateTarget:
    """
    Creates a combiner target (used in MPP arguments like `X0*Z1`).
    Represents the `*` in Stim's MPP syntax.
    """
    return GateTarget.combiner()

def target_inv(index: int) -> GateTarget:
    """
    Shorthand for creating an inverted qubit/qudit target.
    Equivalent to `sdim.target_qudit(index, invert=True)`.
    """
    return GateTarget.qudit(index, invert=True)

# --- Define __all__ for explicit public API ---
__all__ = [
    # Core Classes
    "Circuit",
    "CompiledMeasurementSampler",
    "CompiledDetectorSampler",
    "GateTarget", # The class for type hinting and direct instantiation if needed

    # Module-level target functions
    "target_qudit",
    "target_rec",
    "target_x",
    "target_y",
    "target_z",
    "target_sweep_bit",
    "target_combiner",
    "target_inv",

    # Circuit I/O and Generation
    "read_circuit",
    "write_circuit",
    "circuit_to_cirq_circuit",
    "cirq_statevector_from_circuit",
    "generate_random_clifford_circuit",
    "generate_and_write_random_circuit",

    # Gate Data Utilities (useful for advanced users or introspection)
    "GATE_DATA",
    "gate_name_to_id",
    "gate_id_to_name",
    "is_gate_noisy",
    "is_gate_two_qubit",
    "is_gate_pauli",
    "is_gate_records",
    "is_gate_collapsing",
    "is_not_a_gate",

    # Tableau (for advanced users or if they need direct access)
    "TableauSimulator",

    # Potentially other high-level functions or classes you add
    # "Program", # If you re-introduce or complete the Program class
]