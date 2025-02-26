import pytest
import numpy as np
from sdim.sampler import CompiledMeasurementSampler
from sdim.circuit import Circuit


def test_deterministic_measurement_sampler():
    # Create a circuit with deterministic measurements
    circuit = Circuit(2, 3)
    circuit.append("X", 0)
    circuit.append("CNOT", 0, 1)
    circuit.append("M", 0)
    circuit.append("M", 1)

    # Initialize the sampler
    sampler = CompiledMeasurementSampler(circuit=circuit)

    # Sample the measurement results
    shots = 10
    results = sampler.sample(shots)

    expected = np.ones((2, 10))
    # Check that the results are deterministic
    assert np.array_equal(results, expected)

    