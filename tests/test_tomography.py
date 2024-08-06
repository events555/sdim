import pytest
from sdim.program import Program
from sdim.chp_parser import read_circuit
from sdim.random_circuit import generate_chp_file, cirq_statevector_from_circuit
import numpy as np
import os

def generate_and_test_circuit(depth, seed):
    # Generate the chp file
    generate_chp_file(20, 40, 40, 0, 3, depth, 2, 1, seed=seed)
    
    # Read the circuit
    circuit = read_circuit("circuits/random_circuit.chp")
    
    # Run the simulation
    statevector = cirq_statevector_from_circuit(circuit)
    amplitudes = np.abs(statevector)**2
    
    num_samples = 1000  # Increased for better statistical significance
    n = circuit.num_qudits
    dimension = circuit.dimension
    num_states = dimension**n
    measurement_counts = np.zeros(num_states, dtype=int)

    for _ in range(num_samples):
        program = Program(circuit)
        measurements = program.simulate()
        key = sum(m.measurement_value * (dimension**i) for i, m in enumerate(measurements))
        measurement_counts[key] += 1

    probabilities = measurement_counts / num_samples
    
    # Clean the amplitudes
    threshold = 1e-14
    cleaned_amp = np.where(np.abs(amplitudes) < threshold, 0, amplitudes)

    # Calculate the Total Variation Distance
    tvd = np.sum(np.abs(probabilities - cleaned_amp)) / 2
    
    return tvd, cleaned_amp, probabilities

@pytest.mark.parametrize("depth,seed", [
    (10, 42)
    # (20, 123),
    # (30, 456),
    # (40, 789),
    # (50, 101112)
])
def test_circuit_simulation(depth, seed):
    tvd, amplitudes, probabilities = generate_and_test_circuit(depth, seed)
    
    # Assertions
    assert np.isclose(np.sum(probabilities), 1, atol=1e-6), "The sum of the probabilities is not approximately 1"
    assert np.isclose(np.sum(amplitudes), 1, atol=1e-6), "The sum of the amplitudes is not approximately 1"
    assert tvd < 0.05, f"Total Variation Distance ({tvd}) is not less than 5%"

    # Optional: Print detailed information for debugging
    print(f"Depth: {depth}, Seed: {seed}")
    print(f"Total Variation Distance: {tvd}")
    print(f"Amplitudes: {amplitudes}")
    print(f"Probabilities: {probabilities}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])