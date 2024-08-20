import pytest
from sdim.program import Program
from sdim.chp_parser import read_circuit
from sdim.random_circuit import generate_chp_file, cirq_statevector_from_circuit
import numpy as np

def create_key(measurements, dimension):
    # Sort measurements by qudit_index in descending order (MSB first)
    sorted_measurements = sorted(measurements, key=lambda m: m.qudit_index)
    
    key = 0
    for m in sorted_measurements:
        key = key * dimension + m.measurement_value
    
    return key

def generate_and_test_circuit(depth, dimension, num_qudits):
    generate_chp_file(20, 40, 40, 0, num_qudits, depth, dimension, 1)
    
    circuit = read_circuit("circuits/random_circuit.chp")
    statevector = cirq_statevector_from_circuit(circuit)
    amplitudes = np.abs(statevector)**2
    
    num_samples = 5000
    num_states = dimension**num_qudits
    measurement_counts = np.zeros(num_states, dtype=int)

    for _ in range(num_samples):
        program = Program(circuit)
        measurements = program.simulate()
        key = create_key(measurements, dimension)
        measurement_counts[key] += 1

    probabilities = measurement_counts / num_samples
    threshold = 1e-14
    cleaned_amp = np.where(np.abs(amplitudes) < threshold, 0, amplitudes)
    tvd = np.sum(np.abs(probabilities - cleaned_amp)) / 2
    
    return tvd, cleaned_amp, probabilities

@pytest.mark.parametrize("dimension", [2, 3, 4, 9])
@pytest.mark.parametrize("depth", [10, 50])
def test_random_circuits(dimension, depth):
    num_qudits = 3
    num_circuits = 50

    for i in range(num_circuits):
        tvd, amplitudes, probabilities = generate_and_test_circuit(depth, dimension, num_qudits)
        
        assert np.isclose(np.sum(probabilities), 1, atol=1e-6), f"Circuit {i+1}: The sum of the probabilities is not approximately 1"
        assert np.isclose(np.sum(amplitudes), 1, atol=1e-6), f"Circuit {i+1}: The sum of the amplitudes is not approximately 1"
        assert tvd < 0.15, f"Circuit {i+1}: Total Variation Distance ({tvd}) is not less than 15%"

        print(f"Circuit {i+1} - Dimension: {dimension}, Depth: {depth}, Qudits: {num_qudits}")
        print(f"Total Variation Distance: {tvd}")
        print(f"Sum of Amplitudes: {np.sum(amplitudes)}")
        print(f"Sum of Probabilities: {np.sum(probabilities)}")
        print("---")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])