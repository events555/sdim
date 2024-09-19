import pytest
from sdim.program import Program
from sdim.circuit_io import write_circuit, cirq_statevector_from_circuit
from sdim.random_circuit import generate_random_circuit
import numpy as np

def create_key(measurements, dimension):
    sorted_measurements = sorted(measurements, key=lambda m: m.qudit_index)
    key = 0
    for m in sorted_measurements:
        key = key * dimension + m.measurement_value
    return key

def generate_and_test_circuit(depth, dimension, num_qudits):
    circuit = generate_random_circuit(20, 40, 40, 0, num_qudits, depth, dimension, 1)

    statevector = cirq_statevector_from_circuit(circuit)
    amplitudes = np.abs(statevector)**2
    
    num_samples = 1000
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
    
    return tvd, cleaned_amp, probabilities, circuit


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("depth", [15, 30])
def test_random_circuits(dimension, depth):
    num_qudits = 4
    num_circuits = 1000

    for i in range(num_circuits):
        try:
            tvd, amplitudes, probabilities, circuit = generate_and_test_circuit(depth, dimension, num_qudits)
            assert np.isclose(np.sum(probabilities), 1, atol=1e-6), f"Circuit {i+1}: The sum of the probabilities is not approximately 1"
            assert np.isclose(np.sum(amplitudes), 1, atol=1e-6), f"Circuit {i+1}: The sum of the amplitudes is not approximately 1"
            assert tvd < 0.20, f"Circuit {i+1}: Total Variation Distance ({tvd}) is not less than 20%"
        except Exception as e:
            file_name = f"failed_circuit_{dimension}_{depth}_{i+1}.chp"
            comment = f"Failed circuit - Dimension: {dimension}, Depth: {depth}, Circuit: {i+1}"
            write_circuit(circuit, file_name, comment)
            raise e

        print(f"Circuit {i+1} - Dimension: {dimension}, Depth: {depth}, Qudits: {num_qudits}")
        print(f"Total Variation Distance: {tvd}")
        print(f"Sum of Amplitudes: {np.sum(amplitudes)}")
        print(f"Sum of Probabilities: {np.sum(probabilities)}")
        print("---")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "auto"])
