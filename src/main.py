import cProfile
import pstats
from pstats import SortKey
from sdim.program import Program
from sdim.chp_parser import read_circuit
from sdim.random_circuit import generate_chp_file, cirq_statevector_from_circuit
import numpy as np

def generate_and_test_circuit(depth, seed):
    # Generate the chp file
    generate_chp_file(20, 40, 40, 0, 3, depth, 2, 1, seed=seed)
    
    # Read the circuit
    circuit = read_circuit("circuits/random_circuit.chp")
    
    # Run the simulation
    statevector = cirq_statevector_from_circuit(circuit)
    amplitudes = np.abs(statevector)**2
    
    num_samples = 100
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

def main():
    depth = 10
    seed = 42
    generate_and_test_circuit(depth, seed)

if __name__ == "__main__":
    cProfile.run('main()', 'output.prof')
    
    # Print sorted stats
    with open('profiling_results.txt', 'w') as f:
        p = pstats.Stats('output.prof', stream=f)
        p.sort_stats(SortKey.TIME).print_stats(20)  # Print top 20 time-consuming functions
        p.sort_stats(SortKey.CUMULATIVE).print_stats(20)  # Print top 20 cumulative time-consuming functions
    
    print("Profiling complete. Results written to 'profiling_results.txt'")