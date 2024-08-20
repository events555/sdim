import cProfile
import pstats
from pstats import SortKey
from sdim.circuit import Circuit
from sdim.program import Program
from sdim.chp_parser import read_circuit
from sdim.random_circuit import generate_chp_file, cirq_statevector_from_circuit
from sdim.diophantine import solve
import numpy as np

def create_key(measurements, dimension):
    # Sort measurements by qudit_index in descending order (MSB first)
    sorted_measurements = sorted(measurements, key=lambda m: m.qudit_index)
    
    key = 0
    for m in sorted_measurements:
        key = key * dimension + m.measurement_value
    
    return key

def generate_and_test_circuit(depth, seed):
    # Generate the chp file
    # generate_chp_file(20, 40, 40, 0, 3, depth, 4, 1)
    
    # Read the circuit
    circuit = read_circuit("circuits/random_circuit.chp")
    
    # Run the simulation
    statevector = cirq_statevector_from_circuit(circuit)
    amplitudes = np.abs(statevector)**2
    
    num_samples = 1000
    n = circuit.num_qudits
    dimension = circuit.dimension
    num_states = dimension**n
    measurement_counts = np.zeros(num_states, dtype=int)

    for i in range(num_samples):
        program = Program(circuit)
        measurements = program.simulate(show_measurement=True, verbose=False, show_gate=False)
        print("Simulation #", i)
        key = create_key(measurements, dimension)
        measurement_counts[key] += 1

    probabilities = measurement_counts / num_samples
    
    # Clean the amplitudes
    threshold = 1e-13
    cleaned_amp = np.where(np.abs(amplitudes) < threshold, 0, amplitudes)


    # Calculate the Total Variation Distance
    tvd = np.sum(np.abs(probabilities - cleaned_amp)) / 2
    
    return tvd, cleaned_amp, probabilities

def main():
    # generate_chp_file(20, 40, 40, 0, 5, 20, 3, 1)
    # circuit = read_circuit("circuits/random_circuit.chp")
    # program = Program(circuit)
    # program.simulate(show_measurement=True, verbose=False, show_gate=False)
    # print(cirq_statevector_from_circuit(circuit))
    depth = 10
    seed = 123
    tvd, cleaned_amp, probabilities = generate_and_test_circuit(depth, seed)
    print(f"Total Variation Distance: {tvd}")
    print(f"Amplitudes: {cleaned_amp}")
    print(f"Probabilities: {probabilities}")


if __name__ == "__main__":
    cProfile.run('main()', 'output.prof')
    
    # Print sorted stats
    with open('profiling_results.txt', 'w') as f:
        p = pstats.Stats('output.prof', stream=f)
        p.sort_stats(SortKey.TIME).print_stats(20)  # Print top 20 time-consuming functions
        p.sort_stats(SortKey.CUMULATIVE).print_stats(20)  # Print top 20 cumulative time-consuming functions
    
    print("Profiling complete. Results written to 'profiling_results.txt'")