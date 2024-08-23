import cProfile
import pstats
from pstats import SortKey
from sdim.circuit import Circuit
from sdim.program import Program
from sdim.chp_parser import read_circuit
from sdim.random_circuit import generate_random_circuit, cirq_statevector_from_circuit
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
    # circuit = generate_random_circuit(20, 40, 40, 0, 3, depth, 9, 1)
    circuit = read_circuit("circuits/random_circuit.chp")
    
    # Run the simulation
    statevector = cirq_statevector_from_circuit(circuit)
    amplitudes = np.abs(statevector)**2
    
    num_samples = 2000
    n = circuit.num_qudits
    dimension = circuit.dimension
    num_states = dimension**n
    measurement_counts = np.zeros(num_states, dtype=int)

    for i in range(num_samples):
        program = Program(circuit)
        measurements = program.simulate(show_measurement=False, verbose=False, show_gate=False, exact=False)
        # print("Simulation #", i)
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
    # circuit = Circuit(2, 4)
    # circuit.add_gate("X", 0)
    # circuit.add_gate("X", 1)
    # circuit.add_gate("CNOT", 0, 1)
    # circuit.add_gate("M", 0)
    # circuit.add_gate("M", 1)
    # program = Program(circuit)
    # program.simulate(show_measurement=True, exact=True)
    # generate_chp_file(20, 40, 40, 0, 5, 20, 3, 1)
    # circuit = read_circuit("circuits/random_circuit.chp")
    # program = Program(circuit)
    num_circuits = 1
    depth = 18
    seed = 123
    for i in range(num_circuits):
        tvd, cleaned_amp, probabilities = generate_and_test_circuit(depth, seed)
        print(f"Total Variation Distance: {tvd}", " for circuit", i+1)
        print(f"Amplitudes: {cleaned_amp}")
        print(f"Probabilities: {probabilities}")
        assert tvd < 0.20, f"Circuit {i+1}: Total Variation Distance ({tvd}) is not less than 20%"


if __name__ == "__main__":
    cProfile.run('main()', 'output.prof')
    
    # Print sorted stats
    with open('profiling_results.txt', 'w') as f:
        p = pstats.Stats('output.prof', stream=f)
        p.sort_stats(SortKey.TIME).print_stats(20)  # Print top 20 time-consuming functions
        p.sort_stats(SortKey.CUMULATIVE).print_stats(20)  # Print top 20 cumulative time-consuming functions
    
    print("Profiling complete. Results written to 'profiling_results.txt'")