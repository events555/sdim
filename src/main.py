import cProfile
import pstats
from pstats import SortKey
from sdim.circuit import Circuit
from sdim.program import Program
from sdim.circuit_io import read_circuit, write_circuit, cirq_statevector_from_circuit, circuit_to_cirq_circuit
from sdim.random_circuit import generate_random_circuit, generate_and_write_random_circuit
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
    # circuit = generate_and_write_random_circuit(20, 40, 40, 0, 3, depth, 3, 1)
    circuit = read_circuit("circuits/css_steane_final.chp")
    
    # Run the simulation
    statevector = cirq_statevector_from_circuit(circuit, print_circuit=False)
    amplitudes = np.abs(statevector)**2
    
    num_samples = 1000
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
    
    return tvd, cleaned_amp, probabilities, circuit

def main():
    # Create a new quantum circuit
    circuit = Circuit(4, 2) # Create a circuit with 4 qubits and dimension 2

    # Add gates to the circuit
    circuit.add_gate('X', 0)  # X gate on qubit 0
    circuit.add_gate('CNOT', 0, [1, 2, 3])  # CNOT gate with control on qubit 0 and target on qubit 1
    circuit.add_gate('MR', [0, 1])
    circuit.add_gate('MEASURE', [0, 1, 2, 3]) # Short-hand for multiple single-qubit gates
    # Create a program and add the circuit
    program = Program(circuit) # Must be given an initial circuit as a constructor argument

    # Execute the program
    result = program.simulate(show_measurement=True, show_reset=True) # Runs the program and prints the measurement results. Also returns the results as a list of MeasurementResult objects.
    # num_circuits = 1
    # depth = 18
    # seed = 123
    # for i in range(num_circuits):
    #     tvd, cleaned_amp, probabilities, circuit = generate_and_test_circuit(depth, seed)
    #     for i in range(len(cleaned_amp)):
    #         if cleaned_amp[i] > 0:
    #             print(i, cleaned_amp[i])
    #     for i in range(len(probabilities)):
    #         if probabilities[i] > 0:
    #             print(i, probabilities[i])
    #     print(f"Total Variation Distance: {tvd}", " for circuit", i+1)
    #     print(f"Amplitudes: {cleaned_amp}")
    #     print(f"Probabilities: {probabilities}")
    #     assert tvd < 0.20, f"Circuit {i+1}: Total Variation Distance ({tvd}) is not less than 20%"


if __name__ == "__main__":
    cProfile.run('main()', 'output.prof')
    
    # Print sorted stats
    with open('profiling_results.txt', 'w') as f:
        p = pstats.Stats('output.prof', stream=f)
        p.sort_stats(SortKey.TIME).print_stats(20)  # Print top 20 time-consuming functions
        p.sort_stats(SortKey.CUMULATIVE).print_stats(20)  # Print top 20 cumulative time-consuming functions
    
    print("Profiling complete. Results written to 'profiling_results.txt'")