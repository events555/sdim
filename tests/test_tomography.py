from sdim.program import Program
from sdim.chp_parser import read_circuit
from sdim.random_circuit import generate_chp_file, cirq_statevector_from_circuit
import numpy as np
import itertools
def test_circuit(circuit, num_samples=5000, print_prob=False):
    statevector = cirq_statevector_from_circuit(circuit)
    amplitudes = np.abs(statevector)**2

    # Get the number of qudits and the dimension from the circuit.
    n = circuit.num_qudits
    dimension = circuit.dimension
    num_states = dimension**n
    measurement_counts = np.zeros(num_states, dtype=int)


    # Run the simulation multiple times.
    for _ in range(num_samples):
        program = Program(circuit)
        measurements = program.simulate()
        # Calculate the index from the measurement results
        key = 0
        for result in measurements:
            key = key * dimension + result.measurement_value

        # Increment the count for this index
        measurement_counts[key] += 1

    probabilities = measurement_counts / num_samples
    if print_prob:
        for i, prob in enumerate(probabilities):
            key = "".join(map(str, np.base_repr(i, base=dimension, padding=n)))
            print(f"State {key}: Probability {prob}")
    if not np.isclose(np.sum(probabilities), 1, atol=1e-6):
        raise ValueError("The sum of the probabilities is not approximately 1")
    if not np.isclose(np.sum(amplitudes), 1, atol=1e-6):
        raise ValueError("The sum of the amplitudes is not approximately 1")

    # Return the probabilities and amplitudes.
    return probabilities, amplitudes

if __name__ == "__main__":
    # Parameters for generate_chp_file
    depth = 10
    seed = 42

    successful_circuits = 0

    while True:
        # Generate the chp file
        generate_chp_file(20, 40, 40, 0, 3, 15, 3, 1, seed=None)

        # Run the test_rand_circuit function
        circuit = read_circuit("circuits/random_circuit.chp")
        prob, amp = test_circuit(circuit)

        # Clean the amplitudes
        threshold = 1e-14
        cleaned_amp = np.where(np.abs(amp) < threshold, 0, amp)

        # Calculate the Total Variation Distance
        tvd = np.sum(np.abs(prob - cleaned_amp)) / 2
        print(f"Total Variation Distance: {tvd}")

        # Check if the TVD is less than 5%
        if tvd < 0.05:
            successful_circuits += 1
        else:
            print("TVD is not less than 0.05")
            break

        if successful_circuits >= 100:
            break

    # Uncomment the following lines if you want to print the amplitudes and probabilities
    print(f"Amplitudes: {cleaned_amp}")
    print(f"Probabilities: {prob}")
