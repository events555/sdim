from sdim.program import Program
from sdim.chp_parser import read_circuit
from tomography import cirq_statevector_from_circuit, numpy_statevector_from_circuit
import numpy as np
import itertools
def test_rand_circuit(num_samples=10000, print_prob=False):
    circuit = read_circuit("profiling/random_circuit.chp")
    statevector = cirq_statevector_from_circuit(circuit)
    amplitudes = np.abs(statevector)**2

    # Get the number of qudits and the dimension from the circuit.
    n = circuit.num_qudits
    basis_states = [str(i) for i in range(circuit.dimension)]

    # Generate all possible keys.
    all_possible_keys = [''.join(key) for key in itertools.product(basis_states, repeat=n)]

    # Initialize the dictionary with all possible keys.
    measurement_results = {key: 0 for key in all_possible_keys}

    # Run the simulation multiple times.
    for _ in range(num_samples):
        program = Program(circuit)
        program.simulate()
        measurements = program.measurement_results
        # Construct a key for each sample.
        key = ''.join(str(result) for _, _, result in measurements)


        # Increment the count for this key.
        if key not in measurement_results:
            measurement_results[key] = 0
        measurement_results[key] += 1

    probabilities = np.array([count / num_samples for key, count in sorted(measurement_results.items())])
    if print_prob:
        for key, prob in zip(sorted(measurement_results.keys()), probabilities):
            print(f"State {key}: Probability {prob}")
    if not np.isclose(np.sum(probabilities), 1, atol=1e-6):
        raise ValueError("The sum of the probabilities is not approximately 1")
    if not np.isclose(np.sum(amplitudes), 1, atol=1e-6):
        raise ValueError("The sum of the amplitudes is not approximately 1")

    # Return the probabilities and amplitudes.
    return probabilities, amplitudes

prob, amp = test_rand_circuit()
threshold = 1e-14
cleaned_amp = np.where(np.abs(amp) < threshold, 0, amp)
tvd = np.sum(np.abs(prob - cleaned_amp)) / 2
print(f"Total Variation Distance: {tvd}")
# print(f"Amplitudes: {cleaned_amp}")
# print(f"Probabilities: {prob}")
