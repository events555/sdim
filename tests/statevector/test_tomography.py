from ...src.program import Program
from ...src.chp_parser import read_circuit
from tomography import state_vector, calculate_amplitudes, check_closeness
def test_epr(num_samples=1000):
    circuit = read_circuit("circuits/epr.chp")
    statevector = state_vector(circuit)
    amplitudes = calculate_amplitudes(statevector)
    program = Program(circuit)

    # Initialize a dictionary to store the measurement results.
    measurement_results = {}

    # Run the simulation multiple times.
    for _ in range(num_samples):
        program.simulate()
        measurements = program.measurement_results

        # Record the measurement results.
        for qudit_index, is_deterministic, result in measurements:
            if qudit_index not in measurement_results:
                measurement_results[qudit_index] = {}
            if result not in measurement_results[qudit_index]:
                measurement_results[qudit_index][result] = 0
            measurement_results[qudit_index][result] += 1

    # Normalize the measurement results to get probabilities.
    probabilities = {}
    for qudit_index, results in measurement_results.items():
        probabilities[qudit_index] = {result: count / num_samples for result, count in results.items()}

    # Print the estimated probabilities.
    for qudit_index, probs in probabilities.items():
        print(f"Qudit {qudit_index}:")
        for result, prob in probs.items():
            print(f"  Result {result}: Probability {prob}")
    check_closeness(amplitudes, probabilities)
    return probabilities

test_epr()