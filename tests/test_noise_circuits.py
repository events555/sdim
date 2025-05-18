# test_depolarize1_expectations.py
import pytest
from collections import Counter

from sdim.circuit import Circuit # Adjust import if necessary

N_SHOTS_EXPECT = 15000
PROB_TOL_EXPECT = 6 / (N_SHOTS_EXPECT**0.5)

def expected_probs_qudit(
    d: int,
    p: float,
    initial_zero: bool,   # |0…> (True) or |+…> (False)
    measure_Z: bool       # True → measure Z; False → measure X
) -> dict[int, float]:
    """
    Outcome distribution for a single qudit after DEPOLARIZE1(p).

    Keys are integers 0 … d-1.
    Works for any d ≥ 2
    """

    lam = 1.0 - (d**2 / (d**2 - 1.0)) * p      
    uniform_prob = 1.0 / d

    if (initial_zero and not measure_Z) or (not initial_zero and measure_Z):
        return {k: uniform_prob for k in range(d)}

    p0  = (1.0 + (d - 1) * lam) / d              
    p_others = (1.0 - lam) / d                   

    probs = {0: p0}
    probs.update({k: p_others for k in range(1, d)})
    return probs


# --- Main Test Function ---
@pytest.mark.parametrize("dimension", [2]) # Start with d=2 for simplicity
@pytest.mark.parametrize("error_prob", [0.0, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("num_noisy_qubits", [1, 2, 3])
@pytest.mark.parametrize("initial_state_is_zero", [True, False]) # True for |000>, False for |+++>
@pytest.mark.parametrize("measure_Z_basis", [True, False])      # True for Z-basis, False for X-basis
def test_depolarize1_outcome_distributions_3qubit(
    dimension: int,
    error_prob: float,
    num_noisy_qubits: int,
    initial_state_is_zero: bool,
    measure_Z_basis: bool
):
    d = dimension
    num_qudits_total = 3
    if d==1 and num_noisy_qubits > 0 and error_prob > 0:
        pytest.skip("Depolarizing on d=1 qudit has no effect other than identity")


    circuit = Circuit(num_qudits=num_qudits_total, dimension=d)
    measured_qubit_indices = list(range(num_qudits_total))
    if not initial_state_is_zero: # Prepare |+++>
        circuit.append("H", measured_qubit_indices)  # Apply Hadamard to all qubits to create |+++>

    for i in range(num_noisy_qubits):
        circuit.append("DEPOLARIZE1", i, args=[error_prob])

    if not measure_Z_basis:
        circuit.append("MX", measured_qubit_indices)
    else:
        circuit.append("M", measured_qubit_indices)

    sampler = circuit.compile_sampler()
    all_measurement_results = sampler.sample(shots=N_SHOTS_EXPECT)

    for q_idx in range(num_qudits_total):
        is_this_qubit_noisy = (q_idx < num_noisy_qubits)
        
        current_depolarize_prob = error_prob if is_this_qubit_noisy else 0.0

        initial_state_for_qubit_channel_is_zero = initial_state_is_zero

        expected_probs_dist = expected_probs_qudit(
            d,
            current_depolarize_prob,
            initial_state_for_qubit_channel_is_zero,
            measure_Z_basis
        )

        observed_outcomes_this_qubit = all_measurement_results[:, q_idx]
        counts_this_qubit = Counter(observed_outcomes_this_qubit)

        for outcome_val, expected_p in expected_probs_dist.items():
            observed_freq = counts_this_qubit.get(outcome_val, 0) / N_SHOTS_EXPECT
            
            tol_abs = PROB_TOL_EXPECT
            if expected_p < PROB_TOL_EXPECT * 2 and expected_p > 1e-9:
                 tol_abs = expected_p * 0.6 
            elif expected_p <= 1e-9 and observed_freq > PROB_TOL_EXPECT * 0.1 : # Expected zero, but saw non-trivial amount
                 pass # Assertion below will handle it. If expected_p is 0, approx needs abs=small_value.
            if expected_p == 0.0:
                tol_abs = PROB_TOL_EXPECT * 0.1 # Allow small statistical noise if expected is exactly 0

            assert observed_freq == pytest.approx(expected_p, abs=tol_abs), \
                (f"Qubit {q_idx}, Outcome {outcome_val}, d={d}, p_err={error_prob}, "
                 f"num_noisy={num_noisy_qubits}, init_zero={initial_state_is_zero}, meas_Z={measure_Z_basis}\n"
                 f"Expected Dist: {expected_probs_dist}\nObserved Counts: {counts_this_qubit}")