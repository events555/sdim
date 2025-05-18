# test_noise_channels.py
import pytest
import numpy as np
from collections import Counter
from sdim.circuit import Circuit # To access the _sample_* methods
# If sdim.circuit is not found, adjust path or install sdim package if it's structured that way
# from ..sdim.circuit import Circuit # Example if tests are in a subfolder

N_SHOTS_MONTE_CARLO = 20000  # Number of samples for statistical tests
# Tolerance for probability checks, roughly 5-sigma for binomial
# (can be adjusted based on N_SHOTS_MONTE_CARLO)
PROB_TOLERANCE_ABS = 5 / (N_SHOTS_MONTE_CARLO**0.5)

# Helper to create a circuit instance just for dimension
@pytest.fixture
def noise_circuit(request):
    """Fixture to create a Circuit instance with a specific dimension."""
    dimension = request.param
    return Circuit(num_qudits=2, dimension=dimension) # Need at least 2 for DEPOLARIZE2

def pauli_to_tuple(pauli_arr):
    """Converts a 1D numpy array representing a Pauli error to a hashable tuple."""
    return tuple(pauli_arr)

# --- Test Cases ---

@pytest.mark.parametrize("noise_circuit, prob", [
    ((2), 0.0), ((2), 0.1), ((2), 0.5), ((2), 1.0), # Qubit cases
    ((3), 0.0), ((3), 0.1), ((3), 0.5), ((3), 1.0), # Qudit (d=3) cases
], indirect=["noise_circuit"])
def test_x_error(noise_circuit, prob):
    d = noise_circuit.dimension
    samples = noise_circuit._sample_x_error(N_SHOTS_MONTE_CARLO, prob)
    counts = Counter(pauli_to_tuple(s) for s in samples)

    # Expected outcomes: (x_exp, z_exp)
    # Identity: (0, 0)
    # X-type errors: (k, 0) for k in 1..d-1

    # Check Identity probability
    observed_prob_I = counts[(0, 0)] / N_SHOTS_MONTE_CARLO
    assert observed_prob_I == pytest.approx(1 - prob, abs=PROB_TOLERANCE_ABS)

    if prob > 0:
        # Check probability of any X-type error
        total_x_errors = sum(counts[(k, 0)] for k in range(1, d) if (k,0) in counts)
        observed_prob_any_X = total_x_errors / N_SHOTS_MONTE_CARLO
        assert observed_prob_any_X == pytest.approx(prob, abs=PROB_TOLERANCE_ABS)

        if d > 2 and prob > 1e-9: # If d=2, only one X^k (X^1)
            # For d > 2, X errors are X^1, X^2, ..., X^(d-1), each should be uniform
            expected_prob_specific_X = prob / (d - 1)
            for k in range(1, d):
                observed_prob_Xk = counts[(k, 0)] / N_SHOTS_MONTE_CARLO
                assert observed_prob_Xk == pytest.approx(expected_prob_specific_X, abs=PROB_TOLERANCE_ABS)
        elif d == 2 and prob > 1e-9:
            observed_prob_X1 = counts[(1, 0)] / N_SHOTS_MONTE_CARLO
            assert observed_prob_X1 == pytest.approx(prob, abs=PROB_TOLERANCE_ABS)

@pytest.mark.parametrize("noise_circuit, prob", [
    ((2), 0.0), ((2), 0.1), ((2), 0.5), ((2), 1.0),
    ((3), 0.0), ((3), 0.1), ((3), 0.5), ((3), 1.0),
], indirect=["noise_circuit"])
def test_z_error(noise_circuit, prob):
    d = noise_circuit.dimension
    samples = noise_circuit._sample_z_error(N_SHOTS_MONTE_CARLO, prob)
    counts = Counter(pauli_to_tuple(s) for s in samples)

    observed_prob_I = counts[(0, 0)] / N_SHOTS_MONTE_CARLO
    assert observed_prob_I == pytest.approx(1 - prob, abs=PROB_TOLERANCE_ABS)

    if prob > 0:
        total_z_errors = sum(counts[(0, k)] for k in range(1, d) if (0,k) in counts)
        observed_prob_any_Z = total_z_errors / N_SHOTS_MONTE_CARLO
        assert observed_prob_any_Z == pytest.approx(prob, abs=PROB_TOLERANCE_ABS)

        if d > 2 and prob > 1e-9:
            expected_prob_specific_Z = prob / (d - 1)
            for k in range(1, d):
                observed_prob_Zk = counts[(0, k)] / N_SHOTS_MONTE_CARLO
                assert observed_prob_Zk == pytest.approx(expected_prob_specific_Z, abs=PROB_TOLERANCE_ABS)
        elif d == 2 and prob > 1e-9:
            observed_prob_Z1 = counts[(0, 1)] / N_SHOTS_MONTE_CARLO
            assert observed_prob_Z1 == pytest.approx(prob, abs=PROB_TOLERANCE_ABS)


@pytest.mark.parametrize("noise_circuit, prob", [
    ((2), 0.0), ((2), 0.1), ((2), 0.5), ((2), 1.0),
    ((3), 0.0), ((3), 0.1), ((3), 0.5), ((3), 1.0),
], indirect=["noise_circuit"])
def test_y_error(noise_circuit, prob):
    d = noise_circuit.dimension
    samples = noise_circuit._sample_y_error(N_SHOTS_MONTE_CARLO, prob)
    counts = Counter(pauli_to_tuple(s) for s in samples)

    observed_prob_I = counts[(0, 0)] / N_SHOTS_MONTE_CARLO
    assert observed_prob_I == pytest.approx(1 - prob, abs=PROB_TOLERANCE_ABS)

    if prob > 0:
        # Y error is defined as (a, a) for random non-zero a (if d>2) or (1,1) if d=2
        if d == 2:
            observed_prob_Y = counts[(1, 1)] / N_SHOTS_MONTE_CARLO
            assert observed_prob_Y == pytest.approx(prob, abs=PROB_TOLERANCE_ABS)
        else: # d > 2
            total_y_type_errors = sum(counts[(k, k)] for k in range(1, d) if (k,k) in counts)
            observed_prob_any_Y_type = total_y_type_errors / N_SHOTS_MONTE_CARLO
            assert observed_prob_any_Y_type == pytest.approx(prob, abs=PROB_TOLERANCE_ABS)

            expected_prob_specific_Y = prob / (d - 1)
            for k in range(1, d):
                observed_prob_Yk = counts[(k, k)] / N_SHOTS_MONTE_CARLO
                assert observed_prob_Yk == pytest.approx(expected_prob_specific_Y, abs=PROB_TOLERANCE_ABS)

@pytest.mark.parametrize("noise_circuit, prob", [
    ((2), 0.0), ((2), 0.1), ((2), 0.5), ((2), 1.0),
    ((3), 0.0), ((3), 0.1), ((3), 0.5), ((3), 1.0),
], indirect=["noise_circuit"])
def test_depolarize1(noise_circuit, prob):
    d = noise_circuit.dimension
    samples = noise_circuit._sample_depolarize1_noise(N_SHOTS_MONTE_CARLO, prob)
    counts = Counter(pauli_to_tuple(s) for s in samples)

    observed_prob_I = counts[(0, 0)] / N_SHOTS_MONTE_CARLO
    assert observed_prob_I == pytest.approx(1 - prob, abs=PROB_TOLERANCE_ABS)

    if prob > 0:
        num_non_identity_paulis = d**2 - 1
        expected_prob_specific_non_I = prob / num_non_identity_paulis

        total_non_I_count = 0
        for x_exp in range(d):
            for z_exp in range(d):
                if x_exp == 0 and z_exp == 0:
                    continue # Skip Identity
                pauli_op = (x_exp, z_exp)
                observed_prob_op = counts[pauli_op] / N_SHOTS_MONTE_CARLO
                assert observed_prob_op == pytest.approx(expected_prob_specific_non_I, abs=PROB_TOLERANCE_ABS)
                total_non_I_count += counts[pauli_op]
        
        assert total_non_I_count / N_SHOTS_MONTE_CARLO == pytest.approx(prob, abs=PROB_TOLERANCE_ABS)


@pytest.mark.parametrize("noise_circuit, prob", [
    ((2), 0.0), ((2), 0.1), ((2), 0.5), ((2), 1.0),
    ((3), 0.0), ((3), 0.1), ((3), 0.5), ((3), 1.0),
], indirect=["noise_circuit"])
def test_depolarize2(noise_circuit, prob):
    d = noise_circuit.dimension
    samples = noise_circuit._sample_depolarize2_noise(N_SHOTS_MONTE_CARLO, prob)
    counts = Counter(pauli_to_tuple(s) for s in samples)

    # Expected: II with prob 1-p.
    # Otherwise (with prob p), Q1 gets a random non-I Pauli, Q2 gets a random non-I Pauli.
    # There are (d^2-1) non-I Paulis for a single qudit.

    observed_prob_II = counts[(0, 0, 0, 0)] / N_SHOTS_MONTE_CARLO
    assert observed_prob_II == pytest.approx(1 - prob, abs=PROB_TOLERANCE_ABS)

    if prob > 0:
        num_single_qudit_non_I = d**2 - 1
        # Prob of specific (NonI_1, NonI_2) = p * (1/num_single_qudit_non_I) * (1/num_single_qudit_non_I)
        expected_prob_specific_config = prob / (num_single_qudit_non_I * num_single_qudit_non_I)

        total_error_configs_count = 0
        for x1 in range(d):
            for z1 in range(d):
                if x1 == 0 and z1 == 0: continue # Pauli1 must be non-I
                for x2 in range(d):
                    for z2 in range(d):
                        if x2 == 0 and z2 == 0: continue # Pauli2 must be non-I
                        
                        pauli_op_2q = (x1, z1, x2, z2)
                        observed_prob_op = counts[pauli_op_2q] / N_SHOTS_MONTE_CARLO
                        assert observed_prob_op == pytest.approx(expected_prob_specific_config, abs=PROB_TOLERANCE_ABS)
                        total_error_configs_count += counts[pauli_op_2q]
        
        assert total_error_configs_count / N_SHOTS_MONTE_CARLO == pytest.approx(prob, abs=PROB_TOLERANCE_ABS)


@pytest.mark.parametrize("noise_circuit, prob", [
    ((2), 0.0), ((2), 0.1), ((2), 0.5), ((2), 1.0),
    ((3), 0.0), ((3), 0.1), ((3), 0.5), ((3), 1.0),
], indirect=["noise_circuit"])
def test_heralded_erasure(noise_circuit, prob):
    d = noise_circuit.dimension
    pauli_samples, erasure_flags = noise_circuit._sample_heralded_erasure_noise(N_SHOTS_MONTE_CARLO, prob)

    # erasure_flags is (N_SHOTS_MONTE_CARLO,)
    # pauli_samples is (N_SHOTS_MONTE_CARLO, 2)

    # Count erasure flags
    erasure_counts = Counter(erasure_flags)
    observed_prob_no_erasure = erasure_counts[0] / N_SHOTS_MONTE_CARLO
    observed_prob_erasure = erasure_counts[1] / N_SHOTS_MONTE_CARLO

    assert observed_prob_no_erasure == pytest.approx(1 - prob, abs=PROB_TOLERANCE_ABS)
    assert observed_prob_erasure == pytest.approx(prob, abs=PROB_TOLERANCE_ABS)

    # Check Pauli distribution when no erasure occurs (should be Identity)
    if 1 - prob > 1e-9 : # If there's a chance of no erasure
        no_erasure_paulis = pauli_samples[erasure_flags == 0]
        if no_erasure_paulis.size > 0:
            counts_no_erasure = Counter(pauli_to_tuple(p) for p in no_erasure_paulis)
            assert len(counts_no_erasure) == 1, "Only Identity expected when no erasure"
            assert (0,0) in counts_no_erasure, "Identity must be the only Pauli when no erasure"
            assert counts_no_erasure[(0,0)] == no_erasure_paulis.shape[0] # All must be identity


    # Check Pauli distribution when erasure occurs (should be uniform over ALL d*d Paulis)
    if prob > 1e-9: # If there's a chance of erasure
        erasure_paulis = pauli_samples[erasure_flags == 1]
        if erasure_paulis.size > 0:
            counts_erasure = Counter(pauli_to_tuple(p) for p in erasure_paulis)
            num_erasure_events = erasure_paulis.shape[0]
            
            expected_prob_specific_pauli_on_erasure = 1 / (d**2)
            
            for x_exp in range(d):
                for z_exp in range(d):
                    pauli_op = (x_exp, z_exp)
                    observed_freq_op = counts_erasure[pauli_op] / num_erasure_events
                    assert observed_freq_op == pytest.approx(expected_prob_specific_pauli_on_erasure, abs=PROB_TOLERANCE_ABS * (N_SHOTS_MONTE_CARLO/num_erasure_events)**0.5 if num_erasure_events > 0 else PROB_TOLERANCE_ABS)
                    # Tolerance adjustment because num_erasure_events can be smaller than N_SHOTS_MONTE_CARLO


# --- Tests for Pauli Channels (if you want to test them specifically) ---
# These are a bit more involved as you need to define PMFs.

@pytest.mark.parametrize("noise_circuit, pmf_choice", [
    ((2), "uniform_non_I"), ((2), "mostly_I"), ((2), "only_X"),
    ((3), "uniform_non_I"), ((3), "mostly_I"),
], indirect=["noise_circuit"])
def test_pauli_channel1(noise_circuit, pmf_choice):
    d = noise_circuit.dimension
    pmf = []
    expected_probs_map = {}

    if pmf_choice == "uniform_non_I":
        # p for each non-I, 1-(d^2-1)p for I
        p_non_I_each = 0.1 / (d**2 - 1) if d**2 > 1 else 0.1
        pmf_list_non_I = [p_non_I_each] * (d**2 - 1)
        # The function can take d^2-1 probs, with I prob derived
        pmf = np.array(pmf_list_non_I)
        
        p_I_derived = 1.0 - sum(pmf_list_non_I)
        expected_probs_map[(0,0)] = p_I_derived
        idx_non_I = 0
        for x in range(d):
            for z in range(d):
                if x == 0 and z == 0: continue
                expected_probs_map[(x,z)] = pmf_list_non_I[idx_non_I]
                idx_non_I += 1

    elif pmf_choice == "mostly_I":
        # Full PMF of length d^2
        full_pmf = np.full(d**2, 0.01 / (d**2 -1 if d**2 > 1 else 1)) # Small prob for non-I
        full_pmf[0] = 1.0 - sum(full_pmf[1:]) # High prob for I
        pmf = full_pmf
        idx = 0
        for x in range(d):
            for z in range(d):
                expected_probs_map[(x,z)] = full_pmf[idx]
                idx += 1
                
    elif pmf_choice == "only_X" and d==2: # Specific to d=2 for simple X
        # PMF: [p_I, p_X, p_Y, p_Z] for d=2
        # Order is I, X, Z, Y for d=2 if index = x*d+z means I (00), Z (01), X (10), Y (11)
        # The code says index = x * d + z.
        # d=2: (0,0)->0 (I), (0,1)->1 (Z), (1,0)->2 (X), (1,1)->3 (Y)
        pmf = np.array([0.0, 0.0, 1.0, 0.0]) # Only X
        expected_probs_map[(0,0)] = 0.0
        expected_probs_map[(0,1)] = 0.0 # Z
        expected_probs_map[(1,0)] = 1.0 # X
        expected_probs_map[(1,1)] = 0.0 # Y


    if not expected_probs_map: # Skip if pmf_choice not handled for this d
        pytest.skip(f"PMF choice '{pmf_choice}' not implemented for d={d}")

    samples = noise_circuit._sample_pauli_channel1_noise(N_SHOTS_MONTE_CARLO, list(pmf))
    counts = Counter(pauli_to_tuple(s) for s in samples)

    for pauli_op_tuple, expected_prob in expected_probs_map.items():
        observed_prob = counts[pauli_op_tuple] / N_SHOTS_MONTE_CARLO
        # Adjust tolerance if expected_prob is very small, pytest.approx might need rel
        tol = PROB_TOLERANCE_ABS
        if expected_prob < PROB_TOLERANCE_ABS * 2: # Heuristic
            tol = expected_prob * 0.5 if expected_prob > 1e-9 else PROB_TOLERANCE_ABS
        assert observed_prob == pytest.approx(expected_prob, abs=tol)


# PAULI_CHANNEL_2 test would be similar but with d^4 probabilities, making it more complex to define PMFs.
# For brevity, I'll skip the full implementation of PAULI_CHANNEL_2 test here,
# but the structure would mirror test_pauli_channel1 with more complex PMF setup.