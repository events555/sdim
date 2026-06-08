"""Noise channel sampling and circuit-level distribution tests.

Tests two layers:
  1. Low-level: individual sample_* functions produce correct distributions.
  2. Circuit-level: compile_sampler() with noise gates gives expected outcomes.
"""

from collections import Counter

import numpy as np
import pytest

from sdim.circuit import Circuit
from sdim.noise import (
    sample_depolarize1,
    sample_depolarize2,
    sample_heralded_erasure,
    sample_pauli_channel1,
    sample_x_error,
    sample_y_error,
    sample_z_error,
)

N_SHOTS = 20000
PROB_TOL = 5 / (N_SHOTS**0.5)


def pauli_to_tuple(arr):
    return tuple(arr)


@pytest.fixture
def noise_circuit(request):
    dimension = request.param
    return Circuit(num_qudits=2, dimension=dimension)


@pytest.mark.parametrize(
    "noise_circuit, prob",
    [
        ((2), 0.0),
        ((2), 0.1),
        ((2), 0.5),
        ((2), 1.0),
        ((3), 0.0),
        ((3), 0.1),
        ((3), 0.5),
        ((3), 1.0),
    ],
    indirect=["noise_circuit"],
)
def test_x_error(noise_circuit, prob):
    d = noise_circuit.dimension
    samples = sample_x_error(d, N_SHOTS, prob)
    counts = Counter(pauli_to_tuple(s) for s in samples)

    observed_prob_I = counts[(0, 0)] / N_SHOTS
    assert observed_prob_I == pytest.approx(1 - prob, abs=PROB_TOL)

    if prob > 0:
        total_x = sum(counts[(k, 0)] for k in range(1, d) if (k, 0) in counts)
        assert total_x / N_SHOTS == pytest.approx(prob, abs=PROB_TOL)

        if d > 2 and prob > 1e-9:
            expected_each = prob / (d - 1)
            for k in range(1, d):
                assert counts[(k, 0)] / N_SHOTS == pytest.approx(
                    expected_each, abs=PROB_TOL
                )


@pytest.mark.parametrize(
    "noise_circuit, prob",
    [
        ((2), 0.0),
        ((2), 0.1),
        ((2), 0.5),
        ((2), 1.0),
        ((3), 0.0),
        ((3), 0.1),
        ((3), 0.5),
        ((3), 1.0),
    ],
    indirect=["noise_circuit"],
)
def test_z_error(noise_circuit, prob):
    d = noise_circuit.dimension
    samples = sample_z_error(d, N_SHOTS, prob)
    counts = Counter(pauli_to_tuple(s) for s in samples)

    assert counts[(0, 0)] / N_SHOTS == pytest.approx(1 - prob, abs=PROB_TOL)

    if prob > 0:
        total_z = sum(counts[(0, k)] for k in range(1, d) if (0, k) in counts)
        assert total_z / N_SHOTS == pytest.approx(prob, abs=PROB_TOL)

        if d > 2 and prob > 1e-9:
            expected_each = prob / (d - 1)
            for k in range(1, d):
                assert counts[(0, k)] / N_SHOTS == pytest.approx(
                    expected_each, abs=PROB_TOL
                )


@pytest.mark.parametrize(
    "noise_circuit, prob",
    [
        ((2), 0.0),
        ((2), 0.1),
        ((2), 0.5),
        ((2), 1.0),
        ((3), 0.0),
        ((3), 0.1),
        ((3), 0.5),
        ((3), 1.0),
    ],
    indirect=["noise_circuit"],
)
def test_y_error(noise_circuit, prob):
    d = noise_circuit.dimension
    samples = sample_y_error(d, N_SHOTS, prob)
    counts = Counter(pauli_to_tuple(s) for s in samples)

    assert counts[(0, 0)] / N_SHOTS == pytest.approx(1 - prob, abs=PROB_TOL)

    if prob > 0 and d == 2:
        assert counts[(1, 1)] / N_SHOTS == pytest.approx(prob, abs=PROB_TOL)
    elif prob > 0 and d > 2:
        expected_each = prob / (d - 1)
        for k in range(1, d):
            assert counts[(k, k)] / N_SHOTS == pytest.approx(
                expected_each, abs=PROB_TOL
            )


@pytest.mark.parametrize(
    "noise_circuit, prob",
    [
        ((2), 0.0),
        ((2), 0.1),
        ((2), 0.5),
        ((2), 1.0),
        ((3), 0.0),
        ((3), 0.1),
        ((3), 0.5),
        ((3), 1.0),
    ],
    indirect=["noise_circuit"],
)
def test_depolarize1(noise_circuit, prob):
    d = noise_circuit.dimension
    samples = sample_depolarize1(d, N_SHOTS, prob)
    counts = Counter(pauli_to_tuple(s) for s in samples)

    assert counts[(0, 0)] / N_SHOTS == pytest.approx(1 - prob, abs=PROB_TOL)

    if prob > 0:
        num_non_I = d**2 - 1
        expected_each = prob / num_non_I
        for x_exp in range(d):
            for z_exp in range(d):
                if x_exp == 0 and z_exp == 0:
                    continue
                observed = counts[(x_exp, z_exp)] / N_SHOTS
                assert observed == pytest.approx(expected_each, abs=PROB_TOL)


@pytest.mark.parametrize(
    "noise_circuit, prob",
    [
        ((2), 0.0),
        ((2), 0.1),
        ((2), 0.5),
        ((2), 1.0),
        ((3), 0.0),
        ((3), 0.1),
        ((3), 0.5),
        ((3), 1.0),
    ],
    indirect=["noise_circuit"],
)
def test_depolarize2(noise_circuit, prob):
    d = noise_circuit.dimension
    samples = sample_depolarize2(d, N_SHOTS, prob)
    counts = Counter(pauli_to_tuple(s) for s in samples)

    assert counts[(0, 0, 0, 0)] / N_SHOTS == pytest.approx(
        1 - prob, abs=PROB_TOL
    )

    if prob > 0:
        # Standard two-qudit depolarizing: every non-identity two-qudit
        # Pauli is equally likely, including the weight-1 terms (one qudit
        # identity). There are d**4 - 1 of them.
        num_non_I = d**4 - 1
        expected_each = prob / num_non_I
        for x1 in range(d):
            for z1 in range(d):
                for x2 in range(d):
                    for z2 in range(d):
                        if x1 == 0 and z1 == 0 and x2 == 0 and z2 == 0:
                            continue
                        observed = counts[(x1, z1, x2, z2)] / N_SHOTS
                        assert observed == pytest.approx(
                            expected_each, abs=PROB_TOL
                        )


@pytest.mark.parametrize(
    "noise_circuit, prob",
    [
        ((2), 0.0),
        ((2), 0.1),
        ((2), 0.5),
        ((2), 1.0),
        ((3), 0.0),
        ((3), 0.1),
        ((3), 0.5),
        ((3), 1.0),
    ],
    indirect=["noise_circuit"],
)
def test_heralded_erasure(noise_circuit, prob):
    d = noise_circuit.dimension
    pauli_samples, erasure_flags = sample_heralded_erasure(d, N_SHOTS, prob)

    erasure_counts = Counter(erasure_flags)
    assert erasure_counts[0] / N_SHOTS == pytest.approx(1 - prob, abs=PROB_TOL)
    assert erasure_counts[1] / N_SHOTS == pytest.approx(prob, abs=PROB_TOL)

    if 1 - prob > 1e-9:
        no_erasure = pauli_samples[erasure_flags == 0]
        if no_erasure.size > 0:
            counts = Counter(pauli_to_tuple(p) for p in no_erasure)
            assert len(counts) == 1 and (0, 0) in counts

    if prob > 1e-9:
        erased = pauli_samples[erasure_flags == 1]
        if erased.size > 0:
            counts = Counter(pauli_to_tuple(p) for p in erased)
            n_erased = erased.shape[0]
            expected_each = 1 / (d**2)
            scaled_tol = PROB_TOL * (N_SHOTS / n_erased) ** 0.5
            for x_exp in range(d):
                for z_exp in range(d):
                    observed = counts[(x_exp, z_exp)] / n_erased
                    assert observed == pytest.approx(
                        expected_each, abs=scaled_tol
                    )


@pytest.mark.parametrize(
    "noise_circuit, pmf_choice",
    [
        ((2), "uniform_non_I"),
        ((2), "mostly_I"),
        ((2), "only_X"),
        ((3), "uniform_non_I"),
        ((3), "mostly_I"),
    ],
    indirect=["noise_circuit"],
)
def test_pauli_channel1(noise_circuit, pmf_choice):
    d = noise_circuit.dimension
    expected = {}

    if pmf_choice == "uniform_non_I":
        p_each = 0.1 / (d**2 - 1)
        pmf = np.array([p_each] * (d**2 - 1))
        p_I = 1.0 - sum(pmf)
        expected[(0, 0)] = p_I
        idx = 0
        for x in range(d):
            for z in range(d):
                if x == 0 and z == 0:
                    continue
                expected[(x, z)] = pmf[idx]
                idx += 1

    elif pmf_choice == "mostly_I":
        full_pmf = np.full(d**2, 0.01 / (d**2 - 1))
        full_pmf[0] = 1.0 - sum(full_pmf[1:])
        pmf = full_pmf
        idx = 0
        for x in range(d):
            for z in range(d):
                expected[(x, z)] = full_pmf[idx]
                idx += 1

    elif pmf_choice == "only_X" and d == 2:
        pmf = np.array([0.0, 0.0, 1.0, 0.0])
        expected = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 1.0, (1, 1): 0.0}

    if not expected:
        pytest.skip(f"pmf_choice '{pmf_choice}' not implemented for d={d}")

    samples = sample_pauli_channel1(d, N_SHOTS, list(pmf))
    counts = Counter(pauli_to_tuple(s) for s in samples)

    for key, expected_p in expected.items():
        observed = counts[key] / N_SHOTS
        # 5σ binomial tolerance
        sigma = (expected_p * (1 - expected_p) / N_SHOTS) ** 0.5
        tol = max(5 * sigma, PROB_TOL)
        assert observed == pytest.approx(expected_p, abs=tol)


N_SHOTS_CIRCUIT = 15000
PROB_TOL_CIRCUIT = 6 / (N_SHOTS_CIRCUIT**0.5)


def expected_probs_qudit(d, p, initial_zero, measure_Z):
    lam = 1.0 - (d**2 / (d**2 - 1.0)) * p
    uniform = 1.0 / d

    if (initial_zero and not measure_Z) or (not initial_zero and measure_Z):
        return {k: uniform for k in range(d)}

    p0 = (1.0 + (d - 1) * lam) / d
    probs = {0: p0}
    probs.update({k: (1.0 - lam) / d for k in range(1, d)})
    return probs


@pytest.mark.parametrize("dimension", [2])
@pytest.mark.parametrize("error_prob", [0.0, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("num_noisy_qubits", [1, 2, 3])
@pytest.mark.parametrize("initial_state_is_zero", [True, False])
@pytest.mark.parametrize("measure_Z_basis", [True, False])
def test_depolarize1_circuit_distribution(
    dimension,
    error_prob,
    num_noisy_qubits,
    initial_state_is_zero,
    measure_Z_basis,
):
    d = dimension
    n = 3
    circuit = Circuit(num_qudits=n, dimension=d)

    if not initial_state_is_zero:
        circuit.append("H", list(range(n)))

    for i in range(num_noisy_qubits):
        circuit.append("DEPOLARIZE1", i, args=[error_prob])

    if measure_Z_basis:
        circuit.append("M", list(range(n)))
    else:
        circuit.append("MX", list(range(n)))

    sampler = circuit.compile_sampler()
    results = sampler.sample(shots=N_SHOTS_CIRCUIT)

    for q_idx in range(n):
        p = error_prob if q_idx < num_noisy_qubits else 0.0
        expected = expected_probs_qudit(
            d, p, initial_state_is_zero, measure_Z_basis
        )
        outcomes = Counter(results[:, q_idx])

        for val, expected_p in expected.items():
            observed = outcomes.get(val, 0) / N_SHOTS_CIRCUIT
            tol = PROB_TOL_CIRCUIT
            if expected_p < tol * 2 and expected_p > 1e-9:
                tol = expected_p * 0.6
            elif expected_p == 0.0:
                tol = PROB_TOL_CIRCUIT * 0.1
            assert observed == pytest.approx(expected_p, abs=tol), (
                f"q={q_idx} outcome={val} d={d} p={error_prob} "
                f"noisy={num_noisy_qubits} zero={initial_state_is_zero} Z={measure_Z_basis}"
            )
