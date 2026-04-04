"""End-to-end validation against Cirq statevector simulation.

Requires cirq: run with `just test-all`.

Override defaults via environment variables:
    TOMO_QUDITS=3 TOMO_CIRCUITS=2000 uv run pytest tests/test_tomography.py
"""

import os
import pytest
import numpy as np

from sdim.circuit_io import cirq_statevector_from_circuit
from sdim.random_circuit import generate_random_clifford_circuit

NUM_QUDITS = int(os.environ.get("TOMO_QUDITS", 2))
NUM_CIRCUITS = int(os.environ.get("TOMO_CIRCUITS", 500))
NUM_SHOTS = int(os.environ.get("TOMO_SHOTS", 2000))


def tvd_for_circuit(circuit, num_shots=NUM_SHOTS):
    statevector = cirq_statevector_from_circuit(circuit)
    amplitudes = np.abs(statevector) ** 2
    amplitudes = np.where(np.abs(amplitudes) < 1e-14, 0, amplitudes)

    d = circuit.dimension
    n = circuit.num_qudits
    num_states = d ** n

    sampler = circuit.compile_sampler()
    measurements = sampler.sample(shots=num_shots)

    counts = np.zeros(num_states, dtype=int)
    for shot in measurements:
        key = 0
        for m in shot:
            key = key * d + m
        counts[key] += 1

    probs = counts / num_shots
    return np.sum(np.abs(probs - amplitudes)) / 2


@pytest.mark.parametrize("d", [2, 3, 4, 5, 6, 9])
@pytest.mark.parametrize("depth", [5, 20, 50])
def test_random_circuits(d, depth):
    shots = max(NUM_SHOTS, 20 * d ** NUM_QUDITS)

    for i in range(NUM_CIRCUITS):
        circuit = generate_random_clifford_circuit(
            NUM_QUDITS, depth, d, measurement_rounds=1,
        )
        tvd = tvd_for_circuit(circuit, shots)
        assert tvd < 0.20, (
            f"circuit {i} d={d} depth={depth}: TVD={tvd:.3f}"
        )
