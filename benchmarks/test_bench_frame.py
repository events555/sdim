"""Frame simulator and noise sampling benchmarks."""

import pytest

from sdim import generate_random_clifford_circuit
from sdim.noise import (
    build_noise_banks,
    sample_depolarize1,
    sample_depolarize2,
    sample_x_error,
)

DIMENSIONS = [2, 3, 5, 7, 9]


class TestNoiseSampling:
    """Individual noise sampling function costs."""

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("shots", [1000, 10000, 100000])
    def test_x_error(self, benchmark, d, shots):
        benchmark(sample_x_error, d, shots, 0.1)

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("shots", [1000, 10000, 100000])
    def test_depolarize1(self, benchmark, d, shots):
        benchmark(sample_depolarize1, d, shots, 0.1)

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("shots", [1000, 10000])
    def test_depolarize2(self, benchmark, d, shots):
        benchmark(sample_depolarize2, d, shots, 0.1)


class TestBuildNoiseBanks:
    """Full noise bank construction from a circuit."""

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", [5, 10, 25])
    def test_build_noise_banks(self, benchmark, d, n):
        circuit = generate_random_clifford_circuit(
            num_qudits=n,
            num_gates=100,
            dimension=d,
            measurement_rounds=0,
            seed=d * 1000 + n,
        )
        for i in range(n):
            circuit.append("DEPOLARIZE1", i, args=[0.01])
        circuit.append("M", list(range(n)))

        benchmark(build_noise_banks, circuit, 10000)


class TestFrameSimulation:
    """End-to-end frame simulation (compile + sample)."""

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", [5, 10, 25])
    @pytest.mark.parametrize("shots", [1000, 10000])
    def test_sample(self, benchmark, d, n, shots):
        circuit = generate_random_clifford_circuit(
            num_qudits=n,
            num_gates=min(200, n * 10),
            dimension=d,
            measurement_rounds=1,
            seed=d * 1000 + n,
        )
        circuit.append("DEPOLARIZE1", list(range(n)), args=[0.01])
        circuit.append("M", list(range(n)))
        sampler = circuit.compile_sampler()

        benchmark(sampler.sample, shots)


class TestCompilation:
    """IR lowering + reference sample generation."""

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", [5, 10, 25])
    def test_compile_sampler(self, benchmark, d, n):
        circuit = generate_random_clifford_circuit(
            num_qudits=n,
            num_gates=min(200, n * 10),
            dimension=d,
            measurement_rounds=1,
            seed=d * 1000 + n,
        )

        benchmark(circuit.compile_sampler)
