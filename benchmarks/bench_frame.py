"""Frame simulator benchmarks: sampling at varying shots and circuit sizes."""
import pytest
from sdim import generate_random_clifford_circuit


DIMENSIONS = [2, 3, 4, 5, 6, 7, 8, 9]
SHOT_COUNTS = [100, 1000, 10000]


@pytest.mark.parametrize("d", DIMENSIONS)
@pytest.mark.parametrize("n", [5, 10, 25])
@pytest.mark.parametrize("shots", SHOT_COUNTS)
def test_bench_frame_sample(benchmark, d, n, shots):
    """Benchmark frame simulator sampling with noise."""
    circuit = generate_random_clifford_circuit(
        num_qudits=n, num_gates=min(200, n * 10), dimension=d,
        measurement_rounds=1, seed=d * 1000 + n,
    )
    circuit.append("DEPOLARIZE1", list(range(n)), args=[0.01])
    circuit.append("M", list(range(n)))

    sampler = circuit.compile_sampler()
    benchmark(sampler.sample, shots)
