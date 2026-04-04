"""Tableau simulator benchmarks across dimensions and qudit counts.

Run with: just bench
"""
import pytest
from sdim import TableauSimulator, generate_random_clifford_circuit


DIMENSIONS = [2, 3, 4, 5, 6, 7, 8, 9]
QUDIT_COUNTS = [1, 5, 10, 25, 50, 100]


@pytest.mark.parametrize("d", DIMENSIONS)
@pytest.mark.parametrize("n", QUDIT_COUNTS)
def test_bench_tableau_random_circuit(benchmark, d, n):
    """Benchmark applying 200 random Clifford gates via the tableau."""
    import numpy as np
    rng = np.random.default_rng(d * 1000 + n)
    gates_1q = ["hadamard", "phase_gate", "pauli_x", "pauli_z"]
    gates_2q = ["cnot", "cz", "swap"]

    def run():
        tab = TableauSimulator(n, d)
        for _ in range(200):
            if n == 1 or rng.random() < 0.6:
                gate = rng.choice(gates_1q)
                q = int(rng.integers(0, n))
                getattr(tab, gate)(q)
            else:
                gate = rng.choice(gates_2q)
                q1, q2 = rng.choice(n, size=2, replace=False)
                getattr(tab, gate)(int(q1), int(q2))
        return tab

    benchmark(run)


@pytest.mark.parametrize("d", DIMENSIONS)
@pytest.mark.parametrize("n", [1, 5, 10, 25])
def test_bench_tableau_measure_all(benchmark, d, n):
    """Benchmark measuring all qudits after random prep."""
    import numpy as np
    rng = np.random.default_rng(d * 1000 + n)

    tab = TableauSimulator(n, d)
    for _ in range(50):
        q = int(rng.integers(0, n))
        tab.hadamard(q)

    def run():
        t = TableauSimulator(n, d)
        t.X[:] = tab.X.copy()
        t.Z[:] = tab.Z.copy()
        t.tau_exp[:] = tab.tau_exp.copy()
        results = []
        for q in range(n):
            results.append(t.measure(q))
        return results

    benchmark(run)


@pytest.mark.parametrize("d", DIMENSIONS)
@pytest.mark.parametrize("n", QUDIT_COUNTS)
def test_bench_compile_sampler(benchmark, d, n):
    """Benchmark compile_sampler (IR lowering + reference sample)."""
    circuit = generate_random_clifford_circuit(
        num_qudits=n, num_gates=min(200, n * 10), dimension=d,
        measurement_rounds=1, seed=d * 1000 + n,
    )

    def run():
        return circuit.compile_sampler()

    benchmark(run)
