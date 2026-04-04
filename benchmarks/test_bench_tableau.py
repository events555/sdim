"""Tableau operation benchmarks for Rust FFI baseline.

Isolates individual operations so pre/post Rust comparisons
are meaningful. Each benchmark measures one thing.
"""
import numpy as np
import pytest

from sdim import TableauSimulator


DIMENSIONS = [2, 3, 5, 6, 7, 9]
QUDIT_COUNTS = [5, 10, 25, 50, 100]
GATE_REPS = 500


def _prep_tableau(n: int, d: int, seed: int = 42) -> TableauSimulator:
    """Create a non-trivial tableau state for benchmarking."""
    rng = np.random.default_rng(seed)
    tab = TableauSimulator(n, d)
    for _ in range(min(50, n * 3)):
        q = int(rng.integers(0, n))
        tab.hadamard(q)
        if n > 1:
            q1, q2 = rng.choice(n, size=2, replace=False)
            tab.cnot(int(q1), int(q2))
    return tab


def _clone(tab: TableauSimulator) -> TableauSimulator:
    t = TableauSimulator(tab.n, tab.d)
    t.X[:] = tab.X.copy()
    t.Z[:] = tab.Z.copy()
    t.tau_exp[:] = tab.tau_exp.copy()
    return t


class TestSingleGateCost:
    """Cost of one gate call at various n. This is the inner loop
    that Rust would replace."""

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", QUDIT_COUNTS)
    def test_hadamard(self, benchmark, d, n):
        tab = _prep_tableau(n, d)

        def run():
            for _ in range(GATE_REPS):
                tab.hadamard(0)

        benchmark(run)

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", QUDIT_COUNTS)
    def test_phase(self, benchmark, d, n):
        tab = _prep_tableau(n, d)

        def run():
            for _ in range(GATE_REPS):
                tab.phase_gate(0)

        benchmark(run)

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", QUDIT_COUNTS)
    def test_cnot(self, benchmark, d, n):
        tab = _prep_tableau(n, d)

        def run():
            for _ in range(GATE_REPS):
                tab.cnot(0, 1)

        benchmark(run)

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", QUDIT_COUNTS)
    def test_cz(self, benchmark, d, n):
        tab = _prep_tableau(n, d)

        def run():
            for _ in range(GATE_REPS):
                tab.cz(0, 1)

        benchmark(run)


class TestMeasurementCost:
    """Measurement is O(n^3) due to Gaussian elimination / SNF.
    This is the #1 Rust target."""

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", [5, 10, 25])
    def test_single_measure(self, benchmark, d, n):
        """Cost of measuring one qudit."""
        tab = _prep_tableau(n, d)

        def run():
            t = _clone(tab)
            t.measure(0)

        benchmark(run)

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", [5, 10, 25])
    def test_measure_all(self, benchmark, d, n):
        """Cost of measuring all n qudits sequentially."""
        tab = _prep_tableau(n, d)

        def run():
            t = _clone(tab)
            for q in range(n):
                t.measure(q)

        benchmark(run)


class TestModuloCost:
    """Periodic modulo reduction on the 2n x n arrays."""

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", QUDIT_COUNTS)
    def test_modulo(self, benchmark, d, n):
        tab = _prep_tableau(n, d)
        for _ in range(100):
            tab.hadamard(0)
            tab.phase_gate(0)

        def run():
            tab.modulo()

        benchmark(run)
