"""Multi-dimensional test suite for d=2..9.

Three tiers:
  1. Deterministic circuits: exact outcome checks, scales to n=100.
  2. Symplectic commutation: tableau invariant after random gates.
  3. TVD tomography: statistical comparison against Cirq statevector (slow).
"""

import numpy as np
import pytest

from sdim import (
    Circuit,
    TableauSimulator,
    generate_random_clifford_circuit,
)

DIMENSIONS = [2, 3, 4, 5, 6, 7, 8, 9]
DETERMINISTIC_N = [1, 5, 10, 25, 50, 100]
SYMPLECTIC_N = [1, 2, 5, 10, 15, 20]


def check_symplectic(tab: TableauSimulator) -> None:
    d = tab.d
    Z = tab.Z[: tab.l] % d
    X = tab.X[: tab.l] % d
    SS = (Z @ X.T - X @ Z.T) % d
    assert np.all(SS == 0), f"symplectic commutation violated for d={d}"


class TestDeterministicCircuits:
    """Tier 3: exact-outcome tests that scale to n=100."""

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", DETERMINISTIC_N)
    def test_x_flip_measure(self, d, n):
        """Apply X on qudit 0 then measure; expect outcome 1 mod d."""
        c = Circuit(n, d)
        c.append("X", 0)
        c.append("M", 0)
        sampler = c.compile_sampler()
        results = sampler.sample(10)
        assert np.all(results[:, 0] == 1), f"d={d} n={n}: X|0> should measure 1"

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", [2, 5, 10, 25, 50])
    def test_swap_chain(self, d, n):
        """Prepare qudit 0 with X, SWAP across chain, measure last."""
        c = Circuit(n, d)
        c.append("X", 0)
        for i in range(n - 1):
            c.append("SWAP", i, i + 1)
        c.append("M", n - 1)
        sampler = c.compile_sampler()
        results = sampler.sample(10)
        assert np.all(results[:, 0] == 1), (
            f"d={d} n={n}: SWAP chain should carry X|0>=|1>"
        )

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_ghz_all_equal(self, d, n):
        """H on qudit 0, CNOT fan-out, all measurements should be equal."""
        c = Circuit(n, d)
        c.append("H", 0)
        for i in range(1, n):
            c.append("CNOT", 0, i)
        c.append("M", list(range(n)))
        sampler = c.compile_sampler()
        results = sampler.sample(50)
        for shot in range(results.shape[0]):
            assert np.all(results[shot, :] == results[shot, 0]), (
                f"d={d} n={n}: GHZ outcomes not equal: {results[shot, :]}"
            )

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_gate_roundtrip(self, d):
        """Apply gate then inverse; tableau should return to identity."""
        n = 2
        gate_pairs_1q = [
            ("hadamard", False, "hadamard", True),  # H then H†
            ("phase_gate", False, "phase_gate", True),
            ("pauli_x", False, "pauli_x", True),
            ("pauli_z", False, "pauli_z", True),
        ]
        gate_pairs_2q = [
            ("cnot", False, "cnot", True),
            ("cz", False, "cz", True),
        ]
        for method, dag1, inv_method, dag2 in gate_pairs_1q:
            ref = TableauSimulator(n, d)
            tab = TableauSimulator(n, d)
            getattr(tab, method)(0, dagger=dag1)
            getattr(tab, inv_method)(0, dagger=dag2)
            tab.modulo()
            ref.modulo()
            assert np.array_equal(tab.X % d, ref.X % d), (
                f"{method} roundtrip failed for d={d}"
            )
            assert np.array_equal(tab.Z % d, ref.Z % d), (
                f"{method} roundtrip failed for d={d}"
            )
        for method, dag1, inv_method, dag2 in gate_pairs_2q:
            ref = TableauSimulator(n, d)
            tab = TableauSimulator(n, d)
            getattr(tab, method)(0, 1, dagger=dag1)
            getattr(tab, inv_method)(0, 1, dagger=dag2)
            tab.modulo()
            ref.modulo()
            assert np.array_equal(tab.X % d, ref.X % d), (
                f"{method} roundtrip failed for d={d}"
            )
            assert np.array_equal(tab.Z % d, ref.Z % d), (
                f"{method} roundtrip failed for d={d}"
            )

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_identity_measurement(self, d):
        """Measuring a fresh |0> state should always give 0."""
        n = 5
        c = Circuit(n, d)
        c.append("M", list(range(n)))
        sampler = c.compile_sampler()
        results = sampler.sample(20)
        assert np.all(results == 0), (
            f"d={d}: fresh |0> should measure all zeros"
        )


class TestSymplecticInvariant:
    """Tier 2: random circuits preserve stabilizer commutation."""

    @pytest.mark.parametrize("d", DIMENSIONS)
    @pytest.mark.parametrize("n", SYMPLECTIC_N)
    def test_random_gates_preserve_commutation(self, d, n):
        rng = np.random.default_rng(42 + d * 100 + n)
        tab = TableauSimulator(n, d)
        gates_1q = ["hadamard", "phase_gate", "pauli_x", "pauli_z"]
        gates_2q = ["cnot", "cz", "swap"]
        for _ in range(200):
            if n == 1 or rng.random() < 0.6:
                gate = rng.choice(gates_1q)
                q = int(rng.integers(0, n))
                getattr(tab, gate)(q)
            else:
                gate = rng.choice(gates_2q)
                q1, q2 = rng.choice(n, size=2, replace=False)
                getattr(tab, gate)(int(q1), int(q2))
        check_symplectic(tab)


class TestTVDTomography:
    """Tier 1: statistical validation against Cirq statevector.

    Only runs when cirq is available and d^n <= 5000.
    """

    MAX_HILBERT = 5000
    TVD_THRESHOLD = 0.20
    SHOTS_MULTIPLIER = 20
    MIN_SHOTS = 2000

    @pytest.fixture(autouse=True)
    def _skip_without_cirq(self):
        pytest.importorskip("cirq")

    @staticmethod
    def tvd(empirical: dict, exact: np.ndarray, d: int, n: int) -> float:
        """Total variation distance between empirical counts and exact distribution."""
        total_shots = sum(empirical.values())
        dist = 0.0
        for basis_idx in range(d**n):
            p_exact = float(np.abs(exact[basis_idx]) ** 2)
            p_emp = empirical.get(basis_idx, 0) / total_shots
            dist += abs(p_exact - p_emp)
        return dist / 2.0

    @staticmethod
    def outcomes_to_scalar(results: np.ndarray, d: int) -> list[int]:
        """Convert (shots, n) measurement array to base-d scalars."""
        n = results.shape[1]
        powers = np.array([d ** (n - 1 - i) for i in range(n)])
        return list(results @ powers)

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_random_clifford_tvd(self, d):
        from collections import Counter

        from sdim import cirq_statevector_from_circuit

        max_n = 1
        while d ** (max_n + 1) <= self.MAX_HILBERT:
            max_n += 1
        max_n = min(max_n, 12)

        for n in range(2, max_n + 1):
            shots = max(self.MIN_SHOTS, self.SHOTS_MULTIPLIER * d**n)
            circuit = generate_random_clifford_circuit(
                num_qudits=n,
                num_gates=50,
                dimension=d,
                measurement_rounds=0,
                seed=d * 1000 + n,
            )
            sv = cirq_statevector_from_circuit(circuit)

            meas_circuit = generate_random_clifford_circuit(
                num_qudits=n,
                num_gates=50,
                dimension=d,
                measurement_rounds=1,
                seed=d * 1000 + n,
            )
            sampler = meas_circuit.compile_sampler()
            results = sampler.sample(shots)

            scalars = self.outcomes_to_scalar(results, d)
            counts = Counter(scalars)
            distance = self.tvd(counts, sv, d, n)
            assert distance < self.TVD_THRESHOLD, (
                f"TVD={distance:.3f} > {self.TVD_THRESHOLD} for d={d}, n={n}, shots={shots}"
            )
