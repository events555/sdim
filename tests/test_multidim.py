"""Multi-dimensional test suite for d=2..9.

Three tiers:
  1. Deterministic circuits: exact outcome checks, scales to n=100.
  2. Symplectic commutation: tableau invariant after random gates.
  3. TVD tomography: statistical comparison against Cirq statevector (slow).
"""

import math
import random

import numpy as np
import pytest

from sdim import (
    Circuit,
    TableauSimulator,
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
    """Sampler vs Cirq statevector over random Cliffords, all d. Needs cirq."""

    MAX_HILBERT = 4096
    SHOTS_PER_OUTCOME = 40
    MIN_SHOTS = 4000
    SEEDS_PER_CONFIG = 10
    DEPTHS = [20, 60]
    DELTA_TOTAL = 1e-9  # suite-wide false-failure budget (Bonferroni)

    @pytest.fixture(autouse=True)
    def _skip_without_cirq(self):
        pytest.importorskip("cirq")

    @staticmethod
    def tvd_threshold(
        num_outcomes: int,
        shots: int,
        num_comparisons: int,
        *,
        delta_total: float = DELTA_TOTAL,
        two_sample: bool = False,
    ) -> float:
        # E[TVD] <= 0.5*sqrt((K-1)/S) (multinomial) + McDiarmid tail margin,
        # delta split Bonferroni-style over the sweep. two_sample doubles it.
        K, S = num_outcomes, shots
        base = math.sqrt((K - 1) / S)
        expected = base if two_sample else 0.5 * base
        delta = delta_total / max(num_comparisons, 1)
        margin_factor = math.sqrt(2.0) if two_sample else 1.0
        margin = margin_factor * math.sqrt(math.log(1.0 / delta) / (2 * S))
        return expected + margin

    @staticmethod
    def outcomes_to_scalars(results: np.ndarray, d: int) -> np.ndarray:
        """Convert a (shots, m) measurement array to base-d scalars."""
        m = results.shape[1]
        powers = np.array([d ** (m - 1 - i) for i in range(m)], dtype=np.int64)
        return results.astype(np.int64) @ powers

    @staticmethod
    def empirical_tvd(scalars: np.ndarray, ref: np.ndarray, K: int) -> float:
        """TVD between sampled scalars and a reference probability vector."""
        counts = np.bincount(scalars, minlength=K).astype(float)
        emp = counts / counts.sum()
        return 0.5 * float(np.abs(emp - ref).sum())

    @staticmethod
    def _random_circuit(
        n: int,
        depth: int,
        d: int,
        seed: int,
        measurement_rounds: int,
    ) -> Circuit:
        # Private RNG: a given seed yields the same unitary for any
        # measurement_rounds, so statevector and sampling circuits match.
        rng = random.Random(seed)
        coprimes = [a for a in range(2, d) if math.gcd(a, d) == 1]
        one_q = ["H", "P", "X", "Z", "H_INV", "P_INV", "X_INV", "Z_INV"]
        two_q = ["CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"]
        c = Circuit(n, d)
        for _ in range(depth):
            roll = rng.random()
            if n > 1 and roll < 0.4:
                g = rng.choice(two_q)
                a, b = rng.sample(range(n), 2)
                c.append(g, a, b)
            elif coprimes and roll < 0.55:
                c.append("MULTIPLY", rng.randrange(n), args=rng.choice(coprimes))
            else:
                c.append(rng.choice(one_q), rng.randrange(n))
        for _ in range(measurement_rounds):
            for q in range(n):
                c.append("M", q)
        return c

    @pytest.mark.parametrize("d", DIMENSIONS)
    def test_random_clifford_tvd(self, d):
        """Terminal-measurement distribution vs exact statevector."""
        from sdim import cirq_statevector_from_circuit

        ns = [n for n in (2, 3) if d**n <= self.MAX_HILBERT]
        num_comparisons = len(DIMENSIONS) * len(ns) * len(self.DEPTHS) * (
            self.SEEDS_PER_CONFIG
        )

        for n in ns:
            K = d**n
            shots = max(self.MIN_SHOTS, self.SHOTS_PER_OUTCOME * K)
            threshold = self.tvd_threshold(K, shots, num_comparisons)
            for depth in self.DEPTHS:
                for s in range(self.SEEDS_PER_CONFIG):
                    seed = d * 100_003 + n * 911 + depth * 13 + s
                    base = self._random_circuit(n, depth, d, seed, 0)
                    sv = cirq_statevector_from_circuit(base)
                    ref = np.abs(sv) ** 2

                    meas = self._random_circuit(n, depth, d, seed, 1)
                    results = meas.compile_sampler().sample(shots)
                    scalars = self.outcomes_to_scalars(results, d)
                    dist = self.empirical_tvd(scalars, ref, K)

                    assert dist < threshold, (
                        f"TVD={dist:.4f} >= {threshold:.4f} for d={d}, n={n}, "
                        f"depth={depth}, seed={seed}, shots={shots}"
                    )
