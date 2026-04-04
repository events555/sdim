import numpy as np
import pytest

from sdim import TableauSimulator


def stab_X(tab):
    return tab.X[: tab.l] % tab.d


def stab_Z(tab):
    return tab.Z[: tab.l] % tab.d


def stab_tau(tab):
    return tab.tau_exp[: tab.l] % (2 * tab.d)


def check_symplectic(tab):
    d = tab.d
    Z = tab.Z[: tab.l] % d
    X = tab.X[: tab.l] % d
    SS = (Z @ X.T - X @ Z.T) % d
    assert np.all(SS == 0), f"stabilizer commutation violated:\n{SS}"


DIMENSIONS = [2, 3, 4, 5, 6]


@pytest.mark.parametrize("d", DIMENSIONS)
def test_random_circuit_preserves_symplectic(d):
    rng = np.random.default_rng(42)
    n = 3
    t = TableauSimulator(n, d)
    gates_1q = ["hadamard", "phase_gate", "pauli_x", "pauli_z"]
    gates_2q = ["cnot", "cz", "swap"]
    for _ in range(50):
        if rng.random() < 0.6:
            gate = rng.choice(gates_1q)
            q = int(rng.integers(0, n))
            getattr(t, gate)(q)
        else:
            gate = rng.choice(gates_2q)
            q1, q2 = rng.choice(n, size=2, replace=False)
            getattr(t, gate)(int(q1), int(q2))
    check_symplectic(t)


@pytest.mark.parametrize("d", DIMENSIONS)
def test_gate_roundtrips(d):
    n = 2
    ref_Z = stab_Z(TableauSimulator(n, d)).copy()
    ref_X = stab_X(TableauSimulator(n, d)).copy()
    ref_tau = stab_tau(TableauSimulator(n, d)).copy()

    for gate, args in [
        ("hadamard", (0,)),
        ("phase_gate", (0,)),
        ("pauli_x", (0,)),
        ("pauli_z", (0,)),
        ("cnot", (0, 1)),
        ("cz", (0, 1)),
        ("swap", (0, 1)),
    ]:
        t = TableauSimulator(n, d)
        getattr(t, gate)(*args)
        if gate != "swap":
            getattr(t, gate)(*args, dagger=True)
        else:
            getattr(t, gate)(*args)
        t.modulo()
        assert np.array_equal(stab_Z(t), ref_Z), f"{gate} roundtrip Z (d={d})"
        assert np.array_equal(stab_X(t), ref_X), f"{gate} roundtrip X (d={d})"
        assert np.array_equal(stab_tau(t), ref_tau), (
            f"{gate} roundtrip tau (d={d})"
        )


def test_hadamard_values():
    t = TableauSimulator(1, 2)
    t.hadamard(0)
    t.modulo()
    assert np.array_equal(stab_X(t), [[1]])
    assert np.array_equal(stab_Z(t), [[0]])

    t2 = TableauSimulator(2, 2)
    t2.hadamard(0)
    t2.modulo()
    assert np.array_equal(stab_X(t2), [[1, 0], [0, 0]])
    assert np.array_equal(stab_Z(t2), [[0, 0], [0, 1]])


def test_cnot_values():
    t1 = TableauSimulator(2, 2)
    t1.pauli_x(0)
    t1.cnot(0, 1)
    t1.modulo()
    assert np.array_equal(stab_X(t1), [[0, 0], [0, 0]])
    assert np.array_equal(stab_Z(t1), [[1, 0], [1, 1]])

    t2 = TableauSimulator(2, 3)
    t2.pauli_x(0)
    t2.pauli_x(0)
    t2.cnot(0, 1)
    t2.modulo()
    assert np.array_equal(stab_X(t2), [[0, 0], [0, 0]])
    assert np.array_equal(stab_Z(t2), [[1, 0], [2, 1]])


def test_cz_values():
    t1 = TableauSimulator(2, 2)
    t1.hadamard(0)
    t1.pauli_x(1)
    t1.cz(0, 1)
    t1.hadamard(0)
    t1.modulo()
    assert np.array_equal(stab_X(t1), [[0, 0], [0, 0]])
    assert np.array_equal(stab_Z(t1), [[1, 1], [0, 1]])


def test_measure_z_basis_deterministic():
    for d in [2, 3, 4, 5, 6]:
        t = TableauSimulator(1, d)
        assert t.measure(0) == 0


def test_measure_z_after_x():
    t = TableauSimulator(1, 2)
    t.pauli_x(0)
    assert t.measure(0) == 1


def test_measure_random_qubit():
    counts = {0: 0, 1: 0}
    for _ in range(200):
        t = TableauSimulator(1, 2)
        t.hadamard(0)
        counts[t.measure(0)] += 1
    assert counts[0] > 20 and counts[1] > 20


@pytest.mark.parametrize("d", [2, 3, 5])
def test_measure_idempotent(d):
    t = TableauSimulator(1, d)
    t.hadamard(0)
    r1 = t.measure(0)
    r2 = t.measure(0)
    assert r1 == r2


def test_measure_bell_pair():
    for _ in range(50):
        t = TableauSimulator(2, 2)
        t.hadamard(0)
        t.cnot(0, 1)
        r0 = t.measure(0)
        r1 = t.measure(1)
        assert r0 == r1


@pytest.mark.parametrize("d", [4, 6])
def test_measure_composite_deterministic(d):
    t = TableauSimulator(2, d)
    assert t.measure(0) == 0
    assert t.measure(1) == 0


@pytest.mark.parametrize("d", [4, 6])
def test_measure_composite_random(d):
    from collections import Counter

    counts = Counter()
    for _ in range(500):
        t = TableauSimulator(1, d)
        t.hadamard(0)
        counts[t.measure(0)] += 1
    assert len(counts) == d
    for v in counts.values():
        assert v > 10
