"""Integration tests comparing Rust backend (_sdim_rs) with Python TableauSimulator."""

import numpy as np
import pytest

from sdim._sdim_rs import run_ir, snapshot
from sdim.simulators.tableau_simulator import TableauSimulator

DIMENSIONS = [2, 3, 4, 5, 6]
INT64_MAX = np.iinfo(np.int64).max

# Gate IDs matching quality branch registry
GATE_I = 0
GATE_X = 1
GATE_X_INV = 2
GATE_Z = 3
GATE_Z_INV = 4
GATE_H = 5
GATE_H_INV = 6
GATE_P = 7
GATE_P_INV = 8
GATE_MULTIPLY = 9
GATE_MULTIPLY_INV = 10
GATE_CNOT = 11
GATE_CNOT_INV = 12
GATE_CZ = 13
GATE_CZ_INV = 14
GATE_SWAP = 15
GATE_M = 16
GATE_MR = 17


def make_ir(*instructions):
    """Build IR array from list of (gate_id, qudit, target, arg0) tuples."""
    return np.array(instructions, dtype=np.int64)


def py_snapshot(instructions, n, d, stop_after):
    """Run Python TableauSimulator up to stop_after instructions, return (X, Z, tau)."""
    from sdim.gates.registry import gate_id_to_name, is_gate_collapsing, is_gate_noisy

    tab = TableauSimulator(n, d)
    count = 0
    for i, inst in enumerate(instructions):
        if i >= stop_after:
            break
        gid, qi, ti, a0 = int(inst[0]), int(inst[1]), int(inst[2]), int(inst[3])
        if is_gate_noisy(gid):
            count += 1
            continue
        if is_gate_collapsing(gid):
            name = gate_id_to_name(gid)
            if name in ("M_X", "MR_X"):
                tab.hadamard(qi, dagger=True)
            tab.measure(qi)
        else:
            if qi >= 0 and ti >= 0 and ti < INT64_MAX:
                tab.apply_gate(gid, qi, ti, a0 if a0 >= 0 else None)
            elif qi >= 0:
                tab.apply_gate(gid, qi, 0, a0 if a0 >= 0 else None)
        count += 1

    tab.modulo()
    l = tab.l
    return tab.X[:l] % d, tab.Z[:l] % d, tab.tau_exp[:l] % (2 * d)


# ---- Gate roundtrip tests via snapshot ----

@pytest.mark.parametrize("d", DIMENSIONS)
def test_gate_roundtrip_hadamard(d):
    """H followed by H_INV should be identity."""
    ir = make_ir(
        [GATE_H, 0, INT64_MAX, -1],
        [GATE_H_INV, 0, INT64_MAX, -1],
    )
    x, z, tau = snapshot(ir, 2, d, len(ir))
    ref = TableauSimulator(2, d)
    ref.modulo()
    np.testing.assert_array_equal(x % d, ref.X[:ref.l] % d)
    np.testing.assert_array_equal(z % d, ref.Z[:ref.l] % d)


@pytest.mark.parametrize("d", DIMENSIONS)
def test_gate_roundtrip_phase(d):
    """P followed by P_INV should be identity."""
    ir = make_ir(
        [GATE_P, 0, INT64_MAX, -1],
        [GATE_P_INV, 0, INT64_MAX, -1],
    )
    x, z, tau = snapshot(ir, 2, d, len(ir))
    ref = TableauSimulator(2, d)
    ref.modulo()
    np.testing.assert_array_equal(x % d, ref.X[:ref.l] % d)
    np.testing.assert_array_equal(z % d, ref.Z[:ref.l] % d)


@pytest.mark.parametrize("d", DIMENSIONS)
def test_gate_roundtrip_cnot(d):
    """CNOT followed by CNOT_INV should be identity."""
    ir = make_ir(
        [GATE_CNOT, 0, 1, -1],
        [GATE_CNOT_INV, 0, 1, -1],
    )
    x, z, tau = snapshot(ir, 2, d, len(ir))
    ref = TableauSimulator(2, d)
    ref.modulo()
    np.testing.assert_array_equal(x % d, ref.X[:ref.l] % d)
    np.testing.assert_array_equal(z % d, ref.Z[:ref.l] % d)


@pytest.mark.parametrize("d", DIMENSIONS)
def test_gate_roundtrip_cz(d):
    """CZ followed by CZ_INV should be identity."""
    ir = make_ir(
        [GATE_CZ, 0, 1, -1],
        [GATE_CZ_INV, 0, 1, -1],
    )
    x, z, tau = snapshot(ir, 2, d, len(ir))
    ref = TableauSimulator(2, d)
    ref.modulo()
    np.testing.assert_array_equal(x % d, ref.X[:ref.l] % d)
    np.testing.assert_array_equal(z % d, ref.Z[:ref.l] % d)


@pytest.mark.parametrize("d", DIMENSIONS)
def test_gate_roundtrip_swap(d):
    """SWAP twice should be identity."""
    ir = make_ir(
        [GATE_SWAP, 0, 1, -1],
        [GATE_SWAP, 0, 1, -1],
    )
    x, z, tau = snapshot(ir, 2, d, len(ir))
    ref = TableauSimulator(2, d)
    ref.modulo()
    np.testing.assert_array_equal(x % d, ref.X[:ref.l] % d)
    np.testing.assert_array_equal(z % d, ref.Z[:ref.l] % d)


# ---- Snapshot comparison: Rust vs Python for gate sequences ----

@pytest.mark.parametrize("d", DIMENSIONS)
def test_snapshot_matches_python_clifford_sequence(d):
    """A sequence of Clifford gates should produce identical tableaux."""
    ir = make_ir(
        [GATE_H, 0, INT64_MAX, -1],
        [GATE_CNOT, 0, 1, -1],
        [GATE_P, 1, INT64_MAX, -1],
        [GATE_H, 1, INT64_MAX, -1],
        [GATE_CZ, 0, 1, -1],
    )
    rust_x, rust_z, rust_tau = snapshot(ir, 2, d, len(ir))
    py_x, py_z, py_tau = py_snapshot(ir, 2, d, len(ir))

    np.testing.assert_array_equal(rust_x % d, py_x % d,
                                  err_msg=f"X mismatch for d={d}")
    np.testing.assert_array_equal(rust_z % d, py_z % d,
                                  err_msg=f"Z mismatch for d={d}")
    np.testing.assert_array_equal(rust_tau % (2*d), py_tau % (2*d),
                                  err_msg=f"tau mismatch for d={d}")


# ---- Measurement tests via run_ir ----

@pytest.mark.parametrize("d", DIMENSIONS)
def test_run_ir_deterministic_zero(d):
    """Measuring |0> should always give 0."""
    ir = make_ir([GATE_M, 0, INT64_MAX, -1])
    result = run_ir(ir, 1, d)
    assert result[0] == 0


def test_run_ir_x_then_measure():
    """X|0> = |1>, measuring should give 1."""
    ir = make_ir(
        [GATE_X, 0, INT64_MAX, -1],
        [GATE_M, 0, INT64_MAX, -1],
    )
    result = run_ir(ir, 1, 2)
    assert result[0] == 1


def test_run_ir_bell_state():
    """Bell pair: both qubits should give the same outcome."""
    for _ in range(50):
        ir = make_ir(
            [GATE_H, 0, INT64_MAX, -1],
            [GATE_CNOT, 0, 1, -1],
            [GATE_M, 0, INT64_MAX, -1],
            [GATE_M, 1, INT64_MAX, -1],
        )
        result = run_ir(ir, 2, 2)
        assert result[0] == result[1], f"Bell pair mismatch: {result}"


def test_run_ir_measure_idempotent():
    """Measuring twice should give the same result."""
    for d in [2, 3, 5]:
        ir = make_ir(
            [GATE_H, 0, INT64_MAX, -1],
            [GATE_M, 0, INT64_MAX, -1],
            [GATE_M, 0, INT64_MAX, -1],
        )
        result = run_ir(ir, 1, d)
        assert result[0] == result[1], f"Idempotent fail d={d}: {result}"


def test_run_ir_measure_reset():
    """MR should record measurement then reset to |0>."""
    ir = make_ir(
        [GATE_X, 0, INT64_MAX, -1],
        [GATE_MR, 0, INT64_MAX, -1],
        [GATE_M, 0, INT64_MAX, -1],
    )
    result = run_ir(ir, 1, 2)
    assert result[0] == 1  # MR records 1
    assert result[1] == 0  # After reset, measure 0


@pytest.mark.parametrize("d", [4, 6])
def test_run_ir_composite_random(d):
    """H|0> in composite dimension should give uniform outcomes."""
    from collections import Counter
    counts = Counter()
    for _ in range(500):
        ir = make_ir(
            [GATE_H, 0, INT64_MAX, -1],
            [GATE_M, 0, INT64_MAX, -1],
        )
        counts[int(run_ir(ir, 1, d)[0])] += 1
    assert len(counts) == d, f"Expected {d} outcomes, got {len(counts)}"
    for v in counts.values():
        assert v > 10, f"Outcome too rare: {counts}"
