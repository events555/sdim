"""CSS codes over F_q, the hypergraph product, and Z-memory circuits built from them."""

from __future__ import annotations

import numpy as np
import pytest

from shepherd import gf_q
from shepherd.codes.css import (CSSCode, classical_distance, hgp, lacross_checks, lacross_code,
                                repetition_checks, repetition_code, surface_code)
from shepherd.qecc import PauliString

PRIMES = (2, 3, 5, 7)


# -- gf_q ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("q", PRIMES)
def test_nullspace_and_rank(q):
    rng = np.random.default_rng(q)
    for _ in range(20):
        a = rng.integers(0, q, size=(rng.integers(1, 7), rng.integers(1, 9)))
        ns = gf_q.nullspace(a, q)
        assert ns.shape == (a.shape[1] - gf_q.rank(a, q), a.shape[1])
        assert not np.any(a @ ns.T % q)
        assert gf_q.rank(ns, q) == ns.shape[0]


@pytest.mark.parametrize("q", PRIMES)
def test_inverse(q):
    rng = np.random.default_rng(10 + q)
    done = 0
    while done < 10:
        a = rng.integers(0, q, size=(4, 4))
        if gf_q.rank(a, q) < 4:
            with pytest.raises(ValueError):
                gf_q.inverse(a, q)
            continue
        assert np.array_equal(a @ gf_q.inverse(a, q) % q, np.eye(4, dtype=np.int64))
        done += 1


def test_quotient_basis():
    q = 5
    w = np.array([[1, 1, 0, 0]])
    v = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2]])
    b = gf_q.quotient_basis(v, w, q)
    assert b.shape[0] == 2       # span(v) has dimension 3, span(w) 1
    assert gf_q.rank(np.vstack([w, b]), q) == 3


# -- codes -----------------------------------------------------------------------------------------

def _check_code(code: CSSCode):
    q = code.dimension
    lx, lz = code.logical_matrices()
    assert lx.shape == lz.shape == (code.k, code.n)
    assert not np.any(code.H_X @ code.H_Z.T % q)
    assert not np.any(code.H_X @ lz.T % q)              # Z-logicals commute with X checks
    assert not np.any(code.H_Z @ lx.T % q)              # X-logicals commute with Z checks
    assert np.array_equal(lx @ lz.T % q, np.eye(code.k, dtype=np.int64))
    for row in lz:                                      # not stabilizers
        assert not gf_q.row_space_contains(code.H_Z, row, q)
    # The PauliString view agrees: X-bar_i and Z-bar_j commute unless i == j.
    xs, zs = code.logical_operators()
    for i, x in enumerate(xs):
        for j, z in enumerate(zs):
            assert PauliString.symplectic_inner_product(x, z) == (1 if i == j else 0)


@pytest.mark.parametrize("q", PRIMES)
@pytest.mark.parametrize("L", (3, 5))
def test_repetition_and_surface(q, L):
    rep = repetition_code(L, q)
    assert (rep.n, rep.k, rep.distance) == (L, 1, L)
    _check_code(rep)
    s = surface_code(L, q)
    assert (s.n, s.k, s.distance) == (L * L + (L - 1) ** 2, 1, L)
    _check_code(s)
    h = repetition_checks(L, q)
    assert hgp(h, h, q).distance == L                   # Tillich-Zemor formula agrees


@pytest.mark.parametrize("q", (2, 3, 5))
def test_lacross_open_matches_published_qubit_parameters(q):
    # [[400, 16, 8]] is the qubit LaCross code from 1 + x + x^4 on 16 bits, open boundary; with
    # an open boundary the classical code has dimension ell over every field.
    code = lacross_code(16, 4, q)
    assert (code.n, code.k) == (400, 16)
    if q == 2:
        assert code.distance == 8
    _check_code(code)


def test_lacross_periodic_dimension_depends_on_field():
    # Periodic: the classical dimension is deg gcd(1 + x + x^2, x^7 - 1) over F_q. Over F_3,
    # 1 + x + x^2 = (x - 1)^2 shares the factor x - 1; over F_2 and F_5 nothing is shared.
    h = lacross_checks(7, 2, periodic=True)
    ks = {q: h.shape[1] - gf_q.rank(h, q) for q in (2, 3, 5)}
    assert ks == {2: 0, 3: 1, 5: 0}


def test_classical_distance():
    assert classical_distance(repetition_checks(5, 3), 3) == 5
    assert classical_distance(np.eye(3, dtype=np.int64), 3) == float("inf")


# -- Z-memory circuits (need sdim's DEM) -----------------------------------------------------------

sdim_dem = pytest.importorskip("sdim.dem")


@pytest.mark.parametrize("q", (2, 3, 5))
@pytest.mark.parametrize("rounds", (0, 1, 3))
def test_memory_dem_shape_and_single_faults(q, rounds):
    from shepherd.benchmarking.ler import z_memory_circuit
    code = surface_code(3, q)
    circuit = z_memory_circuit(code, p=0.01, rounds=rounds)
    dem = sdim_dem.DetectorErrorModel.from_circuit(circuit)   # checks determinism
    m = code.H_Z.shape[0]
    assert dem.num_detectors == m * (rounds + 1)
    assert dem.num_observables == code.k
    # Code capacity: each fault is a shift on one data qudit, whose detector vector is a column
    # of H_Z (up to the line's scale).
    if rounds == 0:
        cols = {tuple(code.H_Z[:, j] % q) for j in range(code.n)}
        for mech in dem.to_lines().mechanisms:
            v = np.zeros(m, dtype=np.int64)
            for t, c in mech.generators[0].items():
                if t < m:
                    v[t] = c
            scaled = {tuple(a * v % q) for a in range(1, q)}
            assert scaled & cols


def test_noiseless_detectors_are_zero():
    from shepherd.benchmarking.ler import z_memory_circuit
    circuit = z_memory_circuit(surface_code(3, 3), p=0.0, rounds=2)
    dem = sdim_dem.DetectorErrorModel.from_circuit(circuit)
    det, obs = dem.sample(100, seed=1)
    assert not det.any() and not obs.any()
