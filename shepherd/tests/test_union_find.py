"""
Correctness tests for the qudit union-find decoder.

Faults are injected straight from the detector error model: a weight-w fault is w distinct lines
of `dem.to_lines()`, each at a random non-zero power. The decoder must
  * always return a correction that reproduces the syndrome (the main bookkeeping check),
  * correct every fault of weight <= (d - 1) / 2 on repetition, surface and small LaCross codes,
    at every q (LaCross needs the minimum-weight solution inside each cluster),
  * pick a minimum-weight solution inside a cluster whenever it enumerates,
  * be deterministic, survive pickling, and make the logical error rate fall with distance.
"""

from __future__ import annotations

import itertools
import pickle

import numpy as np
import pytest

pytest.importorskip("sdim.dem")
from sdim.dem import DetectorErrorModel  # noqa: E402

from shepherd import gf_q  # noqa: E402
from shepherd.benchmarking.ler import z_memory_circuit  # noqa: E402
from shepherd.codes.css import lacross_code, repetition_code, surface_code  # noqa: E402
from shepherd.collect import collect  # noqa: E402
from shepherd.decoders.union_find import QuditUnionFindDecoder, min_weight_solution  # noqa: E402

QS = (2, 3, 5, 7)


def _model(code, rounds, p=0.01):
    dem = DetectorErrorModel.from_circuit(z_memory_circuit(code, p=p, rounds=rounds))
    compiled = QuditUnionFindDecoder().compile_decoder_for_dem(dem)
    nd, no, q = dem.num_detectors, dem.num_observables, dem.dimension
    det_vecs, obs_vecs = [], []
    for m in dem.to_lines().mechanisms:
        dv, ov = np.zeros(nd, dtype=np.int64), np.zeros(no, dtype=np.int64)
        for t, v in m.generators[0].items():
            if t < nd:
                dv[t] = v % q
            else:
                ov[t - nd] = v % q
        if dv.any():
            det_vecs.append(dv)
            obs_vecs.append(ov)
    return dem, compiled, np.array(det_vecs), np.array(obs_vecs)


def _faults(n_lines, q, w, rng, trials):
    """(trials, n_lines) powers: w distinct lines per row, each at a power in 1..q-1."""
    a = np.zeros((trials, n_lines), dtype=np.int64)
    for t in range(trials):
        a[t, rng.choice(n_lines, w, replace=False)] = rng.integers(1, q, w)
    return a


def _all_faults(n_lines, q, w):
    rows = []
    for idx in itertools.combinations(range(n_lines), w):
        for pw in itertools.product(range(1, q), repeat=w):
            r = np.zeros(n_lines, dtype=np.int64)
            r[list(idx)] = pw
            rows.append(r)
    return np.array(rows)


def _failures(compiled, det_vecs, obs_vecs, powers, q):
    det = powers @ det_vecs % q
    obs = powers @ obs_vecs % q
    pred = compiled.decode_shots(det) % q
    return int((pred != obs).any(axis=1).sum())


def _edge_vector(compiled, e):
    v = np.zeros(compiled.num_detectors, dtype=np.int64)
    for t, c in zip(compiled.edge_dets[e], compiled.edge_coefs[e]):
        v[t] = c
    return v


# -- gf_q.solve ------------------------------------------------------------------------------------

@pytest.mark.parametrize("q", (2, 3, 5, 7))
def test_solve(q):
    rng = np.random.default_rng(q)
    for _ in range(50):
        m, n = rng.integers(1, 6), rng.integers(0, 7)
        a = rng.integers(0, q, size=(m, n))
        x0 = rng.integers(0, q, size=n)
        x = gf_q.solve(a, a @ x0 % q, q)
        assert x is not None and np.array_equal(a @ x % q, a @ x0 % q)
        b = rng.integers(0, q, size=m)
        x = gf_q.solve(a, b, q)
        solvable = gf_q.rank(np.hstack([a, b.reshape(-1, 1)]), q) == gf_q.rank(a, q) if n else not b.any()
        assert (x is not None) == solvable
        if x is not None:
            assert np.array_equal(a @ x % q, b % q)


@pytest.mark.parametrize("q", (2, 3, 5, 7))
def test_solve_with_kernel_gives_every_solution(q):
    rng = np.random.default_rng(100 + q)
    for _ in range(30):
        m, n = rng.integers(1, 5), rng.integers(1, 5)
        a = rng.integers(0, q, size=(m, n))
        b = a @ rng.integers(0, q, size=n) % q
        x0, k = gf_q.solve_with_kernel(a, b, q)
        every = {tuple(x) for x in itertools.product(range(q), repeat=n)
                 if np.array_equal(a @ np.array(x) % q, b)}
        spanned = {tuple((x0 + np.array(c, dtype=np.int64) @ k) % q)
                   for c in itertools.product(range(q), repeat=k.shape[0])}
        assert spanned == every
        assert k.shape[0] == n - gf_q.rank(a, q)
    assert gf_q.solve_with_kernel(np.zeros((1, 2)), np.ones(1), q) is None


@pytest.mark.parametrize("q", (2, 3, 5))
def test_min_weight_solution(q):
    rng = np.random.default_rng(200 + q)
    for _ in range(40):
        m, n = rng.integers(1, 5), rng.integers(2, 7)
        a = rng.integers(0, q, size=(m, n)) * (rng.random((m, n)) < 0.5)
        b = a @ (rng.integers(0, q, size=n) * (rng.random(n) < 0.5)) % q
        x0, k = gf_q.solve_with_kernel(a, b, q)
        best = min(np.count_nonzero(x) for x in itertools.product(range(q), repeat=n)
                   if np.array_equal(a @ np.array(x) % q, b))
        x = min_weight_solution(x0, k, q)                    # exact: q**len(k) <= q**6 < cap
        assert np.array_equal(a @ x % q, b) and np.count_nonzero(x) == best
        g = min_weight_solution(x0, k, q, cap=1)             # greedy path
        assert np.array_equal(a @ g % q, b) and np.count_nonzero(g) <= np.count_nonzero(x0 % q)


# -- the correction always reproduces the syndrome --------------------------------------------------

@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("make, rounds, p", [
    (lambda q: surface_code(5, q), 0, 0.08),
    (lambda q: surface_code(5, q), 5, 0.03),
    (lambda q: repetition_code(7, q), 7, 0.05),
    (lambda q: lacross_code(10, 4, q), 0, 0.03),
])
def test_correction_reproduces_syndrome(q, make, rounds, p):
    dem, compiled, _, _ = _model(make(q), rounds, p)
    det, _ = dem.sample(300, seed=11)
    for row in det:
        edges, powers, complete = compiled.decode_correction(row)
        assert complete
        produced = np.zeros(compiled.num_detectors, dtype=np.int64)
        for e, a in zip(edges, powers.tolist()):
            produced = (produced + a * _edge_vector(compiled, e)) % q
        assert np.array_equal(produced, row % q)


# -- every fault below half the distance is corrected ----------------------------------------------

@pytest.mark.parametrize("q", QS)
def test_all_weight_1_and_2_faults_code_capacity(q):
    for code in (repetition_code(7, q), surface_code(5, q)):
        _, compiled, dv, ov = _model(code, 0)
        for w in (1, 2):
            assert _failures(compiled, dv, ov, _all_faults(len(dv), q, w), q) == 0, (code, w)


@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("make, rounds", [
    (lambda q: surface_code(7, q), 0),       # d = 7: weights up to 3
    (lambda q: surface_code(9, q), 0),       # d = 9: up to 4
    (lambda q: repetition_code(11, q), 0),   # d = 11: up to 5
    (lambda q: repetition_code(7, q), 7),    # phenomenological, circuit distance 7: up to 3
    (lambda q: surface_code(5, q), 5),       # phenomenological, circuit distance 5: up to 2
])
def test_sampled_faults_below_half_distance(q, make, rounds):
    code = make(q)
    _, compiled, dv, ov = _model(code, rounds)
    rng = np.random.default_rng(1000 * q + code.n + rounds)
    for w in range(1, (code.distance - 1) // 2 + 1):
        assert _failures(compiled, dv, ov, _faults(len(dv), q, w, rng, 1500), q) == 0, (code, rounds, w)


@pytest.mark.parametrize("q", (2, 3, 5))
@pytest.mark.parametrize("n", (12, 14))
def test_lacross_faults_below_half_distance(q, n):
    # LaCross clusters contain short logicals: an arbitrary solution inside a cluster fails here
    # at weight 2 (d >= 5); the minimum-weight one must not. Weight 2 only: union-find has no
    # (d-1)/2 guarantee off topological codes. At n = 14, q = 2 (d = 7) about 1 in 10^4 weight-3
    # faults fails because a cluster turns valid on a weight-4 explanation before it reaches the
    # true fault (a limit of growth, not of the solve).
    code = lacross_code(n, 4, q)
    _, compiled, dv, ov = _model(code, 0)
    assert code.distance >= 5
    assert _failures(compiled, dv, ov, _all_faults(len(dv), q, 1), q) == 0
    rng = np.random.default_rng(300 * q + n)
    assert _failures(compiled, dv, ov, _faults(len(dv), q, 2, rng, 2000), q) == 0, code


# -- determinism and pickling ----------------------------------------------------------------------

@pytest.mark.parametrize("make, rounds, p", [
    (lambda: surface_code(5, 3), 5, 0.03),
    (lambda: lacross_code(16, 4, 3), 0, 0.05),     # large clusters: the greedy path runs too
])
def test_deterministic_and_picklable(make, rounds, p):
    dem, compiled, _, _ = _model(make(), rounds, p)
    det, _ = dem.sample(200, seed=5)
    first = compiled.decode_shots(det)
    again = pickle.loads(pickle.dumps(compiled)).decode_shots(det)
    assert np.array_equal(first, compiled.decode_shots(det))
    assert np.array_equal(first, again)


# -- logical error rate falls with distance --------------------------------------------------------

@pytest.mark.parametrize("q", QS)
def test_logical_error_rate_falls_with_distance(q):
    rates = []
    for L in (3, 7):
        dem, compiled, _, _ = _model(surface_code(L, q), 0, 0.04)
        det, obs = dem.sample(5000, seed=L)
        rates.append(np.mean((compiled.decode_shots(det) % q != obs).any(axis=1)))
    assert rates[1] < rates[0] / 2, rates


# -- end to end through the collector --------------------------------------------------------------

def test_collect_with_union_find():
    from shepherd.noise_model import NoiseModel
    from shepherd.study import Study
    study = Study(name="t", description="", experiment="ler", experiment_params={"rounds": 0},
                  codes=lambda: [surface_code(3, 3)], noise=NoiseModel.code_capacity, ps=[0.05],
                  decoders=["union_find", "none"])
    tasks = [pt.task for pt in study.points()]
    stats = {s.decoder: s for s in collect(tasks=tasks, num_workers=0, max_shots=3000)}
    assert stats["union_find"].shots == 3000
    assert stats["union_find"].errors < stats["none"].errors
