"""Tests for the array-based fast paths in sdim.dem.

`DetectorErrorModel.from_circuit`, `to_lines` and `sample` run on flat arrays
and numba kernels. These tests check them against straightforward reference
implementations built from the public, dict-based API, and check the sampler
against exact distributions.
"""

import functools
import itertools
import math
import random
import struct
import time

import numpy as np
import pytest

import sdim.dem as dem_module
from sdim.circuit import Circuit
from sdim.dem import (
    DetectorErrorModel,
    ErrorMechanism,
    compile_unit_responses,
    line_probability,
    merge_subgroup_probabilities,
    _detector_coefficients,
    _merge_pair,
    _projective_points,
)
from sdim.program import Program, _compile_detector
from tests.test_dem_gates import random_circuit


# --------------------------------------------------------------------------
# Reference implementations (one dict at a time)


def _reference_from_circuit(circuit, merge=True):
    """from_circuit as a loop over compile_unit_responses, then merge_lines."""
    compiled = compile_unit_responses(circuit)
    dem = DetectorErrorModel(circuit.dimension, compiled.num_detectors, compiled.num_observables,
                             detector_labels=compiled.detector_labels,
                             observable_labels=compiled.observable_labels)
    for loc in compiled.locations:
        generators = [g for g in loc.responses if g]
        if generators and not loc.probability <= 0.0:
            dem.mechanisms.append(ErrorMechanism(loc.subgroup_probability, generators, loc.source))
    if merge:
        _reference_merge_lines(dem)
    return dem


def _reference_merge_lines(dem):
    d = dem.dimension
    merged, parts, order, others = {}, {}, [], []
    for mech in dem.mechanisms:
        if mech.rank != 1:
            others.append(mech)
            continue
        items = sorted((t, v % d) for t, v in mech.generators[0].items() if v % d)
        if not items or math.gcd(items[0][1], d) != 1:
            # No unit leading coefficient (composite d, or a zero vector): never merged.
            key = object()
            merged[key] = ErrorMechanism(mech.probability, [dict(items)], mech.source)
            parts[key] = [mech.source]
            order.append(key)
            continue
        inv = pow(items[0][1], -1, d)
        scaled = {t: (v * inv) % d for t, v in items}
        key = tuple(sorted(scaled.items()))
        if key in merged:
            prev = merged[key]
            prev.probability = merge_subgroup_probabilities(prev.probability, mech.probability)
            parts[key].append(mech.source)
        else:
            merged[key] = ErrorMechanism(mech.probability, [scaled], mech.source)
            parts[key] = [mech.source]
            order.append(key)
    for key in order:
        merged[key].source = "+".join(parts[key])
    dem.mechanisms = [merged[k] for k in order] + others


def _reference_to_lines(dem):
    d = dem.dimension
    out = DetectorErrorModel(d, dem.num_detectors, dem.num_observables, [],
                             list(dem.detector_labels), list(dem.observable_labels))
    for mech in dem.mechanisms:
        k = mech.rank
        pl = line_probability(mech.probability, d, k)
        for direction in _projective_points(d, k):
            combined = {}
            for coeff, gen in zip(direction, mech.generators):
                if coeff:
                    for t, v in gen.items():
                        combined[t] = (combined.get(t, 0) + coeff * v) % d
            combined = {t: v for t, v in combined.items() if v}
            if combined:
                out.mechanisms.append(ErrorMechanism(pl, [combined], mech.source))
    _reference_merge_lines(out)
    return out


def _bits(x):
    return struct.pack("<d", float(x))


def _assert_same_model(a, b):
    """Same mechanisms in the same order: probabilities bit for bit, generators with dict order."""
    assert (a.dimension, a.num_detectors, a.num_observables) == (b.dimension, b.num_detectors, b.num_observables)
    assert list(a.detector_labels) == list(b.detector_labels)
    assert list(a.observable_labels) == list(b.observable_labels)
    assert len(a.mechanisms) == len(b.mechanisms)
    for m, n in zip(a.mechanisms, b.mechanisms):
        assert _bits(m.probability) == _bits(n.probability), (m, n)
        assert [list(g.items()) for g in m.generators] == [list(g.items()) for g in n.generators], (m, n)
        assert m.source == n.source
    assert str(a) == str(b)


# --------------------------------------------------------------------------
# Model building


@pytest.mark.parametrize("d", [2, 3, 5, 7, 1000003])
@pytest.mark.parametrize("seed", [3, 17, 29])
def test_from_circuit_matches_reference(d, seed):
    p = {3: 0.01, 17: 0.3, 29: 1.0 - 1.0 / d}[seed]
    c = random_circuit(d, seed, p=p)
    for merge in (True, False):
        _assert_same_model(DetectorErrorModel.from_circuit(c, merge=merge), _reference_from_circuit(c, merge))


def test_from_circuit_merges_many_lines():
    """Many noise gates on the same line of detector space merge in circuit order."""
    d = 5
    c = Circuit(2, d)
    c.add_gate("RESET", [0, 1])
    for k in range(40):
        c.add_gate("N1", 0, noise_channel="f", prob=0.001 * (k + 1))
        c.add_gate("N1", 1, noise_channel="p", prob=0.002)
        c.add_gate("MUL", 0, a=2)
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="rec[-2]")
    c.add_gate("DETECTOR", expr="2*rec[-2] + rec[-1]")
    fast = DetectorErrorModel.from_circuit(c)
    _assert_same_model(fast, _reference_from_circuit(c))
    assert len(fast.mechanisms) == 1
    # The merged source names every gate, in circuit order (phase noise before M is invisible).
    assert fast.mechanisms[0].source == "+".join(f"N1[f]@{2 + 3 * k}:q0" for k in range(40))


def test_merged_sources_name_every_gate():
    """A cap on merged sources (the first four, then " (+N more)") made them differ from the old code."""
    c = Circuit(1, 3)
    c.add_gate("RESET", 0)
    for _ in range(6):
        c.add_gate("N1", 0, noise_channel="f", prob=0.001)
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-1]")
    expected = "+".join(f"N1[f]@{k}:q0" for k in range(1, 7))
    assert DetectorErrorModel.from_circuit(c).mechanisms[0].source == expected
    dem = DetectorErrorModel.from_circuit(c, merge=False)
    dem.merge_lines()
    assert dem.mechanisms[0].source == expected
    lines = DetectorErrorModel(7, 1, 0, [ErrorMechanism(0.1, [{0: k + 1}], f"s{k}") for k in range(6)]).to_lines()
    assert [m.source for m in lines.mechanisms] == ["+".join(f"s{k}" for k in range(6))]


@pytest.mark.parametrize("d", [2, 3, 5])
def test_to_lines_matches_reference(d):
    c = random_circuit(d, 41, p=0.05)
    dem = DetectorErrorModel.from_circuit(c)
    _assert_same_model(dem.to_lines(), _reference_to_lines(dem))


def test_to_lines_user_models_match_reference():
    models = [
        DetectorErrorModel(3, 3, 1, [ErrorMechanism(0.1, [{0: 1, 2: 2}, {1: 1}], "a"),
                                     ErrorMechanism(0.2, [{2: 1, 0: 2}], "b"),
                                     ErrorMechanism(0.0, [{1: 1}], "zero"),
                                     ErrorMechanism(1.0, [{3: 1}], "one"),
                                     ErrorMechanism(0.3, [], "empty"),
                                     ErrorMechanism(0.4, [{0: 2}, {0: 1}], "dependent")]),
        DetectorErrorModel(5, 2, 0, [ErrorMechanism(0.1, [{0: -1, 1: 7}], "a"),
                                     ErrorMechanism(0.2, [{1: 2, 0: 9}, {1: 10 ** 30}], "b")]),
        DetectorErrorModel(2, 3, 0, [ErrorMechanism(0.001 * (i + 1), [{i % 3: 1, (i + 1) % 3: 1}], f"m{i}")
                                     for i in range(50)]),
    ]
    for dem in models:
        _assert_same_model(dem.to_lines(), _reference_to_lines(dem))


def test_to_lines_rejects_composite_dimensions():
    """The split into independent lines needs a field; it used to fail only on a non-unit coefficient."""
    for d, gen in [(4, {0: 2, 1: 1}), (4, {0: 1, 1: 2}), (9, {0: 1}), (15, {1: 7})]:
        dem = DetectorErrorModel(d, 2, 0, [ErrorMechanism(0.1, [gen], "a")])
        with pytest.raises(ValueError, match="prime"):
            dem.to_lines()


def test_merge_lines_composite_dimension_keeps_non_unit_lines():
    """At composite d a line whose first coefficient is not a unit is left unmerged instead of crashing."""
    d = 4
    mechs = [ErrorMechanism(0.1, [{1: 1, 0: 2}], "a"), ErrorMechanism(0.2, [{0: 3, 1: 2}], "b"),
             ErrorMechanism(0.3, [{0: 2, 1: 1}], "c"), ErrorMechanism(0.4, [{0: 1, 1: 2}], "d"),
             ErrorMechanism(0.5, [{0: 4}], "zero"), ErrorMechanism(0.6, [{0: 1}, {1: 1}], "rank2")]
    dem = DetectorErrorModel(d, 2, 0, [ErrorMechanism(m.probability, m.generators, m.source) for m in mechs])
    dem.merge_lines()
    ref = DetectorErrorModel(d, 2, 0, list(mechs))
    _reference_merge_lines(ref)
    _assert_same_model(dem, ref)
    # "b" (3 * (1, 2)) and "d" are one line; "a" and "c" have leading coefficient 2 and stay apart.
    assert [(m.source, m.generators) for m in dem.mechanisms] == [
        ("a", [{0: 2, 1: 1}]), ("b+d", [{0: 1, 1: 2}]), ("c", [{0: 2, 1: 1}]), ("zero", [{}]),
        ("rank2", [{0: 1}, {1: 1}])]
    assert dem.mechanisms[1].probability == merge_subgroup_probabilities(0.2, 0.4)


@pytest.mark.parametrize("d", [4, 6, 9])
def test_from_circuit_composite_dimension_does_not_crash(d):
    """check_dimension_prime=False on a composite d used to crash in the merge (non-unit leading coefficient)."""
    c = Circuit(3, d)
    c.add_gate("RESET", [0, 1, 2])
    c.add_gate("N1", 0, noise_channel="f", prob=0.1)
    c.add_gate("MUL", 0, a=d - 1)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("N1", 1, noise_channel="f", prob=0.05)
    c.add_gate("M", [0, 1, 2])
    factor = min(f for f in range(2, d) if d % f == 0)
    c.add_gate("DETECTOR", expr=f"{factor}*rec[-3]")     # the first fault's leading coefficient is not a unit
    c.add_gate("DETECTOR", expr="rec[-2] - rec[-3]")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-2]")
    with pytest.raises(ValueError, match="prime"):
        DetectorErrorModel.from_circuit(c)
    for merge in (True, False):
        dem = DetectorErrorModel.from_circuit(c, merge=merge, check_dimension_prime=False)
        _assert_same_model(dem, _reference_from_circuit(c, merge))
        # X on q0 reads -1 on both qudits: D0 = -factor, L0 = -1, left unscaled. X on q1: D1 = L0 = 1.
        assert [(m.source, m.generators) for m in dem.mechanisms] == [
            ("N1[f]@3:q0", [{0: d - factor, 2: d - 1}]), ("N1[f]@6:q1", [{1: 1, 2: 1}])]
    det, obs = dem.sample(1000, seed=1)
    assert det.max() < d and obs.max() < d


def test_merge_lines_matches_reference():
    c = random_circuit(3, 7, p=0.05)
    a = DetectorErrorModel.from_circuit(c, merge=False)
    b = DetectorErrorModel.from_circuit(c, merge=False)
    a.merge_lines()
    _reference_merge_lines(b)
    _assert_same_model(a, b)


def test_merge_pair_is_bit_exact():
    rng = np.random.default_rng(1)
    special = [0.0, -0.0, 5e-324, 1e-300, 1e-17, 1e-3, 0.5, 1 - 2 ** -53, 1.0, 1.5, -0.25, float("nan")]
    values = special + list(rng.random(2000)) + list(10.0 ** rng.uniform(-300, 0, 2000))
    for a, b in itertools.chain(itertools.product(special, special), zip(values, values[::-1])):
        if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
            x, y = _merge_pair(a, b), merge_subgroup_probabilities(a, b)
            assert _bits(x) == _bits(y), (a, b)
        else:
            with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
                _merge_pair(a, b)
            with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
                merge_subgroup_probabilities(a, b)


def test_merge_subgroup_probabilities_edge_cases():
    """No mechanisms (or only pi = 0 ones) merge to +0.0, not -0.0; probabilities outside [0, 1] are rejected."""
    for args in [(), (0.0,), (0.0, -0.0), (-0.0,)]:
        assert _bits(merge_subgroup_probabilities(*args)) == _bits(0.0), args
    assert merge_subgroup_probabilities(0.25, 0.5) == 1 - 0.75 * 0.5
    assert merge_subgroup_probabilities(0.3, 1.0) == 1.0
    for bad in (1.0000001, -1e-9, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            merge_subgroup_probabilities(0.1, bad)
        with pytest.raises(ValueError):
            line_probability(bad, 3, 2)
    dem = DetectorErrorModel(3, 1, 0, [ErrorMechanism(0.2, [{0: 1}], "a"), ErrorMechanism(1.5, [{0: 2}], "b")])
    with pytest.raises(ValueError, match="1.5"):
        dem.merge_lines()


def test_probe_buffer_growth_keeps_responses(monkeypatch):
    """Tiny probe blocks and buffers force the forward kernel's buffer growth path; the responses must not
    change, and must equal the backward sweep's."""
    c = random_circuit(5, 11, p=0.02)
    expected = compile_unit_responses(c)
    monkeypatch.setattr(dem_module, "_BACKWARD", False)
    monkeypatch.setattr(dem_module, "_run_probes", functools.partial(dem_module._run_probes, block=3, cap=1))
    got = compile_unit_responses(c)
    assert [loc.responses for loc in got.locations] == [loc.responses for loc in expected.locations]
    assert [list(r.items()) for loc in got.locations for r in loc.responses] == \
        [list(r.items()) for loc in expected.locations for r in loc.responses]


def _wide_circuit(d, width=40, rounds=3, p=0.01):
    """Faults that fan out to many detectors, so probe responses are long."""
    c = Circuit(width + 1, d)
    c.add_gate("RESET", list(range(width + 1)))
    for _ in range(rounds):
        c.add_gate("N1", list(range(width + 1)), noise_channel="d", prob=p)
        c.add_gate("CNOT", 0, list(range(1, width + 1)))
        c.add_gate("N2", 0, 1, prob=p)
        c.add_gate("M", list(range(width + 1)))
        for q in range(width + 1):
            c.add_gate("DETECTOR", expr=f"rec[{q - (width + 1)}]")
        c.add_gate("RESET", list(range(width + 1)))
    return c


@pytest.mark.parametrize("threads", [1, 2, 3, 8])
@pytest.mark.parametrize("block, cap", [(None, 16), (1, 1), (5, 2), (64, 1), (1000, 16)])
def test_probe_blocks_and_threads_do_not_change_the_model(monkeypatch, threads, block, cap):
    """The forward unit-fault pass gives the same model for every block size, buffer size and thread count,
    and the same as the backward sweep."""
    circuits = [_wide_circuit(5), _wide_circuit(1000003, width=25), random_circuit(3, 13, p=0.05)]
    expected = [DetectorErrorModel.from_circuit(c) for c in circuits]
    monkeypatch.setattr(dem_module, "_BACKWARD", False)
    monkeypatch.setattr(dem_module, "_thread_count", lambda: threads)
    monkeypatch.setattr(dem_module, "_PROBE_PARALLEL_MIN", 0)
    monkeypatch.setattr(dem_module, "_run_probes", functools.partial(dem_module._run_probes, block=block, cap=cap))
    for c, want in zip(circuits, expected):
        _assert_same_model(DetectorErrorModel.from_circuit(c), want)
        _assert_same_model(DetectorErrorModel.from_circuit(c, merge=False), _reference_from_circuit(c, merge=False))


def test_qudit_index_out_of_range_is_rejected():
    c = Circuit(2, 3)
    c.add_gate("N1", 0, noise_channel="f", prob=0.1)
    c.add_gate("CNOT", 0, 5)
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="rec[-1]")
    with pytest.raises(IndexError, match="qudit 5"):
        DetectorErrorModel.from_circuit(c)


def test_bad_noise_gate_errors_keep_their_order():
    """When a noise gate would make Program._build_ir raise, that error still comes first."""
    c = Circuit(2, 3)
    c.add_gate("N1", [0, 1], noise_channel="f", prob=0.1)
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="rec[-1] * rec[-2]")
    c.operations[1].params = dict(c.operations[1].params, noise_channel="x")
    with pytest.raises(ValueError, match="noise_channel"):
        DetectorErrorModel.from_circuit(c)
    c = Circuit(2, 3)
    c.add_gate("N2", 0, 1, prob=0.1)
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="rec[-1]")
    # add_gate rejects a bad prob_dist itself, so put one in place behind its back.
    c.operations[0].params = dict(c.operations[0].params, prob_dist=[1.0, 0.0])
    with pytest.raises(ValueError, match="prob_dist has length"):
        DetectorErrorModel.from_circuit(c)


# --------------------------------------------------------------------------
# Detector coefficients


EXPRESSIONS = [
    ["rec[-3]", "rec[-2] - rec[-1]", "2*rec[-1] - 3*rec[-2]"],
    ["rec[-1] + rec[-1]", "-rec[-2]", "rec[-3] * 2 + 0", "(rec[-1] + rec[-2]) * 3"],
    ["(rec[-1] - rec[-2]) % 5", "rec[-3] % 1000003", "rec[-1] * (rec[-2] - rec[-2] + 2)"],
    ["abs(rec[-1])", "rec[-2] ** 1", "rec[-3] // 1", "rec[-1] - rec[-1]", "0"],
    ["rec[-1] * rec[-2]"],
    ["rec[-1] + 1"],
    ["rec[-3] if rec[-1] >= 0 else 0"],
    ["(rec[-1] % 2) * 2"],
    ["rec[-1] % -5", "rec[-1] * -1", "-(-rec[-2])"],
]


def _expression_circuit(d, exprs):
    c = Circuit(3, d)
    c.add_gate("RESET", [0, 1, 2])
    c.add_gate("N1", [0, 1, 2], noise_channel="f", prob=0.05)
    c.add_gate("M", [2, 0, 1])
    for k, e in enumerate(exprs):
        c.add_gate("DETECTOR", expr=e, label=f"det{k}")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-1] - rec[-2]")
    return c


def _outcome(fn):
    try:
        return ("ok", fn())
    except ValueError as e:
        return ("error", str(e))


@pytest.mark.parametrize("d", [2, 3, 5, 1000003])
@pytest.mark.parametrize("exprs", EXPRESSIONS)
def test_symbolic_coefficients_match_probing(d, exprs, monkeypatch):
    """One symbolic call gives the same coefficients and errors as evaluating on probes."""
    _, _, info = Program._build_ir([_expression_circuit(d, exprs)], 1)
    fast = _outcome(lambda: _detector_coefficients(info, d))
    monkeypatch.setattr(dem_module, "_symbolic_coefficients", lambda *args: None)
    probed = _outcome(lambda: _detector_coefficients(info, d))
    assert fast == probed


def test_symbolic_path_is_used_for_plain_expressions():
    _, _, info = Program._build_ir([_expression_circuit(7, EXPRESSIONS[0])], 1)
    for fn, (_, _, args, _) in zip(info.detector_functions, info.detector_data):
        assert dem_module._symbolic_coefficients(fn, len(args), 7, {}) is not None


def _random_expression(rng, n, depth):
    """A random detector expression over rec[0 .. n - 1]: sums, differences, scalings, negations and %."""
    if depth <= 0 or rng.random() < 0.25:
        if rng.random() < 0.8:
            j = rng.randrange(n)
            return f"rec[{j if rng.random() < 0.6 else j - n}]"
        return str(rng.randint(-30, 30))
    a, b = _random_expression(rng, n, depth - 1), _random_expression(rng, n, depth - 1)
    op = rng.choice(["+", "-", "*", "%", "neg", "scale", "+", "-"])
    if op == "neg":
        return f"-({a})"
    if op == "scale":
        return f"{rng.randint(-5, 9)} * ({a})"
    if op == "%":
        return f"({a}) % {rng.choice([1, 2, 3]) * rng.choice([2, 3, 4, 5, 6, 7, 9, 1000003])}"
    return f"({a}) {op} ({b})"


@pytest.mark.parametrize("d", [2, 3, 4, 6, 7, 9, 1000003])
def test_symbolic_forms_match_evaluation_on_random_expressions(d):
    """The in-place sparse forms give the expression itself mod d, for nested sums and differences, scalings
    (by non-units too, at composite d), a record used twice, and a sum that cancels."""
    rng = random.Random(d)
    checked = 0
    for k in range(500):
        n = rng.randint(1, 6)
        src = _random_expression(rng, n, rng.randint(0, 6))
        if k < 3:
            src = ["rec[0] + rec[0] - 2*rec[-1]", "rec[0] - (rec[0] - (rec[0] - rec[0]))", "-(-rec[0])"][k]
        fn = _compile_detector(src, d)
        form = dem_module._symbolic_coefficients(fn, n, d, {})
        if form is None:
            continue
        checked += 1
        assert len(form) == n + 1 and all(0 <= c < d for c in form), src
        for _ in range(4):
            x = [rng.randint(-50, 50) for _ in range(n)]
            assert fn(x) % d == (form[0] + sum(c * v for c, v in zip(form[1:], x))) % d, src
    assert checked > 200


def test_symbolic_forms_take_time_linear_in_the_expression():
    """Each operation used to build a dense tuple over all the records, so an observable reading every round
    of a long memory took time quadratic in the rounds just to read its coefficients."""
    def best(n):
        fn = _compile_detector(" + ".join(f"{j % 5 + 1}*rec[{j}]" for j in range(n)) + " - rec[0]", 3)
        times = []
        for _ in range(3):
            # CPU time: under load a call longer than a scheduler time slice also waits for a core.
            t = time.process_time()
            form = dem_module._symbolic_coefficients(fn, n, 3, {})
            times.append(time.process_time() - t)
        assert form[:4] == (0, 0, 2, 0)
        return min(times)

    best(50)
    # A busy machine can stall either size, so measure up to three times before failing.
    for _ in range(3):
        small, large = best(250), best(2000)
        if large / small < 20:
            break
    # Linear work gives a ratio near 8; it was about 45.
    assert large / small < 20, (small, large)


# --------------------------------------------------------------------------
# Sampler


def _exact_distribution(dem):
    d = dem.dimension
    n = dem.num_detectors + dem.num_observables
    dist = np.zeros((d,) * n)
    dist[(0,) * n] = 1.0
    for m in dem.mechanisms:
        pi = min(max(m.probability, 0.0), 1.0)
        kernel = np.zeros((d,) * n)
        kernel[(0,) * n] += 1.0 - pi
        for coeffs in itertools.product(range(d), repeat=len(m.generators)):
            v = [0] * n
            for a, g in zip(coeffs, m.generators):
                for t, c in g.items():
                    v[t] = (v[t] + a * c) % d
            kernel[tuple(v)] += pi / d ** len(m.generators)
        dist = np.real(np.fft.ifftn(np.fft.fftn(dist) * np.fft.fftn(kernel)))
    return np.clip(dist, 0.0, None).ravel()


def _chi2_against_exact(dem, shots, seed):
    stats = pytest.importorskip("scipy.stats")
    d = dem.dimension
    n = dem.num_detectors + dem.num_observables
    det, obs = dem.sample(shots, seed=seed)
    vals = np.concatenate([det, obs], axis=1)
    counts = np.bincount(np.ravel_multi_index(tuple(vals.T), (d,) * n), minlength=d ** n)
    expected = _exact_distribution(dem) * shots
    assert counts[expected < 1e-9].sum() == 0  # outcomes the model cannot produce
    big = expected >= 5
    chi2 = ((counts[big] - expected[big]) ** 2 / expected[big]).sum()
    return stats.chi2.sf(chi2, big.sum() - 1)


def test_sampler_matches_exact_distribution_rank_two():
    """One rank-2 mechanism at d = 3: P(v) = (1 - pi) [v = 0] + pi / 9 * #{(a1, a2): a1 g1 + a2 g2 = v}."""
    dem = DetectorErrorModel(3, 2, 1, [ErrorMechanism(0.3, [{0: 1, 1: 2}, {1: 1, 2: 1}], "m")])
    det, obs = dem.sample(90_000, seed=5)
    vals = np.concatenate([det, obs], axis=1)
    for a1, a2 in itertools.product(range(3), repeat=2):
        v = (a1 % 3, (2 * a1 + a2) % 3, a2 % 3)
        p = 0.3 / 9 + (0.7 if v == (0, 0, 0) else 0.0)
        freq = np.all(vals == v, axis=1).mean()
        assert abs(freq - p) < 5 * math.sqrt(p * (1 - p) / len(vals)), (v, freq, p)
    assert _chi2_against_exact(dem, 200_000, seed=6) > 1e-4


def test_sampler_matches_exact_distribution_mixed_model():
    """Several mechanisms per probability bin (thinning), pi = 1, pi = 0, tiny pi, ranks 1 to 3."""
    mechs = [ErrorMechanism(0.30, [{0: 1}], "a"), ErrorMechanism(0.31, [{0: 1, 1: 1}], "b"),
             ErrorMechanism(0.32, [{1: 2}], "c"), ErrorMechanism(0.05, [{2: 1}], "d"),
             ErrorMechanism(1.0, [{3: 1}], "always"), ErrorMechanism(0.0, [{1: 1}], "never"),
             ErrorMechanism(1e-30, [{2: 1}], "tiny"), ErrorMechanism(0.4, [{2: 1}, {1: 1, 2: 2}, {0: 2}], "r3")]
    assert _chi2_against_exact(DetectorErrorModel(3, 3, 1, mechs), 300_000, seed=11) > 1e-4


@pytest.mark.parametrize("d", [2, 7])
def test_sampler_matches_exact_distribution_random_model(d):
    rng = np.random.default_rng(d)
    n = 5 if d == 2 else 3
    mechs = []
    for k in range(20):
        gens = []
        for _ in range(int(rng.integers(1, 4))):
            ts = rng.choice(n, size=int(rng.integers(1, n + 1)), replace=False)
            gens.append({int(t): int(rng.integers(1, d)) for t in ts})
        mechs.append(ErrorMechanism(float(rng.choice([0.01, 0.0201, 0.1, 0.5, 1.0])), gens, f"m{k}"))
    assert _chi2_against_exact(DetectorErrorModel(d, n - 1, 1, mechs), 300_000, seed=d) > 1e-4


def test_sampler_large_dimension_values_are_uniform():
    d = 1000003
    dem = DetectorErrorModel(d, 1, 0, [ErrorMechanism(0.5, [{0: 1}], "m")])
    det, _ = dem.sample(200_000, seed=3)
    v = det[:, 0]
    p0 = 0.5 + 0.5 / d
    assert abs((v == 0).mean() - p0) < 5 * math.sqrt(p0 * (1 - p0) / len(v))
    stats = pytest.importorskip("scipy.stats")
    assert stats.kstest(v[v != 0] / d, "uniform").pvalue > 1e-4


def test_sampler_same_seed_same_samples_on_every_kernel(monkeypatch):
    c = random_circuit(5, 23, p=0.05)
    dem = DetectorErrorModel.from_circuit(c)
    det, obs = dem.sample(1000, seed=8)
    again = dem.sample(1000, seed=8)
    np.testing.assert_array_equal(det, again[0])
    np.testing.assert_array_equal(obs, again[1])
    assert not np.array_equal(det, dem.sample(1000, seed=9)[0])
    monkeypatch.setattr(dem_module, "_SAMPLE_PARALLEL_WORK", -1.0)
    parallel = dem.sample(1000, seed=8)
    monkeypatch.setattr(dem_module, "_SAMPLE_PARALLEL_WORK", float("inf"))
    serial = dem.sample(1000, seed=8)
    for got in (parallel, serial):
        np.testing.assert_array_equal(det, got[0])
        np.testing.assert_array_equal(obs, got[1])


@pytest.mark.parametrize("threads", [1, 2, 3, 5, 16])
def test_sampler_result_does_not_depend_on_thread_count(monkeypatch, threads):
    """Blocks are seeded on their own, so any thread count and task split gives the same samples."""
    dem = DetectorErrorModel.from_circuit(random_circuit(7, 31, p=0.05))
    shots = 20 * dem_module._SAMPLE_CHUNK + 17
    monkeypatch.setattr(dem_module, "_SAMPLE_PARALLEL_WORK", float("inf"))
    det, obs = dem.sample(shots, seed=12)
    monkeypatch.setattr(dem_module, "_SAMPLE_PARALLEL_WORK", -1.0)
    monkeypatch.setattr(dem_module, "_thread_count", lambda: threads)
    for tasks_per_thread in (1, 3, 100):
        monkeypatch.setattr(dem_module, "_SAMPLE_TASKS_PER_THREAD", tasks_per_thread)
        got = dem.sample(shots, seed=12)
        np.testing.assert_array_equal(det, got[0])
        np.testing.assert_array_equal(obs, got[1])


def test_concurrent_calls_from_python_threads():
    """sample() and from_circuit() called from several Python threads at once give the serial results."""
    import threading

    circuit = _wide_circuit(1000003, width=30)
    dem = DetectorErrorModel.from_circuit(circuit)
    shots = 40 * dem_module._SAMPLE_CHUNK
    ref = dem.sample(shots, seed=3)
    results, errors = [], []

    def work():
        try:
            for _ in range(3):
                results.append((dem.sample(shots, seed=3), DetectorErrorModel.from_circuit(circuit)))
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == 12
    for (det, obs), model in results:
        np.testing.assert_array_equal(det, ref[0])
        np.testing.assert_array_equal(obs, ref[1])
        _assert_same_model(model, dem)


def test_run_tasks_runs_every_task_once_and_reraises():
    import threading

    seen = []
    lock = threading.Lock()

    def task(i):
        with lock:
            seen.append(i)

    for n_tasks, n_threads in [(0, 4), (1, 4), (7, 1), (50, 3), (5, 10)]:
        seen.clear()
        dem_module._run_tasks(task, n_tasks, n_threads)
        assert sorted(seen) == list(range(n_tasks))

    def failing(i):
        if i == 3:
            raise KeyError("task 3")

    with pytest.raises(KeyError, match="task 3"):
        dem_module._run_tasks(failing, 20, 4)
    # Every worker was joined before the error came back.
    assert not any(t.name == "sdim-dem-worker" and t.is_alive() for t in threading.enumerate())


_FORK_AND_LAYER_SCRIPT = r"""
import multiprocessing as mp
import sys
import threading
import numba
import numpy as np
import sdim.dem as dm
from sdim.circuit import Circuit
from sdim.dem import DetectorErrorModel

dm._SAMPLE_PARALLEL_WORK = -1.0
dm._PROBE_PARALLEL_MIN = 0

def circuit():
    n = 30
    c = Circuit(n, 1000003)
    c.add_gate("RESET", list(range(n)))
    for _ in range(3):
        for q in range(n - 1):
            c.add_gate("CNOT", q, q + 1)
            c.add_gate("N2", q, q + 1, prob=1e-2)
        c.add_gate("N1", list(range(n)), noise_channel="d", prob=1e-2)
    c.add_gate("M", list(range(n)))
    for q in range(n - 1):
        c.add_gate("DETECTOR", expr=f"rec[{-(n - q)}] - rec[{-(n - q - 1)}]")
    return c

dem = DetectorErrorModel.from_circuit(circuit())
ref = dem.sample(5000, seed=1)

def child(seed):
    model = DetectorErrorModel.from_circuit(circuit())
    det, obs = model.sample(5000, seed=1)
    return str(model) == str(dem) and bool(np.array_equal(det, ref[0]))

if __name__ == "__main__":
    # The DEM code must not start numba's own thread pool.
    try:
        numba.threading_layer()
        print("numba threading layer was started")
        sys.exit(1)
    except ValueError:
        pass
    out = []
    threads = [threading.Thread(target=lambda: out.append(child(0))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with mp.get_context("fork").Pool(2) as pool:
        out += pool.map(child, [1, 2])
    print("results", out)
    sys.exit(0 if out == [True] * 6 else 1)
"""


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs fork()")
@pytest.mark.parametrize("layer", ["default", "workqueue"])
def test_fork_and_threads_after_parallel_use(tmp_path, layer):
    """Using the threaded paths, then threads and a forked pool, must neither hang nor abort.

    numba's parallel=True pool aborts forked children under GNU OpenMP and aborts on concurrent use under
    workqueue; the DEM code must not depend on it.
    """
    import os
    import subprocess
    import sys

    script = tmp_path / "fork_check.py"
    script.write_text(_FORK_AND_LAYER_SCRIPT)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.dirname(os.path.dirname(dem_module.__file__)),
                                         env.get("PYTHONPATH", "")])
    if layer != "default":
        env["NUMBA_THREADING_LAYER"] = layer
    proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "results [True, True, True, True, True, True]" in proc.stdout


def test_sampler_rejects_dimensions_outside_its_arithmetic():
    for d in (2 ** 31, 2 ** 31 + 11, 3000000019, 0, -5):
        dem = DetectorErrorModel(d, 2, 0, [ErrorMechanism(1.0, [{0: 1, 1: 2}], "a")])
        with pytest.raises(ValueError, match="2\\*\\*31 - 1"):
            dem.sample(10, seed=1)
        det, obs = DetectorErrorModel(d, 2, 0, []).sample(4, seed=1)   # nothing to sample: zeros, as before
        assert det.shape == (4, 2) and not det.any()


def test_sampler_output_layout():
    dem = DetectorErrorModel(7, 3, 2, [ErrorMechanism(0.5, [{0: 1, 4: 3}], "a"), ErrorMechanism(0.2, [{2: 6}], "b")])
    for shots in (0, 1, 255, 256, 257, 1000):
        det, obs = dem.sample(shots, seed=1)
        assert det.shape == (shots, 3) and obs.shape == (shots, 2)
        assert det.dtype == np.int64 and obs.dtype == np.int64
        assert det.flags.c_contiguous and obs.flags.c_contiguous
        assert det.min(initial=0) >= 0 and det.max(initial=0) < 7 and obs.max(initial=0) < 7
        assert not det[:, 1].any() and not obs[:, 0].any()   # targets 0, 2 and 4 = L1 only
        assert shots < 100 or (obs[:, 1].any() and det[:, 0].any() and det[:, 2].any())
    det, obs = DetectorErrorModel(3, 0, 1, [ErrorMechanism(1.0, [{0: 1}], "o")]).sample(50, seed=2)
    assert det.shape == (50, 0) and obs.shape == (50, 1) and obs.any()


def test_sampler_rejects_bad_models():
    with pytest.raises(ValueError, match="NaN"):
        DetectorErrorModel(3, 1, 0, [ErrorMechanism(float("nan"), [{0: 1}], "x")]).sample(10, seed=1)
    with pytest.raises(ValueError, match="target 5"):
        DetectorErrorModel(3, 1, 0, [ErrorMechanism(0.1, [{5: 1}], "x")]).sample(10, seed=1)
    with pytest.raises(ValueError, match="target -1"):
        DetectorErrorModel(3, 1, 0, [ErrorMechanism(0.1, [{-1: 1}], "x")]).sample(10, seed=1)


@pytest.mark.parametrize("d", [2, 4, 9, 1000003, 2147483647])
def test_sampler_multiplication_is_exact(d):
    """Target 0 reads the random coefficient a itself, so every other target must be a * v mod d.

    Odd d uses Montgomery multiplication and even d uses %; d = 2**31 - 1 is the largest allowed.
    """
    vs = sorted({v % d for v in [1, 2, 3, d - 1, d - 2, d // 2, d // 3 + 1, 12345678, 2 ** 30 + 3]} - {0})
    gen = {0: 1}
    gen.update({j + 1: v for j, v in enumerate(vs)})
    dem = DetectorErrorModel(d, len(gen), 0, [ErrorMechanism(1.0, [gen], "all")])
    det, _ = dem.sample(5000, seed=4)
    a = det[:, 0]
    for j, v in enumerate(vs):
        np.testing.assert_array_equal(det[:, j + 1], (a * v) % d)
    assert len(np.unique(a)) > min(d, 1000) // 2


def test_sampler_wide_generators():
    """Generators with thousands of entries, so one shot adds thousands of increments."""
    d, n = 7, 3000
    dem = DetectorErrorModel(d, n, 1, [ErrorMechanism(1.0, [{t: 1 for t in range(n)}], "wide"),
                                       ErrorMechanism(1.0, [{t: 2 for t in range(0, n + 1, 2)}], "even")])
    det, obs = dem.sample(300, seed=6)
    odd = det[:, 1::2]
    assert (odd == odd[:, :1]).all()
    even = det[:, 0::2]
    assert (even == even[:, :1]).all()
    # target n is observable 0 and only gets the second generator: even = a + 2 b, obs = 2 b
    np.testing.assert_array_equal((odd[:, 0] + obs[:, 0]) % d, even[:, 0])


def test_long_generators_are_sorted_and_merged_like_reference():
    """Faults that reach many detectors, in an order unrelated to the detector numbering
    (exercises the heapsort path used for generators with more than 32 entries)."""
    d, n = 5, 90
    rng = np.random.default_rng(12)
    c = Circuit(n + 1, d)
    c.add_gate("RESET", list(range(n + 1)))
    c.add_gate("N1", 0, noise_channel="f", prob=0.01)
    c.add_gate("N1", 0, noise_channel="f", prob=0.02)
    c.add_gate("CNOT", 0, [int(q) for q in rng.permutation(np.arange(1, n + 1))])
    c.add_gate("N2", 0, 1, prob=0.03)
    c.add_gate("MUL", 0, a=3)
    c.add_gate("M", list(range(n + 1)))
    for q in rng.permutation(n + 1):
        c.add_gate("DETECTOR", expr=f"{int(rng.integers(1, d))}*rec[{int(q) - (n + 1)}]")
    for merge in (True, False):
        _assert_same_model(DetectorErrorModel.from_circuit(c, merge=merge), _reference_from_circuit(c, merge))
    dem = DetectorErrorModel.from_circuit(c)
    assert max(len(g) for m in dem.mechanisms for g in m.generators) > 32
    _assert_same_model(dem.to_lines(), _reference_to_lines(dem))
    keys = [int(k) for k in rng.permutation(200)]
    wide = DetectorErrorModel(3, 200, 0, [ErrorMechanism(0.1, [{k: 1 + (k % 2) for k in keys},
                                                              {k: 2 for k in keys[:70]}], "w"),
                                          ErrorMechanism(0.2, [{k: 2 for k in keys[::-1]}], "v")])
    _assert_same_model(wide.to_lines(), _reference_to_lines(wide))


def test_sort_pairs_matches_numpy():
    rng = np.random.default_rng(2)
    for n in [0, 1, 2, 5, 32, 33, 34, 100, 1000]:
        keys = rng.permutation(10 * n + 1)[:n].astype(np.int64)
        vals = rng.integers(0, 1000, n).astype(np.int64)
        pad_k = np.concatenate(([7, 3], keys, [5]))
        pad_v = np.concatenate(([1, 2], vals, [3]))
        dem_module._sort_pairs(pad_k, pad_v, 2, 2 + n)
        order = np.argsort(keys)
        np.testing.assert_array_equal(pad_k[2:2 + n], keys[order])
        np.testing.assert_array_equal(pad_v[2:2 + n], vals[order])
        assert list(pad_k[:2]) == [7, 3] and pad_k[-1] == 5


# --------------------------------------------------------------------------
# Sampler with many probability bins


def _many_bins_model(d=3, n_det=3, n_bins=40, seed=0, quiet=0):
    """Mechanisms in n_bins different probability bins (pi from 0.6 down by factors of about 1.3),
    plus `quiet` mechanisms in other bins whose pi is so small they never fire."""
    rng = np.random.default_rng(seed)
    mechs = []
    for k in range(n_bins):
        pi = 0.6 / 1.3 ** k
        for _ in range(int(rng.integers(1, 3))):
            gens = [{int(t): int(rng.integers(1, d)) for t in rng.choice(n_det + 1, size=int(rng.integers(1, 3)),
                                                                          replace=False)}
                    for _ in range(int(rng.integers(1, 3)))]
            mechs.append(ErrorMechanism(float(pi * (1 - 0.1 * rng.random())), gens, f"m{k}"))
    for k in range(quiet):
        mechs.append(ErrorMechanism(2.0 ** (-200 - k), [{int(rng.integers(n_det + 1)): 1}], f"q{k}"))
    rng.shuffle(mechs)
    return DetectorErrorModel(d, n_det, 1, mechs)


def _bins(dem):
    p = np.array([m.probability for m in dem.mechanisms])
    return len(set((p[(p > 0) & (p < 1)].view(np.int64) >> 49).tolist()))


def test_sampler_many_bins_matches_exact_distribution():
    """Shots visit only the bins with a candidate in them; the distribution must stay exact."""
    dem = _many_bins_model(n_bins=40, quiet=300)
    assert _bins(dem) > 300
    assert _chi2_against_exact(dem, 200_000, seed=21) > 1e-4


@pytest.mark.parametrize("threads", [1, 3, 8])
def test_sampler_many_bins_reproducible_across_threads(monkeypatch, threads):
    dem = _many_bins_model(d=5, n_det=6, n_bins=60, quiet=2000, seed=4)
    shots = 9 * dem_module._SAMPLE_CHUNK + 5
    monkeypatch.setattr(dem_module, "_SAMPLE_PARALLEL_WORK", float("inf"))
    det, obs = dem.sample(shots, seed=7)
    monkeypatch.setattr(dem_module, "_SAMPLE_PARALLEL_WORK", -1.0)
    monkeypatch.setattr(dem_module, "_thread_count", lambda: threads)
    got = dem.sample(shots, seed=7)
    np.testing.assert_array_equal(det, got[0])
    np.testing.assert_array_equal(obs, got[1])
    assert det.any() and not np.array_equal(det, dem.sample(shots, seed=8)[0])


def test_sampler_cost_does_not_grow_with_shots_times_bins(monkeypatch):
    """Every shot used to visit every probability bin, so ~8000 bins of mechanisms that almost never fire
    cost 8000 steps per shot. Now a block of 256 shots draws one skip per bin and each shot scans a bitmap."""
    rng = np.random.default_rng(3)
    probs = 10.0 ** rng.uniform(-300, -12, 40000)
    dem = DetectorErrorModel(3, 10, 0, [ErrorMechanism(float(p), [{int(i % 10): 1}], "") for i, p in
                                        enumerate(probs)])
    assert _bins(dem) > 7000
    monkeypatch.setattr(dem_module, "_SAMPLE_PARALLEL_WORK", float("inf"))   # one thread
    dem.sample(256, seed=1)
    times = []
    for _ in range(3):
        t = time.perf_counter()
        det, _ = dem.sample(40 * 1024, seed=2)
        times.append(time.perf_counter() - t)
    assert not det.any()
    # About 0.03 s now; visiting 8000 bins in each of 40960 shots took about 1 s.
    assert min(times) < 0.3, times


def test_sampler_bins_and_always_on_mechanisms_together():
    """Bins that fire in most shots, bins that rarely fire, and mechanisms that always fire, in one model."""
    mechs = [ErrorMechanism(1.0, [{0: 1}], "always"), ErrorMechanism(0.9, [{1: 1}], "often"),
             ErrorMechanism(0.5, [{1: 2, 2: 1}], "half"), ErrorMechanism(1e-3, [{2: 1}], "rare"),
             ErrorMechanism(1e-9, [{0: 2}], "very rare"), ErrorMechanism(0.2, [{0: 1, 1: 1, 2: 1}], "fifth")]
    assert _chi2_against_exact(DetectorErrorModel(3, 2, 1, mechs), 300_000, seed=5) > 1e-4


def test_line_groups_hash_path_matches_exact_grouping():
    """Many lines are grouped by a hash of their entries (checked entry by entry); few are grouped exactly."""
    rng = np.random.default_rng(9)
    for n_lines, solo_rate in [(10, 0.0), (64, 0.2), (65, 0.0), (500, 0.0), (2000, 0.1)]:
        pool = [sorted({int(t): int(rng.integers(1, 5)) for t in rng.choice(30, size=int(rng.integers(1, 5)),
                                                                            replace=False)}.items())
                for _ in range(max(n_lines // 4, 1))]
        lines = [pool[int(rng.integers(len(pool)))] for _ in range(n_lines)]
        cptr = np.zeros(n_lines + 1, dtype=np.int64)
        np.cumsum([len(x) for x in lines], out=cptr[1:])
        ctgt = np.array([t for x in lines for t, _ in x], dtype=np.int64)
        cval = np.array([v for x in lines for _, v in x], dtype=np.int64)
        solo = rng.random(n_lines) < solo_rate
        for marks in (None, solo):
            got = dem_module._line_groups(cptr, ctgt, cval, marks)
            want = dem_module._exact_line_groups(cptr, ctgt, cval, marks)
            np.testing.assert_array_equal(got, want)
            # Numbered in order of first appearance, equal lines together, solo lines alone.
            seen = {}
            for i, x in enumerate(lines):
                key = i if marks is not None and marks[i] else tuple(x)
                assert got[i] == seen.setdefault(key, len(seen))
