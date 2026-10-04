"""Tests for the compact qudit DEM in sdim.dem."""

import math
import re

import numpy as np
import pytest

from sdim.circuit import Circuit
from sdim.dem import (
    DetectorErrorModel,
    compile_unit_responses,
    depolarizing_subgroup_probability,
    line_probability,
    merge_subgroup_probabilities,
    _projective_points,
)
from sdim.program import Program, SimulationOptions, simulate_frame


# --------------------------------------------------------------------------
# Decorrelation math


@pytest.mark.parametrize("p", [1e-4, 0.01, 0.2, 0.7])
def test_gidney_qubit_formulas(p):
    """At d = 2 the line probabilities reduce to Gidney's closed forms."""
    # rel_tol is loose because the closed forms lose digits to cancellation at small p.
    pi1 = depolarizing_subgroup_probability(p, 2, 1)
    assert math.isclose(line_probability(pi1, 2, 2) / 2,
                        0.5 - 0.5 * math.sqrt(1 - 4 * p / 3), rel_tol=1e-9)
    if p <= 15 / 16:
        pi2 = depolarizing_subgroup_probability(p, 2, 2)
        assert math.isclose(line_probability(pi2, 2, 4) / 2,
                            0.5 - 0.5 * (1 - 16 * p / 15) ** 0.125, rel_tol=1e-9)


def _convolve(dist_a, dist_b):
    return np.real(np.fft.ifftn(np.fft.fftn(dist_a) * np.fft.fftn(dist_b)))


@pytest.mark.parametrize("d,k", [(2, 2), (2, 4), (3, 2), (3, 4), (5, 2), (5, 4), (7, 2)])
def test_line_decomposition_reproduces_depolarizing(d, k):
    """Independent line mechanisms with the generalized Gidney probability
    convolve to exactly the depolarizing distribution on Z_d^k."""
    p = 0.137
    pi = depolarizing_subgroup_probability(p, d, k // 2)
    pl = line_probability(pi, d, k)
    shape = (d,) * k
    total = np.zeros(shape)
    total[(0,) * k] = 1.0
    for direction in _projective_points(d, k):
        mech = np.zeros(shape)
        mech[(0,) * k] += 1.0 - pl
        for a in range(d):
            idx = tuple((a * x) % d for x in direction)
            mech[idx] += pl / d
        total = _convolve(total, mech)
    expected = np.full(shape, p / (d ** k - 1))
    expected[(0,) * k] = 1.0 - p
    np.testing.assert_allclose(total, expected, atol=1e-12)


def test_merge_rule_is_xor_for_qubits_and_exact_for_qudits():
    p1, p2 = 0.03, 0.11
    # d = 2: a line fired with probability pi applies P with probability pi / 2.
    merged = merge_subgroup_probabilities(2 * p1, 2 * p2) / 2
    assert math.isclose(merged, p1 * (1 - p2) + (1 - p1) * p2)
    # d = 5: compare against direct convolution on Z_5.
    d = 5
    pis = (0.2, 0.35)
    dist = np.zeros(d)
    dist[0] = 1.0
    for pi in pis:
        mech = np.full(d, pi / d)
        mech[0] += 1 - pi
        dist = _convolve(dist, mech)
    pi = merge_subgroup_probabilities(*pis)
    expected = np.full(d, pi / d)
    expected[0] += 1 - pi
    np.testing.assert_allclose(dist, expected, atol=1e-12)


# --------------------------------------------------------------------------
# Compilation against sdim's own frame simulator


def _small_css_circuit(d, p):
    """Two rounds of a 3-qudit repetition code check, plus an H / H_INV pair.

    Uses CNOT, CNOT_INV, H, H_INV, RESET, M, N1 with all three channels, and N2.
    """
    c = Circuit(5, d)
    data, a0, a1 = [0, 1, 2], 3, 4
    for q in data:
        c.add_gate("RESET", q)
        c.add_gate("N1", q, noise_channel="f", prob=p)
    for t in range(2):
        for a in (a0, a1):
            c.add_gate("RESET", a)
            c.add_gate("N1", a, noise_channel="f", prob=p)
        c.add_gate("CNOT", 0, a0)
        c.add_gate("N2", 0, a0, prob=p)
        c.add_gate("CNOT_INV", 1, a0)
        c.add_gate("N2", 1, a0, prob=p)
        c.add_gate("CNOT", 1, a1)
        c.add_gate("N2", 1, a1, prob=p)
        c.add_gate("CNOT_INV", 2, a1)
        c.add_gate("N2", 2, a1, prob=p)
        c.add_gate("H", 2)
        c.add_gate("N1", 2, noise_channel="d", prob=p)
        c.add_gate("H_INV", 2)
        c.add_gate("N1", 0, noise_channel="p", prob=p)
        for a in (a0, a1):
            c.add_gate("N1", a, noise_channel="f", prob=p)
            c.add_gate("M", a)
        if t == 0:
            c.add_gate("DETECTOR", expr="rec[-2]")
            c.add_gate("DETECTOR", expr="rec[-1]")
        else:
            c.add_gate("DETECTOR", expr="rec[-2] - rec[-4]")
            c.add_gate("DETECTOR", expr="rec[-1] - rec[-3]")
    for q in data:
        c.add_gate("N1", q, noise_channel="f", prob=p)
        c.add_gate("M", q)
    c.add_gate("DETECTOR", expr="rec[-3] - rec[-2] - rec[-5]")
    c.add_gate("DETECTOR", expr="rec[-2] - rec[-1] - rec[-4]")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-1]")
    return c


def _sdim_frame(circuit, shots, noise=None, seed=0):
    np.random.seed(seed)
    prog = Program(circuit)
    prog._simulate_tableau(SimulationOptions(shots=1))
    ref = prog._results_to_array(prog.measurement_results)
    ir, sampled, info = prog._build_ir(prog.circuits, shots)
    _, det = simulate_frame(ir, ref, circuit.num_qudits, circuit.dimension, shots,
                            sampled if noise is None else noise, info)
    d = circuit.dimension
    return np.asarray(det.detection_events) % d, np.asarray(det.logical_operator_shifts) % d


@pytest.mark.parametrize("d", [2, 3, 7, 1000003])
def test_unit_responses_match_frame_simulator(d):
    c = _small_css_circuit(d, 0.01)
    compiled = compile_unit_responses(c)
    probes, mine = [], []
    for g, loc in enumerate(compiled.locations):
        comps = {"d": [0, 1], "f": [0], "p": [1], "d2": [0, 1, 2, 3]}[loc.channel]
        for k, comp in enumerate(comps):
            probes.append((g, comp))
            mine.append(loc.responses[k])
    noise = np.zeros((len(compiled.locations), len(probes), 4), dtype=np.int64)
    for j, (g, comp) in enumerate(probes):
        noise[g, j, comp] = 1
    det, obs = _sdim_frame(c, len(probes), noise)
    nd = compiled.num_detectors
    for j, resp in enumerate(mine):
        ref = {i: int(v) for i, v in enumerate(det[:, j]) if v}
        ref.update({nd + i: int(v) for i, v in enumerate(obs[:, j]) if v})
        assert ref == resp, (j, probes[j])


def test_large_dimension_circuit_and_frame_sampling():
    d = 1000003
    c = _small_css_circuit(d, 0.05)  # must not allocate d**4 tables
    det, obs = _sdim_frame(c, 200)
    assert det.shape[1] == 200


@pytest.mark.parametrize("d", [3, 1000003])
def test_dem_sampler_matches_frame_sampler(d):
    p = 0.04
    c = _small_css_circuit(d, p)
    dem = DetectorErrorModel.from_circuit(c)
    shots = 40000
    ddem, odem = dem.sample(shots, seed=7)
    dfr, ofr = _sdim_frame(c, shots, seed=11)
    a = np.concatenate([ddem, odem], axis=1) != 0
    b = np.concatenate([dfr.T, ofr.T], axis=1) != 0
    for i in range(a.shape[1]):
        pa, pb = a[:, i].mean(), b[:, i].mean()
        sigma = math.sqrt(max(pa * (1 - pa), 1e-4) / shots * 2)
        assert abs(pa - pb) < 5 * sigma, (i, pa, pb)
    # pairwise co-firing (checks correlations, e.g. hyperedges from N2)
    ca = (a[:, :, None] & a[:, None, :]).mean(axis=0)
    cb = (b[:, :, None] & b[:, None, :]).mean(axis=0)
    assert np.max(np.abs(ca - cb)) < 0.01


def test_file_roundtrip(tmp_path):
    c = _small_css_circuit(11, 0.02)
    dem = DetectorErrorModel.from_circuit(c)
    path = tmp_path / "model.qdem"
    dem.write_to_file(path)
    again = DetectorErrorModel.read_from_file(path)
    assert again.dimension == dem.dimension
    assert again.num_detectors == dem.num_detectors
    assert again.num_observables == dem.num_observables
    assert len(again.mechanisms) == len(dem.mechanisms)
    for m1, m2 in zip(dem.mechanisms, again.mechanisms):
        assert m1.probability == m2.probability
        assert m1.generators == m2.generators


def test_mechanism_count_independent_of_dimension():
    counts = {d: len(DetectorErrorModel.from_circuit(_small_css_circuit(d, 0.01)).mechanisms)
              for d in (3, 101, 1000003)}
    assert len(set(counts.values())) == 1, counts


# --------------------------------------------------------------------------
# d = 2 against stim (stim's DEPOLARIZE1/2 use Gidney's decorrelation)


def test_matches_stim_at_qubit_dimension():
    stim = pytest.importorskip("stim")
    p = 0.01
    c = _small_css_circuit(2, p)
    names = {"RESET": "R", "CNOT": "CX", "CNOT_INV": "CX", "H": "H", "H_INV": "H", "M": "M"}
    lines = []
    for op in c.operations:
        n = op.name
        if n in names:
            qs = [op.qudit_index] + ([op.target_index] if op.target_index is not None else [])
            lines.append(f"{names[n]} " + " ".join(map(str, qs)))
        elif n == "N1":
            ch = op.params["noise_channel"]
            gate = {"d": "DEPOLARIZE1", "f": "X_ERROR", "p": "Z_ERROR"}[ch]
            lines.append(f"{gate}({op.params['prob']}) {op.qudit_index}")
        elif n == "N2":
            lines.append(f"DEPOLARIZE2({op.params['prob']}) {op.qudit_index} {op.target_index}")
        elif n in ("DETECTOR", "LOGICAL_OBSERVABLE"):
            recs = [int(x) for x in re.findall(r"rec\[(-?\d+)\]", op.params["expr"])]
            targets = " ".join(f"rec[{r}]" for r in recs)
            lines.append(("DETECTOR " if n == "DETECTOR" else "OBSERVABLE_INCLUDE(0) ") + targets)
    stim_dem = stim.Circuit("\n".join(lines)).detector_error_model(flatten_loops=True)
    expected = {}
    for inst in stim_dem:
        if inst.type != "error":
            continue
        key = frozenset(("D" if t.is_relative_detector_id() else "L") + str(t.val)
                        for t in inst.targets_copy() if not t.is_separator())
        expected[key] = expected.get(key, 0.0) * (1 - inst.args_copy()[0]) + \
            (1 - expected.get(key, 0.0)) * inst.args_copy()[0]
    ours = {}
    lines_dem = DetectorErrorModel.from_circuit(c).to_lines()
    nd = lines_dem.num_detectors
    for m in lines_dem.mechanisms:
        key = frozenset(("D" + str(t)) if t < nd else ("L" + str(t - nd)) for t in m.generators[0])
        ours[key] = m.probability / 2  # a fired line {I, P} applies P with prob pi/2
    assert set(ours) == set(expected)
    for key in expected:
        assert math.isclose(ours[key], expected[key], rel_tol=1e-9), key
