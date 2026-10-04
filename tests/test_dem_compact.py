"""Tests for the compact qudit DEM in sdim.dem."""

import math
import re

import numpy as np
import pytest

from sdim.circuit import Circuit
from sdim.dem import (
    DetectorErrorModel,
    ErrorMechanism,
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
    # Noiseless reference shot, as in Program.simulate's frame mode.
    prog._tableau_noise_enabled = False
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


def _mul_circuit(d, a, p):
    """A deterministic circuit that puts MUL by a and by a^-1 between noise and detectors.

    Uses MUL, CNOT, H, H_INV, RESET, M, N1 with all three channels, and N2. Every
    qudit ends in |0>, so all detectors are deterministic. The H / MUL / H_INV
    sandwich on qudit 1 sends X faults through the Z part of MUL (a^-1).
    """
    a_inv = pow(a, -1, d)
    c = Circuit(3, d)
    anc = 2
    for q in (0, 1, anc):
        c.add_gate("RESET", q)
        c.add_gate("N1", q, noise_channel="d", prob=p)
    c.add_gate("MUL", 0, a=a)
    c.add_gate("N1", 0, noise_channel="f", prob=p)
    c.add_gate("H", 1)
    c.add_gate("MUL", 1, a=a)
    c.add_gate("N1", 1, noise_channel="p", prob=p)
    c.add_gate("H_INV", 1)
    c.add_gate("CNOT", 0, anc)
    c.add_gate("N2", 0, anc, prob=p)
    c.add_gate("MUL", 0, a=a_inv)
    c.add_gate("MUL", anc, a=a)
    c.add_gate("CNOT", 1, anc)
    c.add_gate("N2", 1, anc, prob=p)
    c.add_gate("M", anc)
    c.add_gate("M", 0)
    c.add_gate("M", 1)
    c.add_gate("DETECTOR", expr="rec[-3]")
    c.add_gate("DETECTOR", expr="rec[-2]")
    c.add_gate("DETECTOR", expr="rec[-1]")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-2] + rec[-1]")
    return c


@pytest.mark.parametrize("d,a", [(3, 2), (7, 3), (1000003, 2)])
def test_unit_responses_with_mul_match_frame_simulator(d, a):
    c = _mul_circuit(d, a, 0.01)
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
    if d > 3:
        # MUL really scales the faults: a or a^-1 shows up as a coefficient, not just +-1.
        assert any(v in (a % d, pow(a, -1, d)) for resp in mine for v in resp.values())


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


# --------------------------------------------------------------------------
# Regression tests for edge cases


def test_sampler_handles_tiny_probabilities():
    """The geometric skip used to overflow int64 below pi ~ 4e-18 and write out of bounds."""
    for pi in (1e-20, 1e-25, 1e-300):
        dem = DetectorErrorModel(3, 1, 0, [ErrorMechanism(pi, [{0: 1}], "tiny")])
        for seed in range(5):
            det, obs = dem.sample(1000, seed=seed)
            assert not det.any()


def test_sampler_with_no_mechanisms():
    det, obs = DetectorErrorModel(3, 2, 1).sample(5, seed=1)
    assert det.shape == (5, 2) and obs.shape == (5, 1)
    assert not det.any() and not obs.any()
    c = Circuit(1, 3)
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-1]")
    det, _ = DetectorErrorModel.from_circuit(c).sample(4)
    assert det.shape == (4, 1) and not det.any()


def test_sample_seeds_are_not_truncated():
    dem = DetectorErrorModel.from_circuit(_small_css_circuit(3, 0.2))
    assert not np.array_equal(dem.sample(2000, seed=5)[0], dem.sample(2000, seed=5 + 2 ** 32)[0])
    np.testing.assert_array_equal(dem.sample(50, seed=9)[0], dem.sample(50, seed=9)[0])


def _flip_circuit(d, *probs):
    c = Circuit(1, d)
    c.add_gate("RESET", 0)
    for p in probs:
        c.add_gate("N1", 0, noise_channel="f", prob=p)
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-1]")
    return c


def test_probability_above_full_mixing_is_rejected():
    with pytest.raises(ValueError, match="fully mixing"):
        DetectorErrorModel.from_circuit(_flip_circuit(2, 0.9))
    with pytest.raises(ValueError, match="fully mixing"):
        DetectorErrorModel.from_circuit(_flip_circuit(3, 0.9))


def test_fully_mixing_probability_is_supported():
    """pi = 1 exactly (X with probability 1/2 at d = 2) used to crash the merge."""
    dem = DetectorErrorModel.from_circuit(_flip_circuit(2, 0.5, 0.01))
    assert len(dem.mechanisms) == 1 and dem.mechanisms[0].probability == 1.0
    lines = dem.to_lines()
    assert lines.mechanisms[0].probability == 1.0
    det, _ = dem.sample(20000, seed=3)
    assert abs((det[:, 0] != 0).mean() - 0.5) < 0.02


def test_legacy_channel_key_is_respected():
    c = Circuit(1, 3)
    c.add_gate("N1", 0, channel="p", prob=0.3)
    assert c.operations[0].params["noise_channel"] == "p"
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-1]")
    assert DetectorErrorModel.from_circuit(c).mechanisms == []   # phase noise before M is invisible


def test_invalid_noise_parameters_are_rejected():
    c = Circuit(1, 3)
    with pytest.raises(ValueError, match="noise_channel"):
        c.add_gate("N1", 0, noise_channel="x", prob=0.1)
    with pytest.raises(ValueError, match="prob"):
        c.add_gate("N1", 0, noise_channel="d", prob=1.5)
    with pytest.raises(ValueError, match="prob"):
        Circuit(2, 3).add_gate("N2", 0, 1, prob=-0.1)
    with pytest.raises(ValueError, match="2\\*\\*31"):
        Circuit(1, 2 ** 31 + 11)


def test_non_deterministic_detector_is_rejected():
    c = Circuit(1, 3)
    c.add_gate("RESET", 0)
    c.add_gate("H", 0)
    c.add_gate("N1", 0, noise_channel="f", prob=0.01)
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-1]")
    with pytest.raises(ValueError, match="not deterministic"):
        DetectorErrorModel.from_circuit(c)


def test_non_linear_detector_is_rejected():
    c = Circuit(2, 2)
    c.add_gate("N1", 0, noise_channel="f", prob=0.1)
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="rec[-1]")
    c.add_gate("DETECTOR", expr="rec[-1] * rec[-2]")   # a single random probe used to miss this one
    with pytest.raises(ValueError, match="not linear"):
        DetectorErrorModel.from_circuit(c)


def test_from_circuit_leaves_global_rng_alone():
    c = _small_css_circuit(3, 0.05)
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    DetectorErrorModel.from_circuit(c)
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_wide_fan_out_compiles():
    """One fault reaching more than 512 qudits at once used to exhaust memory."""
    n = 700
    c = Circuit(n + 1, 3)
    c.add_gate("RESET", list(range(n + 1)))
    c.add_gate("N1", 0, noise_channel="f", prob=0.01)
    c.add_gate("CNOT", 0, list(range(1, n + 1)))
    c.add_gate("M", list(range(n + 1)))
    c.add_gate("DETECTOR", expr="rec[-1]")
    dem = DetectorErrorModel.from_circuit(c)
    assert [m.generators for m in dem.mechanisms] == [[{0: 1}]]


def test_read_from_file_validation(tmp_path):
    def read(text):
        path = tmp_path / "m.qdem"
        path.write_text(text)
        return DetectorErrorModel.read_from_file(path)

    with pytest.raises(ValueError, match="out of range"):
        read("DIMENSION 3\nDETECTORS 1\nOBSERVABLES 0\nERROR(0.1) D5=1\n")
    with pytest.raises(ValueError, match="bad target"):
        read("DIMENSION 3\nDETECTORS 1\nOBSERVABLES 1\nERROR(0.1) X0=1\n")
    with pytest.raises(ValueError, match="DIMENSION"):
        read("DETECTORS 1\nERROR(0.1) D0=1\n")
    with pytest.raises(ValueError, match="probability"):
        read("DIMENSION 3\nDETECTORS 1\nOBSERVABLES 0\nERROR(1.5) D0=1\n")
    dem = read("DIMENSION 3\nDETECTORS 2\nOBSERVABLES 0\nERROR(0.1) D0=3 D1=1\nERROR(0.2) D0=3\n")
    assert [m.generators for m in dem.mechanisms] == [[{1: 1}]]   # zero coefficients are dropped
    dem.merge_lines()


def test_file_roundtrip_with_labels_and_numpy_floats(tmp_path):
    dem = DetectorErrorModel(5, 1, 1, [ErrorMechanism(np.float64(0.1), [{0: 2, 1: 4}], "src")],
                             ["Z#1 round 0"], ["logical # 0"])
    path = tmp_path / "m.qdem"
    dem.write_to_file(path)
    again = DetectorErrorModel.read_from_file(path)
    assert again.detector_labels == ["Z#1 round 0"]
    assert again.observable_labels == ["logical # 0"]
    assert again.mechanisms[0].probability == 0.1
    assert again.mechanisms[0].generators == [{0: 2, 1: 4}]
