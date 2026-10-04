"""Randomized gate-by-gate cross-checks of the compact DEM in sdim.dem.

The tests build random circuits whose detectors are deterministic without
noise:

    RESET all qudits, put some of them in an X eigenstate with H, apply a random
    Clifford sequence C and then C^-1, and measure every qudit in the basis it
    was prepared in.

N1 (all three channels) and N2 gates are sprinkled through C, C^-1 and the
preparation / measurement layers, so each fault is pushed through a random
mix of H, H_INV, P, P_INV, CNOT, CNOT_INV, CZ, CZ_INV and SWAP before it is
read out. A second round re-prepares every qudit, either by a mid-circuit
RESET or by carrying the measured state over, and its detectors compare the
two rounds (rec[-a] - rec[-b]). X-basis qudits are read out either with M_X
or with H_INV followed by M, so M, M_X and RESET all sit in the middle of the
circuit as well as at its end.

The unit responses of `compile_unit_responses` are compared exactly with
sdim's own Pauli frame simulator, `DetectorErrorModel.sample` is compared
statistically with frame sampling, and at d = 2 the decorrelated model is
compared with stim's detector error model.

test_reference_tableau_has_no_overflow is a regression test for a bug found
along the way in sdim's tableau simulator, where int64 overflow changed
measurement outcomes. It does not involve the DEM.
"""

import math
import re

import numpy as np
import pytest

from sdim.circuit import Circuit
from sdim.dem import DetectorErrorModel, compile_unit_responses
from sdim.program import Program, SimulationOptions, simulate_frame


ONE_QUDIT_CLIFFORDS = ("H", "H_INV", "P", "P_INV")
TWO_QUDIT_CLIFFORDS = ("CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP")
CLIFFORDS = ONE_QUDIT_CLIFFORDS + TWO_QUDIT_CLIFFORDS
INVERSE = {"H": "H_INV", "H_INV": "H", "P": "P_INV", "P_INV": "P",
           "CNOT": "CNOT_INV", "CNOT_INV": "CNOT", "CZ": "CZ_INV", "CZ_INV": "CZ",
           "SWAP": "SWAP"}
NOISE_KINDS = ("d", "f", "p", "n2")
# Every gate the random circuits are built to contain at least once.
COVERED_GATES = set(CLIFFORDS) | {"M", "M_X", "RESET", "N1", "N2", "DETECTOR", "LOGICAL_OBSERVABLE"}

DIMENSIONS = [2, 3, 5, 7, 1000003]
SEEDS = [11, 23, 37, 41, 53, 67, 79, 97]


# --------------------------------------------------------------------------
# Random deterministic circuits


class _Builder:
    """A Circuit plus the bookkeeping needed to write rec[-k] expressions and random noise."""

    def __init__(self, num_qudits, dimension, rng, p):
        self.circuit = Circuit(num_qudits, dimension)
        self.n = num_qudits
        self.rng = rng
        self.p = p
        self.num_records = 0

    def gate(self, name, *qudits):
        self.circuit.add_gate(name, *[int(q) for q in qudits])

    def noise(self, kind=None):
        kind = kind if kind is not None else NOISE_KINDS[int(self.rng.integers(len(NOISE_KINDS)))]
        if kind == "n2":
            a, b = self.rng.choice(self.n, size=2, replace=False)
            self.circuit.add_gate("N2", int(a), int(b), prob=self.p)
        else:
            self.circuit.add_gate("N1", int(self.rng.integers(self.n)), noise_channel=kind, prob=self.p)

    def maybe_noise(self, rate):
        if self.rng.random() < rate:
            self.noise()

    def measure(self, name, q):
        self.gate(name, q)
        self.num_records += 1
        return self.num_records - 1

    def expr(self, terms):
        """Turns {absolute record index: coefficient} into an expression of rec[-k] terms."""
        parts = []
        for rec, coeff in terms.items():
            ref = f"rec[{rec - self.num_records}]"
            sign = "-" if coeff < 0 else "+"
            mag = abs(coeff)
            parts.append(f"{sign} {ref}" if mag == 1 else f"{sign} {mag}*{ref}")
        text = " ".join(parts)
        return text[2:] if text.startswith("+ ") else text

    def detector(self, terms):
        self.circuit.add_gate("DETECTOR", expr=self.expr(terms))

    def observable(self, terms):
        self.circuit.add_gate("LOGICAL_OBSERVABLE", expr=self.expr(terms))


def random_circuit(d, seed, p=0.01, num_qudits=None, num_gates=None, noise_rate=0.4):
    """
    Two rounds of prepare / C / C^-1 / measure on 4 to 6 qudits.

    Each round's C holds every gate of CLIFFORDS at least once (20 to 40
    gates in all), every N1 channel and N2 appear, at least one X-basis qudit
    is read out with M_X and one with H_INV + M, and between the rounds at
    least one qudit is RESET and at least one keeps its measured state.
    """
    rng = np.random.default_rng(seed)
    n = int(num_qudits if num_qudits is not None else rng.integers(4, 7))
    b = _Builder(n, d, rng, p)

    perm = [int(q) for q in rng.permutation(n)]
    num_x = int(rng.integers(2, n))  # at least two X-basis and one Z-basis qudit
    x_basis = set(perm[:num_x])

    for q in range(n):
        b.gate("RESET", q)
    for kind in NOISE_KINDS:
        b.noise(kind)

    last_record = [None] * n
    for rnd in range(2):
        for q in sorted(x_basis):
            b.gate("H", q)
            b.maybe_noise(noise_rate)

        count = int(num_gates if num_gates is not None else rng.integers(20, 41))
        extra = rng.integers(len(CLIFFORDS), size=count - len(CLIFFORDS))
        names = list(CLIFFORDS) + [CLIFFORDS[int(i)] for i in extra]
        rng.shuffle(names)
        sequence = []
        for name in names:
            if name in ONE_QUDIT_CLIFFORDS:
                qudits = (int(rng.integers(n)),)
            else:
                qudits = tuple(int(q) for q in rng.choice(n, size=2, replace=False))
            sequence.append((name, qudits))
            b.gate(name, *qudits)
            b.maybe_noise(noise_rate)
        for name, qudits in reversed(sequence):
            b.gate(INVERSE[name], *qudits)
            b.maybe_noise(noise_rate)

        # X-basis readout: one qudit with M_X, one with H_INV + M, the rest at random.
        xs = [q for q in perm if q in x_basis]
        style = {xs[0]: "M_X", xs[1]: "H_INV+M"}
        for q in xs[2:]:
            style[q] = "M_X" if rng.random() < 0.5 else "H_INV+M"
        round_records = {}
        for q in (int(v) for v in rng.permutation(n)):
            b.maybe_noise(noise_rate)
            if q in x_basis and style[q] == "M_X":
                rec = b.measure("M_X", q)
            else:
                if q in x_basis:
                    b.gate("H_INV", q)
                rec = b.measure("M", q)
            round_records[q] = rec
            terms = {rec: 1}
            if last_record[q] is not None and rng.random() < 0.75:
                terms[last_record[q]] = -1
            b.detector(terms)
            last_record[q] = rec

        if rnd == 0:
            first_round_records = dict(round_records)
            # An observable between the rounds, interleaved with detectors.
            qs = [int(q) for q in rng.choice(n, size=2, replace=False)]
            b.observable({round_records[qs[0]]: 1, round_records[qs[1]]: -1})
            # Mid-circuit RESET on some qudits; the others carry their state over.
            order = [int(q) for q in rng.permutation(n)]
            to_reset = [order[0]] + [q for q in order[2:] if rng.random() < 0.5]
            for q in to_reset:
                b.gate("RESET", q)
                b.maybe_noise(noise_rate)

    # A final observable with coefficients other than +-1 that spans both rounds.
    qs = [int(q) for q in rng.choice(n, size=3, replace=False)]
    b.observable({last_record[qs[0]]: 1, last_record[qs[1]]: 2, first_round_records[qs[2]]: -1})
    return b.circuit


# --------------------------------------------------------------------------
# sdim's frame simulator


# The reference shot comes from sdim's tableau, run without noise like the
# reference shot of Program.simulate(). simulate_frame only copies the reference
# values into its per-shot measurement results; detection events and observable
# shifts come from the frame alone.


def _tableau_reference(circuit):
    prog = Program(circuit)
    prog._tableau_noise_enabled = False
    prog._simulate_tableau(SimulationOptions(shots=1))
    return prog


def _reference_results(circuit):
    prog = _tableau_reference(circuit)
    return prog._results_to_array(prog.measurement_results)


def _sdim_frame(circuit, shots, noise=None, seed=0):
    """Detection events and observable shifts from sdim's own frame simulator, mod d."""
    np.random.seed(seed)
    ref = _reference_results(circuit)
    prog = Program(circuit)
    ir, sampled, info = prog._build_ir(prog.circuits, shots)
    _, det = simulate_frame(ir, ref, circuit.num_qudits, circuit.dimension, shots,
                            sampled if noise is None else noise, info)
    d = circuit.dimension
    return np.asarray(det.detection_events) % d, np.asarray(det.logical_operator_shifts) % d


def _num_noise_gates(circuit):
    return sum(1 for op in circuit.operations if op.name in ("N1", "N2"))


def assert_deterministic(circuit, shots=64, seed=5):
    """Noiseless frame run (noise array zeroed): every detector and observable must read 0.

    The frame starts with a random Z part and measurements re-randomize it, so
    a detector that depends on a random outcome shows up as non-zero here.
    """
    noise = np.zeros((max(_num_noise_gates(circuit), 1), shots, 4), dtype=np.int64)
    det, obs = _sdim_frame(circuit, shots, noise, seed=seed)
    assert det.shape[0] > 0 and obs.shape[0] > 0
    assert not det.any(), "random circuit has a non-deterministic detector"
    assert not obs.any(), "random circuit has a non-deterministic observable"


def _tableau_outcomes(circuit):
    prog = _tableau_reference(circuit)
    return [(int(r[0].measurement_value), bool(r[0].deterministic))
            for rounds in prog.measurement_results for r in rounds]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("d", [2, 3, 5, 1000003])
def test_random_circuits_are_deterministic_and_cover_every_gate(d, seed):
    c = random_circuit(d, seed)
    assert 4 <= c.num_qudits <= 6
    assert COVERED_GATES <= {op.name for op in c.operations}
    assert {op.params["noise_channel"] for op in c.operations if op.name == "N1"} == {"d", "f", "p"}
    assert_deterministic(c)
    # The reference tableau shot agrees: every measurement and reset is
    # deterministic, with outcome 0 since C^-1 undoes C.
    assert set(_tableau_outcomes(c)) == {(0, True)}


# A C from a random search on 5 qudits. RESET, H on qudits 0 and 1, C, C^-1,
# H_INV on 0 and 1, then M on every qudit must read all zeros. Before the tableau
# reduced mod d after every gate, int64 overflow broke this at d = 101 and 1000003.
_OVERFLOW_SEQUENCE = [
    ("SWAP", 1, 2), ("SWAP", 2, 0), ("CZ", 2, 0), ("P_INV", 0), ("SWAP", 4, 2), ("P_INV", 0),
    ("SWAP", 3, 1), ("SWAP", 4, 0), ("CNOT_INV", 4, 2), ("SWAP", 4, 2), ("CZ", 0, 4), ("P", 3),
    ("CNOT_INV", 2, 1), ("P", 3), ("P_INV", 3), ("SWAP", 0, 4), ("P", 1), ("SWAP", 3, 4), ("H", 1),
    ("CNOT_INV", 1, 3), ("CZ", 4, 0), ("H_INV", 3), ("H_INV", 2), ("SWAP", 4, 1), ("CNOT", 4, 3),
    ("CNOT_INV", 0, 1), ("CZ_INV", 1, 4), ("CZ_INV", 2, 4), ("P_INV", 0), ("SWAP", 3, 1), ("SWAP", 2, 1),
]


def _overflow_circuit(d):
    c = Circuit(5, d)
    c.add_gate("RESET", [0, 1, 2, 3, 4])
    c.add_gate("H", [0, 1])
    for name, *qudits in _OVERFLOW_SEQUENCE:
        c.add_gate(name, *qudits)
    for name, *qudits in reversed(_OVERFLOW_SEQUENCE):
        c.add_gate(INVERSE[name], *qudits)
    c.add_gate("H_INV", [0, 1])
    c.add_gate("M", [0, 1, 2, 3, 4])
    return c


@pytest.mark.parametrize("make", [lambda: random_circuit(7, 11),
                                  lambda: _overflow_circuit(101),
                                  lambda: _overflow_circuit(1000003)],
                         ids=["d7-random_circuit-seed11", "d101", "d1000003"])
def test_reference_tableau_has_no_overflow(make):
    """Every outcome of these noiseless circuits is deterministically 0."""
    assert set(_tableau_outcomes(make())) == {(0, True)}


# --------------------------------------------------------------------------
# Unit responses against the frame simulator


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("d", DIMENSIONS)
def test_unit_responses_match_frame_simulator_random_gates(d, seed):
    """Each unit fault, injected alone into sdim's frame simulator (one shot per
    probe), changes the detectors and observables exactly as
    `compile_unit_responses` says."""
    c = random_circuit(d, seed)
    assert_deterministic(c)
    compiled = compile_unit_responses(c)
    assert len(compiled.locations) == _num_noise_gates(c)
    probes, expected = [], []
    for g, loc in enumerate(compiled.locations):
        comps = {"d": [0, 1], "f": [0], "p": [1], "d2": [0, 1, 2, 3]}[loc.channel]
        assert len(loc.responses) == len(comps)
        for k, comp in enumerate(comps):
            probes.append((g, comp))
            expected.append(loc.responses[k])
    noise = np.zeros((len(compiled.locations), len(probes), 4), dtype=np.int64)
    for j, (g, comp) in enumerate(probes):
        noise[g, j, comp] = 1
    det, obs = _sdim_frame(c, len(probes), noise)
    nd = compiled.num_detectors
    assert det.shape == (nd, len(probes))
    assert obs.shape == (compiled.num_observables, len(probes))
    visible = 0
    for j, resp in enumerate(expected):
        ref = {i: int(v) for i, v in enumerate(det[:, j]) if v}
        ref.update({nd + i: int(v) for i, v in enumerate(obs[:, j]) if v})
        g, comp = probes[j]
        assert ref == resp, (j, compiled.locations[g].source, "XZXZ"[comp], ref, resp)
        visible += bool(resp)
    # Guard against a vacuous pass: most faults must reach some detector.
    assert visible > len(probes) // 2


@pytest.mark.parametrize("d", [3, 1000003])
def test_unit_responses_scale_linearly(d):
    """A fault of power a has a times the unit response, mod d (checked in the frame simulator)."""
    c = random_circuit(d, 101)
    assert_deterministic(c)
    compiled = compile_unit_responses(c)
    rng = np.random.default_rng(5)
    probes = []
    for g, loc in enumerate(compiled.locations):
        comps = {"d": [0, 1], "f": [0], "p": [1], "d2": [0, 1, 2, 3]}[loc.channel]
        k = int(rng.integers(len(comps)))
        probes.append((g, comps[k], loc.responses[k], int(rng.integers(1, d))))
    noise = np.zeros((len(compiled.locations), len(probes), 4), dtype=np.int64)
    for j, (g, comp, _, power) in enumerate(probes):
        noise[g, j, comp] = power
    det, obs = _sdim_frame(c, len(probes), noise)
    nd = compiled.num_detectors
    for j, (g, comp, resp, power) in enumerate(probes):
        ref = {i: int(v) for i, v in enumerate(det[:, j]) if v}
        ref.update({nd + i: int(v) for i, v in enumerate(obs[:, j]) if v})
        scaled = {t: (power * v) % d for t, v in resp.items() if (power * v) % d}
        assert ref == scaled, (j, compiled.locations[g].source, power)


# --------------------------------------------------------------------------
# DEM sampling against frame sampling


@pytest.mark.parametrize("d,seed", [(3, 7), (3, 19), (1000003, 8), (1000003, 29)])
def test_dem_sampler_matches_frame_sampler_random_gates(d, seed):
    """`DetectorErrorModel.sample` and frame sampling agree on firing rates,
    pairwise co-firing rates and, at small d, joint values of target pairs."""
    p = 0.01
    c = random_circuit(d, seed, p=p, num_qudits=4, num_gates=20)
    assert_deterministic(c)
    dem = DetectorErrorModel.from_circuit(c)
    shots = 40000
    ddem, odem = dem.sample(shots, seed=seed)
    dfr, ofr = _sdim_frame(c, shots, seed=seed + 100)
    vals_dem = np.concatenate([ddem, odem], axis=1)
    vals_fr = np.concatenate([dfr.T, ofr.T], axis=1)
    a, b = vals_dem != 0, vals_fr != 0
    rates = a.mean(axis=0)
    assert 0.02 < rates.max() < 0.5, rates  # the comparison is not vacuous

    def close(pa, pb):
        sigma = math.sqrt((max(pa * (1 - pa), 1e-4) + max(pb * (1 - pb), 1e-4)) / shots)
        return abs(pa - pb) < 5 * sigma

    for i in range(a.shape[1]):
        pa, pb = a[:, i].mean(), b[:, i].mean()
        assert close(pa, pb), (i, pa, pb)
    # Pairwise co-firing checks the correlations (hyperedges from N2 and from
    # faults that reach several detectors).
    ca = (a[:, :, None] & a[:, None, :]).mean(axis=0)
    cb = (b[:, :, None] & b[:, None, :]).mean(axis=0)
    for i in range(a.shape[1]):
        for j in range(i + 1, a.shape[1]):
            assert close(ca[i, j], cb[i, j]), (i, j, ca[i, j], cb[i, j])
    if d <= 7:
        # Small d: the joint distribution of the values of every pair of
        # targets. A mechanism adds a uniform multiple of each generator, so
        # one target's distribution is symmetric under v -> -v. Pairs are
        # where relative signs and coefficients show up.
        ind_a = [(vals_dem == v).astype(np.float64) for v in range(d)]
        ind_b = [(vals_fr == v).astype(np.float64) for v in range(d)]
        for v in range(1, d):
            for w in range(1, d):
                ja = ind_a[v].T @ ind_a[w] / shots
                jb = ind_b[v].T @ ind_b[w] / shots
                for i in range(a.shape[1]):
                    for j in range(a.shape[1]):
                        if i < j or (i == j and v == w):
                            assert close(ja[i, j], jb[i, j]), (i, j, v, w, ja[i, j], jb[i, j])


# --------------------------------------------------------------------------
# d = 2 against stim


def _to_stim(circuit):
    """
    Translates a d = 2 sdim circuit to stim.

    At d = 2 the sdim tableau has H = H_INV = H, P = S (X -> i XZ = Y) and
    P_INV = S_DAG, CNOT = CNOT_INV = CX and CZ = CZ_INV = CZ.

    sdim's M_X applies H_INV and then measures Z, and leaves the qudit in the
    rotated basis (the frame simulator does the same). stim's MX leaves the
    qubit in an X eigenstate, so M_X becomes `MX q` followed by `H q`, which is
    the same channel as `H q` then `M q`.
    """
    names = {"H": "H", "H_INV": "H", "P": "S", "P_INV": "S_DAG", "CNOT": "CX", "CNOT_INV": "CX",
             "CZ": "CZ", "CZ_INV": "CZ", "SWAP": "SWAP", "M": "M", "RESET": "R"}
    lines = []
    num_observables = 0
    for op in circuit.operations:
        n = op.name
        qs = [op.qudit_index] + ([op.target_index] if op.target_index is not None else [])
        if n in names:
            lines.append(f"{names[n]} " + " ".join(map(str, qs)))
        elif n == "M_X":
            lines.append(f"MX {op.qudit_index}")
            lines.append(f"H {op.qudit_index}")
        elif n == "N1":
            gate = {"d": "DEPOLARIZE1", "f": "X_ERROR", "p": "Z_ERROR"}[op.params["noise_channel"]]
            lines.append(f"{gate}({op.params['prob']}) {op.qudit_index}")
        elif n == "N2":
            lines.append(f"DEPOLARIZE2({op.params['prob']}) {op.qudit_index} {op.target_index}")
        elif n in ("DETECTOR", "LOGICAL_OBSERVABLE"):
            # Keep the records with an odd coefficient.
            targets = []
            for coeff, rec in re.findall(r"(?:(\d+)\*)?rec\[(-?\d+)\]", op.params["expr"]):
                if int(coeff or 1) % 2:
                    targets.append(f"rec[{rec}]")
            if n == "DETECTOR":
                lines.append("DETECTOR " + " ".join(targets))
            else:
                lines.append(f"OBSERVABLE_INCLUDE({num_observables}) " + " ".join(targets))
                num_observables += 1
        else:
            raise AssertionError(f"no stim translation for {n}")
    return "\n".join(lines)


@pytest.mark.parametrize("seed", SEEDS)
def test_matches_stim_at_qubit_dimension_random_gates(seed):
    """At d = 2, the line expansion of the DEM equals stim's detector error
    model: same symptom sets, same probabilities. No gate is left out."""
    stim = pytest.importorskip("stim")
    c = random_circuit(2, seed, p=0.01)
    assert_deterministic(c)
    stim_dem = stim.Circuit(_to_stim(c)).detector_error_model(flatten_loops=True)
    expected = {}
    for inst in stim_dem:
        if inst.type != "error":
            continue
        key = frozenset(("D" if t.is_relative_detector_id() else "L") + str(t.val)
                        for t in inst.targets_copy() if not t.is_separator())
        q = inst.args_copy()[0]
        expected[key] = expected.get(key, 0.0) * (1 - q) + (1 - expected.get(key, 0.0)) * q
    expected.pop(frozenset(), None)
    ours = {}
    lines = DetectorErrorModel.from_circuit(c).to_lines()
    assert lines.num_detectors == stim_dem.num_detectors
    assert lines.num_observables == stim_dem.num_observables
    nd = lines.num_detectors
    for m in lines.mechanisms:
        key = frozenset(("D" + str(t)) if t < nd else ("L" + str(t - nd)) for t in m.generators[0])
        assert key not in ours
        ours[key] = m.probability / 2  # a fired line {I, P} applies P with prob pi/2
    assert set(ours) == set(expected)
    for key in expected:
        assert math.isclose(ours[key], expected[key], rel_tol=1e-9), key
