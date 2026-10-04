"""Tests for the Pauli frame sampler in sdim.program.

The frame kernel is checked exactly against a slow reference implementation of the frame update
rules that uses Python integers (so it cannot overflow) and draws its random Z frames in the same
order, and the noise it samples by itself is checked statistically against the exact channel
distributions.
"""

import math
import random

import numpy as np
import pytest

from sdim.circuit import Circuit
from sdim.program import (MEASUREMENT_DTYPE, NoiseModel, Program, SimulationOptions,
                          _detector_mod, simulate_frame)
from sdim.tableau.dataclasses import MeasurementResult


# --------------------------------------------------------------------------
# Reference implementation


def _reference_frame(ir_array, reference_results, n_qudits, d, shots, noise_array, detector_info):
    """
    The frame update rules with exact integers, drawing random Z frames from np.random in the order
    simulate_frame does: one (n_qudits, shots) draw, then one row per M / M_X / RESET.  M_X is
    H_INV, a Z measurement, then H.

    Returns {(qudit, round): values} for every measurement record, and the detector and observable
    rows.
    """
    x = np.zeros((n_qudits, shots), dtype=object)
    z = np.random.randint(0, d, size=(n_qudits, shots)).astype(object)
    counts = [0] * n_qudits
    shifts = []
    values = {}
    noise_counter = 0
    detectors, observables = [], []
    det_index = 0
    for gate_id, a, b, scalar in ir_array.tolist():
        if gate_id == 5:
            x[a], z[a] = -z[a], x[a].copy()
        elif gate_id == 6:
            x[a], z[a] = z[a].copy(), -x[a]
        elif gate_id == 7:
            z[a] = z[a] + x[a]
        elif gate_id == 8:
            z[a] = z[a] - x[a]
        elif gate_id == 9:
            x[b] = x[b] + x[a]
            z[a] = z[a] - z[b]
        elif gate_id == 10:
            x[b] = x[b] - x[a]
            z[a] = z[a] + z[b]
        elif gate_id == 11:
            z[b] = z[b] + x[a]
            z[a] = z[a] + x[b]
        elif gate_id == 12:
            z[b] = z[b] - x[a]
            z[a] = z[a] - x[b]
        elif gate_id == 13:
            x[a], x[b] = x[b].copy(), x[a].copy()
            z[a], z[b] = z[b].copy(), z[a].copy()
        elif gate_id in (14, 15, 16):
            if gate_id == 15:
                x[a], z[a] = z[a].copy(), -x[a]
            m = counts[a]
            ref = int(reference_results[a, m]['measurement_value'])
            values[(a, m)] = np.array([(ref + v) % d for v in x[a]], dtype=np.int64)
            if gate_id != 16:
                shifts.append(np.array([v % d for v in x[a]], dtype=np.int64))
            else:
                x[a] = 0
            counts[a] += 1
            z[a] = np.random.randint(0, d, size=shots).astype(object)
            if gate_id == 15:   # rotate back with H: the qudit is left in an X eigenstate
                x[a], z[a] = -z[a], x[a].copy()
        elif gate_id == 17:
            x[a] = x[a] + noise_array[noise_counter, :, 0].astype(object)
            z[a] = z[a] + noise_array[noise_counter, :, 1].astype(object)
            noise_counter += 1
        elif gate_id == 18:
            x[a] = x[a] + noise_array[noise_counter, :, 0].astype(object)
            z[a] = z[a] + noise_array[noise_counter, :, 1].astype(object)
            x[b] = x[b] + noise_array[noise_counter, :, 2].astype(object)
            z[b] = z[b] + noise_array[noise_counter, :, 3].astype(object)
            noise_counter += 1
        elif gate_id == 22:
            s = scalar % d
            x[a] = x[a] * s
            z[a] = z[a] * pow(s, -1, d)
        elif gate_id in (19, 20):
            function_index, _, arguments, _ = detector_info.detector_data[det_index]
            det_index += 1
            row = np.asarray(detector_info.detector_functions[function_index]([shifts[k] for k in arguments]))
            (detectors if gate_id == 19 else observables).append(row)
    return values, detectors, observables


ONE_QUDIT = ["H", "H_INV", "P", "P_INV", "X", "X_INV", "Z", "Z_INV"]
TWO_QUDIT = ["CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"]


def _random_circuit(rng, d, n, length):
    """Random circuit with every gate the frame sampler handles, plus detectors (some non-linear)."""
    c = Circuit(n, d)
    meas = 0
    for step in range(length):
        r = rng.random()
        if r < 0.25:
            c.add_gate(rng.choice(ONE_QUDIT), rng.randrange(n))
        elif r < 0.5:
            a, b = rng.sample(range(n), 2)
            c.add_gate(rng.choice(TWO_QUDIT), a, b)
        elif r < 0.6:
            c.add_gate(rng.choice(["M", "M", "M_X"]), rng.randrange(n))
            meas += 1
        elif r < 0.64:
            c.add_gate("RESET", rng.randrange(n))
        elif r < 0.68:
            while True:
                a = rng.randrange(1, min(d, 10 ** 9))
                if math.gcd(a, d) == 1:
                    break
            c.add_gate("MUL", rng.randrange(n), a=a)
        elif r < 0.78:
            c.add_gate("N1", rng.randrange(n), prob=rng.choice([0.0, 0.3, 1.0]), noise_channel=rng.choice("dfp"))
        elif r < 0.85:
            a, b = rng.sample(range(n), 2)
            c.add_gate("N2", a, b, prob=rng.choice([0.0, 0.3, 1.0]))
        elif r < 0.95 and meas >= 2:
            idx = rng.sample(range(1, meas + 1), rng.randrange(1, min(meas, 4) + 1))
            expr = " + ".join(f"{rng.choice([1, -1, 2, d - 1])}*rec[-{i}]" for i in idx)
            if rng.random() < 0.2:
                expr = f"({expr}) * rec[-1]"
            c.add_gate("DETECTOR" if rng.random() < 0.8 else "LOGICAL_OBSERVABLE", expr=expr)
        else:
            c.add_gate("TICK")
    for q in range(n):
        c.add_gate(rng.choice(["M", "M_X"]), q)
    c.add_gate("DETECTOR", expr="rec[-1] - rec[-2]")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-1]")
    return c


def _reference_shot(circuit, seed):
    """The noiseless reference shot of Program.simulate's frame mode, as a structured array."""
    random.seed(seed)
    np.random.seed(seed)
    prog = Program(circuit)
    prog._tableau_noise_enabled = False
    prog._simulate_tableau(SimulationOptions(shots=1))
    return prog, prog._results_to_array(prog.measurement_results)


def _record_slots(circuit):
    counts = [0] * circuit.num_qudits
    slots = []
    for op in circuit.operations:
        if op.name in ("M", "M_X", "RESET"):
            slots.append((op.qudit_index, counts[op.qudit_index]))
            counts[op.qudit_index] += 1
    return slots


def _check_against_reference(circuit, ref, ir, info, shots, noise, seed):
    d = circuit.dimension
    np.random.seed(seed)
    frame, det = simulate_frame(ir, ref, circuit.num_qudits, d, shots, noise, info)
    after_new = np.random.randint(0, 2 ** 31)
    np.random.seed(seed)
    values, detectors, observables = _reference_frame(ir, ref, circuit.num_qudits, d, shots, noise, info)
    after_ref = np.random.randint(0, 2 ** 31)
    assert after_new == after_ref, "simulate_frame consumed np.random differently"
    for q, m in _record_slots(circuit):
        out = frame[q, m]
        np.testing.assert_array_equal(out['measurement_value'], values[(q, m)])
        assert (out['qudit_index'] == q).all() and (out['meas_round'] == m).all()
        np.testing.assert_array_equal(out['shot'], np.arange(shots))
        assert (out['deterministic'] == ref[q, m]['deterministic']).all()
    assert det.detection_events.dtype == np.int64 and det.logical_operator_shifts.dtype == np.int64
    np.testing.assert_array_equal(det.detection_events, np.array(detectors, dtype=np.int64).reshape(-1, shots))
    np.testing.assert_array_equal(det.logical_operator_shifts, np.array(observables, dtype=np.int64).reshape(-1, shots))


# --------------------------------------------------------------------------
# Exact checks with injected noise


@pytest.mark.parametrize("d", [2, 3, 4, 5, 6, 7, 9, 1000003, 2147483647])
@pytest.mark.parametrize("seed", range(6))
def test_injected_noise_matches_exact_reference(d, seed):
    rng = random.Random(1000 * d + seed)
    n = rng.randrange(2, 6)
    c = _random_circuit(rng, d, n, rng.randrange(20, 160))
    shots = rng.choice([1, 5, 33, 100])
    _, ref = _reference_shot(c, seed)
    ir, sampled, info = Program._build_ir([c], shots)
    num_noise = sum(1 for op in c.operations if op.name in ("N1", "N2"))
    gen = np.random.RandomState(seed)
    injected = gen.randint(-d, 2 * d, size=(max(num_noise, 1), shots, 4)).astype(np.int64)
    for noise in (sampled, injected, np.zeros_like(injected)):
        _check_against_reference(c, ref, ir, info, shots, noise, seed + 17)


def _alternating_cnot_circuit(d, rounds, fault, x_basis):
    """Unit fault on qudit 0, then CNOT(0, 1), CNOT(1, 0) repeated, then both qudits measured."""
    c = Circuit(2, d)
    if x_basis:
        c.add_gate("H", [0, 1])
    c.add_gate("N1", 0, prob=1.0, noise_channel="f" if fault == "x" else "p")
    for _ in range(rounds):
        c.add_gate("CNOT", 0, 1)
        c.add_gate("CNOT", 1, 0)
    if x_basis:
        c.add_gate("H", [0, 1])
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="rec[-2]")
    c.add_gate("DETECTOR", expr="rec[-1]")
    return c


@pytest.mark.parametrize("fault,x_basis", [("x", False), ("z", True)])
def test_long_cnot_chain_does_not_overflow(fault, x_basis):
    """
    Back-and-forth CNOTs grow the frame like Fibonacci numbers.  The sampler used to reduce the
    frame mod d only every 64 gates, which overflowed int64 inside one window at d = 2**31 - 1.
    """
    d = 2147483647
    rounds = 300
    c = _alternating_cnot_circuit(d, rounds, fault, x_basis)
    shots = 4
    _, ref = _reference_shot(c, 0)
    ir, _, info = Program._build_ir([c], shots)
    noise = np.zeros((1, shots, 4), dtype=np.int64)
    noise[0, :, 0 if fault == "x" else 1] = 1

    if fault == "x":
        # x0, x1 under CNOT(0, 1) then CNOT(1, 0)
        a, b = 1, 0
        for _ in range(rounds):
            b += a
            a += b
        expected = (a % d, b % d)
    else:
        # z0, z1 under CNOT(0, 1) (z0 -= z1) then CNOT(1, 0) (z1 -= z0); the final H maps z to -z in X
        a, b = 1, 0
        for _ in range(rounds):
            a -= b
            b -= a
        expected = (-a % d, -b % d)
    assert max(abs(a), abs(b)) > 2 ** 200  # the chain is long enough to overflow int64 many times over

    np.random.seed(1)
    frame, det = simulate_frame(ir, ref, 2, d, shots, noise, info)
    for q in (0, 1):
        assert ref[q, 0]['measurement_value'] == 0 and ref[q, 0]['deterministic']
        np.testing.assert_array_equal(frame[q, 0]['measurement_value'], np.full(shots, expected[q]))
        np.testing.assert_array_equal(det.detection_events[q], np.full(shots, expected[q]))
    _check_against_reference(c, ref, ir, info, shots, noise, 5)


def test_large_dimension_chain_with_sampled_noise_stays_in_range():
    """The sampled-noise path at d = 2**31 - 1 on the same chain: outcomes stay in [0, d) and the fault scales linearly."""
    d = 2147483647
    c = _alternating_cnot_circuit(d, 300, "x", False)
    np.random.seed(3)
    measurements, det = Program(c).simulate(shots=200, raw_detector_output=True)
    detectors, _ = det
    a, b = 1, 0
    for _ in range(300):
        b += a
        a += b
    # N1 'f' with prob 1 applies X^e with e uniform in 1..d-1, and the chain is linear in e.
    values0 = np.array([r.measurement_value for r in measurements[0][0][1:]])
    values1 = np.array([r.measurement_value for r in measurements[1][0][1:]])
    assert ((values0 >= 0) & (values0 < d)).all() and ((values1 >= 0) & (values1 < d)).all()
    np.testing.assert_array_equal(detectors[0], values0)
    np.testing.assert_array_equal(detectors[1], values1)
    # Recover e from qudit 1 and check qudit 0 against it.
    assert b % d != 0
    e = np.array([int(v) * pow(b % d, -1, d) % d for v in values1])
    assert ((e > 0) & (e < d)).all()
    np.testing.assert_array_equal(values0, np.array([int(v) * a % d for v in e]))


# --------------------------------------------------------------------------
# Sampled noise: exact channel distributions


def _bell_readout_circuit(d, add_noise):
    """
    Qudit pairs (0, 2) and (1, 3) are prepared in Bell states, noise acts on qudits 0 and 1, and the
    Bell states are undone, so the outcomes read the noise exactly: qudit 0 gives the Z power on
    qudit 0, qudit 2 gives minus its X power, and likewise for qudits 1 and 3.
    """
    c = Circuit(4, d)
    for a, b in ((0, 2), (1, 3)):
        c.add_gate("H", a)
        c.add_gate("CNOT", a, b)
    add_noise(c)
    for a, b in ((0, 2), (1, 3)):
        c.add_gate("CNOT_INV", a, b)
        c.add_gate("H_INV", a)
    c.add_gate("M", [0, 1, 2, 3])
    return c


def _sampled_paulis(c, shots, seed):
    """(x0, z0, x1, z1) of the noise in each frame shot of the Bell readout circuit."""
    np.random.seed(seed)
    random.seed(seed)
    measurements, _ = Program(c).simulate(shots=shots + 1)
    d = c.dimension
    value = lambda q: np.array([r.measurement_value for r in measurements[q][0][1:]], dtype=np.int64)
    assert all(r.measurement_value == 0 for q in range(4) for r in measurements[q][0][:1])
    return np.stack([(-value(2)) % d, value(0), (-value(3)) % d, value(1)], axis=1)


def _assert_distribution(samples, probabilities, d):
    """Every cell of the empirical distribution of 4-tuples is within 5 sigma of its probability."""
    shots = samples.shape[0]
    index = ((samples[:, 0] * d + samples[:, 1]) * d + samples[:, 2]) * d + samples[:, 3]
    counts = np.bincount(index, minlength=d ** 4)
    expected = shots * np.asarray(probabilities)
    sigma = np.sqrt(np.maximum(shots * np.asarray(probabilities) * (1 - np.asarray(probabilities)), 1e-12))
    bad = np.flatnonzero(np.abs(counts - expected) > 5 * sigma + 1e-9)
    assert bad.size == 0, f"cells {bad[:5]}: counts {counts[bad[:5]]}, expected {expected[bad[:5]]}"


def _n1_probabilities(d, channel, p):
    probs = np.zeros(d ** 4)
    probs[0] = 1 - p
    for xa in range(d):
        for za in range(d):
            if (xa, za) == (0, 0):
                continue
            if channel == "d":
                probs[(xa * d + za) * d * d] = p / (d * d - 1)
            elif channel == "f" and za == 0:
                probs[(xa * d) * d * d] = p / (d - 1)
            elif channel == "p" and xa == 0:
                probs[za * d * d] = p / (d - 1)
    return probs


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("channel", ["d", "f", "p"])
@pytest.mark.parametrize("p", [0.3, 1.0])
def test_sampled_n1_distribution(d, channel, p):
    c = _bell_readout_circuit(d, lambda c: c.add_gate("N1", 0, prob=p, noise_channel=channel))
    samples = _sampled_paulis(c, 20000, seed=11)
    _assert_distribution(samples, _n1_probabilities(d, channel, p), d)


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("p", [0.2, 1.0])
def test_sampled_n2_uniform_distribution(d, p):
    c = _bell_readout_circuit(d, lambda c: c.add_gate("N2", 0, 1, prob=p))
    samples = _sampled_paulis(c, 20000, seed=12)
    probs = np.full(d ** 4, p / (d ** 4 - 1))
    probs[0] = 1 - p
    _assert_distribution(samples, probs, d)


@pytest.mark.parametrize("d", [2, 3])
def test_sampled_n2_prob_dist_distribution(d):
    gen = np.random.default_rng(5)
    dist = gen.random(d ** 4) ** 3
    dist[gen.random(d ** 4) < 0.3] = 0.0
    dist /= dist.sum()
    c = _bell_readout_circuit(d, lambda c: c.add_gate("N2", 0, 1, prob_dist=list(dist)))
    samples = _sampled_paulis(c, 20000, seed=13)
    _assert_distribution(samples, dist, d)


def test_sampled_noise_at_large_dimension_is_never_identity_when_it_fires():
    d = 1000003
    c = _bell_readout_circuit(d, lambda c: (c.add_gate("N1", 0, prob=1.0, noise_channel="d"),
                                            c.add_gate("N2", 0, 1, prob=1.0)))
    samples = _sampled_paulis(c, 3000, seed=14)
    assert ((samples >= 0) & (samples < d)).all()
    # Two independent uniform non-identity Paulis on qudit 0, so (x0, z0) is close to uniform.
    for column in range(4):
        mean = samples[:, column].mean() / d
        assert abs(mean - 0.5) < 5 * math.sqrt(1 / 12 / samples.shape[0])


def test_geometric_skipping_matches_bernoulli_rate():
    """Errors of a low-probability gate land on the right fraction of shots, in every block."""
    d = 3
    p = 0.002
    c = Circuit(1, d)
    for _ in range(50):
        c.add_gate("N1", 0, prob=p, noise_channel="f")
        c.add_gate("M", 0)
        c.add_gate("RESET", 0)
    np.random.seed(21)
    shots = 20000
    measurements, _ = Program(c).simulate(shots=shots + 1)
    fired = np.array([[r.measurement_value != 0 for r in rnd[1:]] for rnd in measurements[0][0::2]])
    assert fired.shape == (50, shots)
    total = fired.sum()
    expected = 50 * shots * p
    assert abs(total - expected) < 5 * math.sqrt(expected * (1 - p))
    # No position bias: every run of 1000 shots gets its share too.
    per_block = fired.reshape(50, -1, 1000).sum(axis=(0, 2))
    assert abs(per_block - 50 * 1000 * p).max() < 5 * math.sqrt(50 * 1000 * p) + 1


def test_sampled_noise_is_reproducible_with_numpy_seed():
    rng = random.Random(4)
    c = _random_circuit(rng, 5, 4, 80)

    def run():
        np.random.seed(99)
        random.seed(99)
        measurements, det = Program(c).simulate(shots=300, raw_detector_output=True)
        values = [[[r.measurement_value for r in shots] for shots in rounds] for rounds in measurements]
        return values, det

    v1, d1 = run()
    v2, d2 = run()
    assert v1 == v2
    np.testing.assert_array_equal(d1[0], d2[0])
    np.testing.assert_array_equal(d1[1], d2[1])


# --------------------------------------------------------------------------
# _build_ir with lazy noise


def test_build_ir_without_sampling_returns_noise_model():
    c = Circuit(3, 3)
    c.add_gate("N1", 0, prob=0.25, noise_channel="d")
    c.add_gate("N1", 1, prob=0.0, noise_channel="f")
    c.add_gate("N1", 2, prob=1.0, noise_channel="p")
    c.add_gate("N2", 0, 1, prob=0.5)
    dist = [0.5] + [0.5 / 80] * 80
    c.add_gate("N2", 1, 2, prob_dist=dist)
    c.add_gate("N2", 0, 2, prob_dist=dist)
    c.add_gate("M", [0, 1, 2])
    c.add_gate("DETECTOR", expr="rec[-1] - rec[-2]")
    state = np.random.get_state()
    ir, noise, info, model = Program._build_ir([c], 10, sample_noise=False)
    after = np.random.get_state()
    assert noise is None
    assert state[2] == after[2] and np.array_equal(state[1], after[1]), "lazy _build_ir must not draw"
    ir_sampled, noise_sampled, info_sampled = Program._build_ir([c], 10)
    np.testing.assert_array_equal(ir, ir_sampled)
    assert noise_sampled.shape == (6, 10, 4)
    assert info.detector_data == info_sampled.detector_data
    assert isinstance(model, NoiseModel)
    np.testing.assert_array_equal(model.kind, [0, 1, 2, 3, 4, 4])
    np.testing.assert_array_equal(model.mode, [2, 0, 1, 2, 3, 3])
    assert model.log_q[0] == pytest.approx(math.log(0.75)) and model.log_q[3] == pytest.approx(math.log(0.5))
    # The two gates share one distribution object, so its cumulative distribution is stored once.
    assert model.cdf_offset[4] == model.cdf_offset[5] == 0 and model.cdf_data.shape == (81,)
    np.testing.assert_allclose(model.cdf_data, np.cumsum(dist) / np.sum(dist))


@pytest.mark.parametrize("dist,message", [
    ([1.0, 0.0], "prob_dist has length"),
    ([0.5] * 81, "do not sum to 1"),
    ([-0.5, 1.5] + [0.0] * 79, "non-negative"),
    ([float("nan")] + [0.0] * 80, "NaN"),
])
def test_prob_dist_validation_is_the_same_with_and_without_sampling(dist, message):
    c = Circuit(2, 3)
    with pytest.raises(ValueError, match=message):
        c.add_gate("N2", 0, 1, prob_dist=dist)
    # Circuits can also carry a bad distribution that never went through add_gate
    # (for example one edited in place), so the simulators check it again.
    valid = np.zeros(81)
    valid[0] = 1.0
    c.add_gate("N2", 0, 1, prob_dist=valid)
    c.operations[0].params["prob_dist"] = dist
    c.add_gate("M", [0, 1])
    with pytest.raises(ValueError, match=message) as sampled:
        Program._build_ir([c], 3)
    with pytest.raises(ValueError, match=message) as lazy:
        Program._build_ir([c], 3, sample_noise=False)
    assert str(sampled.value) == str(lazy.value)
    with pytest.raises(ValueError, match=message):
        Program(c).simulate(shots=3)


def test_simulate_frame_needs_noise_for_noise_gates():
    c = Circuit(1, 3)
    c.add_gate("N1", 0, prob=0.5)
    c.add_gate("M", 0)
    _, ref = _reference_shot(c, 0)
    ir, _, info, model = Program._build_ir([c], 4, sample_noise=False)
    with pytest.raises(ValueError, match="noise"):
        simulate_frame(ir, ref, 1, 3, 4, None, info)
    frame, _ = simulate_frame(ir, ref, 1, 3, 4, None, info, model)
    assert frame.shape == (1, 1, 4)


# --------------------------------------------------------------------------
# Program.simulate behavior in frame mode


def test_frame_mode_result_structure():
    c = Circuit(3, 5)
    c.add_gate("X", 0)
    c.add_gate("MUL", 0, a=3)
    c.add_gate("H", 1)
    c.add_gate("N1", 2, prob=1.0, noise_channel="f")
    c.add_gate("RESET", 2)
    c.add_gate("M", [0, 1, 2])
    c.add_gate("M_X", 1)
    c.add_gate("DETECTOR", expr="rec[-4]", label="a")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-2]", label="b")
    shots = 50
    np.random.seed(0)
    measurements, det = Program(c).simulate(shots=shots)
    assert [len(rounds) for rounds in measurements] == [1, 2, 2]
    for rounds in measurements:
        for shot_list in rounds:
            assert len(shot_list) == shots
            assert all(type(r) is MeasurementResult and r.stabilizer_tableau is None for r in shot_list)
    # MUL: X then multiply by 3 gives 3, deterministically.
    assert all(r.measurement_value == 3 and r.deterministic for r in measurements[0][0])
    # RESET records its noisy pre-reset outcome; the noise is skipped in the reference shot.
    assert measurements[2][0][0].measurement_value == 0
    assert all(r.measurement_value != 0 for r in measurements[2][0][1:])
    assert all(r.measurement_value == 0 for r in measurements[2][1])
    # H then M is random; H then M then M_X is random too.
    assert not measurements[1][0][0].deterministic
    assert {r.qudit_index for rounds in measurements[1] for r in rounds} == {1}
    assert set(det) == {"detectors", "logicals"}
    assert [e["label"] for e in det["detectors"]] == ["a"] and [e["label"] for e in det["logicals"]] == ["b"]
    assert det["detectors"][0]["data"].shape == (shots - 1,)
    np.testing.assert_array_equal(det["detectors"][0]["data"], 0)

    np.random.seed(0)
    _, raw = Program(c).simulate(shots=shots, raw_detector_output=True)
    detectors, logicals = raw
    assert detectors.shape == (1, shots - 1) and logicals.shape == (1, shots - 1)
    assert detectors.dtype == np.int64 and logicals.dtype == np.int64


def _old_combine(measurement_results, frame_results):
    """The previous _combine_results, element by element."""
    for q in range(frame_results.shape[0]):
        for m in range(len(measurement_results[q])):
            measurement_results[q][m].extend(
                MeasurementResult(qudit_index=int(frame_results[q, m, s]['qudit_index']),
                                  deterministic=bool(frame_results[q, m, s]['deterministic']),
                                  measurement_value=int(frame_results[q, m, s]['measurement_value']))
                for s in range(frame_results.shape[2]))
    return measurement_results


def test_combine_results_matches_elementwise_construction():
    gen = np.random.default_rng(3)
    c = Circuit(3, 7)
    c.add_gate("M", [0, 1, 2])
    c.add_gate("M", [0, 2])
    prog = Program(c)
    prog.simulate()
    frame = np.zeros((3, 2, 9), dtype=MEASUREMENT_DTYPE)
    frame['qudit_index'] = gen.integers(0, 3, size=(3, 2, 9))
    frame['deterministic'] = gen.random((3, 2, 9)) < 0.5
    frame['measurement_value'] = gen.integers(-2 ** 62, 2 ** 62, size=(3, 2, 9))
    import copy
    expected = _old_combine(copy.deepcopy(prog.measurement_results), frame)
    got = prog._combine_results(frame)
    assert got == expected
    objects = [r for rounds in got for shot_list in rounds for r in shot_list]
    assert len({id(r) for r in objects}) == len(objects) == 3 * 10 + 2 * 10
    for r in objects:
        assert type(r.qudit_index) is int and type(r.deterministic) is bool and type(r.measurement_value) is int


def test_frame_results_match_simulate_frame_and_combine():
    """Program.simulate builds the same objects as _combine_results on simulate_frame's output."""
    rng = random.Random(8)
    c = _random_circuit(rng, 3, 4, 60)
    shots = 40
    prog, ref = _reference_shot(c, 8)
    ir, noise, info = Program._build_ir([c], shots - 1)
    np.random.seed(2)
    frame, _ = simulate_frame(ir, ref, c.num_qudits, c.dimension, shots - 1, noise, info)
    import copy
    expected = copy.deepcopy(prog)._combine_results(frame)
    from sdim.program import _run_frame
    np.random.seed(2)
    run = _run_frame(ir, ref, c.num_qudits, c.dimension, shots - 1, noise, info, None)
    got = prog._combine_frame_run(run, ref, c.dimension)
    assert got == expected


# --------------------------------------------------------------------------
# Detector modulo


def test_detector_mod_matches_numpy_remainder():
    gen = np.random.default_rng(1)
    edge = np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max, 0, 1, -1,
                     2 ** 52, -2 ** 52, 2 ** 52 - 1, 1 - 2 ** 52, 2 ** 53 + 1, -2 ** 53 - 1], dtype=np.int64)
    for d in [1, 2, 3, 7, 1000003, 2 ** 31 - 1, 2 ** 40 + 15]:
        for scale in [d, 2 ** 30, 2 ** 52, 2 ** 62]:
            v = np.concatenate([gen.integers(-scale, scale, size=5000, dtype=np.int64), edge])
            np.testing.assert_array_equal(_detector_mod(v, d), v % d)
        k = gen.integers(-(2 ** 52) // d, (2 ** 52) // d, size=5000, dtype=np.int64)
        for off in (-1, 0, 1):
            np.testing.assert_array_equal(_detector_mod(k * d + off, d), (k * d + off) % d)
    a = gen.integers(-10 ** 12, 10 ** 12, size=(6, 7), dtype=np.int64)
    np.testing.assert_array_equal(_detector_mod(a[:, ::2], 13), a[:, ::2] % 13)
    for other in (17, -17, np.int64(-5), 2.5, a.astype(np.int32), a.astype(np.float64)):
        r = _detector_mod(other, 5)
        assert type(r) is type(other % 5)
        np.testing.assert_array_equal(r, other % 5)
    assert _detector_mod(2 ** 80 + 3, 7) == (2 ** 80 + 3) % 7
