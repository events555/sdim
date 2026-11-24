"""NoiseModel.apply: where noise goes, and that injected memory circuits give the intended DEMs."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sdim.dem")
from sdim import Circuit, Program  # noqa: E402
from sdim.dem import DetectorErrorModel  # noqa: E402

from shepherd.benchmarking.ler import LogicalErrorRate, _expr, _signed  # noqa: E402
from shepherd.codes.css import surface_code  # noqa: E402
from shepherd.noise_model import Channel, GateNoise, IdleNoise, NoiseModel, NoiseModelError  # noqa: E402


def _ops(c: Circuit) -> list[tuple]:
    return [(op.name, op.qudit_index, op.target_index, op.params.get("noise_channel"), op.params.get("prob"))
            for op in c.operations]


# -- placement -------------------------------------------------------------------------------------

def test_gate_noise_before_and_after():
    c = Circuit(3, 3)
    c.add_gate("H", 0)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="1*rec[-1]")
    m = NoiseModel(3, gates={"H": GateNoise(Channel("phase", 0.1), "after"),
                             "CNOT": GateNoise(Channel("dep", 0.2, True), "after"),
                             "M": GateNoise(Channel("flip", 0.3), "before")})
    out = m.apply(c)
    assert _ops(out) == [
        ("H", 0, None, None, None), ("N1", 0, None, "p", 0.1),
        ("CNOT", 0, 1, None, None), ("N2", 0, 1, None, 0.2),
        ("N1", 0, None, "f", 0.3), ("M", 0, None, None, None),
        ("N1", 1, None, "f", 0.3), ("M", 1, None, None, None),
        ("DETECTOR", None, None, None, None)]
    assert out.operations[-1].params["expr"] == "1*rec[-1]"
    assert len(c.operations) == 5                      # the input is untouched


@pytest.mark.parametrize("target, expected", [("control", [0]), ("target", [1]), ("both", [0, 1])])
def test_single_qudit_noise_on_two_qudit_gates(target, expected):
    c = Circuit(2, 5)
    c.add_gate("CNOT", 0, 1)
    m = NoiseModel(5, gates={"CNOT": GateNoise(Channel("dep", 0.1, False), "after", target)})
    assert [op.qudit_index for op in m.apply(c).operations if op.name == "N1"] == expected


def test_two_qudit_channel_must_be_depolarizing_and_zero_p_is_skipped():
    c = Circuit(2, 3)
    c.add_gate("CNOT", 0, 1)
    with pytest.raises(NoiseModelError, match="N2"):
        NoiseModel(3, gates={"CNOT": GateNoise(Channel("flip", 0.1, True), "after")}).apply(c)
    quiet = NoiseModel(3, gates={"CNOT": GateNoise(Channel("dep", 0.0, True), "after")})
    assert [op.name for op in quiet.apply(c).operations] == ["CNOT"]
    with pytest.raises(NoiseModelError, match="dimension"):
        NoiseModel(5).apply(c)


def test_existing_noise_and_unlisted_gates_are_copied():
    c = Circuit(2, 3)
    c.add_gate("N1", 0, noise_channel="d", prob=0.05)
    c.add_gate("X", 1)
    assert _ops(NoiseModel(3).apply(c)) == [("N1", 0, None, "d", 0.05), ("X", 1, None, None, None)]


def test_idle_noise_needs_ticks_and_uses_the_layer_context():
    m = NoiseModel(3, idle=IdleNoise({"default": Channel("phase", 0.01),
                                      "measurement": Channel("dep", 0.02)}))
    c = Circuit(3, 3)
    c.add_gate("X", 0)
    assert [op.name for op in m.apply(c).operations] == ["X"]          # no TICK: no idle noise
    c.add_gate("TICK")
    c.add_gate("M", 1)
    noise = [(op.qudit_index, op.params["noise_channel"]) for op in m.apply(c).operations
             if op.name == "N1"]
    assert noise == [(1, "p"), (2, "p"),       # layer 1 (X on 0): 1 and 2 idle, default
                     (0, "d"), (2, "d")]       # layer 2 (M on 1): 0 and 2 idle, measurement


def test_both_target_round_trips_through_json():
    m = NoiseModel.circuit_level(3, 0.01)
    again = NoiseModel.loads(m.dumps())
    assert again.gate("CNOT").target == "both"
    assert again.fingerprint() == m.fingerprint()


# -- injected memory circuits give the intended models ---------------------------------------------

def _lines(dem: DetectorErrorModel) -> dict[tuple, float]:
    """Every independent line, normalized (first coefficient 1) and merged: model -> probability."""
    q = dem.dimension
    out: dict[tuple, float] = {}
    for mech in dem.to_lines().mechanisms:
        g = sorted((t, v % q) for t, v in mech.generators[0].items() if v % q)
        inv = pow(g[0][1], -1, q)
        key = tuple((t, v * inv % q) for t, v in g)
        out[key] = 1 - (1 - out.get(key, 0.0)) * (1 - mech.probability)
    return out


def _same_model(a: DetectorErrorModel, b: DetectorErrorModel) -> None:
    la, lb = _lines(a), _lines(b)
    assert (a.num_detectors, a.num_observables) == (b.num_detectors, b.num_observables)
    assert la.keys() == lb.keys()
    assert all(np.isclose(la[k], lb[k], rtol=1e-9, atol=0) for k in la)


def _reference_circuit_level(code, p, rounds, two_qudit):
    """The hand-written Z-syndrome circuit-level memory the injector replaces."""
    q, n, hz = code.dimension, code.n, code.H_Z
    _, lz = code.logical_matrices()
    m = hz.shape[0]
    data, anc = list(range(n)), list(range(n, n + m))
    c = Circuit(n + m, q)
    nm, prev = 0, None
    for _ in range(rounds):
        c.add_gate("N1", data, noise_channel="d", prob=p)
        for i in range(m):
            for j in np.nonzero(hz[i])[0]:
                s = _signed(int(hz[i, j]), q)
                for _ in range(abs(s)):
                    c.add_gate("CNOT" if s > 0 else "CNOT_INV", int(j), anc[i])
                    if two_qudit == "N2":
                        c.add_gate("N2", int(j), anc[i], prob=p)
                    else:
                        c.add_gate("N1", [int(j), anc[i]], noise_channel="d", prob=p)
        c.add_gate("N1", anc, noise_channel="f", prob=p)
        c.add_gate("M", anc)
        cur = list(range(nm, nm + m))
        nm += m
        c.add_gate("RESET", anc)
        c.add_gate("N1", anc, noise_channel="f", prob=p)
        for i in range(m):
            t = [(1, cur[i])] if prev is None else [(1, cur[i]), (q - 1, prev[i])]
            c.add_gate("DETECTOR", expr=_expr(t, q, nm))
        prev = cur
    c.add_gate("N1", data, noise_channel="f", prob=p)
    c.add_gate("M", data)
    first = nm
    nm += n
    for i in range(m):
        t = [(int(hz[i, j]), first + int(j)) for j in np.nonzero(hz[i])[0]] + [(q - 1, prev[i])]
        c.add_gate("DETECTOR", expr=_expr(t, q, nm))
    for row in lz:
        c.add_gate("LOGICAL_OBSERVABLE",
                   expr=_expr([(int(row[j]), first + int(j)) for j in np.nonzero(row)[0]], q, nm))
    return c


@pytest.mark.parametrize("q", (2, 3, 5))
@pytest.mark.parametrize("two_qudit", ("independent", "N2"))
def test_circuit_level_matches_hand_written_circuit(q, two_qudit):
    code, p = surface_code(3, q), 0.01
    injected = NoiseModel.circuit_level(q, p, two_qudit=two_qudit).apply(
        LogicalErrorRate(q, rounds=3).build(code))
    _same_model(DetectorErrorModel.from_circuit(injected),
                DetectorErrorModel.from_circuit(_reference_circuit_level(code, p, 3, two_qudit)))


@pytest.mark.parametrize("q", (2, 3, 5))
def test_code_capacity_is_one_shift_per_data_qudit(q):
    code, p = surface_code(5, q), 0.03
    injected = NoiseModel.code_capacity(q, p).apply(LogicalErrorRate(q, rounds=0).build(code))
    ref = Circuit(code.n, q)
    ref.add_gate("N1", list(range(code.n)), noise_channel="f", prob=p)
    for op in LogicalErrorRate(q, rounds=0).build(code).operations:
        if op.name not in ("I", "TICK"):
            ref.add_gate(op.name, *([op.qudit_index] if op.qudit_index is not None else []), **op.params)
    _same_model(DetectorErrorModel.from_circuit(injected), DetectorErrorModel.from_circuit(ref))


@pytest.mark.parametrize("model", ["code_capacity", "phenomenological", "circuit_independent", "circuit_N2"])
def test_noiseless_versions_are_deterministic(model):
    q, rounds = 3, (0 if model == "code_capacity" else 2)
    make = {"code_capacity": lambda p: NoiseModel.code_capacity(q, p),
            "phenomenological": lambda p: NoiseModel.phenomenological(q, p),
            "circuit_independent": lambda p: NoiseModel.circuit_level(q, p),
            "circuit_N2": lambda p: NoiseModel.circuit_level(q, p, two_qudit="N2")}[model]
    circuit = make(0.0).apply(LogicalErrorRate(q, rounds=rounds).build(surface_code(3, q)))
    det, obs = DetectorErrorModel.from_circuit(circuit).sample(50, seed=1)
    assert not det.any() and not obs.any()


def test_injected_dem_agrees_with_the_frame_simulator():
    q, shots = 3, 20_000
    circuit = NoiseModel.circuit_level(q, 0.02, two_qudit="N2").apply(
        LogicalErrorRate(q, rounds=2).build(surface_code(3, q)))
    _, (det_c, obs_c) = Program(circuit).simulate(shots=shots + 1, raw_detector_output=True)
    det_c, obs_c = np.asarray(det_c).T % q, np.asarray(obs_c).T % q
    det_d, obs_d = DetectorErrorModel.from_circuit(circuit).sample(shots, seed=2)
    fc, fd = (det_c != 0).mean(0), (det_d % q != 0).mean(0)
    se = np.sqrt((fc * (1 - fc) + fd * (1 - fd)) / shots)
    assert np.all(np.abs(fc - fd) < 5 * se + 1e-3)
    oc, od = (obs_c != 0).any(axis=1).mean(), (obs_d % q != 0).any(axis=1).mean()
    assert abs(oc - od) < 5 * np.sqrt(2 * oc * (1 - oc) / shots) + 1e-3
