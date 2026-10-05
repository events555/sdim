"""The backward sweep of sdim.dem against the forward reference kernel.

`DetectorErrorModel.from_circuit` and `compile_unit_responses` read every
unit-fault response from one backward sweep over the circuit. Pushing every
unit fault forward to the end of the circuit (`_BACKWARD = False`) gives the
reference. Both must give exactly the same model: the same mechanisms in the
same order, with the same probabilities, sources and generators, dict order
included. That order is canonical: every response and generator lists its
targets in increasing order.
"""

import functools
import os
import subprocess
import sys
import time

import numpy as np
import pytest

import sdim.dem as dem_module
from sdim.circuit import Circuit
from sdim.dem import DetectorErrorModel, compile_unit_responses
from tests.test_dem_gates import random_circuit


INVERSE = {"H": "H_INV", "H_INV": "H", "P": "P_INV", "P_INV": "P", "CNOT": "CNOT_INV", "CNOT_INV": "CNOT",
           "CZ": "CZ_INV", "CZ_INV": "CZ", "SWAP": "SWAP"}


def _model_items(dem):
    return (dem.dimension, dem.num_detectors, dem.num_observables, list(dem.detector_labels),
            list(dem.observable_labels),
            [(float(m.probability).hex(), [list(g.items()) for g in m.generators], m.source)
             for m in dem.mechanisms])


def _responses(circuit):
    return [(loc.source, loc.channel, loc.qudits, [list(r.items()) for r in loc.responses])
            for loc in compile_unit_responses(circuit).locations]


def _outcomes(circuit, **kw):
    """Everything from_circuit / compile_unit_responses give for a circuit, or the error they raise."""
    out = []
    for fn in (lambda: _model_items(DetectorErrorModel.from_circuit(circuit, **kw)),
               lambda: _model_items(DetectorErrorModel.from_circuit(circuit, merge=False, **kw)),
               lambda: _responses(circuit)):
        try:
            out.append(("ok", fn()))
        except (ValueError, IndexError) as e:
            out.append(("error", type(e).__name__, str(e)))
    return out


def assert_sorted_by_target(outcomes):
    """Every generator and response in `_outcomes` lists its targets in increasing order."""
    merged, unmerged, responses = outcomes
    lists = []
    for model in (merged, unmerged):
        if model[0] == "ok":
            lists += [items for _, generators, _ in model[1][-1] for items in generators]
    if responses[0] == "ok":
        lists += [items for *_, location in responses[1] for items in location]
    for items in lists:
        targets = [t for t, _ in items]
        assert targets == sorted(set(targets)), items


def assert_backward_matches_forward(monkeypatch, circuit, **kw):
    monkeypatch.setattr(dem_module, "_BACKWARD", True)
    backward = _outcomes(circuit, **kw)
    monkeypatch.setattr(dem_module, "_BACKWARD", False)
    forward = _outcomes(circuit, **kw)
    monkeypatch.setattr(dem_module, "_BACKWARD", True)
    assert backward == forward
    assert_sorted_by_target(backward)
    return backward


def mul_circuit(d, seed, n=5, depth=30, p=0.02, rounds=3, unit_mul=True):
    """Prepare / C / C^-1 / measure rounds with MUL among the gates, M and M_X readout, mid-circuit RESET."""
    rng = np.random.default_rng(seed)
    c = Circuit(n, d)
    c.add_gate("RESET", list(range(n)))
    n_rec = 0
    last = [None] * n
    x_basis = {int(q) for q in rng.choice(n, size=max(1, n // 2), replace=False)}

    def noise():
        k = int(rng.integers(5))
        if k == 4:
            a, b = (int(q) for q in rng.choice(n, size=2, replace=False))
            c.add_gate("N2", a, b, prob=float(p * rng.random()))
        else:
            prob = p if k == 3 else float(p * rng.random())     # a shared value merges more
            c.add_gate("N1", int(rng.integers(n)), noise_channel="dfpd"[k], prob=prob)

    for rnd in range(rounds):
        for q in sorted(x_basis):
            c.add_gate("H", q)
        seq = []
        for _ in range(depth):
            r = rng.random()
            if r < 0.25:
                a = int(rng.integers(1, d))
                while unit_mul and np.gcd(a, d) != 1:
                    a = int(rng.integers(1, d))
                seq.append(("MUL", (int(rng.integers(n)),), a))
            elif r < 0.55:
                seq.append((["H", "H_INV", "P", "P_INV"][int(rng.integers(4))], (int(rng.integers(n)),), None))
            else:
                g = ["CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"][int(rng.integers(5))]
                seq.append((g, tuple(int(q) for q in rng.choice(n, size=2, replace=False)), None))
            name, qs, a = seq[-1]
            c.add_gate(name, *qs, **({"a": a} if name == "MUL" else {}))
            if rng.random() < 0.5:
                noise()
        for name, qs, a in reversed(seq):
            if name == "MUL":
                c.add_gate("MUL", qs[0], a=pow(a, -1, d))
            else:
                c.add_gate(INVERSE[name], *qs)
            if rng.random() < 0.5:
                noise()
        records = {}
        for q in (int(v) for v in rng.permutation(n)):
            if rng.random() < 0.5:
                noise()
            if q in x_basis and rng.random() < 0.5:
                c.add_gate("M_X", q)
            else:
                if q in x_basis:
                    c.add_gate("H_INV", q)
                c.add_gate("M", q)
                if q in x_basis:
                    c.add_gate("H", q)
                    c.add_gate("H_INV", q)
            records[q] = n_rec
            n_rec += 1
            expr = f"rec[{records[q] - n_rec}]"
            if last[q] is not None and rng.random() < 0.7:
                expr += f" - {int(rng.integers(1, d))}*rec[{last[q] - n_rec}]"
            c.add_gate("DETECTOR", expr=expr)
            last[q] = records[q]
        a, b = (int(q) for q in rng.choice(n, size=2, replace=False))
        c.add_gate("LOGICAL_OBSERVABLE", expr=f"rec[{records[a] - n_rec}] + 3*rec[{records[b] - n_rec}]")
        for q in range(n):
            if q in x_basis or rng.random() < 0.4:
                c.add_gate("RESET", q)
                if rng.random() < 0.5:
                    noise()
    return c


def rep_code(rounds, n_data=6, d=3, p=0.01, observables="plain", extra=0):
    """A repetition-code memory. observables: "plain" (data 0), "overlap" (three observables sharing data 0,
    read at the end in an interleaved order), "span" (also an observable reading ancilla 0 every round) or
    "span2" (also two observables reading ancilla 0 every round, as sum and twice the sum). `extra` adds
    that many N1 'f' gates on data 0 per round."""
    n_anc = n_data - 1
    c = Circuit(n_data + n_anc, d)
    data = list(range(n_data))
    anc = list(range(n_data, n_data + n_anc))
    c.add_gate("RESET", data + anc)
    for r in range(rounds):
        c.add_gate("N1", data, noise_channel="d", prob=p)
        for _ in range(extra):
            c.add_gate("N1", 0, noise_channel="f", prob=p)
        for i in range(n_anc):
            c.add_gate("CNOT", data[i], anc[i])
            c.add_gate("N2", data[i], anc[i], prob=p)
        for i in range(n_anc):
            c.add_gate("CNOT_INV", data[i + 1], anc[i])
            c.add_gate("N2", data[i + 1], anc[i], prob=p)
        c.add_gate("N1", anc, noise_channel="f", prob=p)
        c.add_gate("M", anc)
        for i in range(n_anc):
            expr = f"rec[{i - n_anc}]" + ("" if r == 0 else f" - rec[{i - 2 * n_anc}]")
            c.add_gate("DETECTOR", expr=expr)
        c.add_gate("RESET", anc)
    c.add_gate("N1", data, noise_channel="f", prob=p)
    c.add_gate("M", data)
    for i in range(n_anc):
        c.add_gate("DETECTOR", expr=f"rec[{i - n_data}] - rec[{i + 1 - n_data}] - rec[{i - n_data - n_anc}]")
    c.add_gate("LOGICAL_OBSERVABLE", expr=f"rec[{-n_data}]")
    if observables == "overlap":
        c.add_gate("LOGICAL_OBSERVABLE", expr=f"rec[{-n_data}] + rec[{2 - n_data}]")
        c.add_gate("LOGICAL_OBSERVABLE", expr=f"rec[{1 - n_data}] + rec[{-n_data}]")
    elif observables in ("span", "span2"):
        total = rounds * n_anc + n_data
        c.add_gate("LOGICAL_OBSERVABLE", expr=" + ".join(f"rec[{r * n_anc - total}]" for r in range(rounds)))
        if observables == "span2":
            c.add_gate("LOGICAL_OBSERVABLE", expr=" + ".join(f"2*rec[{r * n_anc - total}]" for r in range(rounds)))
    return c


def wide_circuit(d, width=60, rounds=3, p=0.01):
    c = Circuit(width + 1, d)
    c.add_gate("RESET", list(range(width + 1)))
    for _ in range(rounds):
        c.add_gate("N1", list(range(width + 1)), noise_channel="d", prob=p)
        c.add_gate("CNOT", 0, list(range(1, width + 1)))
        c.add_gate("N2", 0, 1, prob=p)
        c.add_gate("M", list(range(width + 1)))
        for q in range(width + 1):
            c.add_gate("DETECTOR", expr=f"rec[{q - (width + 1)}] - rec[{(q * 7) % (width + 1) - (width + 1)}]"
                       if q % 3 else f"rec[{q - (width + 1)}]")
        c.add_gate("LOGICAL_OBSERVABLE", expr=f"rec[{-width - 1}] + rec[-1] + rec[-2]")
        c.add_gate("RESET", list(range(width + 1)))
    return c


def _balanced_sum(terms):
    """terms joined by +, nested as a balanced tree so a long expression stays shallow."""
    while len(terms) > 1:
        terms = [f"({terms[i]} + {terms[i + 1]})" if i + 1 < len(terms) else terms[i]
                 for i in range(0, len(terms), 2)]
    return terms[0]


def idle_memory(rounds, n_idle=50, d=3, n_obs=2, p=0.01):
    """Idle data qudits with noise in every round, measured only at the end, and two ancillas measured, checked
    and reset in every round. Observable k reads ancilla 0 in every round (weight k + 1) and every data qudit."""
    n_anc = 2
    c = Circuit(n_anc + n_idle, d)
    anc = list(range(n_anc))
    data = list(range(n_anc, n_anc + n_idle))
    c.add_gate("RESET", anc + data)
    for _ in range(rounds):
        c.add_gate("N1", data, noise_channel="f", prob=p)
        c.add_gate("N1", anc, noise_channel="f", prob=p)
        c.add_gate("M", anc)
        for i in range(n_anc):
            c.add_gate("DETECTOR", expr=f"rec[{i - n_anc}]")
        c.add_gate("RESET", anc)
    c.add_gate("M", data)
    total = rounds * n_anc + n_idle
    for k in range(n_obs):
        terms = [f"{k + 1}*rec[{r * n_anc - total}]" for r in range(rounds)]
        terms += [f"rec[{-1 - j}]" for j in range(n_idle)]
        c.add_gate("LOGICAL_OBSERVABLE", expr=_balanced_sum(terms))
    return c


@pytest.mark.parametrize("d", [2, 3, 5, 7, 1000003])
@pytest.mark.parametrize("seed", range(8))
def test_random_circuits(monkeypatch, d, seed):
    """All gates, every noise channel, M and M_X mid-circuit (with M_X qudits reused), RESET, two observables."""
    for c in (random_circuit(d, seed, p=0.01 if seed % 2 else 0.2),
              random_circuit(d, 100 + seed, num_qudits=6, num_gates=120, noise_rate=0.8)):
        assert assert_backward_matches_forward(monkeypatch, c)[0][0] == "ok"


@pytest.mark.parametrize("d", [2, 3, 5, 7, 11, 1000003])
@pytest.mark.parametrize("seed", range(4))
def test_mul_circuits(monkeypatch, d, seed):
    assert assert_backward_matches_forward(monkeypatch, mul_circuit(d, seed))[0][0] == "ok"


@pytest.mark.parametrize("d", [3, 1000003])
@pytest.mark.parametrize("observables", ["plain", "overlap", "span", "span2"])
def test_repetition_codes_with_observables(monkeypatch, d, observables):
    assert assert_backward_matches_forward(monkeypatch, rep_code(12, d=d, observables=observables))[0][0] == "ok"
    c = rep_code(9, d=d, observables=observables, extra=3)
    assert assert_backward_matches_forward(monkeypatch, c)[0][0] == "ok"


@pytest.mark.parametrize("d", [2, 5, 1000003])
def test_wide_fan_out(monkeypatch, d):
    assert assert_backward_matches_forward(monkeypatch, wide_circuit(d))[0][0] == "ok"


@pytest.mark.parametrize("d", [2, 3, 1000003])
def test_idle_memory(monkeypatch, d):
    assert assert_backward_matches_forward(monkeypatch, idle_memory(15, n_idle=6, d=d))[0][0] == "ok"


def test_targets_are_sorted_when_reached_out_of_order(monkeypatch):
    """A fault that reaches D2 and L0 first, then D1, then D0, lists them by target with either kernel, in
    responses, unmerged and merged models alike."""
    c = Circuit(3, 5)
    c.add_gate("RESET", [0, 1, 2])
    c.add_gate("N2", 0, 1, prob=0.01)
    c.add_gate("N1", 0, noise_channel="f", prob=0.02)
    c.add_gate("CNOT", 0, [1, 2])
    c.add_gate("M", 2)
    c.add_gate("M", [1, 0])
    c.add_gate("DETECTOR", expr="rec[-1]")
    c.add_gate("DETECTOR", expr="rec[-2]")
    c.add_gate("DETECTOR", expr="2*rec[-3]")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-3]")
    x0 = [(0, 1), (1, 1), (2, 2), (3, 1)]
    for backward in (True, False):
        monkeypatch.setattr(dem_module, "_BACKWARD", backward)
        n2, n1 = compile_unit_responses(c).locations
        assert [list(r.items()) for r in n2.responses] == [x0, [], [(1, 1)], []]
        assert [list(r.items()) for r in n1.responses] == [x0]
        for merge in (True, False):
            mechanisms = DetectorErrorModel.from_circuit(c, merge=merge).mechanisms
            assert sorted([list(g.items()) for g in m.generators] for m in mechanisms) == [[x0], [x0, [(1, 1)]]]


@pytest.mark.parametrize("d", [4, 6, 9])
def test_composite_dimensions(monkeypatch, d):
    for c in (random_circuit(d, 3), mul_circuit(d, 1), rep_code(5, d=d)):
        assert assert_backward_matches_forward(monkeypatch, c, check_dimension_prime=False)[0][0] == "ok"


def test_errors_match(monkeypatch):
    """Non-deterministic detectors and observables, bad noise and bad qudits raise the same errors either way."""
    cases = []
    c = Circuit(2, 3)
    c.add_gate("RESET", [0, 1])
    c.add_gate("H", 0)
    c.add_gate("N1", 0, prob=0.01)
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="rec[-2]")
    c.add_gate("DETECTOR", expr="rec[-1]")
    cases.append(c)                                   # random Z after RESET + H reaches D0
    c = Circuit(1, 5)
    c.add_gate("RESET", 0)
    c.add_gate("N1", 0, prob=0.01)
    c.add_gate("M_X", 0)
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-1]")
    cases.append(c)                                   # random X after RESET read by M_X
    c = Circuit(2, 3)
    c.add_gate("N1", 0, noise_channel="f", prob=0.1)
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="rec[-1]")
    cases.append(c)                                   # no RESET: random Z at the start is harmless for M
    c = Circuit(1, 3)
    c.add_gate("M_X", 0)
    c.add_gate("M_X", 0)
    c.add_gate("N1", 0, noise_channel="p", prob=0.1)
    c.add_gate("M_X", 0)
    c.add_gate("DETECTOR", expr="rec[-1] - rec[-2]")
    c.add_gate("DETECTOR", expr="rec[-2] - rec[-3]")
    cases.append(c)                                   # M_X repeated: X eigenstate kept, deterministic
    c = Circuit(2, 3)
    c.add_gate("N1", 0, noise_channel="f", prob=0.1)
    c.add_gate("CNOT", 0, 5)
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="rec[-1]")
    cases.append(c)                                   # qudit out of range
    for c in cases:
        assert_backward_matches_forward(monkeypatch, c)
    assert assert_backward_matches_forward(monkeypatch, cases[0])[0][0] == "error"
    assert "D0" in assert_backward_matches_forward(monkeypatch, cases[0])[0][2]
    assert "L0" in assert_backward_matches_forward(monkeypatch, cases[1])[0][2]
    assert assert_backward_matches_forward(monkeypatch, cases[3])[0][0] == "ok"


def test_two_qudit_gate_on_one_qudit_is_rejected():
    """The forward kernel used to fail with "status 2" on such a gate; it has no frame rule, so say so."""
    c = Circuit(2, 3)
    c.add_gate("RESET", [0, 1])
    c.add_gate("N1", 0, noise_channel="f", prob=0.1)
    c.add_gate("CNOT", 0, 1)
    c.operations[-1].target_index = 0          # add_gate refuses this; make it behind its back
    c.add_gate("M", [0, 1])
    c.add_gate("DETECTOR", expr="rec[-1]")
    with pytest.raises(ValueError, match="twice"):
        DetectorErrorModel.from_circuit(c)


def test_from_circuit_does_not_push_faults_forward(monkeypatch):
    """from_circuit reads the responses from backward sweeps; the forward kernel is only the reference."""
    def refuse(*args, **kwargs):
        raise AssertionError("the forward kernel ran")

    monkeypatch.setattr(dem_module, "_run_probes", refuse)
    monkeypatch.setattr(dem_module, "_probe_kernel", refuse)
    for c in (rep_code(30, observables="overlap"), random_circuit(5, 2), mul_circuit(7, 2)):
        DetectorErrorModel.from_circuit(c)
        DetectorErrorModel.from_circuit(c, merge=False)
        compile_unit_responses(c)


def test_compile_time_is_linear_in_rounds():
    """Faults on data qudits used to be pushed through every later round: time quadratic in the rounds."""
    def best(rounds):
        c = rep_code(rounds, n_data=8, observables="overlap")
        times = []
        for _ in range(3):
            t = time.perf_counter()
            dem_module._compile(c).mechanisms(True)
            times.append(time.perf_counter() - t)
        return min(times)

    best(20)
    # A busy machine can stall either size, so measure up to three times before failing.
    for _ in range(3):
        small, large = best(150), best(1200)
        if large / small < 20:
            break
    # Linear work gives a ratio near 8; quadratic gave about 40 here.
    assert large / small < 20, (small, large)


def test_compile_time_is_linear_in_rounds_with_spanning_observables():
    """Observables that read an ancilla in every round: reading their long expressions, and the responses
    of the many faults that reach them, takes time linear in the rounds."""
    def best(rounds):
        c = rep_code(rounds, n_data=5, observables="span2", extra=10)
        times = []
        for _ in range(3):
            t = time.perf_counter()
            DetectorErrorModel.from_circuit(c, merge=False)
            times.append(time.perf_counter() - t)
        return min(times)

    best(20)
    # A busy machine can stall either size, so measure up to three times before failing.
    for _ in range(3):
        small, large = best(150), best(1200)
        if large / small < 20:
            break
    # Linear work gives a ratio near 8; it was over 100 here.
    assert large / small < 20, (small, large)


def test_compile_time_is_linear_in_rounds_with_idle_qudits():
    """Many idle qudits with noise in every round, measured only at the end, and observables that read an
    ancilla in every round. Listing each response's targets in the order the fault first reached them used
    to take time quadratic in the rounds here, for unmerged models and compile_unit_responses."""
    def best(fn, rounds):
        c = idle_memory(rounds, n_idle=20)
        times = []
        for _ in range(3):
            t = time.perf_counter()
            fn(c)
            times.append(time.perf_counter() - t)
        return min(times)

    for fn in (lambda c: DetectorErrorModel.from_circuit(c, merge=False), compile_unit_responses):
        best(fn, 20)
        # A busy machine can stall either size, so measure up to three times before failing.
        for _ in range(3):
            small, large = best(fn, 200), best(fn, 1600)
            if large / small < 18:
                break
        # Linear work gives a ratio near 8; it was about 30 here.
        assert large / small < 18, (small, large)


_COMPILED_KERNELS_SCRIPT = r"""
import numba
import sdim.dem as dm
from sdim.circuit import Circuit
c = Circuit(4, 3)
c.add_gate("RESET", [0, 1, 2, 3])
c.add_gate("N1", [0, 1], prob=0.01)
c.add_gate("CNOT", 0, [1, 2, 3])
c.add_gate("N2", 0, 1, prob=0.01)
c.add_gate("M", [0, 1, 2, 3])
c.add_gate("DETECTOR", expr="rec[-1] - rec[-2]")
c.add_gate("DETECTOR", expr="rec[-3]")
c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-4] + rec[-1]")
dem = dm.DetectorErrorModel.from_circuit(c)
dm.DetectorErrorModel.from_circuit(c, merge=False)
dm.compile_unit_responses(c)
dem.sample(1000, seed=1)
used = sorted(name for name, obj in vars(dm).items()
              if isinstance(obj, numba.core.registry.CPUDispatcher) and obj.signatures)
print("compiled:", ",".join(used))
"""


def test_first_use_compiles_only_two_kernels(tmp_path):
    """Cold start: building and sampling a model needs only the backward sweep and the sampler compiled
    (each numba function costs seconds to compile on first use with an empty cache)."""
    script = tmp_path / "kernels.py"
    script.write_text(_COMPILED_KERNELS_SCRIPT)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.dirname(os.path.dirname(dem_module.__file__)),
                                         env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "compiled: _backward_kernel,_sample_chunks" in proc.stdout, proc.stdout


def test_probe_kernel_reference_is_unchanged_by_block_size(monkeypatch):
    """The forward reference itself still splits into blocks and threads without changing the result."""
    c = mul_circuit(5, 3)
    monkeypatch.setattr(dem_module, "_BACKWARD", False)
    expected = _responses(c)
    monkeypatch.setattr(dem_module, "_thread_count", lambda: 3)
    monkeypatch.setattr(dem_module, "_PROBE_PARALLEL_MIN", 0)
    monkeypatch.setattr(dem_module, "_run_probes", functools.partial(dem_module._run_probes, block=2, cap=1))
    assert _responses(c) == expected
    monkeypatch.setattr(dem_module, "_BACKWARD", True)
    assert _responses(c) == expected
