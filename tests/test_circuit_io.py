"""Tests for sdim.circuit_io: the .chp round trip and the conversion to Cirq."""

import ast
import collections
import math
import os
import random
import shlex
import subprocess
import sys

import cirq
import numpy as np
import pytest

from sdim.circuit import Circuit
from sdim.circuit_io import (GeneralizedSwapGate, _split_gate_line, circuit_to_cirq_circuit,
                             cirq_statevector_from_circuit, read_circuit, write_circuit)
from sdim.program import Program


# --------------------------------------------------------------------------
# .chp round trip


def _every_parameter_circuit():
    d = 2
    dist = np.zeros(d ** 4)
    dist[0] = 0.25
    dist[5] = 0.5
    dist[10] = 0.25
    c = Circuit(5, d)  # qudit 4 stays idle: the qudit count has to come from the file header
    c.add_gate("RESET", [0, 1, 2])
    c.add_gate("H", 0)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("N1", 0, noise_channel="f", prob=0.1)
    c.add_gate("N1", 1, noise_channel="p", prob=1e-5)
    c.add_gate("N1", 2, prob=0.1 + 0.2)
    c.add_gate("N1", 3, channel="d", prob=np.float64(1 / 3))
    c.add_gate("N2", 0, 1, prob=0.25)
    c.add_gate("N2", 1, 2, prob_dist=dist)
    c.add_gate("N2", 2, 3, prob_dist=dist.tolist())
    c.add_gate("N2", 0, 2, prob_dist=None)
    c.add_gate("MUL", 1, a=10 ** 19 + 1)
    c.add_gate("MUL", 2, scalar=-3)
    c.add_gate("MUL", 3, a=3.0)
    c.add_gate("SWAP", 1, 3)
    c.add_gate("TICK")
    c.add_gate("M", [0, 1])
    c.add_gate("M_X", 2)
    c.add_gate("DETECTOR", expr="(rec[-2] == 0) * rec[-1] + rec[0]", label='say "hi" \\ # = ok')
    c.add_gate("DETECTOR", expr="rec[-1] + 3 + 7")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-3]", label="L0")
    # Labels that are not text, and text that looks like a number or None
    c.add_gate("DETECTOR", expr="rec[-1]", label=5)
    c.add_gate("DETECTOR", expr="rec[-1]", label="5")
    c.add_gate("DETECTOR", expr="rec[-1]", label=None)
    c.add_gate("DETECTOR", expr="rec[-1]", label="None")
    c.add_gate("DETECTOR", expr="rec[-2]", label=np.int64(-7))
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-1]", label=2.5)
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-1]", label=True)
    c.add_gate("N1", 0, prob=1)
    c.add_gate("N1", 0, prob=0, noise_channel="p")
    c.add_gate("N2", 2, 3, prob=np.float32(0.125))
    return c


def _assert_same_params(expected, actual):
    assert (expected is None) == (actual is None)
    if expected is None:
        return
    assert set(expected) == set(actual)
    for key, value in expected.items():
        got = actual[key]
        if key == "prob_dist" and value is not None:
            assert isinstance(got, np.ndarray) and got.dtype == np.float64
            np.testing.assert_array_equal(got, np.asarray(value, dtype=float))
            continue
        if isinstance(value, np.generic):
            value = value.item()  # NumPy scalars come back as the matching Python scalar
        assert type(got) is type(value) and got == value, (key, value, got)


def _typed(value):
    """(type, value), with a NumPy scalar taken as the matching Python scalar."""
    if isinstance(value, np.generic):
        value = value.item()
    return type(value), value


def _round_trip(circuit, tmp_path):
    path = write_circuit(circuit, "round_trip.chp", comment="round trip", directory=str(tmp_path))
    return read_circuit(path)


def test_round_trip_keeps_every_parameter(tmp_path):
    c = _every_parameter_circuit()
    cc = _round_trip(c, tmp_path)
    assert (cc.dimension, cc.num_qudits) == (c.dimension, c.num_qudits)
    assert len(cc.operations) == len(c.operations)
    for op, read in zip(c.operations, cc.operations):
        assert (read.gate_name, read.name, read.gate_id) == (op.gate_name, op.name, op.gate_id)
        assert (read.qudit_index, read.target_index) == (op.qudit_index, op.target_index)
        _assert_same_params(op.params, read.params)


def test_round_trip_simulates_identically(tmp_path):
    """An N2 prob_dist used to come back as a string, which neither simulator could use."""
    c = _every_parameter_circuit()
    cc = _round_trip(c, tmp_path)
    for shots in (1, 200):
        np.random.seed(7)
        random.seed(7)
        expected = Program(c).simulate(shots=shots)
        np.random.seed(7)
        random.seed(7)
        actual = Program(cc).simulate(shots=shots)
        if shots == 1:
            assert actual == expected
        else:
            assert actual[0] == expected[0]
            for kind in ("detectors", "logicals"):
                # The labels keep their type too: label=5 used to come back as "5".
                assert [_typed(e["label"]) for e in actual[1][kind]] == [_typed(e["label"]) for e in expected[1][kind]]
                for a, e in zip(actual[1][kind], expected[1][kind]):
                    np.testing.assert_array_equal(a["data"], e["data"])


def test_round_trip_of_a_larger_prob_dist(tmp_path):
    """A NumPy prob_dist used to be written with str(), which wraps long arrays onto several lines."""
    d = 3
    dist = np.random.default_rng(0).random(d ** 4)
    dist /= dist.sum()
    c = Circuit(2, d)
    c.add_gate("N2", 0, 1, prob_dist=dist)
    c.add_gate("M", [0, 1])
    cc = _round_trip(c, tmp_path)
    np.testing.assert_array_equal(cc.operations[0].params["prob_dist"], dist)


def test_reads_files_written_by_older_versions(tmp_path):
    """Older versions wrote d <dimension> without the qudit count, and every value as key="text"."""
    old = "\n".join([
        "Written by an older sdim",
        "#",
        "d 3",
        "H 0",
        'N1 0 noise_channel="f" prob="0.25"',
        'N2 0 1 prob="0.01"',
        'N2 1 0 prob_dist="[' + ", ".join(["1.0"] + ["0.0"] * 80) + ']" prob="0.01"',
        'MUL 1 a="2"',
        "CNOT 0 1",
        "M 0",
        "M 1",
        'DETECTOR expr="rec[-1] - rec[-2] + 3 + 7" label="det"',
        "",
    ])
    path = tmp_path / "old.chp"
    path.write_text(old)
    c = read_circuit(str(path))
    # The constants inside the quoted expression used to be counted as qudit indices.
    assert (c.dimension, c.num_qudits) == (3, 2)
    assert [op.name for op in c.operations] == ["H", "N1", "N2", "N2", "MUL", "CNOT", "M", "M", "DETECTOR"]
    assert c.operations[1].params == {"noise_channel": "f", "prob": 0.25}
    assert c.operations[2].params == {"prob": 0.01}
    np.testing.assert_array_equal(c.operations[3].params["prob_dist"], [1.0] + [0.0] * 80)
    assert c.operations[4].params == {"a": 2}
    assert c.operations[8].params == {"expr": "rec[-1] - rec[-2] + 3 + 7", "label": "det"}
    measurements, det = Program(c).simulate(shots=20)
    assert len(det["detectors"][0]["data"]) == 19


def test_write_keeps_old_readers_working(tmp_path):
    """The dimension line still starts with d <dimension>, which is all older readers look at."""
    c = Circuit(3, 5)
    c.add_gate("X", 0)
    path = write_circuit(c, "header.chp", directory=str(tmp_path))
    lines = open(path).read().splitlines()
    header = lines[lines.index("#") + 1].split()
    assert header[:2] == ["d", "5"]


def test_write_rejects_line_breaks(tmp_path):
    c = Circuit(1, 3)
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-1]", label="two\nlines")
    with pytest.raises(ValueError, match="line break"):
        write_circuit(c, "bad.chp", directory=str(tmp_path))


@pytest.mark.parametrize("gates", [[], ["TICK"], ["TICK", "TICK"]])
def test_round_trip_of_a_circuit_without_qudit_gates(tmp_path, gates):
    """An empty or TICK-only circuit used to fail to read back: max() arg is an empty sequence."""
    c = Circuit(2, 3)
    for gate in gates:
        c.add_gate(gate)
    cc = _round_trip(c, tmp_path)
    assert (cc.dimension, cc.num_qudits) == (3, 2)
    assert [op.name for op in cc.operations] == gates


def test_reads_a_tick_only_file_without_a_qudit_count(tmp_path):
    path = tmp_path / "tick.chp"
    path.write_text("Written by an older sdim\n#\nd 3\nTICK\n")
    c = read_circuit(str(path))
    assert (c.dimension, c.num_qudits) == (3, 1)
    assert [op.name for op in c.operations] == ["TICK"]


def test_reads_blank_lines_after_the_hash_line(tmp_path):
    """A blank line right after '#' used to raise IndexError."""
    path = tmp_path / "blank.chp"
    path.write_text("comment\n#\n\n  \nd 3\nH 1\n\nM 1\n")
    c = read_circuit(str(path))
    assert (c.dimension, c.num_qudits) == (3, 2)
    assert [(op.name, op.qudit_index) for op in c.operations] == [("H", 1), ("M", 1)]


def test_file_without_a_hash_line_raises_value_error(tmp_path):
    """This used to raise a bare StopIteration."""
    path = tmp_path / "nohash.chp"
    path.write_text("d 3\nH 0\n")
    with pytest.raises(ValueError, match="no line with only '#'"):
        read_circuit(str(path))


def test_round_trip_keeps_parameter_types(tmp_path):
    """label=5 used to come back as "5", and prob=1 as 1.0."""
    labels = [5, "5", None, "None", True, "True", 2.5, "2.5", "", "a b", np.int64(3), np.float64(0.5),
              np.bool_(False), -0.0, 10 ** 30, float("inf"), "x=1"]
    c = Circuit(1, 3)
    c.add_gate("M", 0)
    for label in labels:
        c.add_gate("DETECTOR", expr="rec[-1]", label=label)
    c.add_gate("N1", 0, prob=1)
    c.add_gate("N1", 0, prob=np.float64(0.5))
    c.add_gate("MUL", 0, a=np.int64(2))
    cc = _round_trip(c, tmp_path)
    assert [_typed(op.params["label"]) for op in cc.operations[1:1 + len(labels)]] == [_typed(v) for v in labels]
    assert [_typed(op.params.get("prob", op.params.get("a"))) for op in cc.operations[-3:]] == \
        [(int, 1), (float, 0.5), (int, 2)]
    np.random.seed(0)
    _, det = Program(cc).simulate(shots=3)
    assert _typed(det["detectors"][0]["label"]) == (int, 5)


def test_round_trip_of_int_and_float_subclasses(tmp_path):
    """Their repr was written as is, so an IntEnum label became label=<Kind.A: 3>, which read back as garbage."""
    import enum

    class Kind(enum.IntEnum):
        A = 3

    class Probability(float):
        def __repr__(self):
            return f"Probability({float(self)})"

    c = Circuit(1, 3)
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-1]", label=Kind.A)
    c.add_gate("N1", 0, prob=Probability(0.5))
    cc = _round_trip(c, tmp_path)
    assert [op.params for op in cc.operations] == [op.params for op in c.operations]
    assert type(cc.operations[1].params["label"]) is int and type(cc.operations[2].params["prob"]) is float


def test_other_parameter_values_come_back_as_their_text(tmp_path):
    c = Circuit(1, 3)
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-1]", label=(1, 2))
    assert _round_trip(c, tmp_path).operations[1].params["label"] == "(1, 2)"


def test_text_under_number_keys_comes_back_as_written_in_the_docstring(tmp_path):
    """
    write_circuit used to promise that text always comes back with the same type, but text under
    prob, a and scalar is read as a number when it reads as one (as in files of older versions).
    """
    c = Circuit(1, 3)
    c.add_gate("N1", 0, prob="0.1")
    c.add_gate("MUL", 0, a="2")
    c.add_gate("H", 0, prob="abc", scalar="7", note="7")
    c.add_gate("M", 0)
    c.add_gate("DETECTOR", expr="rec[-1]", label="0.5")
    cc = _round_trip(c, tmp_path)
    params = [{key: _typed(value) for key, value in op.params.items()} for op in cc.operations]
    assert params == [{"prob": (float, 0.1), "noise_channel": (str, "d")}, {"a": (int, 2)},
                      {"prob": (str, "abc"), "scalar": (int, 7), "note": (str, "7")}, {},
                      {"expr": (str, "rec[-1]"), "label": (str, "0.5")}]
    assert "except under the keys prob, a and scalar" in " ".join(write_circuit.__doc__.split())
    outcomes = []
    for circuit in (c, cc):  # the simulators read either form the same way
        np.random.seed(4)
        random.seed(4)
        measurements, _ = Program(circuit).simulate(shots=20)
        outcomes.append([m.measurement_value for m in measurements[0][0]])
    assert outcomes[0] == outcomes[1]


def test_reads_unquoted_values(tmp_path):
    """Values written without quotes are numbers, booleans or None when they read as one, text otherwise."""
    path = tmp_path / "unquoted.chp"
    path.write_text("\n".join([
        "#", "d 3",
        "N1 0 prob=0.25 noise_channel=f",
        "MUL 0 a=2",
        "M 0",
        "DETECTOR expr=rec[-1] label=7",
        "DETECTOR expr=rec[-1] label=None",
        "DETECTOR expr=rec[-1] label=",
        "",
    ]))
    c = read_circuit(str(path))
    assert [op.params for op in c.operations] == [
        {"prob": 0.25, "noise_channel": "f"}, {"a": 2}, {},
        {"expr": "rec[-1]", "label": 7}, {"expr": "rec[-1]", "label": None}, {"expr": "rec[-1]", "label": ""}]


def test_gate_lines_split_like_shlex():
    """read_circuit used shlex.split; its own splitter gives the same tokens, and also tells quoted values apart."""
    rng = random.Random(0)
    alphabet = ["a", "b", "0", "1", "=", '"', "'", "\\", " ", " ", "\t", "#", "[", "-", ",", "."]
    for _ in range(20000):
        line = "".join(rng.choice(alphabet) for _ in range(rng.randrange(14)))
        try:
            expected = shlex.split(line)
        except ValueError:
            with pytest.raises(ValueError):
                _split_gate_line(line)
            continue
        assert [word if value is None else f"{word}={value}" for word, value, _ in _split_gate_line(line)] == expected
    assert _split_gate_line('DETECTOR expr="rec[-1] + \\"1\\"" label=5 x="5"') == [
        ("DETECTOR", None, False), ("expr", 'rec[-1] + "1"', True), ("label", "5", False), ("x", "5", True)]


# --------------------------------------------------------------------------
# Cirq conversion


# What `from sdim.circuit_io import *` exported before Cirq was imported lazily (sdim 1.3.x)
_CIRCUIT_IO_STAR_NAMES = [
    "Circuit", "GeneralizedCNOTGate", "GeneralizedCNOTGateInverse", "GeneralizedCZGate", "GeneralizedCZGateInverse",
    "GeneralizedHadamardGate", "GeneralizedHadamardGateInverse", "GeneralizedMultiplicationGate",
    "GeneralizedPhaseShiftGate", "GeneralizedPhaseShiftGateInverse", "GeneralizedXPauliGate",
    "GeneralizedXPauliGateInverse", "GeneralizedZPauliGate", "GeneralizedZPauliGateInverse", "IdentityGate",
    "circuit_to_cirq_circuit", "cirq", "cirq_statevector_from_circuit", "generate_cnot_matrix", "generate_h_matrix",
    "generate_identity_matrix", "generate_m_matrix", "generate_p_matrix", "generate_tau", "generate_x_matrix",
    "generate_z_matrix", "isprime", "np", "os", "product", "read_circuit", "shlex", "write_circuit",
]


def test_sdim_unitary_and_star_imports_still_work():
    """
    Importing Cirq lazily made sdim.unitary missing after `import sdim`, dropped unitary from
    `from sdim import *`, and left `from sdim.circuit_io import *` with 9 of its 33 names.
    Checked in a fresh interpreter, since other tests import sdim.unitary.
    """
    import sdim
    root = os.path.dirname(os.path.dirname(os.path.abspath(sdim.__file__)))
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    code = "\n".join([
        "import sys, sdim",
        "assert sdim.__file__.startswith(sys.argv[1]), sdim.__file__",
        "print(sdim.unitary.GeneralizedHadamardGate.__name__)",
        "names = {}",
        "exec('from sdim.circuit_io import *', names)",
        "print(sorted(name for name in names if name != '__builtins__'))",
        "names = {}",
        "exec('from sdim import *', names)",
        "print('unitary' in names)",
    ])
    out = subprocess.run([sys.executable, "-c", code, root], env=env, capture_output=True, text=True,
                         cwd=os.path.dirname(root))
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[0] == "GeneralizedHadamardGate"
    assert set(_CIRCUIT_IO_STAR_NAMES) <= set(ast.literal_eval(lines[1]))
    assert lines[2] == "True"


def test_cirq_names_still_resolve_from_circuit_io():
    import sdim
    import sdim.circuit_io as circuit_io
    import sdim.unitary as unitary
    from sdim.circuit_io import GeneralizedHadamardGate, IdentityGate
    assert sdim.unitary is unitary
    assert GeneralizedHadamardGate is unitary.GeneralizedHadamardGate and IdentityGate is unitary.IdentityGate
    assert circuit_io.cirq is cirq
    assert GeneralizedSwapGate.__module__ == "sdim.circuit_io" and issubclass(GeneralizedSwapGate, cirq.Gate)
    with pytest.raises(AttributeError):
        circuit_io.no_such_name


def test_cirq_conversion_handles_every_gate_but_noise():
    """SWAP, M_X, RESET, TICK and LOGICAL_OBSERVABLE used to raise NotImplementedError, DETECTOR a TypeError."""
    c = Circuit(3, 3)
    c.add_gate("H", 0)
    c.add_gate("SWAP", 0, 1)
    c.add_gate("TICK")
    c.add_gate("M_X", 1)
    c.add_gate("RESET", 2)
    c.add_gate("M", [0, 1, 2])
    c.add_gate("DETECTOR", expr="rec[-1]")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-2]")
    for measurement in (False, True):
        cc = circuit_to_cirq_circuit(c, measurement=measurement)
        num_measurements = sum(1 for op in cc.all_operations() if cirq.is_measurement(op))
        assert num_measurements == (5 if measurement else 0)
        assert sum(1 for op in cc.all_operations() if isinstance(op.gate, cirq.ResetChannel)) == 1


def test_cirq_conversion_leaves_noise_out_by_default():
    """N1 was always converted to an identity (the noiseless state); N2 now is too instead of raising ValueError."""
    noisy = Circuit(2, 3)
    noisy.add_gate("H", 0)
    noisy.add_gate("N1", 0, prob=0.1)
    noisy.add_gate("CNOT", 0, 1)
    noisy.add_gate("N2", 0, 1, prob=0.1)
    clean = Circuit(2, 3)
    clean.add_gate("H", 0)
    clean.add_gate("CNOT", 0, 1)
    def ops(c):
        return [(type(op.gate).__name__, op.qubits) for op in circuit_to_cirq_circuit(c).all_operations()]
    assert ops(noisy) == ops(clean)
    expected = np.zeros(9)
    expected[[0, 4, 8]] = 1 / math.sqrt(3)
    np.testing.assert_allclose(cirq_statevector_from_circuit(noisy), expected, atol=1e-6)
    only_noise = Circuit(2, 3)
    only_noise.add_gate("N1", 0, prob=0.1)
    only_noise.add_gate("N2", 0, 1, prob=0.1)
    assert [name for name, _ in ops(only_noise)] == ["IdentityGate", "IdentityGate"]  # an identity on each idle qudit
    for gate in ("N1", "N2"):
        c = Circuit(2, 3)
        c.add_gate(gate, *([0] if gate == "N1" else [0, 1]), prob=0.1)
        with pytest.raises(NotImplementedError, match=gate):
            circuit_to_cirq_circuit(c, ignore_noise=False)
        with pytest.raises(NotImplementedError, match=gate):
            cirq_statevector_from_circuit(c, ignore_noise=False)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_cirq_swap_permutes_qudits(d):
    unitary = cirq.unitary(GeneralizedSwapGate(d))
    for i in range(d):
        for j in range(d):
            column = unitary[:, i * d + j]
            assert column[j * d + i] == 1 and np.count_nonzero(column) == 1
    c = Circuit(2, d)
    c.add_gate("X", 0)
    c.add_gate("SWAP", 0, 1)
    state = cirq_statevector_from_circuit(c)
    assert abs(state[1]) == pytest.approx(1)  # |0, 1>


def test_cirq_mul_reduces_its_scalar():
    c = Circuit(1, 5)
    c.add_gate("X", 0)
    c.add_gate("MUL", 0, a=10 ** 19 + 2)
    state = cirq_statevector_from_circuit(c)
    assert abs(state[2]) == pytest.approx(1)


def _cirq_records(c, reps, seed):
    result = cirq.Simulator(seed=seed).run(circuit_to_cirq_circuit(c, measurement=True), repetitions=reps)
    rows = []
    for s in range(reps):
        row = []
        for q in range(c.num_qudits):
            if f"m_{q}" in result.records:
                row.extend(int(v) for v in result.records[f"m_{q}"][s, :, 0])
        rows.append(tuple(row))
    return rows


def _sdim_records(c, shots):
    measurements, _ = Program(c).simulate(shots=shots + 1)
    return [tuple(rnd[s].measurement_value for q in measurements for rnd in q) for s in range(1, shots + 1)]


def _two_sample_p_value(a, b):
    from scipy.stats import chi2
    ca, cb = collections.Counter(a), collections.Counter(b)
    na, nb = len(a), len(b)
    keys = set(ca) | set(cb)
    stat = sum((ca[k] * math.sqrt(nb / na) - cb[k] * math.sqrt(na / nb)) ** 2 / (ca[k] + cb[k]) for k in keys)
    return chi2.sf(stat, max(len(keys) - 1, 1))


def test_cirq_records_follow_sdim_semantics():
    """M_X leaves the qudit in the X eigenstate it reports, and RESET records the pre-reset value."""
    d = 3
    c = Circuit(2, d)
    c.add_gate("H", 0)
    c.add_gate("M_X", 0)      # H|0> is an X eigenstate: outcome 0, deterministically
    c.add_gate("M_X", 0)      # repeats it
    c.add_gate("X", 1)
    c.add_gate("RESET", 1)    # records 1, then |0>
    c.add_gate("M", 1)
    rows = _cirq_records(c, 20, seed=0)
    assert set(rows) == {(0, 0, 1, 0)}
    np.random.seed(0)
    assert set(_sdim_records(c, 20)) == {(0, 0, 1, 0)}


@pytest.mark.parametrize("d,seed", [(2, 0), (3, 1), (5, 2), (4, 3)])
def test_cirq_distributions_match_sdim(d, seed):
    rng = random.Random(seed)
    one = ["H", "H_INV", "P", "P_INV", "X", "Z_INV"]
    two = ["CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"]
    c = Circuit(2, d)
    for _ in range(14):
        r = rng.random()
        if r < 0.45:
            c.add_gate(rng.choice(one), rng.randrange(2))
        elif r < 0.75:
            c.add_gate(rng.choice(two), *rng.sample(range(2), 2))
        elif r < 0.85:
            c.add_gate("MUL", rng.randrange(2), a=rng.choice([k for k in range(1, d) if math.gcd(k, d) == 1]))
        else:
            c.add_gate(rng.choice(["M_X", "RESET", "TICK"]), rng.randrange(2))
    c.add_gate("M_X", 0)
    c.add_gate("M", 1)
    np.random.seed(seed)
    random.seed(seed)
    assert _two_sample_p_value(_cirq_records(c, 600, seed), _sdim_records(c, 600)) > 1e-4
