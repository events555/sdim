"""End-to-end tests of Task -> worker -> collect -> results file, on a qutrit repetition code."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

import numpy as np
import pytest
from sdim import Circuit

from shepherd.collect import collect
from shepherd.stats import TaskStats, read_stats_file, wilson_interval
from shepherd.task import Task


def repetition_circuit(d: int = 3, p: float = 0.01, rounds: int = 2) -> Circuit:
    """3-qudit repetition code (checks Z0 Z1^-1 and Z1 Z2^-1), data 0..2, ancillas 3..4."""
    c = Circuit(5, d)
    for r in range(rounds):
        for a, (q0, q1) in zip((3, 4), ((0, 1), (1, 2))):
            c.add_gate("CNOT", q0, a)
            c.add_gate("CNOT_INV", q1, a)
            c.add_gate("N2", q1, a, prob=p)
        c.add_gate("TICK")
        c.add_gate("N1", [3, 4], noise_channel="f", prob=p)
        c.add_gate("M", [3, 4])
        c.add_gate("RESET", [3, 4])
        if r == 0:
            c.add_gate("DETECTOR", expr="rec[-2]")
            c.add_gate("DETECTOR", expr="rec[-1]")
        else:
            c.add_gate("DETECTOR", expr="rec[-2] - rec[-4]")
            c.add_gate("DETECTOR", expr="rec[-1] - rec[-3]")
    c.add_gate("N1", [0, 1, 2], noise_channel="f", prob=p)
    c.add_gate("M", [0, 1, 2])
    c.add_gate("DETECTOR", expr="rec[-3] - rec[-2] - rec[-5]")
    c.add_gate("DETECTOR", expr="rec[-2] - rec[-1] - rec[-4]")
    c.add_gate("LOGICAL_OBSERVABLE", expr="rec[-1]")
    return c


# -- test decoders (module level, so worker processes can unpickle them) -------------------------
# Most tests use the registered "none" decoder. LookupDecoder is a plain class, not a Decoder
# subclass, so it also checks that any object with compile_decoder_for_dem works as a custom one.

class _LookupCompiled:
    def __init__(self, table, num_observables):
        self.table = table
        self.num_observables = num_observables

    def decode_shots(self, detections):
        out = np.zeros((detections.shape[0], self.num_observables), dtype=np.int64)
        for row, syndrome in enumerate(detections):
            out[row] = self.table.get(syndrome.tobytes(), 0)
        return out


class LookupDecoder:
    """Predicts the most frequent observable value seen with each syndrome in training samples."""

    def __init__(self, training_shots: int = 200_000):
        self.training_shots = training_shots

    def compile_decoder_for_dem(self, dem):
        det, obs = dem.sample(self.training_shots, seed=12345)
        counts: dict[bytes, Counter] = defaultdict(Counter)
        for s, o in zip(det, obs):
            counts[s.tobytes()][o.tobytes()] += 1
        table = {s: np.frombuffer(c.most_common(1)[0][0], dtype=np.int64) for s, c in counts.items()}
        return _LookupCompiled(table, dem.num_observables)


DECODERS = {"lookup": LookupDecoder()}


def task(p=0.01, decoder="none", **kw) -> Task:
    return Task(repetition_circuit(p=p), decoder, json_metadata={"p": p, "decoder": decoder}, **kw)


def file_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# -- tests ---------------------------------------------------------------------------------------

def test_strong_id_is_stable_and_sensitive():
    assert task().strong_id() == task().strong_id()
    ids = {task().strong_id(), task(p=0.02).strong_id(), task(decoder="lookup").strong_id(),
           task(postselection=(0,)).strong_id(),
           Task(repetition_circuit(), "none", json_metadata={"other": 1}).strong_id()}
    assert len(ids) == 5


def test_composite_dimension_is_rejected():
    with pytest.raises(ValueError, match="prime"):
        Task(Circuit(2, 4), "none")


def test_unknown_decoder_fails_before_sampling():
    with pytest.raises(KeyError):
        collect(tasks=[task(decoder="no_such_decoder")], num_workers=0, max_shots=10)


def test_inline_collect_stops_exactly_at_max_shots():
    (s,) = collect(tasks=[task()], num_workers=0, max_shots=5000, custom_decoders=DECODERS)
    assert s.shots == 5000 and s.discards == 0
    assert 0 < s.errors < s.shots


def test_duplicate_tasks_run_once():
    stats = collect(tasks=[task(), task()], num_workers=0, max_shots=1000, custom_decoders=DECODERS)
    assert len(stats) == 1 and stats[0].shots == 1000


def test_resume_only_samples_what_is_missing(tmp_path):
    path = tmp_path / "results.jsonl"
    collect(tasks=[task()], num_workers=0, max_shots=2000, save_resume_filepath=path,
            custom_decoders=DECODERS)
    rows_after_first = len(file_rows(path))

    # Same limit again: nothing left to do, so nothing is written.
    (s,) = collect(tasks=[task()], num_workers=0, max_shots=2000, save_resume_filepath=path,
                   custom_decoders=DECODERS)
    assert s.shots == 2000 and len(file_rows(path)) == rows_after_first

    # Raised limit: only the extra 3000 shots are sampled, and the file sums to the new total.
    (s,) = collect(tasks=[task()], num_workers=0, max_shots=5000, save_resume_filepath=path,
                   custom_decoders=DECODERS)
    assert s.shots == 5000
    assert read_stats_file(path)[s.strong_id].shots == 5000
    assert sum(r["shots"] for r in file_rows(path)) == 5000


def test_tasks_in_one_file_stay_separate(tmp_path):
    path = tmp_path / "results.jsonl"
    stats = collect(tasks=[task(0.01), task(0.02)], num_workers=0, max_shots=1500,
                    save_resume_filepath=path, custom_decoders=DECODERS)
    on_disk = read_stats_file(path)
    assert {s.strong_id for s in stats} == set(on_disk)
    for s in stats:
        assert on_disk[s.strong_id].shots == 1500
        assert on_disk[s.strong_id].json_metadata == s.json_metadata


def test_truncated_last_row_is_skipped_and_run_continues(tmp_path):
    path = tmp_path / "results.jsonl"
    collect(tasks=[task()], num_workers=0, max_shots=1000, save_resume_filepath=path,
            custom_decoders=DECODERS)
    with open(path, "a") as f:
        f.write('{"strong_id": "abc", "sho')            # a run killed mid-write
    (s,) = collect(tasks=[task()], num_workers=0, max_shots=2000, save_resume_filepath=path,
                   custom_decoders=DECODERS)
    assert s.shots == 2000
    assert read_stats_file(path)[s.strong_id].shots == 2000


def test_max_errors_stops_early_without_large_overshoot():
    (s,) = collect(tasks=[task(p=0.05)], num_workers=0, max_shots=10**7, max_errors=100,
                   custom_decoders=DECODERS)
    assert s.errors >= 100
    assert s.shots < 10**7
    assert s.errors < 100 + 60          # at most about one small batch past the limit


def test_lookup_decoder_beats_no_decoding():
    none, lookup = collect(tasks=[task(p=0.03, decoder="none"), task(p=0.03, decoder="lookup")],
                           num_workers=0, max_shots=40_000, custom_decoders=DECODERS)
    assert none.decoder == "none" and lookup.decoder == "lookup"
    assert lookup.error_rate_interval()[1] < none.error_rate_interval()[0]


def test_postselection_discards_shots_with_any_flagged_detector():
    t = task(p=0.02, postselection=tuple(range(6)))
    (s,) = collect(tasks=[t], num_workers=0, max_shots=40_000, custom_decoders=DECODERS)
    det, _ = t.dem().sample(200_000, seed=7)
    expected = det.any(axis=1).mean()
    sigma = math.sqrt(expected * (1 - expected) / s.shots)
    assert abs(s.discard_rate - expected) < 6 * sigma + 0.005
    assert s.kept == s.shots - s.discards > 0


def test_worker_processes_agree_with_direct_sampling():
    t = task(p=0.02)
    (s,) = collect(tasks=[t], num_workers=2, max_shots=30_000, custom_decoders=DECODERS)
    assert s.shots == 30_000
    _, obs = t.dem().sample(300_000, seed=3)
    expected = (obs != 0).any(axis=1).mean()          # "none" is wrong exactly when obs != 0
    lo, hi = wilson_interval(s.errors, s.shots, z=5)
    assert lo <= expected <= hi


def test_task_stats_rates():
    s = TaskStats("x", "none", None, shots=100, errors=5, discards=50)
    assert s.kept == 50 and s.logical_error_rate == 0.1 and s.discard_rate == 0.5
    with pytest.raises(ValueError):
        s + TaskStats("y", "none")
