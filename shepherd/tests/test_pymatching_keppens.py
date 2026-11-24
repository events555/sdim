"""Tests of the "pymatching_keppens" decoder (Keppens et al.'s eigenvalue-node MWPM)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

pytest.importorskip("pymatching")

from sdim.dem import DetectorErrorModel, ErrorMechanism

from shepherd.collect import collect
from shepherd.decoder import Decoder
from shepherd.decoders.pymatching_keppens import KeppensMatchingDecoder
from shepherd.task import Task

from test_collect import repetition_circuit


def compiled_for(d=3, p=0.01, rounds=2, **settings):
    dem = DetectorErrorModel.from_circuit(repetition_circuit(d=d, p=p, rounds=rounds))
    return KeppensMatchingDecoder(**settings).compile_decoder_for_dem(dem), dem


def test_registered_under_pymatching_keppens():
    assert Decoder.lookup("pymatching_keppens") is KeppensMatchingDecoder
    with pytest.raises(ValueError, match="hyperedges"):
        KeppensMatchingDecoder(hyperedges="ignore")


@pytest.mark.parametrize("d", [3, 5])
def test_graph_has_d_minus_1_nodes_per_detector(d):
    compiled, dem = compiled_for(d=d)
    nodes = set(compiled.edge_a.tolist()) | set(compiled.edge_b[compiled.edge_b >= 0].tolist())
    assert nodes == set(range(dem.num_detectors * (d - 1)))


def test_fired_nodes():
    compiled, dem = compiled_for(d=5)
    det = np.zeros((1, dem.num_detectors), dtype=np.int64)
    det[0, 0], det[0, 3] = 2, 4
    nodes = compiled.fired_nodes(det)[0]
    assert np.flatnonzero(nodes).tolist() == [0 * 4 + 1, 3 * 4 + 3]


@pytest.mark.parametrize("d", [3, 5, 7])
def test_every_edge_is_decoded_to_its_observable_shift(d):
    compiled, dem = compiled_for(d=d)
    for a, b, obs in zip(compiled.edge_a, compiled.edge_b, compiled.edge_obs):
        det = np.zeros((1, dem.num_detectors), dtype=np.int64)
        for n in (a, b):
            if n >= 0:
                det[0, n // (d - 1)] = n % (d - 1) + 1
        np.testing.assert_array_equal(compiled.decode_shots(det)[0], obs)


def test_values_steer_the_matching():
    """D3=2, D5=2 (d = 3) has no (3,2)-(5,2) edge, so both go to the boundary: L0 = 1 + 1 = 2."""
    compiled, dem = compiled_for(d=3, rounds=3)
    det = np.zeros((1, dem.num_detectors), dtype=np.int64)
    det[0, [3, 5]] = 2
    assert compiled.decode_shots(det)[0, 0] == 2


def test_unmatchable_shots_predict_zero_without_failing_the_batch():
    # Two detectors joined by one edge and no boundary: a single event has no perfect matching.
    dem = DetectorErrorModel(3, 2, 1, [ErrorMechanism(0.01, [{0: 1, 1: 2, 2: 1}])])
    compiled = KeppensMatchingDecoder().compile_decoder_for_dem(dem)
    det = np.array([[1, 0], [1, 2], [0, 0]], dtype=np.int64)
    assert compiled.decode_shots(det)[:, 0].tolist() == [0, 1, 0]


def test_compiled_decoder_survives_pickling():
    compiled, dem = compiled_for(d=5)
    det, _ = dem.sample(500, seed=3)
    again = pickle.loads(pickle.dumps(compiled))
    np.testing.assert_array_equal(again.decode_shots(det), compiled.decode_shots(det))


@pytest.mark.parametrize("d", [3, 5])
def test_keppens_beats_no_decoding(d):
    circuit = repetition_circuit(d=d, p=0.02, rounds=3)
    tasks = [Task(circuit, name, json_metadata={}) for name in ("none", "pymatching_keppens")]
    none, keppens = collect(tasks=tasks, num_workers=0, max_shots=20_000)
    # Measured: about 2.4x fewer errors at d = 5 (the binarized decoder gets 4-5x here; chains
    # through intermediate detectors are what eigenvalue-node matching cannot model).
    assert keppens.errors < 0.6 * none.errors, (none.errors, keppens.errors)


def test_runs_in_worker_processes():
    task = Task(repetition_circuit(p=0.02), "pymatching_keppens", json_metadata={})
    (stats,) = collect(tasks=[task], num_workers=2, max_shots=4_000, start_batch_shots=500)
    assert stats.shots == 4_000
