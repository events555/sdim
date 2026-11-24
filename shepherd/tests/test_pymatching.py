"""Tests of the "pymatching" decoder (binarized MWPM lifted to Z_d), on the qudit repetition code."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

pytest.importorskip("pymatching")

from sdim.dem import DetectorErrorModel, ErrorMechanism

from shepherd.collect import collect
from shepherd.decoder import Decoder
from shepherd.decoders.pymatching_binarized_qudit_syndromes import (
    CompiledMatchingDecoder, MatchingDecoder)
from shepherd.task import Task

from test_collect import repetition_circuit


def compiled_for(d=3, p=0.01, rounds=2, **settings) -> tuple[CompiledMatchingDecoder, DetectorErrorModel]:
    dem = DetectorErrorModel.from_circuit(repetition_circuit(d=d, p=p, rounds=rounds))
    return MatchingDecoder(**settings).compile_decoder_for_dem(dem), dem


def test_registered_under_pymatching():
    assert Decoder.lookup("pymatching") is MatchingDecoder
    assert isinstance(Decoder.create("pymatching", hyperedges="drop"), MatchingDecoder)


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError, match="hyperedges"):
        MatchingDecoder(hyperedges="ignore")
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match="matching_passes"):
            MatchingDecoder(matching_passes=bad)


def test_settings_are_in_the_fingerprint():
    base = MatchingDecoder().fingerprint()
    assert base != MatchingDecoder(hyperedges="drop").fingerprint()
    assert base != MatchingDecoder(matching_passes=1).fingerprint()


def test_rematching_resolves_charge_that_one_pass_leaves():
    """D3=2, D5=2 (d = 3) is two boundary errors; one pass pairs D3 with D5 and cannot lift it."""
    compiled, dem = compiled_for(d=3, rounds=3)
    single, _ = compiled_for(d=3, rounds=3, matching_passes=1)
    syndrome = np.zeros((1, dem.num_detectors), dtype=np.int64)
    syndrome[0, [3, 5]] = 2
    assert single.decode_shots(syndrome)[0, 0] == 0
    assert compiled.decode_shots(syndrome)[0, 0] == 2


@pytest.mark.parametrize("d", [3, 5, 7])
def test_every_single_edge_fault_is_corrected_exactly(d):
    """A syndrome made by one edge at power a must be decoded to that edge's observable shift * a."""
    compiled, dem = compiled_for(d=d)
    nd = dem.num_detectors
    assert len(compiled.edge_u) > 0
    for k in range(len(compiled.edge_u)):
        for a in range(1, d):
            syndrome = np.zeros((1, nd), dtype=np.int64)
            syndrome[0, compiled.edge_u[k]] += a * compiled.edge_cu[k]
            if compiled.edge_v[k] >= 0:
                syndrome[0, compiled.edge_v[k]] += a * compiled.edge_cv[k]
            syndrome %= d
            got = compiled.decode_shots(syndrome)[0]
            np.testing.assert_array_equal(got, a * compiled.edge_obs[k] % d)


def test_hyperedges_are_decomposed():
    compiled, dem = compiled_for(d=3)
    lines = dem.to_lines()
    nd = dem.num_detectors
    assert any(sum(t < nd for t in m.generators[0]) >= 3 for m in lines.mechanisms)
    assert compiled.num_dropped_hyperedges == 0


def test_undecomposable_hyperedge_raises_or_is_dropped():
    # D0, D1, D2 each have a boundary edge, plus one line D0+D1+D2+L0 that no sum of them makes
    # (the edges move no observable).
    d = 3
    mechs = [ErrorMechanism(0.01, [{t: 1}]) for t in range(3)]
    mechs.append(ErrorMechanism(0.01, [{0: 1, 1: 1, 2: 1, 3: 1}]))
    dem = DetectorErrorModel(d, 3, 1, mechs)
    with pytest.raises(ValueError, match="hyperedge"):
        MatchingDecoder(hyperedges="raise").compile_decoder_for_dem(dem)
    assert MatchingDecoder().compile_decoder_for_dem(dem).num_dropped_hyperedges == 1
    assert MatchingDecoder(hyperedges="drop").compile_decoder_for_dem(dem).num_dropped_hyperedges == 1


def test_compiled_decoder_survives_pickling():
    compiled, dem = compiled_for(d=5)
    det, _ = dem.sample(500, seed=3)
    again = pickle.loads(pickle.dumps(compiled))
    np.testing.assert_array_equal(again.decode_shots(det), compiled.decode_shots(det))


def test_empty_model_predicts_nothing():
    dem = DetectorErrorModel(3, 2, 1, [])
    compiled = MatchingDecoder().compile_decoder_for_dem(dem)
    assert compiled.decode_shots(np.ones((4, 2), dtype=np.int64)).tolist() == [[0]] * 4


@pytest.mark.parametrize("d", [3, 5, 7])
def test_matching_beats_no_decoding(d):
    circuit = repetition_circuit(d=d, p=0.02, rounds=3)
    tasks = [Task(circuit, name, json_metadata={"decoder": name}) for name in ("none", "pymatching")]
    none, mwpm = collect(tasks=tasks, num_workers=0, max_shots=20_000)
    # Measured: about 4-5x fewer errors at d = 3, 5, 7 with the default matching_passes.
    assert mwpm.errors < 0.4 * none.errors, (none.errors, mwpm.errors)


def test_runs_in_worker_processes():
    task = Task(repetition_circuit(p=0.02), "pymatching", json_metadata={})
    (stats,) = collect(tasks=[task], num_workers=2, max_shots=4_000, start_batch_shots=500)
    assert stats.shots == 4_000
