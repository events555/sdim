"""Tests for `DetectorErrorModel.compile_sampler`, and that `DetectorErrorModel.sample` still returns what it did.

A compiled sampler checks and packs the model once. Its calls continue one random stream: their rows, put
together, are the rows `sample(total, seed)` returns for the same seed.
"""

import copy
import hashlib
import itertools
import threading

import numpy as np
import pytest

import sdim.dem as dem_module
from sdim.dem import CompiledDemSampler, DetectorErrorModel, ErrorMechanism
from tests.test_dem_fast_paths import _exact_distribution


def _model(d, n_det, n_obs, n_mech, probs):
    """A model whose entries follow a fixed arithmetic pattern, so it needs no random number generator."""
    n = n_det + n_obs
    mechs = []
    for k in range(n_mech):
        gens = []
        for j in range(1 + k % 3):
            gen = {}
            for i in range(1 + (k + j) % n):
                gen[(5 * k + 3 * j + 7 * i) % n] = (1 + k + 2 * j + 11 * i) % d or 1
            gens.append(gen)
        mechs.append(ErrorMechanism(probs[k % len(probs)], gens, f"m{k}"))
    return DetectorErrorModel(d, n_det, n_obs, mechs)


# Thinning within a bin (0.3, 0.31, 0.32), pi = 1, pi = 0, and tiny probabilities that never fire.
_MIXED = [0.3, 0.31, 0.32, 0.05, 1.0, 0.0, 1e-30, 0.4, 1e-3, 2.0 ** -200, 0.7, 1e-6]
_MANY_BINS = [0.6 / 1.3 ** k * (1 - 0.01 * (k % 7)) for k in range(60)] + [2.0 ** (-200 - k) for k in range(20)]


def _reference_models():
    return {
        "readme": DetectorErrorModel(1000003, 2, 1, [
            ErrorMechanism(0.01000000999998, [{0: 1}], "N1[f]@3:q0"),
            ErrorMechanism(0.01000000999998, [{0: 1, 2: 1000002}], "N1[f]@4:q1"),
            ErrorMechanism(0.001, [{1: 1, 2: 1}, {0: 1, 1: 1}], "N2@7:q1,q2")]),
        "d2": _model(2, 5, 1, 30, _MIXED),
        "d3": _model(3, 3, 1, 25, _MIXED),
        "d4": _model(4, 4, 2, 25, _MIXED),
        "d5_many_bins": _model(5, 6, 1, 400, _MANY_BINS),
        "d7": _model(7, 3, 1, 40, _MIXED),
        "d2147483647": _model(2 ** 31 - 1, 4, 1, 20, _MIXED),
    }


# sha256 (first 16 hex digits) of `sample(shots, seed)` over _SHOTS x _SEEDS, from the sampler before
# compile_sampler existed, under numpy 2.4.6 and 1.26.4 alike.
_SHOTS = (1, 255, 256, 257, 1000, 3000)
_SEEDS = (0, 1, 2 ** 64 + 3)
_DIGESTS = {
    "readme": "d4f69d5e3f14d5d0",
    "d2": "808d7d70b2c7b427",
    "d3": "5a6759bdfeabfc18",
    "d4": "240c4454e429b1ad",
    "d5_many_bins": "e18e076d4f197a06",
    "d7": "878a1269ca89b5f1",
    "d2147483647": "f7594aca4785bd3e",
}


def _digest(dem):
    h = hashlib.sha256()
    for shots in _SHOTS:
        for seed in _SEEDS:
            det, obs = dem.sample(shots, seed=seed)
            h.update(det.astype("<i8").tobytes())
            h.update(obs.astype("<i8").tobytes())
    return h.hexdigest()[:16]


def _calls(sampler, sizes):
    """The rows of consecutive calls of the given sizes, put together."""
    parts = [sampler.sample(n) for n in sizes]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


_SPLIT = (1, 255, 256, 3, 700, 0, 1000, 2785)


# --------------------------------------------------------------------------
# DetectorErrorModel.sample is unchanged


@pytest.mark.parametrize("name", sorted(_DIGESTS))
def test_sample_returns_what_it_did(name):
    assert _digest(_reference_models()[name]) == _DIGESTS[name]


@pytest.mark.parametrize("threads", [2, 3, 16])
def test_sample_returns_what_it_did_on_any_number_of_threads(monkeypatch, threads):
    monkeypatch.setattr(dem_module, "_SAMPLE_PARALLEL_WORK", -1.0)
    monkeypatch.setattr(dem_module, "_thread_count", lambda: threads)
    for name, dem in _reference_models().items():
        assert _digest(dem) == _DIGESTS[name], name


def test_sample_sees_changes_made_in_place():
    """A model's mechanisms are plain lists and dicts that can change in place, so sample() keeps no cache."""
    dem = DetectorErrorModel(5, 2, 0, [ErrorMechanism(1.0, [{0: 1, 1: 1}], "a")])
    det, _ = dem.sample(200, seed=1)
    np.testing.assert_array_equal(det[:, 1], det[:, 0])
    dem.mechanisms[0].generators[0][1] = 2
    det, _ = dem.sample(200, seed=1)
    np.testing.assert_array_equal(det[:, 1], 2 * det[:, 0] % 5)
    dem.mechanisms[0].probability = 0.0
    assert not dem.sample(200, seed=1)[0].any()


# --------------------------------------------------------------------------
# The compiled sampler's stream


@pytest.mark.parametrize("name", sorted(_DIGESTS))
def test_compiled_calls_continue_the_stream_of_one_sample_call(name):
    dem = _reference_models()[name]
    for seed in (0, 5, 2 ** 64 + 3):
        det, obs = dem.sample(sum(_SPLIT), seed=seed)
        got = _calls(dem.compile_sampler(seed), _SPLIT)
        np.testing.assert_array_equal(got[0], det)
        np.testing.assert_array_equal(got[1], obs)
        # Whole blocks only, and one call for everything.
        for sizes in ([256] * 3 + [5000 - 768], [sum(_SPLIT)]):
            got = _calls(dem.compile_sampler(seed), sizes)
            np.testing.assert_array_equal(got[0], det)
            np.testing.assert_array_equal(got[1], obs)


def test_compiled_sampler_is_reproducible_call_by_call():
    dem = _reference_models()["d3"]
    a, b, c = dem.compile_sampler(11), dem.compile_sampler(11), dem.compile_sampler(12)
    for n in (300, 1, 7, 256, 1000):
        x, y, z = a.sample(n), b.sample(n), c.sample(n)
        np.testing.assert_array_equal(x[0], y[0])
        np.testing.assert_array_equal(x[1], y[1])
        assert n < 100 or not np.array_equal(x[0], z[0])
    # Consecutive calls draw new shots, and so do two samplers without a seed.
    assert not np.array_equal(a.sample(1000)[0], a.sample(1000)[0])
    assert not np.array_equal(dem.compile_sampler().sample(1000)[0], dem.compile_sampler().sample(1000)[0])


@pytest.mark.parametrize("threads", [1, 2, 3, 5, 16])
def test_compiled_sampler_does_not_depend_on_thread_count(monkeypatch, threads):
    dem = _reference_models()["d5_many_bins"]
    sizes = (1, 2000, 300, 5 * 256, 4000)
    monkeypatch.setattr(dem_module, "_SAMPLE_PARALLEL_WORK", float("inf"))
    expected = _calls(dem.compile_sampler(3), sizes)
    monkeypatch.setattr(dem_module, "_SAMPLE_PARALLEL_WORK", -1.0)
    monkeypatch.setattr(dem_module, "_thread_count", lambda: threads)
    for tasks_per_thread in (1, 3, 100):
        monkeypatch.setattr(dem_module, "_SAMPLE_TASKS_PER_THREAD", tasks_per_thread)
        got = _calls(dem.compile_sampler(3), sizes)
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])


def test_calls_from_several_threads_take_turns():
    """Each call gets a run of consecutive rows of the stream, and no two calls get the same run."""
    dem = _reference_models()["d7"]
    det, _ = dem.sample(80 * 100, seed=4)
    sampler = dem.compile_sampler(4)
    results = []

    def work():
        for _ in range(20):
            results.append(sampler.sample(100)[0])

    workers = [threading.Thread(target=work) for _ in range(4)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    runs = {det[k:k + 100].tobytes(): k for k in range(0, len(det), 100)}
    assert sorted(runs[r.tobytes()] for r in results) == list(range(0, len(det), 100))


def test_a_call_that_raises_leaves_the_stream_where_it_was(monkeypatch):
    dem = _reference_models()["d7"]
    det, obs = dem.sample(1000, seed=5)
    sampler = dem.compile_sampler(5)
    first = sampler.sample(100)
    run_tasks = dem_module._run_tasks
    calls = []

    def interrupted(*args):
        # The call below draws one whole block, then fails on its last, partly returned block.
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        run_tasks(*args)

    monkeypatch.setattr(dem_module, "_run_tasks", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        sampler.sample(600)
    monkeypatch.setattr(dem_module, "_run_tasks", run_tasks)
    rest = sampler.sample(900)
    np.testing.assert_array_equal(np.concatenate([first[0], rest[0]]), det)
    np.testing.assert_array_equal(np.concatenate([first[1], rest[1]]), obs)


@pytest.mark.parametrize("error, failing_call", [(KeyboardInterrupt, 1), (MemoryError, 2)])
def test_a_call_that_raises_in_the_states_leaves_the_stream_where_it_was(monkeypatch, error, failing_call):
    """A sampler keeps 64 states ahead; an error while it computes the next ones must not leave the old ones."""
    dem = _reference_models()["d7"]
    # Blocks 64 .. 127 take words from two SeedSequences, so failing call 2 comes after call 1 succeeded.
    monkeypatch.setattr(dem_module, "_STREAM_EPOCH", 100)
    det, obs = dem.sample(256 * 140, seed=5)
    sampler = dem.compile_sampler(5)
    first = sampler.sample(256 * 64)
    block_states = dem_module._block_states
    calls = []

    def interrupted(*args):
        calls.append(args)
        if len(calls) == failing_call:
            raise error
        return block_states(*args)

    monkeypatch.setattr(dem_module, "_block_states", interrupted)
    with pytest.raises(error):
        sampler.sample(256)
    monkeypatch.setattr(dem_module, "_block_states", block_states)
    rest = _calls(sampler, (256, 256 * 75))
    np.testing.assert_array_equal(np.concatenate([first[0], rest[0]]), det)
    np.testing.assert_array_equal(np.concatenate([first[1], rest[1]]), obs)


@pytest.mark.parametrize("d", [2, 3, 5, 7])
def test_compiled_sampler_matches_exact_distribution(d):
    stats = pytest.importorskip("scipy.stats")
    n_det, n_obs = (4, 1) if d < 5 else (2, 1)
    mechs = [ErrorMechanism(0.3, [{0: 1}], "a"), ErrorMechanism(0.31, [{0: 1, 1: d - 1}], "b"),
             ErrorMechanism(0.05, [{1: 1}, {2: 1}], "c"), ErrorMechanism(1.0, [{n_det: 1}], "always"),
             ErrorMechanism(0.0, [{1: 1}], "never"), ErrorMechanism(1e-30, [{0: 1}], "tiny"),
             ErrorMechanism(0.2, [{n_det - 1: 1, n_det: 1}, {1: 1}, {0: 1, n_det - 1: d - 1}], "rank 3")]
    dem = DetectorErrorModel(d, n_det, n_obs, mechs)
    sampler = dem.compile_sampler(d)
    shots = 0
    parts = []
    for n in itertools.cycle((1, 37, 256, 1000, 20_000)):
        n = min(n, 200_000 - shots)
        parts.append(sampler.sample(n))
        shots += n
        if shots == 200_000:
            break
    vals = np.concatenate([np.concatenate(p, axis=1) for p in parts])
    n = n_det + n_obs
    counts = np.bincount(np.ravel_multi_index(tuple(vals.T), (d,) * n), minlength=d ** n)
    expected = _exact_distribution(dem) * shots
    assert counts[expected < 1e-9].sum() == 0  # outcomes the model cannot produce
    big = expected >= 5
    chi2 = ((counts[big] - expected[big]) ** 2 / expected[big]).sum()
    assert stats.chi2.sf(chi2, big.sum() - 1) > 1e-4


# --------------------------------------------------------------------------
# Inputs and edge cases


def test_numpy_integer_shots_and_seeds():
    """Unsigned and 8-bit shot counts too, whose negation wraps or overflows in NumPy arithmetic."""
    dem = _reference_models()["d4"]
    det, obs = dem.sample(700, seed=7)
    for shots, seed in ((np.int64(700), np.uint64(7)), (np.int32(700), np.int8(7)), (np.uint16(700), 7),
                        (np.uint32(700), 7), (np.uint64(700), 7), (np.uint8(200), 7), (np.int8(100), 7)):
        n = int(shots)
        got = dem.sample(shots, seed=seed)
        np.testing.assert_array_equal(got[0], det[:n])
        np.testing.assert_array_equal(got[1], obs[:n])
        sampler = dem.compile_sampler(seed)
        got = _calls(sampler, (type(shots)(n // 2), type(shots)(n - n // 2)))
        np.testing.assert_array_equal(got[0], det[:n])
        np.testing.assert_array_equal(got[1], obs[:n])
    with pytest.raises(TypeError):
        dem.compile_sampler(1).sample(2.0)
    with pytest.raises(ValueError, match="-1 shots"):
        dem.compile_sampler(1).sample(-1)


def test_zero_and_one_shot():
    dem = _reference_models()["d3"]
    det, obs = dem.sample(300, seed=2)
    sampler = dem.compile_sampler(2)
    for n in (0, 1, 0, 1, 1, 0):
        got = sampler.sample(n)
        assert got[0].shape == (n, 3) and got[1].shape == (n, 1)
        assert got[0].dtype == np.int64 and got[0].flags.c_contiguous and got[1].flags.c_contiguous
    # The calls of 0 shots drew nothing, so the next call starts at row 3.
    got = sampler.sample(297)
    np.testing.assert_array_equal(got[0], det[3:])
    np.testing.assert_array_equal(got[1], obs[3:])
    one = dem.compile_sampler(2).sample(1)
    np.testing.assert_array_equal(one[0], dem.sample(1, seed=2)[0])
    np.testing.assert_array_equal(one[0], det[:1])
    zero = dem.sample(0, seed=2)
    assert zero[0].shape == (0, 3) and zero[1].shape == (0, 1)


def test_models_without_mechanisms_give_zeros():
    for d in (3, 2 ** 31, 0):
        sampler = DetectorErrorModel(d, 2, 1).compile_sampler(1)
        det, obs = sampler.sample(5)
        assert det.shape == (5, 2) and obs.shape == (5, 1) and not det.any() and not obs.any()
        assert (sampler.num_detectors, sampler.num_observables) == (2, 1)


def test_compile_sampler_checks_the_model_like_sample():
    bad = [DetectorErrorModel(3, 1, 0, [ErrorMechanism(float("nan"), [{0: 1}], "x")]),
           DetectorErrorModel(3, 1, 0, [ErrorMechanism(0.1, [{5: 1}], "x")]),
           DetectorErrorModel(3, 1, 0, [ErrorMechanism(0.1, [{0: 0.5}], "x")]),
           DetectorErrorModel(2 ** 31, 1, 0, [ErrorMechanism(0.1, [{0: 1}], "x")])]
    for dem in bad:
        with pytest.raises(ValueError) as from_sample:
            dem.sample(10, seed=1)
        with pytest.raises(ValueError) as from_compile:
            dem.compile_sampler(1)
        assert str(from_compile.value) == str(from_sample.value)


def test_compiled_sampler_does_the_model_work_once(monkeypatch):
    dem = _reference_models()["d5_many_bins"]
    expected = dem.sample(3000, seed=9)
    sampler = dem.compile_sampler(9)
    assert isinstance(sampler, CompiledDemSampler)

    def refuse(*args):
        raise AssertionError("the model was flattened or planned again")

    monkeypatch.setattr(DetectorErrorModel, "_flatten", refuse)
    monkeypatch.setattr(dem_module, "_sample_plan", refuse)
    got = _calls(sampler, (1000, 1000, 1000))
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])


def test_compiled_sampler_keeps_its_own_copy_of_the_model():
    dem = _reference_models()["d7"]
    before = copy.deepcopy(dem)
    sampler = dem.compile_sampler(6)
    dem.mechanisms[0].generators[0][0] = 3
    dem.mechanisms[1].probability = 1.0
    dem.mechanisms.append(ErrorMechanism(1.0, [{1: 1}], "new"))
    dem.num_detectors = 5
    det, obs = _calls(sampler, (100, 900))
    expected = before.sample(1000, seed=6)
    np.testing.assert_array_equal(det, expected[0])
    np.testing.assert_array_equal(obs, expected[1])


def test_compiled_sampler_widths_are_read_only():
    """The packed targets split into detectors and observables at the compiled widths, so these cannot change."""
    dem = DetectorErrorModel(7, 3, 2, [ErrorMechanism(1.0, [{4: 1}], "always")])
    sampler = dem.compile_sampler(1)
    for name in ("num_detectors", "num_observables"):
        with pytest.raises(AttributeError):
            setattr(sampler, name, 1)
    det, obs = sampler.sample(4)
    assert det.shape == (4, 3) and not det.any() and obs.shape == (4, 2) and not obs[:, 0].any()
    np.testing.assert_array_equal(obs, dem.sample(4, seed=1)[1])


@pytest.mark.parametrize("seed", [0, 1, 123456789, 2 ** 64 + 3, 10 ** 40, [3, 1, 4], None])
def test_block_states_continue_seed_sequence(seed):
    """The sampler computes SeedSequence.generate_state's words itself, so a stream can continue at any block."""
    seq = np.random.SeedSequence(seed)
    words = seq.generate_state(4 * 300, dtype=np.uint64).reshape(300, 4)
    for first, count in ((0, 1), (0, 300), (1, 2), (7, 1), (255, 45), (299, 1)):
        np.testing.assert_array_equal(dem_module._block_states(seq.pool, first, count),
                                      words[first:first + count])


@pytest.mark.parametrize("seed", [0, 2 ** 64 + 3, [3, 1, 4], None])
def test_stream_takes_new_words_every_2_27_blocks(seed):
    """One SeedSequence's words repeat after 2**27 blocks, so each run of 2**27 blocks has its own SeedSequence."""
    seq = np.random.SeedSequence(seed)
    stream = dem_module._BlockStream(seq.entropy)
    epoch = 2 ** 27
    first = stream.states(0, 3)
    np.testing.assert_array_equal(first, dem_module._block_states(seq.pool, 0, 3))
    # The words alone would start over at block 2**27.
    np.testing.assert_array_equal(dem_module._block_states(seq.pool, epoch, 3), first)
    spawned = [np.random.SeedSequence(seq.entropy, spawn_key=(e,)).pool for e in (1, 2)]
    across = stream.states(epoch - 1, 3)
    np.testing.assert_array_equal(across[0], dem_module._block_states(seq.pool, epoch - 1, 1)[0])
    np.testing.assert_array_equal(across[1:], dem_module._block_states(spawned[0], 0, 2))
    np.testing.assert_array_equal(stream.states(2 * epoch + 5, 2), dem_module._block_states(spawned[1], 5, 2))
    np.testing.assert_array_equal(stream.states(0, 3), first)
    starts = [stream.states(e * epoch, 1)[0].tobytes() for e in range(4)]
    assert len(set(starts)) == 4


def test_stream_states_ahead_are_the_same_states(monkeypatch):
    monkeypatch.setattr(dem_module, "_STREAM_EPOCH", 50)
    requests = [(0, 1), (1, 3), (4, 60), (64, 1), (3, 2), (65, 200), (265, 1), (0, 300), (299, 1), (1000, 0)]
    ahead = dem_module._BlockStream(8, ahead=64)
    for first, count in requests:
        got = ahead.states(first, count)
        assert got.shape == (count, 4) and got.flags.c_contiguous
        np.testing.assert_array_equal(got, dem_module._BlockStream(8).states(first, count))


def test_compiled_sampler_computes_block_states_ahead(monkeypatch):
    """A sampler drawing one block per call computes the blocks' states 64 at a time, not once per call."""
    dem = _reference_models()["d3"]
    expected = dem.sample(256 * 130, seed=2)
    block_states = dem_module._block_states
    counts = []

    def counted(pool, first, count):
        counts.append(count)
        return block_states(pool, first, count)

    monkeypatch.setattr(dem_module, "_block_states", counted)
    got = _calls(dem.compile_sampler(2), [256] * 130)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])
    assert counts == [64, 64, 64]


def test_compiled_sampler_does_not_repeat_after_2_27_blocks():
    dem = _reference_models()["d7"]
    start = dem.compile_sampler(7).sample(256)
    late = dem.compile_sampler(7)
    late._next_block = 2 ** 27  # 2**35 shots on
    assert not np.array_equal(late.sample(256)[0], start[0])


def test_streams_continue_across_new_words(monkeypatch):
    """With new words every 4 blocks, sample() and compiled calls still give one stream, which does not repeat."""
    dem = _reference_models()["d7"]
    det = dem.sample(sum(_SPLIT), seed=3)[0]
    monkeypatch.setattr(dem_module, "_STREAM_EPOCH", 4)
    short = dem.sample(sum(_SPLIT), seed=3)
    np.testing.assert_array_equal(short[0][:1024], det[:1024])
    for k in range(1024, sum(_SPLIT) - 1024, 1024):
        assert not np.array_equal(short[0][k:k + 1024], short[0][:1024])
    got = _calls(dem.compile_sampler(3), _SPLIT)
    np.testing.assert_array_equal(got[0], short[0])
    np.testing.assert_array_equal(got[1], short[1])
    monkeypatch.setattr(dem_module, "_SAMPLE_PARALLEL_WORK", -1.0)
    monkeypatch.setattr(dem_module, "_thread_count", lambda: 3)
    threaded = dem.sample(sum(_SPLIT), seed=3)
    np.testing.assert_array_equal(threaded[0], short[0])
    np.testing.assert_array_equal(threaded[1], short[1])
