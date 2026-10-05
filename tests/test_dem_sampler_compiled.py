"""Tests for `DetectorErrorModel.compile_sampler`, for the random states of the sampler blocks, and for the
values `DetectorErrorModel.sample` returns for fixed seeds.

A compiled sampler checks and packs the model once. Its calls continue one random stream: their rows, put
together, are the rows `sample(total, seed)` returns for the same seed.
"""

import copy
import hashlib
import itertools
import pickle
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


# sha256 (first 16 hex digits) of `sample(shots, seed)` over _SHOTS x _SEEDS, from the sampler with SplitMix64
# block states, under numpy 2.4.6 and 1.26.4 alike.
_SHOTS = (1, 255, 256, 257, 1000, 3000)
_SEEDS = (0, 1, 2 ** 64 + 3)
_DIGESTS = {
    "readme": "2293cd70438dc3f3",
    "d2": "ae4dc0c94adc29d6",
    "d3": "63e6e870678b14dd",
    "d4": "a0409b3aada78aec",
    "d5_many_bins": "8439932daffaec7c",
    "d7": "a9a727ac98fe7d67",
    "d2147483647": "595f407b0e79b1cc",
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
# DetectorErrorModel.sample's values for fixed seeds


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


@pytest.mark.parametrize("error", [KeyboardInterrupt, MemoryError])
def test_a_call_that_raises_in_the_states_leaves_the_stream_where_it_was(monkeypatch, error):
    """A sampler keeps 64 states ahead; an error while it computes the next ones must not leave the old ones."""
    dem = _reference_models()["d7"]
    det, obs = dem.sample(256 * 140, seed=5)
    sampler = dem.compile_sampler(5)
    first = sampler.sample(256 * 64)
    block_states = dem_module._block_states

    def interrupted(*args):
        raise error

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


# --------------------------------------------------------------------------
# The blocks' random states


def test_stream_states_ahead_are_the_same_states():
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

    def counted(key, first, count):
        counts.append(count)
        return block_states(key, first, count)

    monkeypatch.setattr(dem_module, "_block_states", counted)
    got = _calls(dem.compile_sampler(2), [256] * 130)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])
    assert counts == [64, 64, 64]


_M64 = (1 << 64) - 1


def _splitmix64(start, gamma, n):
    """Output n of SplitMix64 with the given seed and gamma, on Python ints."""
    z = (start + (n + 1) * gamma) & _M64
    z = (z ^ z >> 30) * 0xBF58476D1CE4E5B9 & _M64
    z = (z ^ z >> 27) * 0x94D049BB133111EB & _M64
    return z ^ z >> 31


def _first_draws(states):
    """The first xoshiro256** output of each block state, as the sampler maps it to (0, 1). It only reads word 1."""
    s1 = states[:, 1] * np.uint64(5)
    r = ((s1 << np.uint64(7)) | (s1 >> np.uint64(57))) * np.uint64(9)
    return ((r >> np.uint64(11)).astype(np.float64) + 0.5) / 2.0 ** 53


def _transitions(gamma):
    return bin(gamma ^ gamma >> 1).count("1")


@pytest.mark.parametrize("seed", [0, 1, 123456789, 2 ** 64 + 3, 10 ** 40, [3, 1, 4], np.int8(7), None])
def test_block_states_are_splitmix64_outputs(seed):
    """Block c starts from outputs 4c .. 4c + 3 of SplitMix64, whose seed and gamma come from the seed's pool."""
    stream = dem_module._BlockStream(seed)
    start, gamma = stream._key
    if seed is not None:
        assert stream._key == dem_module._stream_key(np.random.SeedSequence(seed))
        pool = np.random.SeedSequence(seed).pool.tolist()
        assert start == pool[0] | pool[1] << 32 and gamma in {pool[2] | pool[3] << 32 | 1,
                                                              (pool[2] | pool[3] << 32 | 1) ^ 0xAAAAAAAAAAAAAAAA}
    assert gamma % 2 == 1 and _transitions(gamma) >= 24
    for first, count in ((0, 3), (7, 1), (255, 2), (2 ** 40 - 1, 2), (2 ** 62 - 1, 1)):
        expected = [[_splitmix64(start, gamma, 4 * c + j) for j in range(4)] for c in range(first, first + count)]
        np.testing.assert_array_equal(stream.states(first, count), np.array(expected, dtype=np.uint64))
    # Block indices wrap around at 2**62.
    np.testing.assert_array_equal(stream.states(2 ** 62 + 5, 1), stream.states(5, 1))


def test_poorly_mixing_gammas_get_every_other_bit_flipped():
    """As in Java's SplittableRandom, a gamma with fewer than 24 changes between neighbouring bits is replaced."""
    flipped = 0
    for seed in range(300):
        pool = np.random.SeedSequence(seed).pool.tolist()
        raw = pool[2] | pool[3] << 32 | 1
        gamma = dem_module._stream_key(seed)[1]
        assert gamma == (raw ^ 0xAAAAAAAAAAAAAAAA if _transitions(raw) < 24 else raw)
        flipped += gamma != raw
    assert flipped > 0


def test_seed_sequences_are_seeds():
    """A SeedSequence (a TypeError before) gives the stream of its entropy and spawn key, so the children that
    `spawn` makes give their own streams."""
    dem = _reference_models()["d3"]
    det, obs = dem.sample(700, seed=11)
    sequence = np.random.SeedSequence(11)
    for got in (dem.sample(700, seed=sequence), _calls(dem.compile_sampler(sequence), (300, 400))):
        np.testing.assert_array_equal(got[0], det)
        np.testing.assert_array_equal(got[1], obs)
    children = sequence.spawn(2)
    assert children[1].spawn_key == (1,)
    np.testing.assert_array_equal(dem.compile_sampler(children[1]).sample(700)[0],
                                  dem.sample(700, seed=np.random.SeedSequence(11, spawn_key=(1,)))[0])
    # SeedSequence pads the entropy with zeros before the spawn key, so child 1 is the seed [11, 0, 0, 0, 1], as
    # the CompiledDemSampler docstring warns.
    np.testing.assert_array_equal(dem.sample(700, seed=children[1])[0], dem.sample(700, seed=[11, 0, 0, 0, 1])[0])
    larger_pool = np.random.SeedSequence(11, pool_size=8)
    streams = [dem_module._BlockStream(s).states(0, 64) for s in [11, larger_pool] + children]
    assert len(np.unique(np.concatenate(streams))) == 4 * 64 * 4


def test_a_zero_word_never_makes_an_all_zero_state():
    """xoshiro256** must not start from the all-zero state. The output function maps 0 to 0, so one SplitMix64
    word in 2**64 is zero, but the other three words of its block are different from it, so not zero."""
    gamma = 0x9E3779B97F4A7C15
    for j in range(4):
        # Word j of block 5 is output 4 * 5 + j, the output function of start + (21 + j) * gamma.
        start = -(21 + j) * gamma % 2 ** 64
        state = dem_module._block_states((start, gamma), 5, 1)[0]
        assert state[j] == 0 and np.count_nonzero(state) == 3


# Seeds whose words in the old stream (SeedSequence.generate_state) repeated: some blocks 2**24 blocks on for 30,
# the whole stream 2**25 blocks on for 563, 658, 923 and 1190 and 2**26 blocks on for 9 and 30.
_SEEDS_OF_OLD_REPEATS = [9, 30, 563, 658, 923, 1190]


def test_blocks_at_power_of_two_offsets_are_unrelated():
    """Blocks 2**k apart used to get states that differed only in the high bits of some words, and for some seeds
    the same states. Now no word comes back, and words of blocks 2**k apart differ in about half their bits,
    by a different amount (difference or XOR) for every block, for k up to 40."""
    n = 256
    for seed in list(range(40)) + _SEEDS_OF_OLD_REPEATS + [2 ** 64 + 3, [3, 1, 4]]:
        stream = dem_module._BlockStream(seed)
        start = stream.states(0, n)
        # Blocks 0 .. 255 and 2**k .. 2**k + 255 for k = 8 .. 40 do not overlap.
        words = np.concatenate([start] + [stream.states(2 ** k, n) for k in range(8, 41)])
        assert len(np.unique(words)) == words.size, seed
        for k in range(41):
            later = stream.states(2 ** k, n)
            assert not (later[:, :, None] == start[:, None, :]).any(), (seed, k)
            flipped = np.unpackbits((later ^ start).view(np.uint8)).mean() * 64
            assert abs(flipped - 32) < 0.75, (seed, k, flipped)  # six standard deviations
            assert len(np.unique(later - start)) == later.size, (seed, k)
            assert len(np.unique(later ^ start)) == later.size, (seed, k)


def test_first_draws_of_blocks_at_power_of_two_offsets_are_not_correlated():
    """A block's first draw depends on its word 1 only. Its correlation with the first draw 2**k blocks on, over
    4096 blocks, stays below five standard deviations for k up to 40. In the old stream it was six standard
    deviations for seed 51 (k = 9) and 5.5 for seed 144 (k = 19), and -0.4 within the first 1024 blocks of
    seed 11523 (k = 9)."""
    n = 4096
    for seed in list(range(20)) + [51, 144, 11523] + _SEEDS_OF_OLD_REPEATS:
        stream = dem_module._BlockStream(seed)
        first = _first_draws(stream.states(0, n))
        for k in range(41):
            z = np.corrcoef(first, _first_draws(stream.states(2 ** k, n)))[0, 1] * np.sqrt(n)
            assert abs(z) < 5, (seed, k, z)


@pytest.mark.parametrize("seed", _SEEDS_OF_OLD_REPEATS[:3])
def test_compiled_sampler_blocks_far_apart_are_not_correlated(seed):
    """The shots before each block's first firing come from the block's first draws, so they used to be correlated
    between blocks 2**k apart for half the seeds or more from about k = 15 on."""
    dem = DetectorErrorModel(1000003, 1, 0, [ErrorMechanism(0.01, [{0: 1}])])
    n_blocks = 1024

    def first_firings(block):
        sampler = dem.compile_sampler(seed)
        sampler._next_block = block
        fired = sampler.sample(256 * n_blocks)[0].reshape(n_blocks, 256) != 0
        return np.where(fired.any(axis=1), fired.argmax(axis=1), 256)

    start = first_firings(0)
    for k in range(41):
        # 0.2 is over six standard deviations of the correlation of independent blocks.
        assert abs(np.corrcoef(start, first_firings(2 ** k))[0, 1]) < 0.2, k


@pytest.mark.parametrize("seed", [0, 1, 7, 12345])
def test_seeds_that_share_their_first_entropy_words_give_unrelated_streams(seed):
    """SeedSequence pads the entropy with zeros before it appends a spawn key, so the old stream's blocks
    2**10 onwards, from spawn key (1,), were the first blocks of the seed [seed, 0, 0, 0, 1], or seed + 2**128.
    Now those seeds, and seed + 2**64, share no word with the stream of `seed`, and their first draws are not
    correlated."""
    n = 4096
    stream = dem_module._BlockStream(seed)
    own = [stream.states(0, n), stream.states(2 ** 10, n)]
    for other in (seed + 2 ** 128, [seed, 0, 0, 0, 1], seed + 2 ** 64):
        theirs = dem_module._BlockStream(other).states(0, n)
        for mine in own:
            assert not np.isin(theirs, mine).any(), other
            z = np.corrcoef(_first_draws(mine), _first_draws(theirs))[0, 1] * np.sqrt(n)
            assert abs(z) < 5, (other, z)


# --------------------------------------------------------------------------
# Copies


def test_compiled_sampler_cannot_be_copied_or_pickled():
    """A copy would continue the same stream, so it would return the rows the original returns. copy.copy used to
    succeed; pickle and copy.deepcopy failed only on the lock."""
    dem = _reference_models()["d3"]
    sampler = dem.compile_sampler(1)
    first = sampler.sample(100)
    for copier in (copy.copy, copy.deepcopy, pickle.dumps, lambda x: pickle.dumps(x, protocol=0)):
        with pytest.raises(TypeError, match="compile one sampler per consumer, each with its own seed"):
            copier(sampler)
    with pytest.raises(TypeError, match="compile one sampler"):
        copy.deepcopy({"sampler": sampler})
    # The stream goes on where it was.
    det, obs = dem.sample(400, seed=1)
    rest = sampler.sample(300)
    np.testing.assert_array_equal(np.concatenate([first[0], rest[0]]), det)
    np.testing.assert_array_equal(np.concatenate([first[1], rest[1]]), obs)
