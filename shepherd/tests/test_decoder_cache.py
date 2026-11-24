"""Tests of decoder fingerprints and the on-disk decoder cache, alone and through collect."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_collect import repetition_circuit

from shepherd import decoder_cache
from shepherd.collect import collect
from shepherd.decoder import Decoder
from shepherd.decoders.none import CompiledZeroDecoder, ZeroDecoder
from shepherd.task import Task


# -- test decoders (module level, so worker processes can unpickle them) -------------------------

class CountingDecoder(ZeroDecoder):
    """Cacheable; appends a line to `log` every time it compiles, from whichever process."""

    cacheable = True

    def __init__(self, log: str, tag: int = 0):
        super().__init__(log=log, tag=tag)
        self.log = Path(log)

    def compile_decoder_for_dem(self, dem):
        with open(self.log, "a") as f:
            f.write("compiled\n")
        return super().compile_decoder_for_dem(dem)


class UncachedCountingDecoder(CountingDecoder):
    cacheable = False


class CountingDecoderV2(CountingDecoder):
    version = 2


class NoSuperInitDecoder(ZeroDecoder):
    def __init__(self, x: int = 1):
        self.x = x


def compiles(log: Path) -> int:
    return len(log.read_text().splitlines()) if log.exists() else 0


def dem(p: float = 0.01):
    return Task(repetition_circuit(p=p), "unused").dem()


# -- fingerprints ---------------------------------------------------------------------------------

def test_fingerprint_follows_class_version_and_settings(tmp_path):
    log = str(tmp_path / "log")
    assert CountingDecoder(log).fingerprint() == CountingDecoder(log).fingerprint()
    prints = {CountingDecoder(log).fingerprint(), CountingDecoder(log, tag=1).fingerprint(),
              CountingDecoderV2(log).fingerprint(), UncachedCountingDecoder(log).fingerprint()}
    assert len(prints) == 4


def test_fingerprint_rejects_unusable_settings():
    with pytest.raises(TypeError, match="super"):
        NoSuperInitDecoder().fingerprint()
    with pytest.raises(TypeError, match="JSON"):
        ZeroDecoder(weights=object()).fingerprint()


def test_create_passes_settings():
    class Named(ZeroDecoder, name="_test_named_decoder"):
        def __init__(self, width: int = 1):
            super().__init__(width=width)
    try:
        assert Decoder.create("_test_named_decoder", width=4).params == {"width": 4}
    finally:
        del Decoder.registry["_test_named_decoder"]


def test_cache_key_follows_dem_and_decoder(tmp_path):
    d = CountingDecoder(str(tmp_path / "log"))
    assert decoder_cache.cache_key(d, dem()) == decoder_cache.cache_key(d, dem())
    assert decoder_cache.cache_key(d, dem()) != decoder_cache.cache_key(d, dem(0.02))
    assert decoder_cache.cache_key(d, dem()) != decoder_cache.cache_key(CountingDecoder(str(d.log), 1), dem())


def test_cacheable_without_fingerprint_is_an_error():
    class Duck:
        cacheable = True
    with pytest.raises(TypeError, match="fingerprint"):
        decoder_cache.is_cacheable(Duck())


# -- the cache on its own ------------------------------------------------------------------------

def test_compiles_once_then_loads(tmp_path):
    log, cache = tmp_path / "log", tmp_path / "cache"
    d, m = CountingDecoder(str(log)), dem()
    first = decoder_cache.load_or_compile(cache, d, m)
    second = decoder_cache.load_or_compile(cache, d, m)
    assert compiles(log) == 1
    assert isinstance(second, CompiledZeroDecoder) and second.num_observables == first.num_observables
    key = decoder_cache.cache_key(d, m)
    meta = json.loads((decoder_cache.entry_path(cache, d, key) / "meta.json").read_text())
    assert meta["key"] == key and meta["decoder_params"] == {"log": str(log), "tag": 0}


def test_damaged_entry_is_replaced(tmp_path):
    log, cache = tmp_path / "log", tmp_path / "cache"
    d, m = CountingDecoder(str(log)), dem()
    decoder_cache.load_or_compile(cache, d, m)
    path = decoder_cache.entry_path(cache, d, decoder_cache.cache_key(d, m))
    (path / "compiled.pkl").write_bytes(b"not a pickle")
    with pytest.warns(RuntimeWarning, match="could not be loaded"):
        decoder_cache.load_or_compile(cache, d, m)
    decoder_cache.load_or_compile(cache, d, m)
    assert compiles(log) == 2           # recompiled once, then loaded again


def test_store_that_loses_a_race_keeps_the_first_entry(tmp_path):
    cache = tmp_path / "cache"
    d, m = CountingDecoder(str(tmp_path / "log")), dem()
    key = decoder_cache.cache_key(d, m)
    decoder_cache.store(cache, d, key, CompiledZeroDecoder(1))
    decoder_cache.store(cache, d, key, CompiledZeroDecoder(1))
    assert [p.name for p in cache.iterdir()] == [decoder_cache.entry_path(cache, d, key).name]


# -- through collect ----------------------------------------------------------------------------

def test_collect_compiles_each_entry_once_across_workers_and_runs(tmp_path):
    log, cache = tmp_path / "log", tmp_path / "cache"
    decoders = {"counting": CountingDecoder(str(log))}
    # The first two share a DEM (only their metadata differs), so they share a cache entry.
    tasks = [Task(repetition_circuit(p=0.01), "counting", json_metadata={"run": "a"}),
             Task(repetition_circuit(p=0.01), "counting", json_metadata={"run": "b"}),
             Task(repetition_circuit(p=0.02), "counting", json_metadata={"run": "c"})]
    stats = collect(tasks=tasks, num_workers=2, max_shots=3000, custom_decoders=decoders,
                    decoder_cache_dir=cache)
    assert [s.shots for s in stats] == [3000] * 3
    assert compiles(log) == 2
    collect(tasks=tasks, num_workers=2, max_shots=3000, custom_decoders=decoders,
            decoder_cache_dir=cache)
    assert compiles(log) == 2


def test_collect_leaves_uncacheable_decoders_alone(tmp_path):
    log, cache = tmp_path / "log", tmp_path / "cache"
    collect(tasks=[Task(repetition_circuit(), "u")], num_workers=0, max_shots=500,
            custom_decoders={"u": UncachedCountingDecoder(str(log))}, decoder_cache_dir=cache)
    assert compiles(log) == 1 and not cache.exists()
