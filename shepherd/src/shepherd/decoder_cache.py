"""
decoder_cache.py -- compiled decoders kept on disk, so that an expensive compile (a trellis, say)
is done once per detector error model instead of once per worker process and per run.

Only decoders with `cacheable = True` are cached. Each entry is a directory:

    <cache_dir>/<DecoderClass>-<key>/
        meta.json       what the entry was built from
        compiled.pkl    or whatever the compiled decoder's `save` wrote

The key is computed, never chosen: a SHA-256 over the decoder's fingerprint (class, version,
settings), the DEM text, and the sdim and shepherd versions. Callers choose only the directory.

An entry is built in a temporary directory and renamed into place, so a reader never sees a
half-written one, and of several processes storing the same entry at once the first rename wins.
An entry that fails to load (a damaged file, a class that has since moved) is removed with a
warning, then compiled again and stored.

Entries are unpickled when loaded, and unpickling can run arbitrary code: only point
`cache_dir` at a directory that you trust.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import tempfile
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shepherd.decoder import CompiledDecoder

if TYPE_CHECKING:
    from sdim.dem import DetectorErrorModel

# Part of every key: changing how entries are laid out or keyed invalidates all of them.
CACHE_SCHEMA = "shepherd-decoder-cache-v1"


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def is_cacheable(decoder: Any) -> bool:
    """Whether the decoder asked to be cached. It then needs a `fingerprint()` method."""
    if not getattr(decoder, "cacheable", False):
        return False
    if not callable(getattr(decoder, "fingerprint", None)):
        raise TypeError(f"{type(decoder).__name__} sets cacheable = True but has no fingerprint() "
                        f"method; subclass shepherd.decoder.Decoder or define one")
    return True


def cache_key(decoder: Any, dem: DetectorErrorModel) -> str:
    """SHA-256 over everything that determines what `compile_decoder_for_dem(dem)` builds."""
    payload = {
        "schema": CACHE_SCHEMA,
        "decoder": decoder.fingerprint(),
        "dem": str(dem),
        "sdim": _package_version("sdim"),
        "shepherd": _package_version("shepherd"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def entry_path(cache_dir: str | Path, decoder: Any, key: str) -> Path:
    return Path(cache_dir) / f"{type(decoder).__name__}-{key}"


def has_entry(cache_dir: str | Path, decoder: Any, key: str) -> bool:
    """Whether an entry exists. Cheap: it does not load the entry, so it may still fail to load."""
    return (entry_path(cache_dir, decoder, key) / "meta.json").exists()


def load(cache_dir: str | Path, decoder: Any, key: str) -> Any | None:
    """The cached compiled decoder, or None if there is none or it cannot be loaded."""
    path = entry_path(cache_dir, decoder, key)
    meta_path = path / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("key") != key:
            raise ValueError(f"meta.json is for key {meta.get('key')!r}")
        module, _, qualname = meta["compiled_class"].partition(":")
        cls: Any = importlib.import_module(module)
        for part in qualname.split("."):
            cls = getattr(cls, part)
        if isinstance(cls, type) and issubclass(cls, CompiledDecoder):
            return cls.load(path)
        return CompiledDecoder.load(path)       # not a CompiledDecoder: pickled by store()
    except Exception as e:
        warnings.warn(f"decoder cache entry {path} could not be loaded ({type(e).__name__}: {e}); "
                      f"removing it to be compiled again", RuntimeWarning, stacklevel=2)
        shutil.rmtree(path, ignore_errors=True)
        return None


def store(cache_dir: str | Path, decoder: Any, key: str, compiled: Any) -> None:
    """Writes an entry, unless another process has just written it."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = entry_path(cache_dir, decoder, key)
    # Same directory as the entry, so the rename below stays on one filesystem and is atomic.
    tmp = Path(tempfile.mkdtemp(dir=cache_dir, prefix=f".tmp-{final.name}-"))
    try:
        if isinstance(compiled, CompiledDecoder):
            compiled.save(tmp)
        else:
            CompiledDecoder.save(compiled, tmp)  # duck-typed compiled decoders are pickled
        cls = type(compiled)
        meta = {
            "schema": CACHE_SCHEMA,
            "key": key,
            "decoder_class": f"{type(decoder).__module__}:{type(decoder).__qualname__}",
            "decoder_version": getattr(decoder, "version", None),
            "decoder_params": getattr(decoder, "params", None),
            "compiled_class": f"{cls.__module__}:{cls.__qualname__}",
            "sdim": _package_version("sdim"),
            "shepherd": _package_version("shepherd"),
        }
        # meta.json last: an entry without it was never finished.
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=repr) + "\n")
        try:
            tmp.rename(final)
        except OSError:
            if not final.exists():
                raise
            # Another process stored the same entry first; keep theirs.
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_or_compile(cache_dir: str | Path, decoder: Any, dem: DetectorErrorModel) -> Any:
    """The compiled decoder for `dem`, from the cache if it is there, else compiled and stored."""
    key = cache_key(decoder, dem)
    compiled = load(cache_dir, decoder, key)
    if compiled is None:
        compiled = decoder.compile_decoder_for_dem(dem)
        store(cache_dir, decoder, key, compiled)
    return compiled
