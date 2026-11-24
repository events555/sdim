from __future__ import annotations

import hashlib
import importlib
import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from sdim.dem import DetectorErrorModel


class CompiledDecoder(ABC):
    """
    A decoder prepared for one detector error model, made by `Decoder.compile_decoder_for_dem`.

    The collector builds one per task in each worker process and calls `decode_shots` on every
    batch, so expensive setup belongs in `compile_decoder_for_dem`, not here.

    A compiled decoder is fixed once built: its structure (graph, trellis, weights, table) depends
    only on the decoder's settings and the DEM, never on the syndromes it has decoded. Workers
    reuse it across batches, the decoder cache stores it to disk and loads it in other workers and
    later runs, and `deduplicate_syndromes` assumes it answers the same way every time; state
    left behind by one shot would silently change the answers for the rest. Decoders that learn
    from data (adaptive weights, drift tracking) should do so in a separate calibration step and
    pass the result in as a setting or a reweighted DEM, so it is part of the cache key.
    """

    # True if equal syndromes always get equal predictions. The worker then decodes each distinct
    # syndrome of a batch once. Set to False for decoders that break ties randomly, or that cannot
    # avoid carrying state from one shot to the next.
    deduplicate_syndromes: ClassVar[bool] = True

    @abstractmethod
    def decode_shots(self, detections: np.ndarray) -> np.ndarray:
        """
        detections: (shots, num_detectors) int64 array of detector values mod d.
        returns:    (shots, num_observables) integer array of predicted observable shifts mod d.

        Must not modify the compiled decoder (see the class docstring). Per-shot work that adapts
        to the syndrome, such as reweighting edges from belief-propagation marginals or a second
        matching pass, belongs in local arrays or a copy, not in the compiled structure.
        """

    def save(self, directory: Path) -> None:
        """
        Writes this compiled decoder into `directory` (which exists and is empty) for the decoder
        cache. The default pickles it. Override together with `load` for a format that loads
        faster, or that several workers can share, e.g. large arrays written with np.save and
        read back with np.load(..., mmap_mode="r").
        """
        with open(Path(directory) / "compiled.pkl", "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, directory: Path) -> CompiledDecoder:
        """Reads back what `save` wrote into `directory`."""
        with open(Path(directory) / "compiled.pkl", "rb") as f:
            return pickle.load(f)


class Decoder(ABC):

    registry: ClassVar[dict[str, type[Decoder]]] = {}

    # name -> module that defines it; imported on first lookup only
    _lazy: ClassVar[dict[str, str]] = {
        "none":             "shepherd.decoders.none",
        "pymatching":       "shepherd.decoders.pymatching_binarized_qudit_syndromes",
        "pymatching_keppens": "shepherd.decoders.pymatching_keppens",
        "perfect":          "shepherd.decoders.perfect",
        "trellis":          "shepherd.decoders.trellis",
        "union_find":       "shepherd.decoders.union_find",
    }

    # Bump when compile_decoder_for_dem changes what it builds, so that cached decoders built by
    # the old code are not loaded.
    version: ClassVar[int] = 1

    # True to keep compiled decoders on disk when collect is given a decoder_cache_dir (see
    # decoder_cache.py). Worth it only when compiling takes longer than loading the result.
    cacheable: ClassVar[bool] = False

    @classmethod
    def lookup(cls, name: str) -> type[Decoder]:
        if name not in cls.registry:
            module_path = cls._lazy.get(name)
            if module_path is None:
                raise KeyError(f"Unknown decoder {name!r}; available: {cls.available()}")
            importlib.import_module(module_path)   # class statement runs -> __init_subclass__ registers it
            if name not in cls.registry:
                raise RuntimeError(f"{module_path} was imported but did not register {name!r}")
        return cls.registry[name]

    @classmethod
    def create(cls, name: str, **params) -> Decoder:
        return cls.lookup(name)(**params)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(set(cls.registry) | set(cls._lazy))

    @classmethod
    def register_lazy(cls, name: str, module_path: str) -> None:
        """Let users add their own decoders without editing this file."""
        if name in cls._lazy or name in cls.registry:
            raise ValueError(f"Duplicate key: {name}")
        cls._lazy[name] = module_path

    def __init_subclass__(cls, name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is None:
            return
        if name in Decoder.registry:
            raise ValueError(f"Duplicate key: {name}")
        Decoder.registry[name] = cls

    def __init__(self, **params):
        """
        `params` are the decoder's settings and must be JSON-serializable. Subclasses that take
        settings should pass all of them here, e.g. `super().__init__(beam_width=beam_width)`:
        they are part of the fingerprint, so a setting left out could load a cached decoder
        compiled with a different value.
        """
        self.params = params

    def fingerprint(self) -> str:
        """
        SHA-256 of the decoder's class, version and settings: decoders with equal fingerprints
        compile equal decoders from equal DEMs. Override to leave out settings that do not affect
        compilation (a thread count, say) or to add inputs that are not settings (a weights file).
        """
        if not hasattr(self, "params"):
            raise TypeError(f"{type(self).__name__}.__init__ must call super().__init__(**settings)")
        payload = {"class": f"{type(self).__module__}.{type(self).__qualname__}",
                   "version": self.version, "params": self.params}
        try:
            blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            raise TypeError(f"{type(self).__name__} settings must be JSON-serializable: {e}") from None
        return hashlib.sha256(blob.encode()).hexdigest()

    @abstractmethod
    def compile_decoder_for_dem(self, dem: DetectorErrorModel) -> CompiledDecoder:
        """
        Expensive setup for one detector error model (matching graph, trellis, lookup table, ...).
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement compile_decoder_for_dem")