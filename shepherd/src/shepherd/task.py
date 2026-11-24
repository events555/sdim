"""
task.py -- one unit of work for the collector, modeled on sinter.Task.

A Task is a noisy sdim circuit plus the name of the decoder to run on it. Its strong_id is a
hash of everything that determines its statistics: the detector error model the shots are drawn
from, the decoder (its name, and its `version` when it is a registered Decoder, so results from
before and after a change to a decoder are not added together), the postselection, and the
metadata. Two tasks with the same strong_id are the
same experiment, so their results can be added together; change any of those and the id changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from sdim import Circuit
from sdim.dem import DetectorErrorModel

# Prepend to strong_id to avoid diluting results from different versions
STRONG_ID_SCHEMA = "shepherd-task-v2"     # v2: decoder version in the id


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def _decoder_version(name: str) -> int | None:
    """`version` of the registered Decoder called `name`; None for a name only given to collect."""
    from shepherd.decoder import Decoder
    try:
        return Decoder.lookup(name).version
    except KeyError:
        return None


@dataclass(frozen=True, eq=False)
class Task:
    """
    Attributes:
        circuit: Noisy circuit with N1/N2 noise and DETECTOR / LOGICAL_OBSERVABLE instructions.
        decoder: Name of the decoder: a key of `collect(custom_decoders=...)`, or a name in the
            `Decoder` registry.
        json_metadata: Anything JSON can hold. Not used by the collector; it's copied into every
            results row so you can group and plot by it (e.g. {"d": 3, "p": 1e-3}).
        postselection: Detector indices. A shot in which any of them is non-zero is discarded,
            not decoded, and counted in `discards`.
        detector_error_model: The circuit's DEM, if you already have it. Built from the circuit
            when omitted.
    """

    circuit: Circuit
    decoder: str
    json_metadata: Any = None
    postselection: tuple[int, ...] = ()
    detector_error_model: DetectorErrorModel | None = None
    _strong_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        d = self.circuit.dimension
        
        # sanity checking
        if not _is_prime(d):
            raise ValueError(f"shepherd supports only prime qudit dimensions for now; got d = {d}")
        if not isinstance(self.decoder, str) or not self.decoder:
            raise TypeError(f"decoder must be a non-empty string, got {self.decoder!r}")
        try:
            json.dumps(self.json_metadata, sort_keys=True)
        except (TypeError, ValueError) as e:
            raise TypeError(f"json_metadata must be JSON-serializable: {e}") from None
        
        post = tuple(sorted({int(i) for i in self.postselection}))
        object.__setattr__(self, "postselection", post)

        if self.detector_error_model is not None and self.detector_error_model.dimension != d:
            raise ValueError("detector_error_model and circuit have different dimensions")

    @property
    def dimension(self) -> int:
        return self.circuit.dimension

    def dem(self) -> DetectorErrorModel:
        """The task's detector error model, built from the circuit on first use."""
        if self.detector_error_model is None:
            object.__setattr__(self, "detector_error_model", DetectorErrorModel.from_circuit(self.circuit))
        dem = self.detector_error_model
        bad = [i for i in self.postselection if not 0 <= i < dem.num_detectors]
        if bad:
            raise ValueError(f"postselection refers to detectors {bad}, but the model has "
                             f"{dem.num_detectors} detectors")
        return dem

    def strong_id(self) -> str:
        """SHA-256 over the DEM text, decoder and its version, postselection and metadata."""
        if self._strong_id is None:
            payload = {
                "schema": STRONG_ID_SCHEMA,
                "dem": str(self.dem()),
                "decoder": self.decoder,
                "decoder_version": _decoder_version(self.decoder),
                "postselection": list(self.postselection),
                "json_metadata": self.json_metadata,
            }

            blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            object.__setattr__(self, "_strong_id", hashlib.sha256(blob.encode()).hexdigest())

        return self._strong_id
