"""
worker.py -- what each worker process does: sample a batch from a task's DEM, decode it, and
count the errors.

The collector starts every worker with `init_worker`, which hands it the DEM and decoder of every
task once. After that each batch is just `run_batch(strong_id, shots)`, so DEMs are not sent
again with every batch. The first batch of a task in a worker compiles the DEM sampler and the
decoder; later batches of that task reuse them. With a decoder cache directory, cacheable decoders
are loaded from it instead of compiled (see decoder_cache.py).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sdim.dem import DetectorErrorModel

from shepherd import decoder_cache
from shepherd.stats import AnonTaskStats

# Compiled (sampler, decoder) pairs kept per worker. Older ones are dropped beyond this.
MAX_COMPILED_TASKS = 8


@dataclass(frozen=True)
class WorkerTask:
    """What a worker needs to know about a task. Sent to each worker once, at startup."""

    strong_id: str
    dimension: int
    dem: DetectorErrorModel
    decoder: str
    postselection: tuple[int, ...]


@dataclass
class _Compiled:
    sampler: Any   # sdim CompiledDemSampler: cannot be pickled, so it is made inside the worker
    decoder: Any   # CompiledDecoder
    postselection: np.ndarray
    dimension: int


_tasks: dict[str, WorkerTask] = {}
_decoders: dict[str, Any] = {}
_compiled: OrderedDict[str, _Compiled] = OrderedDict()
_cache_dir: Path | None = None


def init_worker(tasks: dict[str, WorkerTask], custom_decoders: dict[str, Any],
                decoder_cache_dir: Path | None = None) -> None:
    """Process initializer: stores the tasks and decoders this worker may be asked to run."""
    global _cache_dir
    _tasks.clear()
    _tasks.update(tasks)
    _decoders.clear()
    _decoders.update(custom_decoders)
    _compiled.clear()
    _cache_dir = decoder_cache_dir


def _resolve_decoder(name: str) -> Any:
    if name in _decoders:
        return _decoders[name]
    # Imported here so a run with only custom decoders never imports the registry's modules.
    from shepherd.decoder import Decoder
    decoder = Decoder.lookup(name)()
    _decoders[name] = decoder
    return decoder


def _get_compiled(strong_id: str) -> _Compiled:
    if strong_id in _compiled:
        _compiled.move_to_end(strong_id)
        return _compiled[strong_id]
    task = _tasks[strong_id]
    decoder = _resolve_decoder(task.decoder)
    if _cache_dir is not None and decoder_cache.is_cacheable(decoder):
        compiled_decoder = decoder_cache.load_or_compile(_cache_dir, decoder, task.dem)
    else:
        compiled_decoder = decoder.compile_decoder_for_dem(task.dem)
    compiled = _Compiled(
        # seed=None: fresh entropy, so every worker draws its own stream.
        sampler=task.dem.compile_sampler(seed=None),
        decoder=compiled_decoder,
        postselection=np.asarray(task.postselection, dtype=np.int64),
        dimension=task.dimension,
    )
    _compiled[strong_id] = compiled
    while len(_compiled) > MAX_COMPILED_TASKS:
        _compiled.popitem(last=False)
    return compiled


def warm_decoder_cache(strong_id: str) -> None:
    """Compiles a task's decoder into the decoder cache, so other workers load it from there."""
    _get_compiled(strong_id)


def decode_batch(decoder: Any, detections: np.ndarray, num_observables: int) -> np.ndarray:
    """Runs the decoder, decoding each distinct syndrome once when the decoder allows it."""
    shots = detections.shape[0]

    if shots == 0:
        return np.zeros((0, num_observables), dtype=np.int64)
    
    if getattr(decoder, "deduplicate_syndromes", True) and detections.shape[1] > 0:
        unique, inverse = np.unique(detections, axis=0, return_inverse=True)
        predictions = np.asarray(decoder.decode_shots(unique))
        _check_shape(predictions, unique.shape[0], num_observables)
        return predictions[inverse.reshape(-1)]
    
    predictions = np.asarray(decoder.decode_shots(detections))
    _check_shape(predictions, shots, num_observables)
    return predictions


def _check_shape(predictions: np.ndarray, shots: int, num_observables: int) -> None:
    if predictions.shape != (shots, num_observables):
        raise ValueError(f"decode_shots returned shape {predictions.shape}, expected "
                         f"({shots}, {num_observables})")


def run_batch(strong_id: str, shots: int) -> tuple[str, AnonTaskStats]:
    """Samples `shots` shots of a task, decodes them, and counts errors and discards."""
    compiled = _get_compiled(strong_id)
    start = time.perf_counter()
    detections, observables = compiled.sampler.sample(shots)

    discards = 0
    if compiled.postselection.size:
        keep = ~detections[:, compiled.postselection].any(axis=1)
        discards = int(shots - keep.sum())
        detections, observables = detections[keep], observables[keep]

    predictions = decode_batch(compiled.decoder, detections, observables.shape[1])
    wrong = ((predictions - observables) % compiled.dimension != 0).any(axis=1)
    seconds = time.perf_counter() - start
    return strong_id, AnonTaskStats(shots=shots, errors=int(wrong.sum()),
                                    discards=discards, seconds=seconds)
