"""
collect.py -- run many tasks across worker processes until each reaches its shot or error limit,
modeled on sinter.collect.

    from shepherd.collect import collect
    from shepherd.task import Task

    if __name__ == "__main__":          # required: workers are started with "spawn"
        stats = collect(tasks=[Task(circuit, "my_decoder", json_metadata={"p": p}) for ...],
                        num_workers=8, max_shots=10**6, max_errors=500,
                        save_resume_filepath="results.jsonl",
                        custom_decoders={"my_decoder": MyDecoder()})

Each finished batch is appended to `save_resume_filepath` straight away. Running the same call
again reads the file, counts what each task already has, and only samples what is missing, so an
interrupted run resumes where it stopped, and raising max_shots extends a finished one.
"""

from __future__ import annotations

import multiprocessing
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable

from shepherd.stats import StatsWriter, TaskStats, read_stats_file
from shepherd.task import Task
from shepherd import decoder_cache
from shepherd.worker import WorkerTask, init_worker, run_batch, warm_decoder_cache


class _InlineExecutor(Executor):
    """Runs each batch in this process as it is submitted (num_workers=0): easy to debug and profile."""

    def submit(self, fn, /, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as e:
            future.set_exception(e)
        return future


class _NullWriter:
    def __enter__(self):
        return self

    def write(self, stats: TaskStats) -> None:
        pass

    def __exit__(self, *exc) -> None:
        pass


def collect(
    *,
    tasks: Iterable[Task],
    num_workers: int,
    max_shots: int,
    max_errors: int | None = None,
    save_resume_filepath: str | Path | None = None,
    custom_decoders: dict[str, Any] | None = None,
    decoder_cache_dir: str | Path | None = None,
    start_batch_shots: int = 256,
    max_batch_shots: int = 1 << 20,
    target_batch_seconds: float = 2.0,
    print_progress: bool = False,
) -> list[TaskStats]:
    """
    Samples and decodes every task until it has `max_shots` shots or `max_errors` errors.

    Args:
        tasks: The tasks. Tasks with equal strong_ids are run once.
        num_workers: Worker processes. 0 runs everything in this process.
        max_shots: Shot limit per task, counting shots already in the results file.
        max_errors: Error limit per task, or None for no limit. Batches already running when a
            task reaches it still finish, so a task can end slightly above it.
        save_resume_filepath: JSONL results file to append to and resume from. None keeps
            results in memory only.
        custom_decoders: Decoder objects by name, each with `compile_decoder_for_dem(dem)`.
            Names not given here are looked up in the `Decoder` registry. With num_workers > 0
            they are pickled and sent to every worker.
        decoder_cache_dir: Directory where compiled decoders with `cacheable = True` are kept
            (see decoder_cache.py), so each is compiled once and then loaded, by every worker and
            by later runs. With num_workers > 0, missing entries are compiled before sampling
            starts, each by one worker. None compiles every decoder in every worker.
        start_batch_shots: Size of a task's first batches, before its speed is known.
        max_batch_shots: Upper limit on any batch.
        target_batch_seconds: Batches are sized to take about this long once a task's speed is
            known: long enough that scheduling overhead is small, short enough that the shot and
            error limits are not overshot by much.
        print_progress: Print a progress line to stderr about once a second.

    Returns:
        Totals for each distinct task, in the order the tasks were given, including results
        that were already in the file.
    """
    if max_shots < 0:
        raise ValueError("max_shots must be non-negative")
    if max_errors is not None and max_errors < 0:
        raise ValueError("max_errors must be non-negative or None")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    
    custom_decoders = dict(custom_decoders or {})

    # Strong ids (which build the DEMs) are computed here once, in this process.
    by_id: dict[str, Task] = {}
    for task in tasks:
        by_id.setdefault(task.strong_id(), task)
    _check_decoders(by_id.values(), custom_decoders, pickled=num_workers > 0)

    previous = read_stats_file(save_resume_filepath) if save_resume_filepath is not None else {}
    totals = {i: previous.get(i) or TaskStats(i, t.decoder, t.json_metadata) for i, t in by_id.items()}
    in_flight = defaultdict(int)          # strong_id -> shots submitted but not finished
    run_seconds = defaultdict(float)      # this run's measured time per task, for batch sizing
    run_shots = defaultdict(int)

    def remaining_shots(i: str) -> int:
        return max_shots - totals[i].shots - in_flight[i]

    def next_batch(i: str) -> int:
        """Size of the next batch of task i, or 0 if it should get no more batches right now."""
        s = totals[i]
        left = remaining_shots(i)
        if left <= 0 or (max_errors is not None and s.errors >= max_errors):
            return 0
        if run_seconds[i] > 0:
            n = int(run_shots[i] / run_seconds[i] * target_batch_seconds)
        else:
            n = start_batch_shots
        
        if max_errors is not None and s.errors > 0:
            shots_per_error = s.shots / s.errors
            expected_left = max_errors - s.errors - in_flight[i] / shots_per_error
            if expected_left <= 0:
                return 0                  # enough errors are probably already on their way
            n = min(n, int(expected_left * shots_per_error))
        return max(1, min(max(n, start_batch_shots), max_batch_shots, left))

    cache_dir = Path(decoder_cache_dir) if decoder_cache_dir is not None else None
    # Inline, the first batch of each task fills the cache itself.
    to_warm = _missing_cache_entries(by_id, custom_decoders, cache_dir) if cache_dir and num_workers else []

    payloads = {i: WorkerTask(i, t.dimension, t.dem(), t.decoder, t.postselection) for i, t in by_id.items()}
    if num_workers == 0:
        init_worker(payloads, custom_decoders, cache_dir)
        executor: Executor = _InlineExecutor()
        max_pending = 1
    else:
        executor = ProcessPoolExecutor(max_workers=num_workers,
                                       mp_context=multiprocessing.get_context("spawn"),
                                       initializer=init_worker,
                                       initargs=(payloads, custom_decoders, cache_dir))
        max_pending = 2 * num_workers     # keep every worker busy while results come back

    writer = StatsWriter(Path(save_resume_filepath)) if save_resume_filepath is not None else _NullWriter()
    pending: dict[Future, tuple[str, int]] = {}
    last_print = 0.0

    def fill() -> None:
        while len(pending) < max_pending:
            candidates = [(totals[i].shots + in_flight[i], i) for i in by_id]
            candidates = [(c, i) for c, i in sorted(candidates) if next_batch(i) > 0]
            if not candidates:
                return
            i = candidates[0][1]          # least-sampled task first
            n = next_batch(i)
            in_flight[i] += n
            pending[executor.submit(run_batch, i, n)] = (i, n)

    try:
        with writer:
            # Compile each missing cache entry once, spread over the workers, before any batch
            # asks for it; otherwise every worker would compile it at the same moment.
            for future in [executor.submit(warm_decoder_cache, i) for i in to_warm]:
                future.result()
            fill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    i, n = pending.pop(future)
                    in_flight[i] -= n
                    _, batch = future.result()
                    totals[i] = totals[i] + batch
                    run_shots[i] += batch.shots
                    run_seconds[i] += batch.seconds
                    writer.write(TaskStats(i, by_id[i].decoder, by_id[i].json_metadata,
                                           batch.shots, batch.errors, batch.discards, batch.seconds))
                if print_progress and time.monotonic() - last_print > 1.0:
                    last_print = time.monotonic()
                    _print_progress(totals, max_shots, max_errors)
                fill()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if print_progress:
        _print_progress(totals, max_shots, max_errors)
    return list(totals.values())


def _check_decoders(tasks: Iterable[Task], custom_decoders: dict[str, Any], pickled: bool) -> None:
    """Fails fast, in this process, on a decoder name that cannot be resolved or sent to workers."""
    for name in sorted({t.decoder for t in tasks}):
        if name in custom_decoders:
            if not hasattr(custom_decoders[name], "compile_decoder_for_dem"):
                raise TypeError(f"custom decoder {name!r} has no compile_decoder_for_dem method")
            if pickled:
                try:
                    pickle.dumps(custom_decoders[name])
                except Exception as e:
                    raise TypeError(f"custom decoder {name!r} cannot be pickled to send to the "
                                    f"workers ({e}); use num_workers=0, or define it at module "
                                    f"level") from None
        else:
            from shepherd.decoder import Decoder
            Decoder.lookup(name)          # raises KeyError with the available names


def _missing_cache_entries(by_id: dict[str, Task], custom_decoders: dict[str, Any],
                           cache_dir: Path) -> list[str]:
    """One task id for each decoder cache entry the tasks need but the cache lacks."""
    from shepherd.decoder import Decoder
    decoders: dict[str, Any] = {}
    missing: dict[str, str] = {}             # cache key -> a task that needs it
    for i, t in by_id.items():
        if t.decoder not in decoders:
            decoders[t.decoder] = (custom_decoders[t.decoder] if t.decoder in custom_decoders
                                   else Decoder.lookup(t.decoder)())
        decoder = decoders[t.decoder]
        if not decoder_cache.is_cacheable(decoder):
            continue
        key = decoder_cache.cache_key(decoder, t.dem())
        if key not in missing and not decoder_cache.has_entry(cache_dir, decoder, key):
            missing[key] = i
    return list(missing.values())


def _print_progress(totals: dict[str, TaskStats], max_shots: int, max_errors: int | None) -> None:
    finished = sum(1 for s in totals.values()
                   if s.shots >= max_shots or (max_errors is not None and s.errors >= max_errors))
    shots = sum(s.shots for s in totals.values())
    errors = sum(s.errors for s in totals.values())
    print(f"[shepherd] {finished}/{len(totals)} tasks done, {shots} shots, {errors} errors",
          file=sys.stderr, flush=True)
