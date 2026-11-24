"""
study.py -- a sweep (codes x noise strengths x decoders) as one named, resumable unit of work.

A Study says what to run; `run_study` runs it into one results file, `<results_dir>/<name>.jsonl`,
and can be stopped and restarted at any time (collect resumes from the file).

    from shepherd.study import Study, run_study
    from shepherd.codes.css import surface_code
    from shepherd.noise_model import NoiseModel

    study = Study(
        name="surface_cc",
        description="surface code, code capacity",
        experiment="ler", experiment_params={"rounds": 0},
        codes=lambda: [surface_code(L, q) for q in (2, 3) for L in (5, 7, 9)],
        noise=NoiseModel.code_capacity,              # (q, p) -> NoiseModel
        ps=[0.06, 0.08, 0.10, 0.12],
        decoders=["union_find"],
        max_shots=100_000, max_errors=300,
    )
    run_study(study, results_dir="results", num_workers=8)

Every task's json_metadata is the code's metadata (code, family, n, k, d, q), the experiment's
(experiment, rounds, basis), the noise model's name and fingerprint, and p. Results files can
therefore be grouped and plotted by any of those keys (see analysis.py).

`ps`, `max_shots` and `max_errors` may be functions of the code (and, for the budgets, of p),
so one study can give each q its own grid and the biggest codes smaller budgets.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from sdim.dem import DetectorErrorModel

from shepherd.codes.css import CSSCode
from shepherd.collect import collect
from shepherd.experiment import Experiment
from shepherd.noise_model import NoiseModel
from shepherd.stats import TaskStats, read_stats_file
from shepherd.task import Task


def clean_p(p: float) -> float:
    """p to 6 significant digits, so grids built with logspace or arithmetic group cleanly."""
    return float(f"{float(p):.6g}")


@dataclass(frozen=True)
class Point:
    """One task of a study, with the budget it runs to."""
    task: Task
    code: CSSCode
    p: float
    max_shots: int
    max_errors: int | None

    @property
    def metadata(self) -> dict:
        return self.task.json_metadata


@dataclass(frozen=True)
class Study:
    """
    name: also the results file's stem.
    experiment, experiment_params: an Experiment registry name and its parameters (the
        dimension is taken from each code).
    codes: a function returning the codes (called when the study is run, so that defining a
        study costs nothing).
    noise: (q, p) -> NoiseModel.
    ps: noise strengths, or a function code -> noise strengths.
    max_shots, max_errors: per task; ints or functions (code, p) -> int.
    """
    name: str
    description: str
    experiment: str
    codes: Callable[[], Iterable[CSSCode]]
    noise: Callable[[int, float], NoiseModel]
    ps: Sequence[float] | Callable[[CSSCode], Sequence[float]]
    experiment_params: Mapping[str, Any] = field(default_factory=dict)
    decoders: Sequence[str] = ("union_find",)
    max_shots: int | Callable[[CSSCode, float], int] = 100_000
    max_errors: int | None | Callable[[CSSCode, float], int | None] = 300

    def results_path(self, results_dir: str | Path) -> Path:
        return Path(results_dir) / f"{self.name}.jsonl"

    def points(self, *, where: Mapping[str, Iterable[Any]] | None = None,
               ps: Sequence[float] | None = None) -> list[Point]:
        """
        The study's tasks. `where` keeps only tasks whose metadata (or decoder) matches, e.g.
        {"q": [3], "n": [12, 16]}; `ps` replaces the study's noise strengths.
        Each (code, p) builds its circuit and DEM once, shared by all its decoders.
        """
        where = {k: {_norm(v) for v in vs} for k, vs in (where or {}).items()}
        out: list[Point] = []
        for code in self.codes():
            q = code.dimension
            exp = Experiment.create(self.experiment, q, **dict(self.experiment_params))
            base = {**code.metadata(), **exp.metadata(code)}
            if not _matches(base, where, partial=True):
                continue
            grid = ps if ps is not None else (self.ps(code) if callable(self.ps) else self.ps)
            noiseless = None
            for p in (clean_p(x) for x in grid):
                model = self.noise(q, p)
                meta = {**base, "noise": model.name, "noise_fingerprint": model.fingerprint()[:16],
                        "p": p}
                decoders = [d for d in self.decoders if _matches({**meta, "decoder": d}, where)]
                if not decoders:
                    continue
                if noiseless is None:
                    noiseless = exp.build(code)
                circuit = model.apply(noiseless)
                dem = DetectorErrorModel.from_circuit(circuit)
                for dec in decoders:
                    out.append(Point(
                        task=Task(circuit, dec, json_metadata=dict(meta), detector_error_model=dem),
                        code=code, p=p,
                        max_shots=_budget(self.max_shots, code, p),
                        max_errors=_budget(self.max_errors, code, p)))
        return out


def _norm(v: Any) -> Any:
    """Make '3' and 3, '0.01' and 0.01 compare equal (command-line filters arrive as strings)."""
    if isinstance(v, str):
        for conv in (int, float):
            try:
                return conv(v)
            except ValueError:
                pass
    if isinstance(v, float):
        return clean_p(v)
    return v


def _matches(meta: Mapping[str, Any], where: Mapping[str, set], *, partial: bool = False) -> bool:
    for k, allowed in where.items():
        if k not in meta:
            if partial:
                continue                  # decided later, once the key exists
            return False
        if _norm(meta[k]) not in allowed:
            return False
    return True


def _budget(b, code: CSSCode, p: float):
    return b(code, p) if callable(b) else b


def run_study(study: Study, *, results_dir: str | Path, num_workers: int,
              where: Mapping[str, Iterable[Any]] | None = None, ps: Sequence[float] | None = None,
              max_shots: int | None = None, max_errors: int | None = None,
              decoder_cache_dir: str | Path | None = None,
              print_progress: bool = True) -> list[TaskStats]:
    """
    Runs every point of the study until it reaches its budget (or the overrides `max_shots` /
    `max_errors`), appending to `study.results_path(results_dir)`. Codes run one at a time,
    smallest first, so results for the small codes are complete early; within a code all
    noise strengths and decoders share the workers.

    Do not run two processes on the same results file at once.
    """
    points = study.points(where=where, ps=ps)
    path = study.results_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_code: dict[str, list[Point]] = {}
    for pt in points:
        by_code.setdefault(pt.code.label + f"_q{pt.code.dimension}", []).append(pt)
    order = sorted(by_code.values(), key=lambda pts: (pts[0].code.n * (pts[0].metadata.get("rounds", 0) + 1),
                                                      pts[0].code.dimension))
    stats: list[TaskStats] = []
    for pts in order:
        budgets: dict[tuple, list[Point]] = {}
        for pt in pts:
            key = (max_shots if max_shots is not None else pt.max_shots,
                   max_errors if max_errors is not None else pt.max_errors)
            budgets.setdefault(key, []).append(pt)
        for (ms, me), group in budgets.items():
            if print_progress:
                c = group[0].code
                print(f"[shepherd] {study.name}: {c.label} q={c.dimension}, {len(group)} tasks, "
                      f"up to {ms} shots / {me} errors each", file=sys.stderr, flush=True)
            stats.extend(collect(tasks=[pt.task for pt in group], num_workers=num_workers,
                                 max_shots=ms, max_errors=me, save_resume_filepath=path,
                                 decoder_cache_dir=decoder_cache_dir,
                                 print_progress=print_progress))
    return stats


def plan(study: Study, *, results_dir: str | Path, where=None, ps=None,
         max_shots: int | None = None, max_errors: int | None = None) -> list[dict]:
    """
    What `run_study` would do, without sampling: one row per point with its detector count,
    budget and what the results file already holds.
    """
    done = read_stats_file(study.results_path(results_dir))
    rows = []
    for pt in study.points(where=where, ps=ps):
        s = done.get(pt.task.strong_id())
        ms = max_shots if max_shots is not None else pt.max_shots
        me = max_errors if max_errors is not None else pt.max_errors
        finished = s is not None and (s.shots >= ms or (me is not None and s.errors >= me))
        rows.append({**pt.metadata, "decoder": pt.task.decoder,
                     "detectors": pt.task.dem().num_detectors,
                     "max_shots": ms, "max_errors": me,
                     "shots": s.shots if s else 0, "errors": s.errors if s else 0,
                     "done": finished})
    return rows
