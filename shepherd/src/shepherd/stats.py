"""
stats.py -- sampling statistics and the JSONL results file.

Every finished batch is appended to the results file as one JSON object on its own line:

    {"strong_id": "...", "decoder": "...", "json_metadata": {...},
     "shots": 1024, "errors": 3, "discards": 0, "seconds": 0.41}

Rows with the same strong_id are summed to get a task's totals, so a file can be appended to by
any number of runs, and resuming a run is just reading the file back.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class AnonTaskStats:
    """Counts with no task attached, as a worker returns them for one batch."""

    shots: int = 0
    errors: int = 0      # kept shots where the decoder got an observable wrong
    discards: int = 0    # shots rejected by postselection
    seconds: float = 0.0

    def __add__(self, other: AnonTaskStats) -> AnonTaskStats:
        return AnonTaskStats(self.shots + other.shots, self.errors + other.errors,
                             self.discards + other.discards, self.seconds + other.seconds)


@dataclass
class TaskStats:
    """Totals for one task."""

    strong_id: str
    decoder: str
    json_metadata: Any = None
    shots: int = 0
    errors: int = 0
    discards: int = 0
    seconds: float = 0.0

    def __add__(self, other: TaskStats | AnonTaskStats) -> TaskStats:
        if isinstance(other, TaskStats) and other.strong_id != self.strong_id:
            raise ValueError(f"cannot add stats of different tasks ({self.strong_id[:12]}... and "
                             f"{other.strong_id[:12]}...)")
        return TaskStats(self.strong_id, self.decoder, self.json_metadata,
                         self.shots + other.shots, self.errors + other.errors,
                         self.discards + other.discards, self.seconds + other.seconds)

    @property
    def kept(self) -> int:
        """Shots that survived postselection."""
        return self.shots - self.discards

    @property
    def logical_error_rate(self) -> float:
        """Errors per kept shot (nan if no shot was kept)."""
        return self.errors / self.kept if self.kept else math.nan

    @property
    def discard_rate(self) -> float:
        return self.discards / self.shots if self.shots else math.nan

    def error_rate_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Wilson score interval for the logical error rate (z = 1.96 gives about 95%)."""
        return wilson_interval(self.errors, self.kept, z)

    def to_row(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict) -> TaskStats:
        return cls(strong_id=row["strong_id"], decoder=row["decoder"],
                   json_metadata=row.get("json_metadata"), shots=int(row["shots"]),
                   errors=int(row["errors"]), discards=int(row.get("discards", 0)),
                   seconds=float(row.get("seconds", 0.0)))


def wilson_interval(errors: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate; (nan, nan) when trials == 0."""
    if trials <= 0:
        return math.nan, math.nan
    p = errors / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def merge_stats(stats: Iterable[TaskStats]) -> dict[str, TaskStats]:
    """Sums stats by strong_id, keeping the first-seen order."""
    totals: dict[str, TaskStats] = {}
    for s in stats:
        totals[s.strong_id] = totals[s.strong_id] + s if s.strong_id in totals else s
    return totals


def read_stats_file(path: str | Path) -> dict[str, TaskStats]:
    """
    Reads a results file and returns the totals for each strong_id.

    A last line that does not parse is skipped: it is a row that was being written when a run was
    killed. A bad line anywhere else raises, since that means the file is damaged.
    """
    path = Path(path)
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(TaskStats.from_row(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            if i == len(lines) - 1:
                break
            raise ValueError(f"{path}:{i + 1}: bad results row: {e}") from None
    return merge_stats(rows)


@dataclass
class StatsWriter:
    """Appends rows to a results file, flushing each one so a killed run loses at most one row."""

    path: Path
    _file: Any = field(default=None, repr=False)

    def __enter__(self) -> StatsWriter:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # If an earlier run died mid-row, cut that partial row off before appending. Its counts
        # were never complete, and rows written after it would otherwise hide it mid-file.
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "r+b") as f:
                data = f.read()
                if not data.endswith(b"\n"):
                    f.truncate(data.rfind(b"\n") + 1)
        self._file = open(self.path, "a")
        return self

    def write(self, stats: TaskStats) -> None:
        self._file.write(json.dumps(stats.to_row(), sort_keys=True) + "\n")
        self._file.flush()

    def __exit__(self, *exc) -> None:
        self._file.close()
