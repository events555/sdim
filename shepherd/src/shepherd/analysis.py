"""
analysis.py -- results files to tables, threshold crossings and plots.

    from shepherd import analysis
    rows = analysis.load_results("results/surface_cc.jsonl")
    for c in analysis.crossings(rows):
        print(c)
    analysis.plot_thresholds(rows, "plots/")

A row is one task's totals: its json_metadata (code, family, n, k, d, q, experiment, rounds,
basis, noise, noise_fingerprint, p), plus decoder, shots, errors, discards, seconds, and the
logical error rate with a 95% Wilson interval (ler, ler_lo, ler_hi). The logical error rate is per
shot: a shot fails if any logical observable is predicted wrong.

Curves and crossings
--------------------
A curve is one code size within a group, a group being the rows that agree on GROUP_KEYS
(family, noise, q, decoder, experiment, basis): the LER of that code as a function of p. Sizes
are ordered by n. For each pair of consecutive sizes, the crossing is where log LER of the
larger code rises above that of the smaller, found by linear interpolation in (log p, log LER)
between the grid points that bracket the first sign change. Its range comes from repeating
this with the interval ends that pull the crossing furthest each way (larger code at the upper
end and smaller at the lower, and the reverse). Points with no errors are left out.

Crossings of consecutive sizes drifting with size are a finite-size effect; the largest pair is
the best estimate of the threshold. A crossing near the saturation value 1 - 1/q**k is not a
threshold.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from shepherd.stats import read_stats_file, wilson_interval

GROUP_KEYS = ("family", "noise", "q", "decoder", "experiment", "basis")
CURVE_KEY = "n"

# Categorical series colors, in fixed order (validated light-mode palette).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_results(*paths: str | Path) -> list[dict]:
    """One row per task across the given results files (see the module docstring)."""
    rows = []
    for path in paths:
        for s in read_stats_file(path).values():
            meta = s.json_metadata if isinstance(s.json_metadata, dict) else {}
            lo, hi = wilson_interval(s.errors, s.kept) if s.kept else (0.0, 1.0)
            rows.append({**meta, "decoder": s.decoder, "shots": s.shots, "errors": s.errors,
                         "discards": s.discards, "seconds": s.seconds,
                         "ler": s.errors / s.kept if s.kept else float("nan"),
                         "ler_lo": lo, "ler_hi": hi, "strong_id": s.strong_id})
    return rows


def group_rows(rows: Iterable[dict], keys: Sequence[str] = GROUP_KEYS) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    return dict(sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])))


def curves(rows: Iterable[dict], curve_key: str = CURVE_KEY) -> dict:
    """{size: rows sorted by p}, sizes ascending."""
    out: dict = defaultdict(list)
    for r in rows:
        out[r.get(curve_key)].append(r)
    return {k: sorted(v, key=lambda r: r["p"]) for k, v in sorted(out.items())}


def _crossing(a: dict[float, float], b: dict[float, float]) -> float | None:
    """First p where log b - log a goes from <= 0 to > 0, interpolated in log-log."""
    ps = sorted(p for p in set(a) & set(b) if a[p] > 0 and b[p] > 0)
    diff = [math.log(b[p]) - math.log(a[p]) for p in ps]
    for i in range(len(ps) - 1):
        if diff[i] <= 0 < diff[i + 1]:
            t = -diff[i] / (diff[i + 1] - diff[i])
            return math.exp(math.log(ps[i]) + t * (math.log(ps[i + 1]) - math.log(ps[i])))
    return None


def crossings(rows: Iterable[dict], *, group_keys: Sequence[str] = GROUP_KEYS,
              curve_key: str = CURVE_KEY) -> list[dict]:
    """One dict per group and pair of consecutive sizes: the crossing p and its range."""
    out = []
    for gkey, grows in group_rows(rows, group_keys).items():
        cs = curves(grows, curve_key)
        sizes = list(cs)
        for small, large in zip(sizes, sizes[1:]):
            mid = {k: {r["p"]: r[k] for r in cs[small]} for k in ("ler", "ler_lo", "ler_hi")}
            big = {k: {r["p"]: r[k] for r in cs[large]} for k in ("ler", "ler_lo", "ler_hi")}
            p = _crossing(mid["ler"], big["ler"])
            lo = _crossing(mid["ler_lo"], big["ler_hi"])     # larger code looks worse: earlier
            hi = _crossing(mid["ler_hi"], big["ler_lo"])     # larger code looks better: later
            label = lambda size: next(r.get("code") for r in cs[size])
            out.append({**dict(zip(group_keys, gkey)), curve_key + "_small": small,
                        curve_key + "_large": large, "code_small": label(small),
                        "code_large": label(large), "p_cross": p, "p_lo": lo, "p_hi": hi})
    return out


def write_csv(rows: Sequence[dict], path: str | Path) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def format_table(rows: Sequence[dict], columns: Sequence[str]) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return "" if v is None else str(v)
    cells = [[fmt(r.get(c)) for c in columns] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(columns)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(columns, widths))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in cells]
    return "\n".join(lines)


def _title(gkey: tuple, keys: Sequence[str]) -> str:
    g = dict(zip(keys, gkey))
    parts = [str(g.get("family")), str(g.get("noise")).replace("_", " "), f"q = {g.get('q')}",
             f"decoder: {g.get('decoder')}"]
    return ", ".join(parts)


def plot_thresholds(rows: Iterable[dict], out_dir: str | Path, *,
                    group_keys: Sequence[str] = GROUP_KEYS, curve_key: str = CURVE_KEY,
                    logx: bool = True) -> list[Path]:
    """
    One PNG per group: LER against p, one curve per code size with 95% Wilson bars, and the
    largest pair's crossing range shaded. Also writes each group's points and crossings as CSV
    next to the PNG. Returns the PNG paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    all_cross = crossings(rows, group_keys=group_keys, curve_key=curve_key)
    written = []
    for gkey, grows in group_rows(rows, group_keys).items():
        cs = curves(grows, curve_key)
        fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        for i, (size, crows) in enumerate(cs.items()):
            pts = [r for r in crows if r["errors"] > 0]
            if not pts:
                continue
            color = SERIES[i % len(SERIES)]
            x = [r["p"] for r in pts]
            y = [r["ler"] for r in pts]
            yerr = [[r["ler"] - r["ler_lo"] for r in pts], [r["ler_hi"] - r["ler"] for r in pts]]
            d = crows[0].get("d")
            label = f"{crows[0].get('code')}  (n={crows[0].get('n')}, d={d})"
            ax.errorbar(x, y, yerr=yerr, color=color, marker=MARKERS[i % len(MARKERS)],
                        markersize=5, linewidth=2, elinewidth=1, capsize=2, label=label,
                        markeredgecolor=SURFACE, markeredgewidth=1)
        mine = [c for c in all_cross if tuple(c.get(k) for k in group_keys) == gkey]
        best = next((c for c in reversed(mine) if c["p_cross"] is not None), None)
        if best is not None:
            lo = best["p_lo"] or best["p_cross"]
            hi = best["p_hi"] or best["p_cross"]
            ax.axvspan(lo, hi, color=INK_2, alpha=0.10, linewidth=0)
            ax.axvline(best["p_cross"], color=INK_2, linewidth=1, linestyle="--")
            ax.text(best["p_cross"], 0.02, f" crossing ≈ {best['p_cross']:.4g}",
                    transform=ax.get_xaxis_transform(), color=INK_2, fontsize=8)
        from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
        plain = FuncFormatter(lambda v, _: f"{v:.3g}")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(NullFormatter())
        if logx:
            ax.set_xscale("log")
        grid_ps = sorted({r["p"] for r in grows})
        step = max(1, math.ceil(len(grid_ps) / 8))          # at most ~8 labeled ticks
        ax.xaxis.set_major_locator(FixedLocator(grid_ps[::step]))
        ax.xaxis.set_major_formatter(plain)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("physical error rate p", color=INK)
        ax.set_ylabel("logical error rate per shot", color=INK)
        ax.set_title(_title(gkey, group_keys), color=INK, fontsize=10, loc="left")
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2, which="both", labelsize=8)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK)
        fig.tight_layout()
        stem = "_".join(str(v) for v in gkey).replace("/", "-").replace(" ", "")
        png = out_dir / f"{stem}.png"
        fig.savefig(png, facecolor=SURFACE)
        plt.close(fig)
        write_csv(sorted(grows, key=lambda r: (r.get(curve_key), r["p"])), out_dir / f"{stem}_points.csv")
        if mine:
            write_csv(mine, out_dir / f"{stem}_crossings.csv")
        written.append(png)
    return written
