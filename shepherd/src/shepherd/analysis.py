"""
analysis.py -- results files to tables, threshold estimates and plots.

    from shepherd import analysis
    rows = analysis.load_results("results/surface_cc.jsonl")
    for t in analysis.thresholds(rows):          # one joint fit per group
        print(t["q"], t["p_th"], t["p_lo"], t["p_hi"])
    analysis.plot_thresholds(rows, "plots/")

A row is one task's totals: its json_metadata (code, family, n, k, d, q, experiment, rounds,
basis, noise, noise_fingerprint, p), plus decoder, shots, errors, discards, seconds, and the
logical error rate with a 95% Wilson interval (ler, ler_lo, ler_hi). The logical error rate is per
shot: a shot fails if any logical observable is predicted wrong.

Groups and curves
-----------------
A group is the rows that agree on GROUP_KEYS (family, noise, q, decoder, experiment, basis); within
a group each code size (ordered by n) gives one curve, LER against p.

Threshold: a joint finite-size-scaling fit (`thresholds`)
---------------------------------------------------------
All sizes of a group are fitted at once to the form used since Wang, Harrington and Preskill
(2003),

    LER ~ A + B x + C x^2,   x = (p - p_th) * d^(1/nu),

weighting each point by its binomial variance. For a trial (p_th, nu) the best A, B, C follow from
weighted linear least squares, so (p_th, nu) are found by a grid search (nu in [0.3, 5]) that
zooms in around the best point.

The quadratic form only holds near the threshold, so the fit uses only points near it: it starts
from the median of the pairwise crossings (below) and keeps the points within +-WINDOW (15%) of
it, widening to +-25% and then +-35% when fewer than 3 points remain for at least 2 sizes or
the fit lands on the window's edge (the status then says which), and leaving out points above 60% of saturation (1 - 1/q^k). On
synthetic curves with a known threshold and binomial noise, a 15% window showed no measurable
bias and its interval covered the truth in 20 of 20 runs; at 30% the estimate was about 1.5% low
and covered it in 12 of 20. So for a final number, sample at least 3 values of p within 15% of
the threshold. The 95% interval is a parametric bootstrap: each point's error count is redrawn
from Binomial(shots, LER) and the fit repeated; the 2.5th and 97.5th percentiles are kept.

Status: "ok", or "ok (window +-25%)" / "(+-35%)"; ", poor fit" is added when chi^2 per degree of
freedom exceeds 4, ", interval clipped by window" when the interval reaches the edge of the
fitted p range (the fit only searches inside it, so the true interval is wider), and ", pairwise
crossings drift (a-b)" when the crossings of consecutive sizes span more than a factor DRIFT
(1.25). When the curves have no common crossing at all, so the fit cannot place one, the status is
"no common crossing (pairwise a-b)" and no threshold is given; quote the range instead. Without a usable fit: "above grid" or "below grid" when no pair of sizes crosses
(the threshold lies outside the sampled p, on the side the curves' order shows), "no common
crossing (...)", "too few points near the crossing", "fit at the window edge", or "too few
points".

Pairwise crossings (`crossings`), a diagnostic
---------------------------------------------
For consecutive sizes, each curve's log LER against log p is fitted with a weighted quadratic (a
line with 2 points; points with no errors are left out), and the crossing is where the larger
code's fitted curve rises above the smaller one's. `num_crossings` counts how many times the fitted
curves cross inside the shared p range: more than one means the pair does not separate cleanly
and its crossing should not be trusted. The range is a bootstrap interval as above.

Crossings of consecutive sizes drifting with size are a finite-size effect; the joint fit accounts
for it through nu. A crossing near saturation is not a threshold.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from shepherd.stats import read_stats_file, wilson_interval

GROUP_KEYS = ("family", "noise", "q", "decoder", "experiment", "basis")
CURVE_KEY = "n"
WINDOW = 0.15              # second-stage fit: p within +-15% of the first-stage estimate
SATURATION_CUT = 0.60      # first stage: points below this fraction of 1 - 1/q^k
BOOTSTRAP = 200
DRIFT = 1.25               # pairwise crossings spread wider than this ratio are reported as drift

# Categorical series colors, in fixed order (validated light-mode palette).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# -- loading and grouping ---------------------------------------------------------------------------

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


def _kept(r: dict) -> int:
    return int(r["shots"]) - int(r.get("discards", 0) or 0)


def _arrays(rows: Sequence[dict]):
    p = np.array([r["p"] for r in rows], dtype=float)
    d = np.array([r["d"] for r in rows], dtype=float)
    n = np.array([_kept(r) for r in rows], dtype=float)
    e = np.array([r["errors"] for r in rows], dtype=float)
    return p, d, n, e


def _weights(n: np.ndarray, e: np.ndarray) -> np.ndarray:
    """1 / binomial variance of each LER, with a floor so points with 0 or all errors count."""
    ph = (e + 0.5) / (n + 1.0)
    return n / (ph * (1.0 - ph))


# -- the joint finite-size-scaling fit --------------------------------------------------------------

def _scaling_chi2(p, d, y, w, pth, nu):
    """chi^2 of the best quadratic in x for every (pth, nu) on the grids; shape (len(pth), len(nu))."""
    x = (p[None, None, :] - pth[:, None, None]) * d[None, None, :] ** (1.0 / nu[None, :, None])
    X = np.stack([np.ones_like(x), x, x * x], axis=-1)                 # (P, V, N, 3)
    Xw = X * w[None, None, :, None]
    A = np.einsum("pvni,pvnj->pvij", Xw, X) + 1e-12 * np.eye(3)
    b = np.einsum("pvni,pvn->pvi", Xw, y[None, None, :])
    coef = np.linalg.solve(A, b[..., None])[..., 0]
    resid = y[None, None, :] - np.einsum("pvni,pvi->pvn", X, coef)
    return np.einsum("pvn,n->pv", resid * resid, w)


def _fit_scaling(p, d, y, w, p_range):
    """(p_th, nu, chi2) minimizing chi2 over p_th in p_range and nu in [0.3, 5]."""
    lo, hi = p_range
    pth = np.linspace(lo, hi, 41)
    lognu = np.linspace(np.log(0.3), np.log(5.0), 25)
    for _ in range(4):
        chi2 = _scaling_chi2(p, d, y, w, pth, np.exp(lognu))
        i, j = np.unravel_index(np.argmin(chi2), chi2.shape)
        best = (pth[i], np.exp(lognu[j]), chi2[i, j])
        dp, dl = 2 * (pth[1] - pth[0]), 2 * (lognu[1] - lognu[0])
        pth = np.linspace(max(lo, pth[i] - dp), min(hi, pth[i] + dp), 21)
        lognu = np.linspace(max(np.log(0.3), lognu[j] - dl), min(np.log(5.0), lognu[j] + dl), 21)
    return best


def _enough(d: np.ndarray, min_sizes: int = 2, min_points: int = 3) -> bool:
    sizes, counts = np.unique(d, return_counts=True)
    return int((counts >= min_points).sum()) >= min_sizes


def fit_threshold(rows: Sequence[dict], *, window: float = WINDOW, bootstrap: int = BOOTSTRAP,
                  seed: int = 0) -> dict:
    """The joint fit for one group's rows (see the module docstring)."""
    rows = [r for r in rows if _kept(r) > 0 and r.get("d") is not None]
    out = {"p_th": None, "p_lo": None, "p_hi": None, "nu": None, "chi2_per_dof": None,
           "points": 0, "sizes": 0, "window": None, "pairwise_range": None,
           "status": "too few points"}
    if len({r["d"] for r in rows}) < 2:
        return out
    q, k = rows[0].get("q") or 2, rows[0].get("k") or 1
    saturation = 1.0 - float(q) ** -float(k)
    p, d, n, e = _arrays(rows)
    use = e / n <= SATURATION_CUT * saturation
    # Starting point: the median of the pairwise crossings. None at all means the curves do not
    # cross inside the grid, and which way they are ordered says on which side the threshold is.
    pairs = crossings(rows, bootstrap=0)
    found = [c["p_cross"] for c in pairs if c["p_cross"] is not None]
    if not found:
        cs = curves(rows)
        small, large = cs[min(cs)], cs[max(cs)]
        common = sorted({r["p"] for r in small} & {r["p"] for r in large})
        if common:
            by_p = lambda rs: {r["p"]: r["errors"] / _kept(r) for r in rs}
            a, b = by_p(small), by_p(large)
            better = sum(b[x] < a[x] for x in common)
            out["status"] = "above grid" if better > len(common) / 2 else "below grid"
        return out
    p0 = float(np.median(found))
    spread = (min(found), max(found)) if len(found) >= 2 else None
    drifting = spread is not None and spread[1] > DRIFT * spread[0]
    out["pairwise_range"] = spread
    # Narrowest window first; widen when it holds too few points or the fit lands on its edge.
    attempt = None
    for frac in (window, 0.25, 0.35):
        win = use & (np.abs(p - p0) <= frac * p0)
        if not _enough(d[win]):
            continue
        P, D, N, E = p[win], d[win], n[win], e[win]
        span = (float(P.min()), float(P.max()))
        pth, nu, chi2 = _fit_scaling(P, D, E / N, _weights(N, E), span)
        edge = 0.02 * (span[1] - span[0])
        attempt = (frac, P, D, N, E, span, pth, nu, chi2, edge)
        if span[0] + edge < pth < span[1] - edge:
            break
    if attempt is None:
        out["status"] = "too few points near the crossing"
        return out
    frac, P, D, N, E, span, pth, nu, chi2, edge = attempt
    dof = max(1, len(P) - 5)
    status = "ok" if frac == window else f"ok (window +-{frac:.0%})"
    if chi2 / dof > 4:
        status += ", poor fit"
    if not span[0] + edge < pth < span[1] - edge:
        status = "fit at the window edge"
        if drifting:
            status = f"no common crossing (pairwise {spread[0]:.3g}-{spread[1]:.3g})"
    elif drifting:
        status += f", pairwise crossings drift ({spread[0]:.3g}-{spread[1]:.3g})"
    out.update(p_th=float(pth), nu=float(nu), chi2_per_dof=float(chi2 / dof), points=int(len(P)),
               sizes=int(len(np.unique(D))), window=span, status=status)
    if not status.startswith("ok"):
        out["p_th"] = None                  # no threshold to report; the status says why
    if bootstrap and status.startswith("ok"):
        rng = np.random.default_rng(seed)
        ph = E / N
        boots = []
        for _ in range(bootstrap):
            Eb = rng.binomial(N.astype(np.int64), ph).astype(float)
            boots.append(_fit_scaling(P, D, Eb / N, _weights(N, Eb), span)[0])
        out["p_lo"], out["p_hi"] = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
        if out["p_lo"] <= span[0] + edge or out["p_hi"] >= span[1] - edge:
            out["status"] += ", interval clipped by window"     # the true interval is wider
    return out


def thresholds(rows: Iterable[dict], *, group_keys: Sequence[str] = GROUP_KEYS,
               bootstrap: int = BOOTSTRAP) -> list[dict]:
    """One joint fit per group."""
    out = []
    for gkey, grows in group_rows(rows, group_keys).items():
        try:
            fit = fit_threshold(grows, bootstrap=bootstrap)
        except Exception as e:                  # a fit must never stop plots or tables
            fit = {"p_th": None, "p_lo": None, "p_hi": None, "status": f"fit failed: {e}"}
        out.append({**dict(zip(group_keys, gkey)), **fit})
    return out


# -- pairwise crossings ------------------------------------------------------------------------------

def _log_fit(rows: Sequence[dict], errors: np.ndarray | None = None):
    """Weighted polynomial (degree <= 2) of log LER against log p, or None with < 2 usable points."""
    p, _, n, e = _arrays(rows)
    if errors is not None:
        e = errors
    keep = e > 0
    if keep.sum() < 2:
        return None
    x, y = np.log(p[keep]), np.log(e[keep] / n[keep])
    sigma = np.sqrt((1 - e[keep] / n[keep]) / e[keep]) + 1e-12      # sd of log LER
    deg = min(2, int(keep.sum()) - 1)
    return np.polynomial.Polynomial.fit(x, y, deg, w=1 / sigma)


def _pair_crossings(fa, fb, lo: float, hi: float) -> list[float]:
    """p values in [lo, hi] where fitted curve b rises above fitted curve a."""
    xs = np.linspace(np.log(lo), np.log(hi), 400)
    diff = fb(xs) - fa(xs)
    up = np.nonzero((diff[:-1] <= 0) & (diff[1:] > 0))[0]
    return [float(np.exp(xs[i] - diff[i] * (xs[i + 1] - xs[i]) / (diff[i + 1] - diff[i]))) for i in up]


def crossings(rows: Iterable[dict], *, group_keys: Sequence[str] = GROUP_KEYS,
              curve_key: str = CURVE_KEY, bootstrap: int = BOOTSTRAP, seed: int = 0) -> list[dict]:
    """One dict per group and pair of consecutive sizes (see the module docstring)."""
    out = []
    rng = np.random.default_rng(seed)
    for gkey, grows in group_rows(rows, group_keys).items():
        cs = curves(grows, curve_key)
        sizes = list(cs)
        for small, large in zip(sizes, sizes[1:]):
            a, b = cs[small], cs[large]
            row = {**dict(zip(group_keys, gkey)), curve_key + "_small": small,
                   curve_key + "_large": large, "code_small": a[0].get("code"),
                   "code_large": b[0].get("code"), "p_cross": None, "p_lo": None, "p_hi": None,
                   "num_crossings": 0}
            lo = max(min(r["p"] for r in a), min(r["p"] for r in b))
            hi = min(max(r["p"] for r in a), max(r["p"] for r in b))
            fa, fb = _log_fit(a), _log_fit(b)
            if fa is not None and fb is not None and hi > lo:
                found = _pair_crossings(fa, fb, lo, hi)
                row["num_crossings"] = len(found)
                if found:
                    row["p_cross"] = float(np.median(found))
                    if bootstrap:
                        _, _, na, ea = _arrays(a)
                        _, _, nb, eb = _arrays(b)
                        boots = []
                        for _ in range(bootstrap):
                            ga = _log_fit(a, rng.binomial(na.astype(np.int64), ea / na).astype(float))
                            gb = _log_fit(b, rng.binomial(nb.astype(np.int64), eb / nb).astype(float))
                            if ga is not None and gb is not None:
                                f = _pair_crossings(ga, gb, lo, hi)
                                if f:
                                    boots.append(float(np.median(f)))
                        if len(boots) >= bootstrap // 2:
                            row["p_lo"], row["p_hi"] = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
            out.append(row)
    return out


# -- tables -------------------------------------------------------------------------------------------

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
        if isinstance(v, tuple):
            return "-".join(fmt(x) for x in v)
        return "" if v is None else str(v)
    cells = [[fmt(r.get(c)) for c in columns] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(columns)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(columns, widths))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in cells]
    return "\n".join(lines)


# -- plots ---------------------------------------------------------------------------------------------

def _title(gkey: tuple, keys: Sequence[str]) -> str:
    g = dict(zip(keys, gkey))
    parts = [str(g.get("family")), str(g.get("noise")).replace("_", " "), f"q = {g.get('q')}",
             f"decoder: {_decoder_label(str(g.get('decoder')))}"]
    return ", ".join(parts)


def _style_axes(ax, rows: Sequence[dict], title: str, logx: bool) -> None:
    """Shared look of every plot: log axes with plain tick labels at the grid's p values."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
    plain = FuncFormatter(lambda v, _: f"{v:.3g}")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(NullFormatter())
    if logx:
        ax.set_xscale("log")
    grid_ps = sorted({r["p"] for r in rows if r["p"] > 0})
    # Label grid values spread out on the log axis (dense points near a threshold would overlap).
    gap = math.log10(grid_ps[-1] / grid_ps[0]) / 8 if len(grid_ps) > 1 else 1.0
    ticks: list[float] = []
    for v in grid_ps:
        if not ticks or math.log10(v / ticks[-1]) >= gap:
            ticks.append(v)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("physical error rate p", color=INK)
    ax.set_ylabel("logical error rate per shot", color=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, which="both", labelsize=8)


def plot_thresholds(rows: Iterable[dict], out_dir: str | Path, *,
                    group_keys: Sequence[str] = GROUP_KEYS, curve_key: str = CURVE_KEY,
                    logx: bool = True, bootstrap: int = BOOTSTRAP) -> list[Path]:
    """
    One PNG per group: LER against p, one curve per code size with 95% Wilson bars, and the joint
    fit's threshold with its 95% interval shaded. Next to each PNG: the group's points, its joint
    fit (`_threshold.csv`) and its pairwise crossings (`_crossings.csv`). Returns the PNG paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fits = {tuple(t.get(k) for k in group_keys): t
            for t in thresholds(rows, group_keys=group_keys, bootstrap=bootstrap)}
    pairs = crossings(rows, group_keys=group_keys, curve_key=curve_key, bootstrap=bootstrap)
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
            yerr = [[r["ler"] - r["ler_lo"] for r in pts], [r["ler_hi"] - r["ler"] for r in pts]]
            label = f"{crows[0].get('code')}  (n={crows[0].get('n')}, d={crows[0].get('d')})"
            ax.errorbar([r["p"] for r in pts], [r["ler"] for r in pts], yerr=yerr, color=color,
                        marker=MARKERS[i % len(MARKERS)], markersize=5, linewidth=2, elinewidth=1,
                        capsize=2, label=label, markeredgecolor=SURFACE, markeredgewidth=1)
        fit = fits.get(gkey, {})
        if str(fit.get("status", "")).startswith("ok") and fit.get("p_th") is not None:
            if fit.get("p_lo") is not None:
                ax.axvspan(fit["p_lo"], fit["p_hi"], color=INK_2, alpha=0.10, linewidth=0)
            ax.axvline(fit["p_th"], color=INK_2, linewidth=1, linestyle="--")
            ci = (f" (95%: {fit['p_lo']:.4g}-{fit['p_hi']:.4g})" if fit.get("p_lo") is not None else "")
            note = f" threshold ≈ {fit['p_th']:.4g}{ci}"
            if "drift" in str(fit.get("status")) and fit.get("pairwise_range"):
                lo, hi = fit["pairwise_range"]
                note += f"\n pairs of sizes cross between {lo:.3g} and {hi:.3g} (finite-size drift)"
        else:
            note = f" threshold: {fit.get('status', 'no fit')}"
        ax.text(0.01, 0.02, note, transform=ax.transAxes, color=INK_2, fontsize=8)
        _style_axes(ax, grows, _title(gkey, group_keys), logx)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK)
        fig.tight_layout()
        stem = "_".join(str(v) for v in gkey).replace("/", "-").replace(" ", "")
        png = out_dir / f"{stem}.png"
        fig.savefig(png, facecolor=SURFACE)
        plt.close(fig)
        write_csv(sorted(grows, key=lambda r: (r.get(curve_key), r["p"])), out_dir / f"{stem}_points.csv")
        if fit:
            write_csv([fit], out_dir / f"{stem}_threshold.csv")
        mine = [c for c in pairs if tuple(c.get(k) for k in group_keys) == gkey]
        if mine:
            write_csv(mine, out_dir / f"{stem}_crossings.csv")
        written.append(png)
    return written


# -- decoder comparisons -------------------------------------------------------------------------------

COMPARE_KEYS = ("family", "noise", "q", "experiment", "basis")     # GROUP_KEYS without the decoder
# Display names. The registry names stay as they are: a decoder's name is part of every task's
# strong_id, so renaming one would orphan the results already collected.
DECODER_NAMES = {"union_find": "union-find", "pymatching_keppens": "lifted MWPM",
                 "pymatching": "matching (binarized syndromes)", "none": "no decoding"}
DECODER_ORDER = ("union_find", "pymatching_keppens", "pymatching", "none")
LINESTYLES = ["-", "--", ":", "-."]


def _decoder_label(name: str) -> str:
    return DECODER_NAMES.get(name, name)


def _threshold_note(fit: dict) -> str:
    status = str(fit.get("status", ""))
    if status.startswith("ok") and fit.get("p_th") is not None:
        ci = f" (95%: {fit['p_lo']:.4g}-{fit['p_hi']:.4g})" if fit.get("p_lo") is not None else ""
        drift = ""
        if "drift" in status and fit.get("pairwise_range"):
            drift = f"; pairs cross {fit['pairwise_range'][0]:.3g}-{fit['pairwise_range'][1]:.3g}"
        return f"threshold ≈ {fit['p_th']:.4g}{ci}{drift}"
    if status == "below grid":
        return "no threshold in range: larger codes are worse at every p sampled"
    if status == "above grid":
        return "threshold above the sampled range: larger codes are better at every p"
    return f"no threshold: {status or 'no fit'}"


def plot_comparison(rows: Iterable[dict], out_dir: str | Path, *,
                    group_keys: Sequence[str] = COMPARE_KEYS, curve_key: str = CURVE_KEY,
                    logx: bool = True, bootstrap: int = BOOTSTRAP) -> list[Path]:
    """
    One PNG per group with two or more decoders: LER against p for every code size and decoder on
    the same axes. Color is the code size (as in plot_thresholds); line style and marker fill are
    the decoder (union-find solid with filled markers, then dashed, dotted ... with hollow ones).
    Each decoder's own joint threshold fit is stated in the corner and written, one row per
    decoder, to `<stem>_compare_thresholds.csv`. Returns the PNG paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    written = []
    for gkey, grows in group_rows(rows, group_keys).items():
        present = {r["decoder"] for r in grows}
        decoders = [d for d in DECODER_ORDER if d in present] + sorted(present - set(DECODER_ORDER))
        if len(decoders) < 2:
            continue
        sizes = sorted({r.get(curve_key) for r in grows})
        color = {n: SERIES[i % len(SERIES)] for i, n in enumerate(sizes)}
        fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        notes = []
        fits = []
        for j, dec in enumerate(decoders):
            drows = [r for r in grows if r["decoder"] == dec]
            for i, (size, crows) in enumerate(curves(drows, curve_key).items()):
                pts = [r for r in crows if r["errors"] > 0]
                if not pts:
                    continue
                yerr = [[r["ler"] - r["ler_lo"] for r in pts], [r["ler_hi"] - r["ler"] for r in pts]]
                ax.errorbar([r["p"] for r in pts], [r["ler"] for r in pts], yerr=yerr,
                            color=color[size], linestyle=LINESTYLES[j % len(LINESTYLES)],
                            marker=MARKERS[sizes.index(size) % len(MARKERS)], markersize=5,
                            markerfacecolor=color[size] if j == 0 else SURFACE,
                            markeredgecolor=color[size], linewidth=2 if j == 0 else 1.5,
                            elinewidth=1, capsize=2)
            fit = fit_threshold(drows, bootstrap=bootstrap)
            fits.append({**dict(zip(group_keys, gkey)), "decoder": dec, **fit})
            notes.append(f"{_decoder_label(dec)}: {_threshold_note(fit)}")
            if str(fit.get("status", "")).startswith("ok") and fit.get("p_th") is not None:
                ax.axvline(fit["p_th"], color=INK_2, linewidth=0.8, alpha=0.6,
                           linestyle=LINESTYLES[j % len(LINESTYLES)])
        ax.text(0.01, 0.02, "\n".join(notes), transform=ax.transAxes, color=INK_2, fontsize=7.5,
                va="bottom")
        title = ", ".join([str(dict(zip(group_keys, gkey)).get("family")),
                           str(dict(zip(group_keys, gkey)).get("noise")).replace("_", " "),
                           f"q = {dict(zip(group_keys, gkey)).get('q')}", "decoders compared"])
        _style_axes(ax, grows, title, logx)
        by_size = {}
        for r in grows:
            by_size.setdefault(r.get(curve_key), r)
        size_handles = [Line2D([], [], color=color[n], marker=MARKERS[i % len(MARKERS)], linewidth=2,
                               label=f"{by_size[n].get('code')} (d={by_size[n].get('d')})")
                        for i, n in enumerate(sizes)]
        dec_handles = [Line2D([], [], color=INK_2, linestyle=LINESTYLES[j % len(LINESTYLES)],
                              marker="o", markerfacecolor=INK_2 if j == 0 else SURFACE,
                              markeredgecolor=INK_2, linewidth=2 if j == 0 else 1.5,
                              label=_decoder_label(d)) for j, d in enumerate(decoders)]
        first = ax.legend(handles=size_handles, fontsize=7.5, frameon=False, labelcolor=INK,
                          loc="upper left", title="code", title_fontsize=7.5)
        ax.add_artist(first)
        ax.legend(handles=dec_handles, fontsize=7.5, frameon=False, labelcolor=INK,
                  loc="lower right", title="decoder", title_fontsize=7.5)
        fig.tight_layout()
        stem = "_".join(str(v) for v in gkey).replace("/", "-").replace(" ", "") + "_compare"
        png = out_dir / f"{stem}.png"
        fig.savefig(png, facecolor=SURFACE)
        plt.close(fig)
        write_csv(fits, out_dir / f"{stem}_thresholds.csv")
        written.append(png)
    return written
