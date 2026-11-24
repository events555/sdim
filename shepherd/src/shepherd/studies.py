"""
studies.py -- the built-in studies, by name. `python -m shepherd list` prints them.

Grids are placed around where coarse scans put the crossings (union-find decoder, Nov 2026). A
grid is a starting point: run a coarse pass first (`--max-errors 50`), look at the plot, then
run the full budget, adding points with `--ps` where the crossing turned out to be. The coarse
shots stay in the results file and count toward the full budget.

Your own studies: define Study objects in any Python file and run them with
`python -m shepherd run path/to/file.py:NAME`.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from shepherd.codes.css import lacross_code, repetition_code, surface_code
from shepherd.noise_model import NoiseModel
from shepherd.study import Study


def _grid(lo: float, hi: float, num: int) -> list[float]:
    return [float(x) for x in np.linspace(lo, hi, num)]


def _by_q(grids: dict[int, list[float]]):
    return lambda code: grids[code.dimension]


# Thresholds from the coarse pass (2026-11-24; union-find, ~50 errors per point). Each study's
# grid gets four extra points at 0.90, 0.95, 1.05 and 1.10 times these, so the full run has enough
# points within +-15% of the threshold for the joint fit's narrow, unbiased window (analysis.py).
COARSE_THRESHOLDS: dict[str, dict[int, float]] = {
    "rep_phenom": {2: 0.103, 3: 0.127, 5: 0.138, 7: 0.145},
    "surface_cc": {2: 0.0966, 3: 0.123, 5: 0.136},
    "surface_phenom": {2: 0.0270, 3: 0.0319, 5: 0.0354},
    "lacross_cc": {2: 0.0338, 3: 0.0350, 5: 0.0384},
}


def _near(study: str, base):
    """`base` (a grid, or a function of the code) plus points close to the study's threshold."""
    def grid(code):
        pts = set(base(code) if callable(base) else base)
        center = COARSE_THRESHOLDS[study].get(code.dimension)
        if center is not None:
            pts |= {float(f"{center * f:.4g}") for f in (0.90, 0.95, 1.05, 1.10)}
        return sorted(pts)
    return grid


def _large_near_threshold(base, *, min_qudits: int, lo: float, hi: float):
    """`base`, but codes with more than `min_qudits` qudits only get the points in [lo, hi]."""
    def grid(code):
        pts = base(code) if callable(base) else base
        return [p for p in pts if code.n <= min_qudits or lo <= p <= hi]
    return grid


def _circuit(two_qudit: str):
    return lambda q, p: NoiseModel.circuit_level(q, p, two_qudit=two_qudit)


STUDIES: dict[str, Study] = {s.name: s for s in [
    Study(
        name="rep_phenom",
        description="repetition code, phenomenological noise, d rounds, q = 2, 3, 5, 7",
        experiment="ler", experiment_params={"rounds": "d"},
        codes=lambda: [repetition_code(L, q) for q in (2, 3, 5, 7) for L in (5, 9, 13, 17)],
        noise=NoiseModel.phenomenological,
        ps=_near("rep_phenom", _grid(0.04, 0.16, 7)),
        max_shots=100_000, max_errors=300),
    Study(
        name="surface_cc",
        description="planar surface code (HGP of repetition codes), code capacity, q = 2, 3, 5",
        experiment="ler", experiment_params={"rounds": 0},
        codes=lambda: [surface_code(L, q) for q in (2, 3, 5) for L in (5, 7, 9, 11, 13)],
        noise=NoiseModel.code_capacity,
        ps=_near("surface_cc", _by_q({2: _grid(0.07, 0.13, 7), 3: _grid(0.08, 0.20, 7),
                                       5: _grid(0.08, 0.24, 9)})),
        max_shots=100_000, max_errors=300),
    Study(
        name="surface_phenom",
        description="planar surface code, phenomenological noise, d rounds, q = 2, 3, 5",
        experiment="ler", experiment_params={"rounds": "d"},
        codes=lambda: [surface_code(L, q) for q in (2, 3, 5) for L in (5, 7, 9, 11)],
        noise=NoiseModel.phenomenological,
        ps=_near("surface_phenom", _grid(0.015, 0.055, 9)),
        max_shots=30_000, max_errors=300),
    Study(
        name="lacross_cc",
        description="open-boundary LaCross codes (HGP of 1 + x + x^4), n = 12-36, code capacity, "
                    "q = 2, 3, 5",
        experiment="ler", experiment_params={"rounds": 0},
        # n = 28, 32, 36 (1360-2320 qudits) were added because n = 12-24 showed no common
        # crossing at q = 2, 3. They decode at only ~1-10 shots/s per core near threshold, so they
        # get just the points in [0.025, 0.045] and at most 20,000 shots each.
        codes=lambda: [lacross_code(n, 4, q) for q in (2, 3, 5) for n in (12, 16, 20, 24, 28, 32, 36)],
        noise=NoiseModel.code_capacity,
        ps=_large_near_threshold(_near("lacross_cc", _grid(0.015, 0.05, 8)),
                                 min_qudits=1000, lo=0.025, hi=0.045),
        max_shots=lambda code, p: 20_000 if code.n > 1000 else 100_000,
        max_errors=lambda code, p: 200 if code.n > 400 else 300),
    Study(
        name="lacross_phenom",
        description="open-boundary LaCross codes, phenomenological noise, d rounds, q = 2, 3, 5",
        experiment="ler", experiment_params={"rounds": "d"},
        codes=lambda: [lacross_code(n, 4, q) for q in (2, 3, 5) for n in (12, 16, 20)],
        noise=NoiseModel.phenomenological,
        # Coarse thresholds ~0.0115 (q=2), ~0.0173 (q=3), ~0.013 (q=5, least certain). Points far
        # above threshold are nearly saturated and the slowest to decode (n=20 spent ~85% of
        # the coarse pass at p >= 0.016), so each grid stops ~30-40% above its threshold.
        ps=_by_q({2: [0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.016],
                  3: [0.012, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.021],
                  5: [0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.016, 0.018]}),
        max_shots=20_000, max_errors=200),
    Study(
        name="surface_circuit",
        description="planar surface code, circuit-level noise on the Z-syndrome circuit "
                    "(independent single-qudit noise on each SUM qudit), d rounds, q = 2, 3, 5",
        experiment="ler", experiment_params={"rounds": "d"},
        codes=lambda: [surface_code(L, q) for q in (2, 3, 5) for L in (3, 5, 7, 9)],
        noise=_circuit("independent"),
        ps=_grid(0.003, 0.015, 7),
        max_shots=30_000, max_errors=300),
    Study(
        name="surface_circuit_n2",
        description="as surface_circuit, with two-qudit depolarizing (N2) after each SUM",
        experiment="ler", experiment_params={"rounds": "d"},
        codes=lambda: [surface_code(L, q) for q in (2, 3, 5) for L in (3, 5, 7, 9)],
        noise=_circuit("N2"),
        ps=_grid(0.004, 0.016, 7),
        max_shots=30_000, max_errors=300),
    Study(
        name="lacross_circuit",
        description="open-boundary LaCross codes, circuit-level noise on the Z-syndrome circuit "
                    "(independent), d rounds, q = 3",
        experiment="ler", experiment_params={"rounds": "d"},
        codes=lambda: [lacross_code(n, 4, 3) for n in (12, 16, 20)],
        noise=_circuit("independent"),
        # Threshold ~0.003 (earlier scans). Points far above it are nearly saturated and decode
        # at <1 shot/s per core at n = 20, so the grid stops ~1.3x above it.
        ps=[0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004],
        max_shots=20_000, max_errors=200),

]}


def _with_decoder(study: Study, decoder: str, *, label: str | None = None,
                  max_qudits: int | None = None) -> Study:
    """The same study (codes, grids, budgets) with another decoder, under its own name, so its
    results go to their own file. `max_qudits` leaves out larger codes."""
    codes, limit = study.codes, ""
    if max_qudits is not None:
        codes = lambda: [c for c in study.codes() if c.n <= max_qudits]
        limit = f" (only codes up to {max_qudits} qudits)"
    return replace(study, name=f"{study.name}_{decoder.removeprefix('pymatching_')}",
                   description=f"{study.description}; decoder {label or decoder}{limit}",
                   decoders=(decoder,), codes=codes)


# Lifted-MWPM (Keppens; registered as pymatching_keppens) versions of the core studies, for
# comparison with union-find
# (`shepherd compare surface_cc surface_cc_keppens`). Matching decodes ~100x faster, so these are
# cheap; for LaCross only n <= 24, since matching cannot represent its hyperedges anyway.
for _name, _cap in (("rep_phenom", None), ("surface_cc", None), ("surface_phenom", None),
                    ("lacross_cc", 1000)):
    _s = _with_decoder(STUDIES[_name], "pymatching_keppens", label="lifted MWPM (pymatching_keppens)",
                       max_qudits=_cap)
    STUDIES[_s.name] = _s


def get_study(name: str) -> Study:
    """A built-in study by name, or `path/to/file.py:NAME` for a Study defined in a file."""
    if ":" in name and name.split(":", 1)[0].endswith(".py"):
        import runpy
        path, attr = name.split(":", 1)
        obj = runpy.run_path(path).get(attr)
        if not isinstance(obj, Study):
            raise KeyError(f"{path} defines no Study called {attr!r}")
        return obj
    if name not in STUDIES:
        raise KeyError(f"unknown study {name!r}; built-in studies: {sorted(STUDIES)}")
    return STUDIES[name]
