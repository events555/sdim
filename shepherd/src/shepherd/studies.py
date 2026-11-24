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

import numpy as np

from shepherd.codes.css import lacross_code, repetition_code, surface_code
from shepherd.noise_model import NoiseModel
from shepherd.study import Study


def _grid(lo: float, hi: float, num: int) -> list[float]:
    return [float(x) for x in np.linspace(lo, hi, num)]


def _by_q(grids: dict[int, list[float]]):
    return lambda code: grids[code.dimension]


def _circuit(two_qudit: str):
    return lambda q, p: NoiseModel.circuit_level(q, p, two_qudit=two_qudit)


STUDIES: dict[str, Study] = {s.name: s for s in [
    Study(
        name="rep_phenom",
        description="repetition code, phenomenological noise, d rounds, q = 2, 3, 5, 7",
        experiment="ler", experiment_params={"rounds": "d"},
        codes=lambda: [repetition_code(L, q) for q in (2, 3, 5, 7) for L in (5, 9, 13, 17)],
        noise=NoiseModel.phenomenological,
        ps=_grid(0.04, 0.16, 7),
        max_shots=100_000, max_errors=300),
    Study(
        name="surface_cc",
        description="planar surface code (HGP of repetition codes), code capacity, q = 2, 3, 5",
        experiment="ler", experiment_params={"rounds": 0},
        codes=lambda: [surface_code(L, q) for q in (2, 3, 5) for L in (5, 7, 9, 11, 13)],
        noise=NoiseModel.code_capacity,
        ps=_by_q({2: _grid(0.07, 0.13, 7), 3: _grid(0.08, 0.20, 7), 5: _grid(0.08, 0.24, 9)}),
        max_shots=100_000, max_errors=300),
    Study(
        name="surface_phenom",
        description="planar surface code, phenomenological noise, d rounds, q = 2, 3, 5",
        experiment="ler", experiment_params={"rounds": "d"},
        codes=lambda: [surface_code(L, q) for q in (2, 3, 5) for L in (5, 7, 9, 11)],
        noise=NoiseModel.phenomenological,
        ps=_grid(0.015, 0.055, 9),
        max_shots=30_000, max_errors=300),
    Study(
        name="lacross_cc",
        description="open-boundary LaCross codes (HGP of 1 + x + x^4), code capacity, q = 2, 3, 5",
        experiment="ler", experiment_params={"rounds": 0},
        codes=lambda: [lacross_code(n, 4, q) for q in (2, 3, 5) for n in (12, 16, 20, 24)],
        noise=NoiseModel.code_capacity,
        ps=_grid(0.015, 0.05, 8),
        max_shots=100_000, max_errors=lambda code, p: 200 if code.n > 400 else 300),
    Study(
        name="lacross_phenom",
        description="open-boundary LaCross codes, phenomenological noise, d rounds, q = 2, 3, 5",
        experiment="ler", experiment_params={"rounds": "d"},
        codes=lambda: [lacross_code(n, 4, q) for q in (2, 3, 5) for n in (12, 16, 20)],
        noise=NoiseModel.phenomenological,
        ps=_grid(0.006, 0.022, 9),
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
        ps=_grid(0.0015, 0.005, 6),
        max_shots=20_000, max_errors=200),
]}


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
