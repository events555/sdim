"""The ler experiment, Study / run_study, built-in studies, analysis, and the command line."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("sdim.dem")

from shepherd import analysis  # noqa: E402
from shepherd.__main__ import main  # noqa: E402
from shepherd.codes.css import repetition_code, surface_code  # noqa: E402
from shepherd.decoder import Decoder  # noqa: E402
from shepherd.experiment import Experiment  # noqa: E402
from shepherd.noise_model import NoiseModel  # noqa: E402
from shepherd.stats import read_stats_file  # noqa: E402
from shepherd.studies import STUDIES, get_study  # noqa: E402
from shepherd.study import Study, plan, run_study  # noqa: E402
from shepherd.task import Task  # noqa: E402


# -- the ler experiment ----------------------------------------------------------------------------

def test_ler_is_registered_and_builds_noiseless_circuits():
    cls = Experiment.lookup("ler")
    assert cls.experiment_name == "ler"
    code = surface_code(3, 5)
    exp = Experiment.create("ler", 5, rounds="d")
    assert exp.rounds_for(code) == 3
    circuit = exp.build(code)
    assert not any(op.name in ("N1", "N2") for op in circuit.operations)
    assert sum(op.name == "I" for op in circuit.operations) == 3 * code.n   # data waits once per round
    assert exp.metadata(code) == {"experiment": "ler", "rounds": 3, "basis": "Z"}
    assert np.array_equal(exp.expected_observables(code), np.zeros(code.k))


def test_ler_rejects_bad_parameters():
    with pytest.raises(NotImplementedError):
        Experiment.create("ler", 3, basis="X")
    with pytest.raises(ValueError):
        Experiment.create("ler", 3, rounds=-1)
    with pytest.raises(ValueError):
        Experiment.create("ler", 3).build(surface_code(3, 5))


# -- Study -----------------------------------------------------------------------------------------

def _tiny(**kw) -> Study:
    args = dict(name="tiny", description="", experiment="ler", experiment_params={"rounds": "d"},
                codes=lambda: [repetition_code(L, q) for q in (2, 3) for L in (3, 5)],
                noise=NoiseModel.phenomenological, ps=[0.05, 0.1],
                decoders=["union_find", "none"], max_shots=600, max_errors=None)
    args.update(kw)
    return Study(**args)


def test_points_metadata_filters_and_budgets():
    pts = _tiny().points()
    assert len(pts) == 2 * 2 * 2 * 2
    meta = pts[0].metadata
    assert set(meta) == {"code", "family", "n", "k", "d", "q", "experiment", "rounds", "basis",
                         "noise", "noise_fingerprint", "p"}
    assert meta["noise"] == "phenomenological" and meta["rounds"] == meta["d"]
    some = _tiny().points(where={"q": ["3"], "decoder": ["union_find"], "p": ["0.05"]})
    assert {(pt.code.dimension, pt.task.decoder, pt.p) for pt in some} == {(3, "union_find", 0.05)}
    assert len(some) == 2
    assert {pt.p for pt in _tiny().points(ps=[0.02])} == {0.02}
    by_q = _tiny(ps=lambda c: [0.01 * c.dimension], max_shots=lambda c, p: 100 * c.n).points()
    assert {(pt.code.dimension, pt.p) for pt in by_q} == {(2, 0.02), (3, 0.03)}
    assert {pt.max_shots for pt in by_q} == {300, 500}


def test_run_study_writes_resumes_and_plans(tmp_path):
    study = _tiny()
    stats = run_study(study, results_dir=tmp_path, num_workers=0, print_progress=False)
    assert len(stats) == 16 and all(s.shots == 600 for s in stats)
    on_disk = read_stats_file(study.results_path(tmp_path))
    assert sum(s.shots for s in on_disk.values()) == 16 * 600
    again = run_study(study, results_dir=tmp_path, num_workers=0, print_progress=False)
    assert all(s.shots == 600 for s in again)                     # nothing re-run
    assert sum(s.shots for s in read_stats_file(study.results_path(tmp_path)).values()) == 16 * 600
    rows = plan(study, results_dir=tmp_path, max_shots=1000)
    assert all(not r["done"] and r["shots"] == 600 for r in rows)
    more = run_study(study, results_dir=tmp_path, num_workers=0, max_shots=1000,
                     where={"q": [2]}, print_progress=False)
    assert len(more) == 8 and all(s.shots == 1000 for s in more)


def test_decoder_version_is_part_of_the_strong_id(monkeypatch):
    pt = _tiny().points(where={"decoder": ["union_find"]})[0]
    before = pt.task.strong_id()
    cls = Decoder.lookup("union_find")
    monkeypatch.setattr(cls, "version", cls.version + 1)
    again = Task(pt.task.circuit, "union_find", json_metadata=pt.metadata,
                 detector_error_model=pt.task.dem())
    assert again.strong_id() != before
    custom = Task(pt.task.circuit, "my_own_decoder", detector_error_model=pt.task.dem())
    assert custom.strong_id()                                    # unregistered names still work


def test_builtin_studies_build():
    assert {"rep_phenom", "surface_cc", "surface_phenom", "lacross_cc", "lacross_phenom",
            "surface_circuit", "surface_circuit_n2", "lacross_circuit"} <= set(STUDIES)
    for study in STUDIES.values():
        codes = list(study.codes())
        q, n = min(c.dimension for c in codes), min(c.n for c in codes)
        pts = study.points(where={"q": [q], "n": [n]})
        assert pts, study.name
        grid = study.ps(pts[0].code) if callable(study.ps) else study.ps
        assert len(pts) == len(grid) * len(study.decoders)
        assert all(pt.task.json_metadata["noise"] for pt in pts)


# -- analysis --------------------------------------------------------------------------------------

def _synthetic(threshold=0.1, sizes=(5, 9, 13), ps=(0.06, 0.08, 0.1, 0.12, 0.14)):
    """Noiseless curves 0.1 * (p / threshold)^(n/2): every pair crosses exactly at the threshold."""
    rows = []
    for n in sizes:
        for p in ps:
            ler = 0.1 * (p / threshold) ** (n / 2)
            rows.append({"family": "toy", "noise": "x", "q": 3, "decoder": "d", "experiment": "ler",
                         "basis": "Z", "n": n, "d": n, "code": f"toy{n}", "p": p, "shots": 10**6,
                         "errors": round(ler * 10**6), "ler": ler,
                         "ler_lo": ler * 0.97, "ler_hi": ler * 1.03})
    return rows


def _noisy(seed, threshold=0.1, ds=(5, 9, 13, 17), errors=300):
    """Binomially sampled curves 0.08 * (p / threshold)^((d+1)/2) on a grid around the threshold."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in ds:
        for p in np.linspace(0.07, 0.13, 7):
            ler = min(0.75, 0.08 * (p / threshold) ** ((d + 1) / 2))
            shots = int(min(10**6, errors / ler))
            e = int(rng.binomial(shots, ler))
            rows.append({"family": "toy", "noise": "x", "q": 3, "k": 1, "decoder": "d",
                         "experiment": "ler", "basis": "Z", "n": d * d, "d": d, "code": f"toy{d}",
                         "p": float(p), "shots": shots, "errors": e, "discards": 0, "ler": e / shots,
                         "ler_lo": 0.0, "ler_hi": 1.0})
    return rows


def test_crossings_find_a_known_threshold():
    cross = analysis.crossings(_synthetic())
    assert len(cross) == 2
    for c in cross:
        assert math.isclose(c["p_cross"], 0.1, rel_tol=1e-3)
        assert c["num_crossings"] == 1
        assert c["p_lo"] <= 0.1 <= c["p_hi"]
    assert analysis.crossings(_synthetic(ps=(0.02, 0.04)))[0]["p_cross"] is None   # all below


def test_joint_fit_recovers_a_known_threshold():
    # Noisy data, 300 errors per point. The estimate is within 2% and the 95% interval covers the
    # truth in most runs (20 of 20 when the method was chosen).
    covered = 0
    for seed in range(6):
        fit = analysis.fit_threshold(_noisy(seed), bootstrap=60, seed=seed)
        assert fit["status"].startswith("ok"), fit["status"]
        assert abs(fit["p_th"] - 0.1) < 0.002
        covered += fit["p_lo"] <= 0.1 <= fit["p_hi"]
    assert covered >= 5


def test_joint_fit_reports_a_threshold_outside_the_grid():
    below = [r for r in _noisy(0) if r["p"] < 0.095]        # every p below the threshold
    assert analysis.fit_threshold(below, bootstrap=0)["status"] == "above grid"
    above = [r for r in _noisy(0) if r["p"] > 0.105]
    assert analysis.fit_threshold(above, bootstrap=0)["status"] == "below grid"


def test_noise_does_not_pick_an_early_crossing():
    # The old method took the first raw sign change; noise made it report a crossing far below the
    # threshold. Fitted curves cross once, near the truth, even at 50 errors per point.
    for seed in range(4):
        for c in analysis.crossings(_noisy(seed, errors=50), bootstrap=0):
            assert c["num_crossings"] == 1
            assert abs(c["p_cross"] - 0.1) < 0.015


def test_plot_writes_png_and_tables(tmp_path):
    pngs = analysis.plot_thresholds(_noisy(0), tmp_path, bootstrap=20)
    assert len(pngs) == 1 and pngs[0].stat().st_size > 1000
    for kind in ("points", "threshold", "crossings"):
        assert len(list(tmp_path.glob(f"*_{kind}.csv"))) == 1, kind


# -- command line ----------------------------------------------------------------------------------

def test_command_line(tmp_path, capsys):
    study_file = tmp_path / "mine.py"
    study_file.write_text(
        "from shepherd.study import Study\n"
        "from shepherd.noise_model import NoiseModel\n"
        "from shepherd.codes.css import surface_code\n"
        "S = Study(name='mine', description='', experiment='ler', experiment_params={'rounds': 0},\n"
        "          codes=lambda: [surface_code(L, 3) for L in (3, 5)], noise=NoiseModel.code_capacity,\n"
        "          ps=[0.08, 0.12], max_shots=400, max_errors=None)\n")
    target = f"{study_file}:S"
    results = str(tmp_path / "results")
    assert main(["list"]) == 0
    assert main(["run", target, "--results", results, "--dry-run"]) == 0
    assert "4 tasks, 0 done" in capsys.readouterr().out
    assert main(["run", target, "--results", results, "--workers", "0"]) == 0
    assert main(["status", target, "--results", results]) == 0
    out = capsys.readouterr().out
    assert "surface_L5" in out and "0.12" in out
    assert main(["plot", str(tmp_path / "results" / "mine.jsonl"), "--out", str(tmp_path / "plots")]) == 0
    assert list((tmp_path / "plots").glob("*.png"))
    with pytest.raises(KeyError):
        get_study("no_such_study")
