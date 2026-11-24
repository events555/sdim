"""
python -m shepherd -- run studies and look at their results.

    python -m shepherd list
    python -m shepherd run surface_cc --workers 8                  # resumable; Ctrl-C is safe
    python -m shepherd run surface_cc --workers 8 --max-errors 50  # quick coarse pass first
    python -m shepherd run lacross_cc --where q=3 --where n=12,16 --ps 0.02,0.03,0.04
    python -m shepherd run surface_cc --dry-run                    # what would run, what's done
    python -m shepherd status surface_cc                           # per-task shots / errors / LER
    python -m shepherd plot surface_cc                             # PNG + CSV per group, crossings

A STUDY is a built-in name (see `list`) or `path/to/file.py:NAME`. Results go to
<results>/<study>.jsonl (default results/), plots to <plots>/<study>/ (default plots/).
`status` and `plot` also take a results file path instead of a study.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _where(items: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--where expects key=value[,value...], got {item!r}")
        k, v = item.split("=", 1)
        out.setdefault(k.strip(), []).extend(x.strip() for x in v.split(","))
    return out


def _ps(text: str | None) -> list[float] | None:
    return None if text is None else [float(x) for x in text.split(",") if x.strip()]


def _results_file(target: str, results_dir: str) -> Path:
    if target.endswith(".jsonl") or Path(target).is_file():
        return Path(target)
    from shepherd.studies import get_study
    return get_study(target).results_path(results_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m shepherd", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list the built-in studies")

    run = sub.add_parser("run", help="run (or resume) a study")
    run.add_argument("study")
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--results", default="results", help="results directory")
    run.add_argument("--where", action="append", default=[], metavar="KEY=V[,V...]",
                     help="keep only tasks whose metadata matches (repeatable): q=3, n=12,16, decoder=union_find")
    run.add_argument("--ps", help="comma-separated noise strengths, replacing the study's grid")
    run.add_argument("--max-shots", type=int, help="override every task's shot budget")
    run.add_argument("--max-errors", type=int, help="override every task's error budget")
    run.add_argument("--decoder-cache", help="directory for compiled decoders")
    run.add_argument("--dry-run", action="store_true", help="list the tasks and what is already done")

    for name, help_ in (("status", "per-task totals of a results file"),
                        ("plot", "threshold plots, CSV tables and crossing estimates")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("target", help="study name or results file")
        p.add_argument("--results", default="results", help="results directory")
        if name == "plot":
            p.add_argument("--out", default=None, help="output directory (default plots/<stem>)")
            p.add_argument("--linear-x", action="store_true", help="linear p axis")

    args = parser.parse_args(argv)

    if args.command == "list":
        from shepherd.studies import STUDIES
        for s in STUDIES.values():
            print(f"{s.name:20s} {s.description}")
        return 0

    from shepherd import analysis

    if args.command == "run":
        from shepherd.study import plan, run_study
        from shepherd.studies import get_study
        study = get_study(args.study)
        where, ps = _where(args.where), _ps(args.ps)
        if args.dry_run:
            rows = plan(study, results_dir=args.results, where=where, ps=ps,
                        max_shots=args.max_shots, max_errors=args.max_errors)
            cols = ["code", "q", "d", "rounds", "noise", "p", "decoder", "detectors",
                    "max_shots", "max_errors", "shots", "errors", "done"]
            print(analysis.format_table(rows, cols))
            print(f"\n{len(rows)} tasks, {sum(r['done'] for r in rows)} done; results file "
                  f"{study.results_path(args.results)}")
            return 0
        run_study(study, results_dir=args.results, num_workers=args.workers, where=where, ps=ps,
                  max_shots=args.max_shots, max_errors=args.max_errors,
                  decoder_cache_dir=args.decoder_cache)
        print(f"results in {study.results_path(args.results)}")
        return 0

    path = _results_file(args.target, args.results)
    rows = analysis.load_results(path)
    if not rows:
        print(f"no results in {path}")
        return 1
    if args.command == "status":
        rows.sort(key=lambda r: tuple(str(r.get(k)) for k in analysis.GROUP_KEYS) + (r.get("n"), r.get("p")))
        cols = ["code", "q", "d", "rounds", "noise", "p", "decoder", "shots", "errors", "ler",
                "ler_lo", "ler_hi", "seconds"]
        print(analysis.format_table(rows, cols))
        return 0

    out = Path(args.out) if args.out else Path("plots") / path.stem
    pngs = analysis.plot_thresholds(rows, out, logx=not args.linear_x)
    cross = analysis.crossings(rows)
    if cross:
        print(analysis.format_table(cross, ["family", "noise", "q", "decoder", "code_small",
                                            "code_large", "p_cross", "p_lo", "p_hi"]))
    print(f"\n{len(pngs)} plots in {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
