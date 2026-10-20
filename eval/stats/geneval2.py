"""Aggregate GenEval2 (soft-TIFA) numbers across seeds and plot them.

Reads each seed's ``log.txt`` -- which holds one ``ITER-n:`` block per
iteration followed by the summary tables -- and reports AM score, GM score and
the all-questions-correct rate, plus the per-skill AM scores, as mean +-
standard error over seeds together with the per-seed paired differences
against iter-1.

It also writes paper figures (score vs. iteration, mean +- s.e. over seeds) to
``out_dir`` as PDF (vector, for LaTeX) and PNG (preview):

* ``geneval2_overall``      -- AM / GM / all-correct side by side (\\textwidth)
* ``geneval2_<metric>``     -- one panel per metric (\\columnwidth)
* ``geneval2_skills``       -- per-skill AM, one small multiple per skill

Run with the ``tar`` env (the only one with matplotlib).
"""

import re
from pathlib import Path

from iter_plots import COLUMN_WIDTH, TEXT_WIDTH, complete_seeds, mean_sem, plot_panels

# label -> results dir holding seed_*/log.txt. Several entries are drawn as
# separate lines on the same axes (e.g. to compare methods).
RUNS = {
    "RL-t8 (MDP) IID": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/GenEval2/iter_adhoc_MDP_n4_slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1_rl_ft_oracle_t8_500",
    # "RL-t8 (w/o MDP) IID": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/GenEval2/iter_adhoc_niter4_slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1_rl_ft_oracle_t8_500",
    # "SFT": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/GenEval2/iter_adhoc_slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1",
    "RL-PM-t1 (MDP) OOD": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/GenEval2/iter_adhoc_MDP_n4_slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1_rl_ft_promptsmix_oracle_t1_500",
    # "RL-PM-t1 (w/o MDP) OOD": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/GenEval2/iter_adhoc_niter4_slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1_rl_ft_promptsmix_oracle_t1_500",
    "T2I OOD": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/GenEval2/iter_adhoc_niter4_slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1_bl_rl_t2i_t1_200"
}
out_dir = "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/GenEval2/figures"

METRICS = ["AM", "GM", "all-correct%"]
METRIC_LABELS = {
    "AM": "AM score",
    "GM": "GM score",
    "all-correct%": "All-correct prompts (%)",
}

ITER_RE = re.compile(r"^\s*ITER-(\d+)\s*:")
# full-precision per-iteration lines, e.g. "AM Score: 59.953261526490586"
SCORE_RE = re.compile(r"^(AM|GM) Score:\s*([\d.]+)", re.IGNORECASE)
# summary table row, one "count (pct%)" cell per iteration:
# "| All-questions correct | 138 (17.25%) | 173 (21.62%) | ... |  +56 |"
ALL_CORRECT_ROW = "| All-questions correct"
PCT_RE = re.compile(r"\(([\d.]+)%\)")
SKILL_HEADER = "Per-skill question scores"
# per-skill row "| attribute | 1214 | 72.56 | 80.24 | ... | 9.31 |": the cells
# after n are one per iteration, then the delta column
SKILL_RE = re.compile(r"^\|\s*([a-z_]+)\s*\|\s*\d+\s*\|(.*)\|\s*$")


def parse_log(path):
    """Return ({iter: {metric: value}}, {iter: {skill: AM}})."""
    lines = path.read_text(errors="replace").splitlines()
    per_iter = {}
    skills = {}
    cur_iter = None
    in_skills = False
    for line in lines:
        if line.startswith(SKILL_HEADER):
            in_skills = True
            continue
        if in_skills:
            # the section ends at the blank line after its table
            if not line.strip():
                in_skills = False
                continue
            m = SKILL_RE.match(line.strip())
            if m and m.group(1) != "skill":
                cells = [c.strip() for c in m.group(2).split("|")]
                for it, val in zip(sorted(per_iter), cells):
                    skills.setdefault(it, {})[m.group(1)] = float(val)
            continue

        m = ITER_RE.match(line)
        if m:
            cur_iter = int(m.group(1))
            continue

        m = SCORE_RE.match(line.strip())
        if m and cur_iter is not None:
            per_iter.setdefault(cur_iter, {})[m.group(1).upper()] = float(
                m.group(2)
            )
            continue

        # the summary tables come after the last ITER block and carry all
        # iterations side by side
        if line.strip().startswith(ALL_CORRECT_ROW):
            for it, val in zip(sorted(per_iter), PCT_RE.findall(line)):
                per_iter[it]["all-correct%"] = float(val)
            continue
    return per_iter, skills


def load_run(input_dir):
    """Return (runs, skill_runs): seed -> {iter: {metric|skill: value}}."""
    root = Path(input_dir)
    seed_dirs = sorted(
        root.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])
    )
    if not seed_dirs:
        raise SystemExit(f"No seed_* directories under {root}")

    runs = {}
    skill_runs = {}
    for seed_dir in seed_dirs:
        log = seed_dir / "log.txt"
        if not log.exists():
            print(f"[skip] missing {log}")
            continue
        per_iter, skills = parse_log(log)
        if not per_iter:
            print(f"[skip] no scores parsed from {log}")
            continue
        runs[seed_dir.name] = per_iter
        skill_runs[seed_dir.name] = skills
    return runs, skill_runs


def print_table(title, columns, rows):
    print(title)
    header = f"{'group':<20}" + "".join(f"{c:>22}" for c in columns)
    print(header)
    print("-" * len(header))
    for name, vals in rows:
        cells = ""
        for c in columns:
            mean, sem = mean_sem(vals[c])
            cells += f"{mean:>15.4f} +- {sem:<4.4f}"
        print(f"{name:<20}{cells}")
    print()


def build_rows(runs, iters, columns):
    """Rows of (label, {column: [per-seed values]}) incl. paired diffs vs iter-1."""
    rows = []
    for it in iters:
        rows.append(
            (
                f"iter-{it}",
                {c: [r[it][c] for r in runs.values() if it in r] for c in columns},
            )
        )
    a = iters[0]
    for b in iters[1:]:
        rows.append(
            (
                f"iter-{b} - iter-{a}",
                {
                    c: [
                        r[b][c] - r[a][c]
                        for r in runs.values()
                        if a in r and b in r
                    ]
                    for c in columns
                },
            )
        )
    return rows


def report(label, input_dir, runs, skill_runs):
    iters = sorted({it for r in runs.values() for it in r})
    metrics = [m for m in METRICS if all(m in r[it] for r in runs.values() for it in r)]
    skill_names = sorted(
        {s for r in skill_runs.values() for d in r.values() for s in d}
    )

    print(f"run      : {label}")
    print(f"input_dir: {input_dir}")
    print(f"seeds    : {', '.join(runs)} (n={len(runs)})\n")

    print("Per-seed scores")
    header = f"{'seed':<10}{'iter':>6}" + "".join(f"{m:>14}" for m in metrics)
    print(header)
    print("-" * len(header))
    for seed, per_iter in runs.items():
        for it in sorted(per_iter):
            row = "".join(f"{per_iter[it][m]:>14.4f}" for m in metrics)
            print(f"{seed:<10}{it:>6}{row}")
    print()

    print_table(
        "Overall (mean +- standard error across seeds)",
        metrics,
        build_rows(runs, iters, metrics),
    )

    if skill_names:
        usable = complete_seeds(skill_runs, iters, skill_names)
        if usable:
            print_table(
                "Per-skill question scores (mean +- standard error across seeds)",
                skill_names,
                build_rows(usable, iters, skill_names),
            )


def make_figures(loaded):
    out = Path(out_dir)
    overall = {}
    skills = {}
    for label, (runs, skill_runs) in loaded.items():
        iters = sorted({it for r in runs.values() for it in r})
        overall[label] = (complete_seeds(runs, iters, METRICS, label), iters)
        names = sorted({s for r in skill_runs.values() for d in r.values() for s in d})
        skills[label] = (complete_seeds(skill_runs, iters, names), iters, names)

    metrics = [m for m in METRICS if all(seeds for seeds, _ in overall.values())]
    plot_panels(
        overall, metrics, METRIC_LABELS, out / "geneval2_overall",
        TEXT_WIDTH, 1.85,
    )
    for m in metrics:
        slug = m.replace("%", "").replace("-", "_").lower()
        plot_panels(
            overall, [m], METRIC_LABELS, out / f"geneval2_{slug}",
            COLUMN_WIDTH, 2.0,
        )

    if all(seeds for seeds, _, _ in skills.values()):
        names = sorted(set.intersection(*(set(n) for _, _, n in skills.values())))
        plot_panels(
            {label: (seeds, iters) for label, (seeds, iters, _) in skills.items()},
            names,
            {s: s.replace("_", " ").capitalize() for s in names},
            out / "geneval2_skills",
            TEXT_WIDTH,
            1.75,
            titles=True,
            ylabel="AM score",
            compact=True,
        )


def parse_args():
    """Runs and output folder; without --run, the RUNS dict above is used."""
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run", action="append", metavar="LABEL=DIR",
        help="A results dir holding seed_*/ (repeat to draw several runs on the "
             "same axes). Default: the RUNS dict in this file.")
    parser.add_argument("--out_dir", default=out_dir,
                        help="Where the figures go (default: %(default)s).")
    args = parser.parse_args()
    if args.run:
        bad = [r for r in args.run if "=" not in r]
        if bad:
            parser.error(f"--run expects LABEL=DIR, got {bad}")
        args.runs = dict(r.split("=", 1) for r in args.run)
    else:
        args.runs = RUNS
    return args


def main():
    global out_dir
    args = parse_args()
    out_dir = args.out_dir
    loaded = {}
    for label, input_dir in args.runs.items():
        if not input_dir:  # placeholder entry in RUNS
            print(f"[skip] {label}: no results dir")
            continue
        runs, skill_runs = load_run(input_dir)
        report(label, input_dir, runs, skill_runs)
        loaded[label] = (runs, skill_runs)
    make_figures(loaded)


if __name__ == "__main__":
    main()
