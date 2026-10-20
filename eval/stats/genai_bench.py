"""Aggregate GenAI-Bench overall (basic / advanced / all) numbers across seeds
and plot them.

Parses the per-seed judge logs, which hold one "Tag Group: overall" block per
iteration, and reports mean +- standard error over seeds for every iteration
and the per-seed paired differences against iter-1.

It also writes paper figures (score vs. iteration, mean +- s.e. over seeds) to
``out_dir`` as PDF (vector, for LaTeX) and PNG (preview):

* ``genai_overall``  -- basic / advanced / all side by side (\\textwidth)
* ``genai_all``      -- the "all" score alone (\\columnwidth)

The judge log prints scores with two decimals only, so per-seed values (and
the plotted gains) move in steps of 0.01 (1 point on the x100 axis).

Run with the ``tar`` env (the only one with matplotlib).
"""

import re
from pathlib import Path

from iter_plots import COLUMN_WIDTH, TEXT_WIDTH, complete_seeds, mean_sem, plot_panels

# label -> results dir holding seed_*/<score_model>_log.txt. Several entries
# are drawn as separate lines on the same axes (e.g. to compare methods).
RUNS = {
    "RL (MDP)": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/genai-bench-800/iter_adhoc_MDP_n4_slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1_rl_ft_oracle_t8_500",
    "RL (w/o MDP)": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/genai-bench-800/iter_adhoc_slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1_rl_ft_oracle_t8_500",
    "SFT": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/genai-bench-800/iter_adhoc_slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1",
    "RL-PM-t1 (MDP) OOD": "",
    "RL-PM-t1 (w/o MDP) OOD": "",
    "T2I OOD": ""
}
out_dir = "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/genai-bench-800/figures"
score_model = "qwen3.5-27b"
# scores are in [0, 1]; the figures show them x100 so the gain labels keep
# their resolution (the per-seed values are logged with two decimals)
PLOT_SCALE = 100

METRICS = ["basic", "advanced", "all"]
METRIC_LABELS = {"basic": "Basic", "advanced": "Advanced", "all": "All"}
# The model column runs into the first value ("<path>/10.43 +- 0.30"), so anchor
# on the "+-" and keep only the trailing "<digit>.<digits>" before it.
VALUE_RE = re.compile(r"(\d\.\d+)\s*\+-\s*\d\.\d+")


def parse_log(path):
    """Return {iter_index: {metric: value}} for the overall tag group."""
    lines = path.read_text(errors="replace").splitlines()
    per_iter = {}
    cur_iter = None
    for i, line in enumerate(lines):
        m = re.match(r"\s*ITER-(\d+)\s*:", line)
        if m:
            cur_iter = int(m.group(1))
            continue
        if line.startswith("Tag Group: overall") and cur_iter is not None:
            # next line is the header, the one after holds the values
            values = VALUE_RE.findall(lines[i + 2])
            if len(values) != len(METRICS):
                raise ValueError(
                    f"{path}: expected {len(METRICS)} values, got {values}"
                )
            per_iter[cur_iter] = dict(zip(METRICS, map(float, values)))
    return per_iter


def load_run(input_dir):
    """Return seed -> {iter: {metric: value}}."""
    root = Path(input_dir)
    seed_dirs = sorted(
        root.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])
    )
    if not seed_dirs:
        raise SystemExit(f"No seed_* directories under {root}")

    runs = {}
    for seed_dir in seed_dirs:
        log = seed_dir / f"{score_model}_log.txt"
        if not log.exists():
            print(f"[skip] missing {log}")
            continue
        per_iter = parse_log(log)
        if not per_iter:
            print(f"[skip] no scores parsed from {log}")
            continue
        runs[seed_dir.name] = per_iter
    return runs


def report(label, input_dir, runs):
    iters = sorted({it for r in runs.values() for it in r})

    print(f"run        : {label}")
    print(f"input_dir  : {input_dir}")
    print(f"score_model: {score_model}")
    print(f"seeds      : {', '.join(runs)} (n={len(runs)})\n")

    print("Per-seed overall scores")
    header = f"{'seed':<10}{'iter':>6}" + "".join(f"{m:>12}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for seed, per_iter in runs.items():
        for it in sorted(per_iter):
            row = "".join(f"{per_iter[it][m]:>12.4f}" for m in METRICS)
            print(f"{seed:<10}{it:>6}{row}")
    print()

    rows = []
    for it in iters:
        vals = {m: [r[it][m] for r in runs.values() if it in r] for m in METRICS}
        rows.append((f"iter-{it}", vals))

    a = iters[0]
    for b in iters[1:]:
        paired = {
            m: [r[b][m] - r[a][m] for r in runs.values() if a in r and b in r]
            for m in METRICS
        }
        rows.append((f"iter-{b} - iter-{a}", paired))

    print("Mean +- standard error across seeds")
    header = f"{'group':<20}" + "".join(f"{m:>22}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for name, vals in rows:
        cells = ""
        for m in METRICS:
            mean, sem = mean_sem(vals[m])
            cells += f"{mean:>15.4f} +- {sem:<4.4f}"
        print(f"{name:<20}{cells}")
    print()


def make_figures(loaded):
    data = {}
    for label, runs in loaded.items():
        iters = sorted({it for r in runs.values() for it in r})
        data[label] = (complete_seeds(runs, iters, METRICS, label), iters)
    out = Path(out_dir)

    plot_panels(
        data,
        METRICS,
        METRIC_LABELS,
        out / "genai_overall",
        TEXT_WIDTH,
        1.85,
        titles=True,
        ylabel="GenAI-Bench score (\u00d7100)",
        scale=PLOT_SCALE,
    )
    plot_panels(
        data,
        ["all"],
        {"all": "GenAI-Bench score (\u00d7100)"},
        out / "genai_all",
        COLUMN_WIDTH,
        2.0,
        scale=PLOT_SCALE,
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
    parser.add_argument("--score_model", default=score_model,
                        help="Judge name in seed_*/<score_model>_log.txt "
                             "(default: %(default)s).")
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
    global out_dir, score_model
    args = parse_args()
    out_dir, score_model = args.out_dir, args.score_model
    loaded = {}
    for label, input_dir in args.runs.items():
        if not input_dir:  # placeholder entry in RUNS
            print(f"[skip] {label}: no results dir")
            continue
        runs = load_run(input_dir)
        report(label, input_dir, runs)
        loaded[label] = runs
    make_figures(loaded)


if __name__ == "__main__":
    main()
