"""Aggregate TIIF-Bench overall numbers across seeds and plot them.

Reads each seed's ``eval_results/result_summary_dimension.txt`` and reports the
per-iteration following-overall scores (basic / advanced / real-world) plus the
overall average, as mean +- standard error over seeds, together with the
per-seed paired differences against iter-1. Short and long prompts are
reported in separate tables.

It also writes paper figures (score vs. iteration, mean +- s.e. over seeds) to
``out_dir`` as PDF (vector, for LaTeX) and PNG (preview):

* ``tiif_overall``      -- 9-sub-attribute average, short / long (\\columnwidth)
* ``tiif_<prompt>``     -- basic / advanced / real-world / overall (\\textwidth)

Run with the ``tar`` env (the only one with matplotlib).
"""

import re
from pathlib import Path

from iter_plots import COLUMN_WIDTH, TEXT_WIDTH, complete_seeds, mean_sem, plot_panels

# label -> results dir holding seed_*/eval_results. Several entries are drawn
# as separate lines on the same axes (e.g. to compare methods).
RUNS = {
    "RL (MDP) OOD": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/tiif-bench-testmini_eval/iter_adhoc_MDP_n4_sslf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1_rl_ft_oracle_t8_500",
    # "RL (w/o MDP) OOD": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/tiif-bench-testmini_eval/iter_adhoc_sslf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1_rl_ft_oracle_t8_500",
    # "SFT OOD": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/tiif-bench-testmini_eval/iter_adhoc_sslf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1",
    "RL-PM-t1 (MDP) OOD": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/tiif-bench-testmini_eval/iter_adhoc_MDP_n4_sslf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1_rl_ft_promptsmix_oracle_t1_500",
    # "RL-PM-t1 (w/o MDP) OOD": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/tiif-bench-testmini_eval/iter_adhoc_n4_sslf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1_rl_ft_promptsmix_oracle_t1_500",
    "T2I OOD": "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/tiif-bench-testmini_eval/iter_adhoc_MDP_n4_sslf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px_repeat1_bl_rl_t2i_t1_200",
}
out_dir = "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/tiif-bench-testmini_eval/figures"
results_subdir = "eval_results"   # per-seed judge folder (eval_results_qwen72b for the official 72B judge)
# the summary holds accuracies in [0, 1]; the figures show them in percent
PLOT_SCALE = 100

# section title in result_summary_dimension.txt -> short column name
SECTIONS = {
    "Basic_Following_Overall - Overall": "basic",
    "Advanced_Following_Overall - Overall": "advanced",
    "Real_World_Following_Overall - Overall": "real-world",
    "Overall average score of 9 sub-attributes": "overall",
}
METRICS = ["basic", "advanced", "real-world", "overall"]
METRIC_LABELS = {
    "basic": "Basic",
    "advanced": "Advanced",
    "real-world": "Real world",
    "overall": "Overall",
}
PROMPTS = ["short", "long"]

SECTION_RE = re.compile(r"^====\s*(.+?)\s*====\s*$")
# a data row: model name then the short and long values. pandas truncates long
# names ("..._S0_i..."), so the iteration comes from result_summary.xlsx.
ROW_RE = re.compile(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s*$")
ITER_NAME_RE = re.compile(r"_iter(\d+)$")
# first-column cells of the xlsx sheet, inline or shared strings
XLSX_CELL_RE = re.compile(r'<c r="A(\d+)"([^>]*)>(.*?)</c>', re.S)


def xlsx_model_iters(path):
    """Iteration of each model row of result_summary.xlsx, in row order.

    The summary txt prints its rows in the xlsx row order, which is not
    sorted by iteration. Reads the sheet XML directly (no openpyxl here).
    """
    import zipfile

    with zipfile.ZipFile(path) as z:
        sheet = z.read("xl/worksheets/sheet1.xml").decode()
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = re.findall(
                r"<si>.*?<t[^>]*>(.*?)</t>.*?</si>",
                z.read("xl/sharedStrings.xml").decode(),
                re.S,
            )
    iters = []
    for _, attrs, inner in sorted(
        XLSX_CELL_RE.findall(sheet), key=lambda c: int(c[0])
    ):
        m = re.search(r"<t[^>]*>(.*?)</t>", inner) or re.search(r"<v>(.*?)</v>", inner)
        if not m:
            continue
        text = shared[int(m.group(1))] if 't="s"' in attrs else m.group(1)
        m = ITER_NAME_RE.search(text.strip())
        if m:
            iters.append(int(m.group(1)))
    return iters


def parse_summary(path, row_iters):
    """Return {iter_index: {(metric, prompt): value}}.

    ``row_iters`` gives the iteration of each data row of a section, in order.
    """
    per_iter = {}
    section = None
    k = 0
    for line in path.read_text(errors="replace").splitlines():
        m = SECTION_RE.match(line)
        if m:
            section = SECTIONS.get(m.group(1))
            k = 0
            continue
        if section is None:
            continue
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        # a full name carries its own iteration; otherwise go by row position
        m_it = ITER_NAME_RE.search(m.group(1))
        if m_it:
            it = int(m_it.group(1))
        else:
            if k >= len(row_iters):
                raise ValueError(f"{path}: more rows than xlsx models {row_iters}")
            it = row_iters[k]
        k += 1
        vals = per_iter.setdefault(it, {})
        vals[(section, "short")] = float(m.group(2))
        vals[(section, "long")] = float(m.group(3))
    return per_iter


def load_run(input_dir):
    """Return seed -> {iter: {(metric, prompt): value}}."""
    root = Path(input_dir)
    seed_dirs = sorted(
        root.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])
    )
    if not seed_dirs:
        raise SystemExit(f"No seed_* directories under {root}")

    runs = {}
    for seed_dir in seed_dirs:
        results = seed_dir / results_subdir
        summary = results / "result_summary_dimension.txt"
        if not summary.exists():
            print(f"[skip] missing {summary}")
            continue
        xlsx = results / "result_summary.xlsx"
        row_iters = xlsx_model_iters(xlsx) if xlsx.exists() else []
        per_iter = parse_summary(summary, row_iters)
        if not per_iter:
            print(f"[skip] no scores parsed from {summary}")
            continue
        missing = [
            k
            for it in per_iter
            for k in ((m, p) for m in METRICS for p in PROMPTS)
            if k not in per_iter[it]
        ]
        if missing:
            raise ValueError(f"{summary}: missing entries {sorted(set(missing))}")
        runs[seed_dir.name] = per_iter
    return runs


def print_overall_table(rows):
    """Compact table: the 9-sub-attribute average for short and long prompts."""
    cols = [("overall", "short"), ("overall", "long")]
    print("Overall average score of 9 sub-attributes "
          "(mean +- standard error across seeds)")
    header = f"{'group':<20}" + "".join(
        f"{'overall-' + p:>22}" for _, p in cols
    )
    print(header)
    print("-" * len(header))
    for name, vals in rows:
        cells = ""
        for key in cols:
            mean, sem = mean_sem(vals[key])
            cells += f"{mean:>15.4f} +- {sem:<4.4f}"
        print(f"{name:<20}{cells}")
    print()


def print_table(prompt, rows):
    print(f"Mean +- standard error across seeds ({prompt} prompts)")
    header = f"{'group':<20}" + "".join(f"{m:>22}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for name, vals in rows:
        cells = ""
        for m in METRICS:
            mean, sem = mean_sem(vals[(m, prompt)])
            cells += f"{mean:>15.4f} +- {sem:<4.4f}"
        print(f"{name:<20}{cells}")
    print()


def report(label, input_dir, runs):
    iters = sorted({it for r in runs.values() for it in r})

    print(f"run      : {label}")
    print(f"input_dir: {input_dir}")
    print(f"seeds    : {', '.join(runs)} (n={len(runs)})\n")

    print("Per-seed scores")
    header = (
        f"{'seed':<10}{'iter':>6}{'prompt':>8}"
        + "".join(f"{m:>12}" for m in METRICS)
    )
    print(header)
    print("-" * len(header))
    for seed, per_iter in runs.items():
        for it in sorted(per_iter):
            for prompt in PROMPTS:
                row = "".join(
                    f"{per_iter[it][(m, prompt)]:>12.4f}" for m in METRICS
                )
                print(f"{seed:<10}{it:>6}{prompt:>8}{row}")
    print()

    keys = [(m, p) for m in METRICS for p in PROMPTS]
    rows = []
    for it in iters:
        vals = {k: [r[it][k] for r in runs.values() if it in r] for k in keys}
        rows.append((f"iter-{it}", vals))

    a = iters[0]
    for b in iters[1:]:
        paired = {
            k: [r[b][k] - r[a][k] for r in runs.values() if a in r and b in r]
            for k in keys
        }
        rows.append((f"iter-{b} - iter-{a}", paired))

    for prompt in PROMPTS:
        print_table(prompt, rows)

    print_overall_table(rows)


def make_figures(loaded):
    keys = [(m, p) for m in METRICS for p in PROMPTS]
    data = {}
    for label, runs in loaded.items():
        iters = sorted({it for r in runs.values() for it in r})
        data[label] = (complete_seeds(runs, iters, keys, label), iters)
    out = Path(out_dir)

    plot_panels(
        data,
        [("overall", p) for p in PROMPTS],
        {("overall", p): f"{p.capitalize()} prompts" for p in PROMPTS},
        out / "tiif_overall",
        COLUMN_WIDTH,
        1.9,
        titles=True,
        ylabel="Overall accuracy (%)",
        compact=True,
        scale=PLOT_SCALE,
    )
    for prompt in PROMPTS:
        plot_panels(
            data,
            [(m, prompt) for m in METRICS],
            {(m, prompt): METRIC_LABELS[m] for m in METRICS},
            out / f"tiif_{prompt}",
            TEXT_WIDTH,
            1.8,
            titles=True,
            ylabel=f"Accuracy, {prompt} prompts (%)",
            compact=True,
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
    parser.add_argument("--results_subdir", default=results_subdir,
                        help="Per-seed folder holding result_summary_dimension.txt (default: %(default)s; "
                             "eval_results_qwen72b = the official Qwen2.5-VL-72B judge, 2026-10-30).")
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
    global out_dir, results_subdir
    args = parse_args()
    out_dir = args.out_dir
    results_subdir = args.results_subdir
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
