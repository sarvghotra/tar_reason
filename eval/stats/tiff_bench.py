"""Aggregate TIIF-Bench overall numbers across seeds.

Reads each seed's ``eval_results/result_summary_dimension.txt`` and reports the
per-iteration following-overall scores (basic / advanced / real-world) plus the
overall average, as mean +- standard error over seeds, together with the
per-seed paired difference (iter-2 - iter-1). Short and long prompts are
reported in separate tables.
"""

import math
import re
from pathlib import Path

input_dir = "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/tiif-bench-testmini_eval/iter_adhoc_slf_ref_edit_t20_16K_greedy_slf_ref_draft_S0_512px_repeat1_RL_rl_ft_oracle_t2_ckpt_500"

# section title in result_summary_dimension.txt -> short column name
SECTIONS = {
    "Basic_Following_Overall - Overall": "basic",
    "Advanced_Following_Overall - Overall": "advanced",
    "Real_World_Following_Overall - Overall": "real-world",
    "Overall average score of 9 sub-attributes": "overall",
}
METRICS = ["basic", "advanced", "real-world", "overall"]
PROMPTS = ["short", "long"]

SECTION_RE = re.compile(r"^====\s*(.+?)\s*====\s*$")
# e.g. "7B_iter1  0.7601  0.7546" -- the two values are short then long
ROW_RE = re.compile(r"^(\S+)_iter(\d+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse_summary(path):
    """Return {iter_index: {(metric, prompt): value}}."""
    per_iter = {}
    section = None
    for line in path.read_text(errors="replace").splitlines():
        m = SECTION_RE.match(line)
        if m:
            section = SECTIONS.get(m.group(1))
            continue
        if section is None:
            continue
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        it = int(m.group(2))
        vals = per_iter.setdefault(it, {})
        vals[(section, "short")] = float(m.group(3))
        vals[(section, "long")] = float(m.group(4))
    return per_iter


def mean_sem(xs):
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return mean, float("nan")
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var / n)


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


def main():
    root = Path(input_dir)
    seed_dirs = sorted(
        root.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])
    )
    if not seed_dirs:
        raise SystemExit(f"No seed_* directories under {root}")

    runs = {}  # seed -> {iter: {(metric, prompt): value}}
    for seed_dir in seed_dirs:
        summary = seed_dir / "eval_results" / "result_summary_dimension.txt"
        if not summary.exists():
            print(f"[skip] missing {summary}")
            continue
        per_iter = parse_summary(summary)
        missing = [
            k
            for it in per_iter
            for k in ((m, p) for m in METRICS for p in PROMPTS)
            if k not in per_iter[it]
        ]
        if missing:
            raise ValueError(f"{summary}: missing entries {sorted(set(missing))}")
        runs[seed_dir.name] = per_iter

    iters = sorted({it for r in runs.values() for it in r})

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

    if len(iters) >= 2:
        a, b = iters[0], iters[1]
        paired = {
            k: [r[b][k] - r[a][k] for r in runs.values() if a in r and b in r]
            for k in keys
        }
        rows.append((f"iter-{b} - iter-{a}", paired))

    for prompt in PROMPTS:
        print_table(prompt, rows)

    print_overall_table(rows)


if __name__ == "__main__":
    main()
