"""Aggregate GenAI-Bench overall (basic / advanced / all) numbers across seeds.

Parses the per-seed judge logs, which hold one "Tag Group: overall" block per
iteration, and reports mean +- standard error over seeds for iter-1, iter-2 and
the per-seed paired difference (iter-2 - iter-1).
"""

import math
import re
from pathlib import Path

input_dir = "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/genai-bench-800/iter_adhoc_slf_ref_edit_t20_16K_greedy_slf_ref_draft_S0_512px_rl_ft_oracle_t2_ckpt_500"
score_model = "qwen3.5-27b"

METRICS = ["basic", "advanced", "all"]
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


def mean_sem(xs):
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return mean, float("nan")
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var / n)


def main():
    root = Path(input_dir)
    seed_dirs = sorted(
        root.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])
    )
    if not seed_dirs:
        raise SystemExit(f"No seed_* directories under {root}")

    runs = {}  # seed -> {iter: {metric: value}}
    for seed_dir in seed_dirs:
        log = seed_dir / f"{score_model}_log.txt"
        if not log.exists():
            print(f"[skip] missing {log}")
            continue
        runs[seed_dir.name] = parse_log(log)

    iters = sorted({it for r in runs.values() for it in r})

    print(f"input_dir : {input_dir}")
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

    if len(iters) >= 2:
        a, b = iters[0], iters[1]
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


if __name__ == "__main__":
    main()
