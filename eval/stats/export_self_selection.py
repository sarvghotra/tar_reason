"""Export self-verified selection (highest self-AM image) for the paper repo.

CLAIRE-Tar = E450_ORC (`iter_adhoc_MDP_n4_E-450_ORC_512px_repeat1_vllm`, the runs behind the paper's CLAIRE-Tar rows).
For budget n the plain chain reports image n; "+ self-selection" reports, per prompt, the image among 1..n with the
highest self-AM: the policy predicts each benchmark question's answer from the prompt text alone, then scores
P(predicted answer | image) per question; self-AM = mean over the questions (no ground-truth answers are read).
The chosen image is scored with the run's existing judge outputs (no re-judging).

Inputs (Mila):
  TIIF      $E/tiif_rollback[_judgeT0]_am_pol/seed_S/eval_results/result_summary_dimension.txt
            (eval/analyze/tiif_rollback.py; rows plain_iter{n}, best_B{n})
  GenEval2  $E/geneval2_rollback_Qwen3.5-27B.json (eval/analyze/geneval2_rollback.py; plain_iter{n}, am_ans:best_B{n})

Writes per_seed_scores.csv, summary.csv (mean, se = std(ddof=1)/sqrt(n_seeds)) and per_round_deltas.csv (paired by
seed: value at n minus value at n-1, and selection minus plain at the same n).

  python export_self_selection.py --out_dir <paper repo>/results/self_verified_selection
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

E = Path("/network/scratch/s/singhsd/tar_reason/results_e450orc")
TIIF_SEEDS, GENEVAL2_SEEDS = [13, 17, 91], [2213, 713, 13]
PLAIN, SELECT = "CLAIRE-Tar", "CLAIRE-Tar + self-selection"
TIIF_JUDGES = {"": "Qwen2.5-VL-72B-Instruct",
               "judgeT0_": "Qwen2.5-VL-72B-Instruct (temperature-0 re-judge)"}
GENEVAL2_JUDGE = "Qwen3.5-27B (soft-TIFA)"
SECTION = "Overall average score of 9 sub-attributes"


def tiif_overall(path):
    out, on = {}, False
    for line in open(path):
        if line.startswith("===="):
            on = SECTION in line
            continue
        m = re.match(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s*$", line) if on else None
        if m:
            out[m.group(1)] = (100 * float(m.group(2)), 100 * float(m.group(3)))
    return out


def rows():
    out = []
    for prefix, judge in TIIF_JUDGES.items():
        d = E / f"tiif_rollback_{prefix}am_pol"
        for seed in TIIF_SEEDS:
            v = tiif_overall(d / f"seed_{seed}" / "eval_results" / "result_summary_dimension.txt")
            for n in range(1, 5):
                for method, key in ((PLAIN, f"plain_iter{n}"), (SELECT, "plain_iter1" if n == 1 else f"best_B{n}")):
                    for j, metric in enumerate(("overall_short", "overall_long")):
                        out.append({"method": method, "benchmark": "TIIF-testmini", "n": n, "metric": metric,
                                    "seed": seed, "value": f"{v[key][j]:.4f}", "judge": judge,
                                    "source_dir": str(d / f"seed_{seed}" / "eval_results")})
    g = E / "geneval2_rollback_Qwen3.5-27B.json"
    res = json.load(open(g))
    for seed in GENEVAL2_SEEDS:
        for n in range(1, 5):
            for method, key in ((PLAIN, f"plain_iter{n}"), (SELECT, "plain_iter1" if n == 1 else f"am_ans:best_B{n}")):
                for metric, src in (("AM", "AM"), ("GM", "GM"), ("all_correct_pct", "all_correct")):
                    out.append({"method": method, "benchmark": "GenEval2", "n": n, "metric": metric, "seed": seed,
                                "value": f"{res[key][str(seed)][src]:.4f}", "judge": GENEVAL2_JUDGE,
                                "source_dir": str(g)})
    return out


def mean_se(xs):
    m = sum(xs) / len(xs)
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs)) if len(xs) > 1 else math.nan
    return m, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_seed = rows()
    vals = {}
    for r in per_seed:
        vals.setdefault((r["method"], r["benchmark"], r["metric"], r["judge"]), {}).setdefault(r["n"], {})[
            r["seed"]] = float(r["value"])
    summary, deltas = [], []
    for (method, bench, metric, judge), by_n in vals.items():
        for n, by_seed in sorted(by_n.items()):
            m, se = mean_se(list(by_seed.values()))
            summary.append({"method": method, "benchmark": bench, "n": n, "metric": metric, "mean": f"{m:.4f}",
                            "se": f"{se:.4f}", "n_seeds": len(by_seed), "judge": judge})
            seeds = sorted(by_seed)
            if n > 1:
                m, se = mean_se([by_seed[s] - by_n[n - 1][s] for s in seeds])
                deltas.append({"method": method, "benchmark": bench, "metric": metric, "judge": judge,
                               "comparison": f"n{n} - n{n - 1}", "mean": f"{m:+.4f}", "se": f"{se:.4f}",
                               "n_seeds": len(seeds)})
            if method == SELECT:
                plain = vals[(PLAIN, bench, metric, judge)][n]
                m, se = mean_se([by_seed[s] - plain[s] for s in seeds])
                deltas.append({"method": method, "benchmark": bench, "metric": metric, "judge": judge,
                               "comparison": f"n{n}: selection - plain", "mean": f"{m:+.4f}", "se": f"{se:.4f}",
                               "n_seeds": len(seeds)})
    for name, data in (("per_seed_scores.csv", per_seed), ("summary.csv", summary), ("per_round_deltas.csv", deltas)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"wrote {len(per_seed)} per-seed rows, {len(summary)} summary rows, {len(deltas)} delta rows to {out}")


if __name__ == "__main__":
    main()
