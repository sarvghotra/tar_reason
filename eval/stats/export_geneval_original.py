"""Export the original GenEval (Ghosh et al. 2023) results of iterative runs for the paper repo.

Reads ``<run>/seed_S/<k>/geneval_summary.txt`` (scripts/eval/geneval_score_mila.sh = GenEval's own
evaluation/evaluate_images.py + summary_scores.py) and writes per_seed_scores.csv, summary.csv and per_round_deltas.csv
(paired by seed, n{k} - n{k-1}). Scores in percent; ``overall`` = GenEval's mean over its 6 tasks.

  python export_geneval_original.py --out_dir <paper repo>/results/geneval_original
"""

import argparse
import csv
import math
import re
from pathlib import Path

ROOT = Path("/network/scratch/s/singhsd/tar_reason/results_t4/GenEval")
RUNS = {"CLAIRE-Tar-t4": "iter_adhoc_MDP_n4_E-450_ORC_512px_repeat4_Cric-SFT-t4",
        "CLAIRE-Tar-t4 + stage-3 RL (run A step 500)": "iter_adhoc_MDP_n4_t4_e2eA500_512px_repeat4"}
SEEDS = [13, 17, 91]
METRICS = ["overall", "single_object", "two_object", "counting", "colors", "position", "color_attr"]
JUDGE = "GenEval official (Mask2Former Swin-S + CLIP ViT-L-14)"


def parse(path):
    out = {}
    for line in open(path):
        m = re.match(r"^(\w+)\s+=\s+([\d.]+)%", line)
        if m:
            out[m.group(1)] = float(m.group(2))
        m = re.match(r"^Overall score.*:\s*([\d.]+)", line)
        if m:
            out["overall"] = 100 * float(m.group(1))
    return out


def mean_se(xs):
    m = sum(xs) / len(xs)
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs)) if len(xs) > 1 else math.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    per_seed, vals = [], {}
    for method, run in RUNS.items():
        for seed in SEEDS:
            for n in range(1, 5):
                f = ROOT / run / f"seed_{seed}" / str(n) / "geneval_summary.txt"
                v = parse(f)
                for metric in METRICS:
                    per_seed.append({"method": method, "benchmark": "GenEval", "n": n, "metric": metric, "seed": seed,
                                     "value": f"{v[metric]:.4f}", "judge": JUDGE, "source_dir": str(f.parent)})
                    vals.setdefault((method, metric), {}).setdefault(n, {})[seed] = v[metric]
    summary, deltas = [], []
    for (method, metric), by_n in vals.items():
        for n, by_seed in sorted(by_n.items()):
            m, se = mean_se(list(by_seed.values()))
            summary.append({"method": method, "benchmark": "GenEval", "n": n, "metric": metric, "mean": f"{m:.4f}",
                            "se": f"{se:.4f}", "n_seeds": len(by_seed), "judge": JUDGE})
            if n > 1:
                m, se = mean_se([by_seed[s] - by_n[n - 1][s] for s in SEEDS])
                deltas.append({"method": method, "benchmark": "GenEval", "metric": metric, "judge": JUDGE,
                               "comparison": f"n{n} - n{n - 1}", "mean": f"{m:+.4f}", "se": f"{se:.4f}",
                               "n_seeds": len(SEEDS)})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("per_seed_scores.csv", per_seed), ("summary.csv", summary), ("per_round_deltas.csv", deltas)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"wrote {len(per_seed)} per-seed rows to {out}")


if __name__ == "__main__":
    main()
