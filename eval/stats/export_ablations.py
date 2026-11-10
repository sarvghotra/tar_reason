"""Export the E-450 ablation results (official benchmarks) as per-experiment CSVs.

Writes one tidy CSV per experiment for the paper repo (LatentDCR_cvpr_paper/ablations/): one row per
checkpoint x benchmark x metric x iteration, with the mean and standard error over seeds, the paired
change vs the draft (iter 1), and every seed's value. Scores are in percent.

Inputs are the folders written by scripts/eval/bench_official_tamia.sh (+ official_judges_tamia.sh), laid
out as under results_jeet/: <ROOT>/<bench folder>/iter_adhoc_MDP_n5_<label>_512px_repeat1/seed_<S>/...
  GenEval2 : seed_<S>/log.txt                                    (Qwen3-VL-8B soft-TIFA; AM, GM, all-correct)
  TIIF     : seed_<S>/eval_results_qwen72b/result_summary_dimension.txt (+ .xlsx)  (official Qwen2.5-VL-72B)
  GenAI    : seed_<S>/<iter>/clip-flant5-xxl_1600_prompts.pt     (official CLIP-FlanT5-XXL VQAScore, exact mean)

Usage (tar env, from eval/stats):  python export_ablations.py --root <ROOT> --out_dir <paper repo>/ablations
"""

import argparse
import csv
from pathlib import Path

import geneval2
import tiff_bench
from iter_plots import mean_sem

FOLDER = "iter_adhoc_MDP_n5_{label}_512px_repeat1"
BENCH_DIRS = {"GenEval2": "GenEval2", "TIIF": "tiif-bench-testmini_eval", "GenAI": "genai-bench-1600"}
JUDGES = {"GenEval2": "Qwen3-VL-8B (soft-TIFA)", "TIIF": "Qwen2.5-VL-72B", "GenAI": "CLIP-FlanT5-XXL (VQAScore)"}

# file name -> [(checkpoint step, eval label, benchmarks)]
EXPERIMENTS = {
    "abl1_policy": [(200, "darshan_pm1_E_policy_200", ["GenEval2", "TIIF"]),
                    (450, "darshan_pm1_E_policy_450", ["GenEval2", "TIIF"])],
    "abl2_mixed": [(450, "darshan_pm1_E_mixed_450", ["GenEval2", "TIIF"])],
    "gemma_reward": [(450, "darshan_pm1_E_gemma_450", ["GenEval2", "TIIF", "GenAI"])],
}

COLUMNS = ["experiment", "checkpoint_step", "benchmark", "judge", "metric", "iter", "images",
           "mean", "se", "delta_vs_draft", "delta_se", "n_seeds", "seeds", "per_seed"]


def load_geneval2(d):
    runs, _ = geneval2.load_run(d)
    keys = {"AM": "AM", "GM": "GM", "all-correct%": "all_correct_pct"}
    return {s: {it: {keys[k]: v for k, v in m.items() if k in keys} for it, m in per.items()}
            for s, per in runs.items()}


def load_tiif(d):
    tiff_bench.results_subdir = "eval_results_qwen72b"
    runs = tiff_bench.load_run(d)
    return {s: {it: {f"{m}_{p}": 100 * v for (m, p), v in vals.items()} for it, vals in per.items()}
            for s, per in runs.items()}


def load_genai(d):
    import torch
    runs = {}
    for seed_dir in sorted(Path(d).glob("seed_*")):
        per = {}
        for f in sorted(seed_dir.glob("*/clip-flant5-xxl_1600_prompts.pt")):
            per[int(f.parent.name)] = {"all": 100 * torch.load(f).float().mean().item()}
        if per:
            runs[seed_dir.name] = per
    return runs


LOADERS = {"GenEval2": load_geneval2, "TIIF": load_tiif, "GenAI": load_genai}


def rows_for(exp, step, label, bench, root):
    d = Path(root) / BENCH_DIRS[bench] / FOLDER.format(label=label)
    runs = LOADERS[bench](d)
    if not runs:
        print(f"[skip] {exp} {step} {bench}: no seeds under {d}")
        return []
    seeds = list(runs)
    iters = sorted(set.intersection(*(set(r) for r in runs.values())))
    metrics = list(runs[seeds[0]][iters[0]])
    out = []
    for m in metrics:
        for it in iters:
            vals = [runs[s][it][m] for s in seeds]
            mean, se = mean_sem(vals)
            deltas = [runs[s][it][m] - runs[s][iters[0]][m] for s in seeds]
            dmean, dse = mean_sem(deltas)
            out.append({
                "experiment": exp, "checkpoint_step": step, "benchmark": bench, "judge": JUDGES[bench],
                "metric": m, "iter": it, "images": it,
                "mean": f"{mean:.2f}", "se": "" if se != se else f"{se:.2f}",
                "delta_vs_draft": f"{round(dmean, 2) + 0.0:+.2f}", "delta_se": "" if dse != dse else f"{dse:.2f}",
                "n_seeds": len(seeds), "seeds": "/".join(s.split("_")[1] for s in seeds),
                "per_seed": "/".join(f"{v:.2f}" for v in vals),
            })
    print(f"{exp} step {step} {bench}: seeds {[s.split('_')[1] for s in seeds]}, iters {iters}, metrics {metrics}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder laid out like results_jeet/")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for exp, entries in EXPERIMENTS.items():
        rows = [r for step, label, benches in entries for b in benches
                for r in rows_for(exp, step, label, b, args.root)]
        with open(Path(args.out_dir) / f"{exp}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
