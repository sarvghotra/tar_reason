"""Export the system-level baseline results (paper Table 2a) for the paper repo.

Writes ``per_seed_scores.csv`` and ``summary.csv`` in the format of the paper repo's
``results/TTS_perf_n_compute_comparison/`` (minus the FLOPs columns) to
``LatentDCR_cvpr_paper/results/system_level_baselines/``:

  per_seed_scores.csv: method,benchmark,n,metric,seed,value,judge,source_dir
  summary.csv:         method,benchmark,n,metric,mean,se,n_seeds,judge   (se = std(ddof=1)/sqrt(n_seeds))

Every row is single pass, so ``n`` is 1 (image 1 of the n2 folders; for these models image 2 = image 1).

GenEval2 rows are computed here from the Qwen3.5-27B re-scoring mirrors written by
scripts/eval/geneval2_rescore_mila.sh (``<run>_judge-Qwen3.5-27B/seed_*/1/results_geneval2.jsonl``: one list of
soft-TIFA question probabilities per prompt): AM and GM as in evaluation_optz.py (mean over prompts of the
per-prompt mean / geometric mean, checked against the judge's ``eval_log.txt``) and the all-questions-correct
rate (every probability >= 0.5, as viz_geneval2_iter_results.py), in percent. Only iteration 1 is scored, so
there is no ``log.txt`` summary to read. TIIF (Qwen2.5-VL-72B) and GenAI-Bench (CLIP-FlanT5-XXL)
rows were judged on Nibi and are exported by Fir in the same per-seed format; pass those CSVs with
``--extra_per_seed`` and they are merged. A (method, benchmark, metric) cell with no seeds is written to
``summary.csv`` as ``nan`` with ``n_seeds`` 0.

Usage (any env with scipy, from eval/stats):
  python export_system_level.py --out_dir <paper repo>/results/system_level_baselines [--extra_per_seed fir.csv]
"""

import argparse
import csv
import math
from pathlib import Path

import json
import re

from iter_plots import mean_sem
from scipy.stats import gmean

GENEVAL2_ROOT = "/network/scratch/s/singhsd/tar_reason/results_jeet_rescore/GenEval2"
GENEVAL2_JUDGE = "Qwen3.5-27B (soft-TIFA)"
TAG = "Qwen3.5-27B"

# method name in the paper -> run folder (same name under every benchmark of results_jeet/)
METHODS = {
    "FLUX.1-dev": "diffusers_n2_flux1_dev_1024px_repeat1",
    "SD3.5-Large": "diffusers_n2_sd35_large_1024px_repeat1",
    "Qwen-Image": "diffusers_n2_qwen_image_1024px_repeat1",
    "Janus-Pro-7B": "janus_single_n2_janus_pro_7b_384px_repeat1",
    "Show-o2-7B": "showo2_n2_showo2_7b_432px_repeat1",
    "BAGEL-14B": "bagel_n2_bagel_7b_1024px_repeat1",
    "BLIP3-o-8B": "blip3o_n2_blip3o_8b_1024px_repeat1",
}

# the cells summary.csv must have, filled with nan when nothing was exported for them
EXPECTED = {
    "GenEval2": (["AM", "GM", "all_correct_pct"], GENEVAL2_JUDGE),
    "TIIF-testmini": (["overall_short", "overall_long"], "Qwen2.5-VL-72B-Instruct"),
    "GenAI-Bench-1600": (["basic", "advanced", "all"], "clip-flant5-xxl"),
    "GenAI-Bench-800": (["basic", "advanced", "all"], "clip-flant5-xxl"),
}

PER_SEED_COLUMNS = ["method", "benchmark", "n", "metric", "seed", "value", "judge", "source_dir"]
SUMMARY_COLUMNS = ["method", "benchmark", "n", "metric", "mean", "se", "n_seeds", "judge"]
CORRECT_THRESHOLD = 0.5
LOG_RE = re.compile(r"^(AM|GM) Score:\s*([\d.]+)")


def geneval2_seed(iter_dir):
    """{metric: value} for one seed's iteration-1 scores, AM / GM checked against eval_log.txt."""
    scores = json.load(open(iter_dir / "results_geneval2.jsonl"))
    if len(scores) != 800 or any(p is None for s in scores for p in s):
        raise SystemExit(f"{iter_dir}: {len(scores)} prompts or missing question scores")
    vals = {"AM": 100 * sum(sum(s) / len(s) for s in scores) / len(scores),
            "GM": 100 * sum(gmean(s) for s in scores) / len(scores),
            "all_correct_pct": 100 * sum(all(p >= CORRECT_THRESHOLD for p in s) for s in scores) / len(scores)}
    for line in open(iter_dir / "eval_log.txt"):
        m = LOG_RE.match(line.strip())
        if m and abs(float(m.group(2)) - vals[m.group(1)]) > 1e-6:
            raise SystemExit(f"{iter_dir}: {m.group(1)} {vals[m.group(1)]} != eval_log.txt {m.group(2)}")
    return vals


def geneval2_rows():
    rows = []
    for method, run in METHODS.items():
        d = Path(GENEVAL2_ROOT) / f"{run}_judge-{TAG}"
        seed_dirs = sorted(d.glob("seed_*/1/results_geneval2.jsonl"))
        if not seed_dirs:
            print(f"[skip] GenEval2 {method}: no seed_*/1/results_geneval2.jsonl under {d}")
            continue
        seeds = []
        for f in seed_dirs:
            seed = int(f.parent.parent.name.split("_")[1])
            seeds.append(seed)
            for metric, v in geneval2_seed(f.parent).items():
                rows.append({"method": method, "benchmark": "GenEval2", "n": 1, "metric": metric, "seed": seed,
                             "value": f"{v:.4f}", "judge": GENEVAL2_JUDGE, "source_dir": str(d)})
        print(f"GenEval2 {method}: seeds {sorted(seeds)}")
    return rows


def summarize(per_seed):
    groups = {}
    for r in per_seed:
        groups.setdefault((r["method"], r["benchmark"], int(r["n"]), r["metric"], r["judge"]), []).append(
            float(r["value"]))
    for method in METHODS:
        for bench, (metrics, judge) in EXPECTED.items():
            for m in metrics:
                if not any(k[:4] == (method, bench, 1, m) for k in groups):
                    groups[(method, bench, 1, m, judge)] = []
    out = []
    for (method, bench, n, metric, judge), vals in groups.items():
        if vals:
            mean, se = mean_sem(vals)
        else:
            mean, se = math.nan, math.nan
        out.append({"method": method, "benchmark": bench, "n": n, "metric": metric,
                    "mean": "nan" if mean != mean else f"{mean:.4f}",
                    "se": "nan" if se != se else f"{se:.4f}", "n_seeds": len(vals), "judge": judge})
    order = {m: i for i, m in enumerate(METHODS)}
    bench_order = {b: i for i, b in enumerate(EXPECTED)}
    out.sort(key=lambda r: (order.get(r["method"], 99), bench_order.get(r["benchmark"], 99), r["benchmark"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--extra_per_seed", nargs="*", default=[],
                    help="per-seed CSVs in the same columns (Fir's TIIF / GenAI rows), merged as they are")
    args = ap.parse_args()
    per_seed = geneval2_rows()
    for f in args.extra_per_seed:
        with open(f) as fh:
            per_seed += list(csv.DictReader(fh))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_seed_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_SEED_COLUMNS)
        w.writeheader()
        w.writerows(per_seed)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(summarize(per_seed))
    print(f"wrote {len(per_seed)} per-seed rows to {out}")


if __name__ == "__main__":
    main()
