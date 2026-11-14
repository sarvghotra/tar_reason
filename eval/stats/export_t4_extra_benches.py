"""Export t4 (CLAIRE after critique SFT) on GenEval++ and T2I-CompBench++ for the paper repo.

Writes ``per_seed_scores.csv`` (method,benchmark,n,metric,seed,value,judge,source_dir) and ``summary.csv``
(method,benchmark,n,metric,mean,se,n_seeds,judge; se = std(ddof=1)/sqrt(n_seeds), nan for one seed), the format of
``results/system_level_baselines/``. ``n`` is the image index in the refinement chain (1 = draft).

Inputs (scripts/eval/t4_gen_mila.sh + scoring scripts), under ROOT/<bench>/<RUN>/seed_<S>/<k>/:
  GenEval++        geneval_pp_qwen72b_summary.json  (eval/geneval_pp_judge.py: per-tag accuracy + overall = mean of
                                                     the 7 tags, 0-1; open judge Qwen2.5-VL-72B, not GPT-4.1)
  T2I-CompBench++  t2icb/score_<cat>.json            (official evaluators via t2icompbench_eval_mila.sh, 0-1)

Usage (any python3 with the stdlib, from eval/stats):
  python export_t4_extra_benches.py --out_dir <paper repo>/results/t4_geneval_pp_t2icompbench
"""

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path("/network/scratch/s/singhsd/tar_reason/results_t4")
RUN = "iter_adhoc_MDP_n3_E-450_ORC_512px_repeat1_Cric-SFT-t4"
METHOD = "CLAIRE-Tar-t4"
GENEVAL_PP_JUDGE = "Qwen2.5-VL-72B-Instruct (Echo-4o GenEval++ prompt; official judge is GPT-4.1)"
# Geneval++.jsonl tags (40 prompts each), in Echo-4o Table 3's column order:
# Color, Count, Color/Count, Color/Pos, Pos/Count, Pos/Size, Multi-Count
GENEVAL_PP_TAGS = ["color_attr", "counting", "color_count_attr", "color_spatial_attr", "spatial_count_attr",
                   "size_spatial_attr", "multi_object_count_attr"]
CATS = ["color", "shape", "texture", "spatial", "3d_spatial", "non_spatial", "numeracy", "complex"]
CAT_JUDGE = {"color": "BLIP-VQA", "shape": "BLIP-VQA", "texture": "BLIP-VQA", "spatial": "UniDet",
             "3d_spatial": "UniDet + DPT depth", "numeracy": "UniDet", "non_spatial": "CLIPScore",
             "complex": "3-in-1 (BLIP-VQA, UniDet, CLIPScore)"}
PER_SEED_COLUMNS = ["method", "benchmark", "n", "metric", "seed", "value", "judge", "source_dir"]
SUMMARY_COLUMNS = ["method", "benchmark", "n", "metric", "mean", "se", "n_seeds", "judge"]


def iter_dirs(seed_dir):
    return sorted((d for d in seed_dir.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))


def rows():
    out = []
    for seed_dir in sorted((ROOT / "GenEval++" / RUN).glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])
        for it in iter_dirs(seed_dir):
            f = it / "geneval_pp_qwen72b_summary.json"
            if not f.exists():
                print(f"[missing] {f}")
                continue
            s = json.load(open(f))
            if s["n"] != 280 or s["n_failed"]:
                raise SystemExit(f"{f}: n={s['n']} n_failed={s['n_failed']}")
            unknown = set(s["tag_accuracy"]) - set(GENEVAL_PP_TAGS)
            if unknown:
                raise SystemExit(f"{f}: unexpected tags {unknown}")
            vals = {t: s["tag_accuracy"][t] for t in GENEVAL_PP_TAGS} | {"overall": s["overall"]}
            for m, v in vals.items():
                out.append({"method": METHOD, "benchmark": "GenEval++", "n": int(it.name), "metric": m, "seed": seed,
                            "value": f"{v:.4f}", "judge": GENEVAL_PP_JUDGE, "source_dir": str(it)})
    for cat in CATS:
        for seed_dir in sorted((ROOT / "T2I-CompBench++" / cat / RUN).glob("seed_*")):
            seed = int(seed_dir.name.split("_")[1])
            for it in iter_dirs(seed_dir):
                f = it / "t2icb" / f"score_{cat}.json"
                if not f.exists():
                    print(f"[missing] {f}")
                    continue
                s = json.load(open(f))
                if s["n_images"] != 300:
                    raise SystemExit(f"{f}: {s['n_images']} images")
                out.append({"method": METHOD, "benchmark": "T2I-CompBench++", "n": int(it.name), "metric": cat,
                            "seed": seed, "value": f"{s['score']:.4f}", "judge": CAT_JUDGE[cat],
                            "source_dir": str(it / "t2icb")})
    return out


def summarize(per_seed):
    groups = {}
    for r in per_seed:
        groups.setdefault((r["method"], r["benchmark"], r["n"], r["metric"], r["judge"]), []).append(float(r["value"]))
    out = []
    for (method, bench, n, metric, judge), vals in groups.items():
        mean = sum(vals) / len(vals)
        se = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) / len(vals)) if len(vals) > 1 else math.nan
        out.append({"method": method, "benchmark": bench, "n": n, "metric": metric, "mean": f"{mean:.4f}",
                    "se": "nan" if se != se else f"{se:.4f}", "n_seeds": len(vals), "judge": judge})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    per_seed = rows()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, cols, data in [("per_seed_scores.csv", PER_SEED_COLUMNS, per_seed),
                             ("summary.csv", SUMMARY_COLUMNS, summarize(per_seed))]:
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(data)
    print(f"wrote {len(per_seed)} per-seed rows to {out}")


if __name__ == "__main__":
    main()
