"""Export the paper's Table 2a baseline rows (system level, single pass) with the official judges (2026-11-12).

Rows for results/system_level_baselines/ in the paper repo, in the per-seed format of
results/TTS_perf_n_compute_comparison/per_seed_scores.csv:
    method,benchmark,n,metric,seed,value,judge,source_dir
n = 1 (image 1: these models are single pass; their n2 folders repeat or regenerate image 2, not used).
  TIIF-testmini   : overall_short, overall_long (9-sub-attribute average, 0-100), Qwen2.5-VL-72B,
                    from seed_<S>/eval_results_qwen72b/result_summary_dimension.txt (tiff_bench.load_run).
  GenAI-Bench-1600: basic, advanced, all (VQAScore, 0-1), CLIP-FlanT5-XXL, from seed_<S>/1/clip-flant5-xxl_1600_prompts.pt
                    (rows in the key order of GenAI-Image-1600/genai_image.json; basic / advanced = mean over the
                    row indices in genai_skills.json). Each seed is checked against the judge's own printed
                    basic / advanced / all (clip-flant5-xxl_log.txt, 2 decimals), which also confirms the row order.
  GenAI-Bench-800 : the same tensors restricted to the 800 ids of GenAI-Image-800/genai_image.json (an exact subset
                    of the 1600 file, same prompts), basic / advanced from the 800 genai_skills.json. Only with --genai800_dir.
GenEval2 is not exported here (Mila re-scores it with Qwen3.5-27B).

Usage (tar env, from eval/stats):
    python export_system_baselines.py --root /scratch/dars11/share/agrpo_tar_eval/results \
        --genai1600_dir /scratch/dars11/benchmarks/GenAI-Bench/GenAI-Image-1600 [--genai800_dir .../GenAI-Image-800] \
        --out per_seed_scores_fir.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path

import tiff_bench

# method name (as Mila's exporter uses them) -> run folder
RUNS = {
    "FLUX.1-dev": "diffusers_n2_flux1_dev_1024px_repeat1",
    "SD3.5-Large": "diffusers_n2_sd35_large_1024px_repeat1",
    "Qwen-Image": "diffusers_n2_qwen_image_1024px_repeat1",
    "Janus-Pro-7B": "janus_single_n2_janus_pro_7b_384px_repeat1",
    "Show-o2-7B": "showo2_n2_showo2_7b_432px_repeat1",
    "BAGEL-14B": "bagel_n2_bagel_7b_1024px_repeat1",
    "BLIP3-o-8B": "blip3o_n2_blip3o_8b_1024px_repeat1",
}
TIIF_JUDGE = "Qwen2.5-VL-72B-Instruct"
GENAI_JUDGE = "clip-flant5-xxl"
COLUMNS = ["method", "benchmark", "n", "metric", "seed", "value", "judge", "source_dir"]


def tiif_rows(method, run_dir):
    tiff_bench.results_subdir = "eval_results_qwen72b"
    if not run_dir.exists():
        print(f"[missing] TIIF {method}: {run_dir}")
        return []
    out = []
    for seed, per_iter in sorted(tiff_bench.load_run(run_dir).items(), key=lambda kv: int(str(kv[0]).split("_")[-1])):
        if 1 not in per_iter:
            print(f"[skip] TIIF {method} {seed}: no iteration 1")
            continue
        s = int(str(seed).split("_")[-1])
        for p in ("short", "long"):
            out.append([method, "TIIF-testmini", 1, f"overall_{p}", s, round(100 * per_iter[1][("overall", p)], 4),
                        TIIF_JUDGE, str(run_dir / f"seed_{s}" / "eval_results_qwen72b")])
    return out


def log_iter1(log):
    """basic / advanced / all of ITER-1 as printed by the judge (2 decimals)."""
    text = log.read_text(errors="ignore")
    m = re.search(r"overall \(metric performance\)\s*\n.*?\n[^\n]*?/1([0-9.]+) \+- [0-9.]+\s+([0-9.]+) \+- [0-9.]+\s+([0-9.]+) \+-", text)
    return tuple(float(x) for x in m.groups()) if m else None


def genai_rows(method, run_dir, keys1600, skills1600, sub800):
    import torch
    if not run_dir.exists():
        print(f"[missing] GenAI {method}: {run_dir}")
        return []
    out = []
    for seed_dir in sorted(run_dir.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
        f = seed_dir / "1" / "clip-flant5-xxl_1600_prompts.pt"
        if not f.exists():
            print(f"[skip] GenAI {method} {seed_dir.name}: no image-1 scores")
            continue
        t = torch.load(f, map_location="cpu").float().reshape(-1)
        assert t.numel() == len(keys1600) == 1600, (f, t.numel())
        s = int(seed_dir.name.split("_")[1])
        vals = {"basic": t[skills1600["basic"]].mean().item(), "advanced": t[skills1600["advanced"]].mean().item(),
                "all": t.mean().item()}
        printed = log_iter1(seed_dir / "clip-flant5-xxl_log.txt")
        assert printed is not None, f"no ITER-1 summary in {seed_dir}/clip-flant5-xxl_log.txt"
        for (k, v), pv in zip(vals.items(), printed):
            # 2-decimal log; 0.006 allows a value just past the rounding boundary (BLIP3-o seed 17 all: 0.7751 vs 0.77)
            assert abs(v - pv) <= 0.006, f"{method} {s} {k}: tensor {v:.4f} vs log {pv}"
        for k, v in vals.items():
            out.append([method, "GenAI-Bench-1600", 1, k, s, round(v, 6), GENAI_JUDGE, str(seed_dir)])
        if sub800:
            # The 800 genai_skills.json lists ids of the 1600 set (= row positions of the 1600 tensor), not 0..799.
            idx, sk = sub800
            v8 = {"basic": t[sk["basic"]].mean().item(), "advanced": t[sk["advanced"]].mean().item(),
                  "all": t[idx].mean().item()}
            for k, v in v8.items():
                out.append([method, "GenAI-Bench-800", 1, k, s, round(v, 6), GENAI_JUDGE, str(seed_dir)])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="holds tiif-bench-testmini_eval/ and genai-bench-1600/")
    p.add_argument("--genai1600_dir", required=True)
    p.add_argument("--genai800_dir", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root = Path(a.root)
    img1600 = json.load(open(Path(a.genai1600_dir) / "genai_image.json"))
    keys1600 = list(img1600)
    skills1600 = json.load(open(Path(a.genai1600_dir) / "genai_skills.json"))
    sub800 = None
    if a.genai800_dir:
        img800 = json.load(open(Path(a.genai800_dir) / "genai_image.json"))
        skills800 = json.load(open(Path(a.genai800_dir) / "genai_skills.json"))
        pos = {k: i for i, k in enumerate(keys1600)}
        k800 = list(img800)
        assert all(k in pos and img800[k]["prompt"] == img1600[k]["prompt"] for k in k800), "800 not a subset of 1600"
        ids800 = {pos[k] for k in k800}
        assert all(i in ids800 for l in skills800.values() for i in l), "800 skill ids outside the 800 set"
        sub800 = ([pos[k] for k in k800], skills800)
    rows = []
    for method, run in RUNS.items():
        rows += tiif_rows(method, root / "tiif-bench-testmini_eval" / run)
        rows += genai_rows(method, root / "genai-bench-1600" / run, keys1600, skills1600, sub800)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
