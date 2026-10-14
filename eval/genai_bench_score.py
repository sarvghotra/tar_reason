"""VQAScore for GenAI-Bench (image) with t2v_metrics, one generation stage at a time.

Same scoring as t2v_metrics' genai_bench/evaluate.py (score_func.batch_forward on
(image, prompt) pairs with the model's default "Does this figure show ...?" / "Yes"
templates; per-skill means over the prompts tagged in genai_skills.json, "all" = every
tagged prompt), but it also accepts a subset of prompts (--limit, for smoke tests) and
writes the scores and the skill table to JSON instead of only printing them.

Run it in the t2v_metrics env, without tar_reason on PYTHONPATH (t2v_metrics has its own
`llava` package):

    env -u PYTHONPATH ~/envs/genai_judge/bin/python eval/genai_bench_score.py \
        --set_dir $SCRATCH/benchmarks/GenAI-Bench/GenAI-Image-1600 \
        --image_dir <output_dir>/<gen_model>/1 --out_json <...>/scores_qwen3.5-27b.json
"""

import argparse
import json
import os

import t2v_metrics

TAG_GROUPS = {
    "basic": ["attribute", "scene", "spatial relation", "action relation", "part relation", "basic"],
    "advanced": ["counting", "comparison", "differentiation", "negation", "universal", "advanced"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set_dir", required=True, help="folder with genai_image.json and genai_skills.json")
    p.add_argument("--image_dir", required=True, help="folder with <prompt_idx>.jpeg")
    p.add_argument("--out_json", required=True)
    p.add_argument("--model", default="qwen3.5-27b")
    p.add_argument("--cache_dir", default=os.path.join(os.environ.get("HF_HOME", "./hf_cache"), "hub"))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="score only the first N prompts (0 = all)")
    args = p.parse_args()

    prompts = json.load(open(os.path.join(args.set_dir, "genai_image.json")))
    skills = json.load(open(os.path.join(args.set_dir, "genai_skills.json")))
    keys = sorted(prompts)[:args.limit] if args.limit else sorted(prompts)
    items = []
    for key in keys:
        path = os.path.join(args.image_dir, f"{key}.jpeg")
        assert os.path.exists(path), f"missing image {path}"
        items.append({"images": [path], "texts": [prompts[key]["prompt"]]})

    score_func = t2v_metrics.get_score_model(model=args.model, device="cuda", cache_dir=args.cache_dir)
    scores = score_func.batch_forward(items, batch_size=args.batch_size).cpu().mean(dim=1).flatten().tolist()
    by_idx = {int(key): s for key, s in zip(keys, scores)}

    table = {}
    for tag, idxs in skills.items():
        vals = [by_idx[i] for i in idxs if i in by_idx]
        table[tag] = sum(vals) / len(vals) if vals else None
    tagged = sorted({i for idxs in skills.values() for i in idxs if i in by_idx})
    table["all"] = sum(by_idx[i] for i in tagged) / len(tagged) if tagged else None

    out = {"model": args.model, "n_prompts": len(keys), "skills": table,
           "scores": {key: s for key, s in zip(keys, scores)}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=1)
    for group, tags in TAG_GROUPS.items():
        print(group, "  ".join(f"{t}={table[t]:.4f}" for t in tags if table.get(t) is not None))
    print(f"all={table['all']:.4f} over {len(tagged)} prompts")


if __name__ == "__main__":
    main()
