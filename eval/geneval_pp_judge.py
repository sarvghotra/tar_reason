"""GenEval++ (Echo-4o, arXiv 2508.09987) scoring with an open judge served by vLLM.

Echo-4o's official scorer is ``test_scripts/Eval-gpt-4.1-geneval++.py`` (yejy53/Echo-4o): GPT-4.1, temperature 0,
max_tokens 200, one call per image with their system prompt and the prompt's checklist built from
``Geneval++.jsonl``; a prompt counts as correct when the judge returns {"correct": 1}; overall = mean of the 7 tag
accuracies. This script keeps all of that (system prompt, checklist text, JPEG base64 image, parsing, 3 retries,
-1 on failure counted as wrong) and only swaps the endpoint: an OpenAI-compatible ``vllm serve`` of an open model
(Qwen2.5-VL-72B-Instruct by default). Scores are therefore NOT comparable to published GenEval++ numbers.

Images are read from the iterative generator's layout, ``<run>/seed_S/<k>/<NNNNN>/samples/00000.png`` (prompt
NNNNN = line NNNNN of Geneval++.txt, 0-based), for every iteration k present. Writes per iteration
``<run>/seed_S/<k>/geneval_pp_<tag>.json`` (per-prompt results) and ``geneval_pp_<tag>_summary.json``.

  python eval/geneval_pp_judge.py --seed_dirs <run>/seed_13 ... --base_url http://localhost:8000/v1 --model <path>
"""

import argparse
import importlib.util
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

# Echo-4o's own SYSTEM_PROMPT, encode_image, extract_json_from_response and metadata_to_explanation, imported from
# their script so the judge prompt is byte-identical (importing it only builds an unused OpenAI client).
ECHO4O_SCORER = "/network/scratch/s/singhsd/benchmarks/Echo-4o/test_scripts/Eval-gpt-4.1-geneval++.py"


def load_echo4o(path=ECHO4O_SCORER):
    spec = importlib.util.spec_from_file_location("echo4o_geneval_pp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E4O = load_echo4o()


def process_sample(client, model, idx, metadata, image_path, max_retries=3):
    instruction = metadata.get("prompt", "").strip()
    explanation = E4O.metadata_to_explanation(metadata)
    base = {"id": idx + 1, "tag": metadata.get("tag", ""), "prompt": instruction}
    if not image_path.exists():
        return {**base, "result": {"correct": 0, "reason": "Image not found"}}
    image_b64 = E4O.encode_image(image_path)
    messages = [
        {"role": "system", "content": E4O.SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"Instruction:\n{instruction}\n\nExpectation checklist:\n{explanation}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]},
    ]
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(model=model, messages=messages, max_tokens=200,
                                                      temperature=0.0)
            return {**base, "result": E4O.extract_json_from_response(response.choices[0].message.content)}
        except Exception as e:  # as the official script: retry, then -1
            if attempt == max_retries:
                return {**base, "result": {"correct": -1, "reason": f"not evaluated after {max_retries} retries: {e}"}}


def summarize(results):
    tag_counter = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        tag_counter[r["tag"]]["total"] += 1
        if r["result"].get("correct", 0) == 1:
            tag_counter[r["tag"]]["correct"] += 1
    tag_accuracy = {t: c["correct"] / c["total"] for t, c in tag_counter.items() if c["total"]}
    return {"tag_accuracy": tag_accuracy,
            "overall": sum(tag_accuracy.values()) / len(tag_accuracy) if tag_accuracy else 0.0,
            "n": len(results), "n_failed": sum(r["result"].get("correct") == -1 for r in results)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_dirs", nargs="+", required=True)
    ap.add_argument("--meta", default="/network/scratch/s/singhsd/benchmarks/Echo-4o/test_scripts/Geneval++.jsonl")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True, help="served model name (the path given to vllm serve)")
    ap.add_argument("--tag", default="qwen72b", help="output file suffix")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    metas = [json.loads(l) for l in open(args.meta) if l.strip()]
    client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY")
    for seed_dir in map(Path, args.seed_dirs):
        for it_dir in sorted((d for d in seed_dir.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name)):
            out = it_dir / f"geneval_pp_{args.tag}.json"
            if out.exists():
                print(f"[skip] {out}")
                continue
            paths = [it_dir / f"{i:05d}" / "samples" / "00000.png" for i in range(len(metas))]
            with ThreadPoolExecutor(args.workers) as pool:
                results = list(pool.map(lambda a: process_sample(client, args.model, *a),
                                        [(i, m, p) for i, (m, p) in enumerate(zip(metas, paths))]))
            summary = {"judge": args.model, **summarize(results)}
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(results, indent=1))
            (it_dir / f"geneval_pp_{args.tag}_summary.json").write_text(json.dumps(summary, indent=1))
            tmp.replace(out)
            print(f"{it_dir}: overall {summary['overall']:.4f} "
                  f"{ {t: round(a, 3) for t, a in summary['tag_accuracy'].items()} } failed {summary['n_failed']}",
                  flush=True)


if __name__ == "__main__":
    main()
