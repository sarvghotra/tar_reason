"""Check the batched Qwen3.5 VQAScore path against saved per-image scores.

Scores the first --n images of one GenAI-Bench iteration dir with the batched
t2v_metrics path and compares them to the `<model>_<num_prompts>_prompts.pt`
the old per-image path saved there. Needs one GPU and the t2v_metric env:

    python eval/tests/check_batched_vqascore.py \
        --images_dir results/genai-bench-800/<gen_model>/seed_13/1
"""
import argparse
import os
import sys
import time

import torch

T2V_METRICS_DIR = "/network/scratch/s/sarvjeet-singh.ghotra/git/t2v_metrics"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--num_prompts", type=int, default=800)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # The env's site-packages has an older t2v_metrics; use the repo the eval scripts use.
    sys.path.insert(0, T2V_METRICS_DIR)
    import t2v_metrics
    from dataset import GenAIBench_Image

    dataset = GenAIBench_Image(root_dir=f"{T2V_METRICS_DIR}/datasets", num_prompts=args.num_prompts)
    keys = list(dataset.dataset.keys())[:args.n]
    items = [{"images": [os.path.join(args.images_dir, f"{k}.jpeg")],
              "texts": [dataset.dataset[k]["prompt"]]} for k in keys]
    saved = torch.load(os.path.join(args.images_dir, f"{args.model}_{args.num_prompts}_prompts.pt"))[:args.n]

    score_func = t2v_metrics.get_score_model(model=args.model, device="cuda")
    start = time.perf_counter()
    scores = score_func.batch_forward(items, batch_size=args.batch_size).cpu()
    elapsed = time.perf_counter() - start

    diff = (scores - saved).abs().flatten()
    print(f"{args.n} images in {elapsed:.1f}s ({elapsed / args.n:.2f} s/image; old path ~2.3 s/image)")
    print(f"|batched - saved|: max {diff.max():.4f}, mean {diff.mean():.5f}")
    print(f"mean score: batched {scores.mean():.4f}, saved {saved.mean():.4f}")


if __name__ == "__main__":
    main()
