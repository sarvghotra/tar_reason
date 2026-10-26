"""Unified-model baselines for scripts/eval/bench_jeet_mila.sh: one driver, one backend per model family.

Drop-in GEN_SCRIPT like eval/baselines/janus_pro_r1_gen.py and diffusers_gen.py: same command line as
jeet's iterative generators (Tar-only flags accepted and ignored), the same prompt sharding over torchrun
ranks (prompt i -> rank i % world_size) and the same per-iteration output layout (see the Janus module's
docstring), so the launcher's completeness checks and the three scorers read the results unchanged.

    --family {diffusers, showo, showo2, bagel, blip3o, emu3, unicot}

A backend (eval/baselines/backends/<family>.py) exposes ``add_args(parser)`` and a class ``Backend(args,
device)`` whose ``run(prompts, seeds)`` returns, per prompt, ``(images, rounds)``: the list of images the
model produced for that prompt (one for single-pass models; first ... final for self-correcting ones such
as Uni-CoT) and a list of ``{"accepted": bool, "reason": str}`` per self-check round (empty for single
pass). Iteration k = 1..--n_iters gets image min(k, len(images)) and the last iteration the final image,
exactly as the Janus generator maps its loop. Per-prompt seeds are seed * 100003 + prompt index.

Usage (normally via the launcher: GEN_FAMILY=<family> ... scripts/eval/bench_jeet_mila.sh):
    torchrun --standalone --nproc_per_node=4 eval/baselines/unified_gen.py --family bagel \\
        --n_iters 2 --prompts_file $BENCH/GenEval2/geneval2_data.jsonl --geneval2 --out_dir OUT --seed 713
CPU check of arguments / layout without loading a model (dummy images):
    python eval/baselines/unified_gen.py --family bagel --dry_run --limit_prompts 3 <same args>
"""

import argparse
import importlib
import json
import os
import sys
import time

# Families whose library hard-codes cuda:0: under torchrun give each rank one visible GPU (before torch loads).
SINGLE_GPU_FAMILIES = ("blip3o",)
if "--family" in sys.argv and sys.argv[sys.argv.index("--family") + 1] in SINGLE_GPU_FAMILIES \
        and "LOCAL_RANK" in os.environ and "--dry_run" not in sys.argv:
    _lr = int(os.environ["LOCAL_RANK"])
    _vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = _vis.split(",")[_lr] if _vis else str(_lr)
    os.environ["LOCAL_RANK"] = "0"

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from janus_pro_r1_gen import (  # noqa: E402  (shared prompt loading, sharding and output layout)
    TIIF_DESCRIPTIONS, barrier, chunks, gather, genai_path, init_distributed, iteration_images,
    load_geneval2, load_genai, load_tiif, reset_output_dirs, tiif_path)

FAMILIES = ("diffusers", "showo", "showo2", "bagel", "blip3o", "emu3", "unicot")


def load_backend(family):
    return importlib.import_module(f"backends.{family}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--model_path", default=None, help="HF id or local dir (default: the backend's).")
    parser.add_argument("--gen_batch", type=int, default=None, help="Prompts per backend call (default: the backend's).")
    parser.add_argument("--save_size", type=int, default=0, help="Resize saved images to this square size (0 = native).")
    parser.add_argument("--dry_run", action="store_true", help="No model: dummy images to check arguments and layout.")
    parser.add_argument("--limit_prompts", type=int, default=0, help="Use only the first N prompts (0 = all).")
    parser.add_argument("--bench", choices=["geneval2", "genai", "tiif"], help="Default: inferred from the flags.")
    # --- flags shared with jeet's generators (same names and defaults)
    parser.add_argument("--n_iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompts_file")
    parser.add_argument("--out_dir", default="results/iterative")
    parser.add_argument("--geneval2", action="store_true")
    parser.add_argument("--strip_photo_prefix", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--gen_model")
    parser.add_argument("--root_dir", default="./datasets")
    parser.add_argument("--eval_set_dir")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--prompts_dir")
    parser.add_argument("--dimensions", nargs="*")
    parser.add_argument("--descriptions", nargs="*", choices=TIIF_DESCRIPTIONS, default=list(TIIF_DESCRIPTIONS))
    parser.add_argument("--eval_model_name", default="tar")
    # the family's own flags (parsed from the same argv; --family is read first)
    pre, _ = parser.parse_known_args(argv)
    backend = load_backend(pre.family)
    backend.add_args(parser)
    args, unknown = parser.parse_known_args(argv)
    args.ignored = unknown
    if args.bench is None:
        if args.prompts_dir:
            args.bench = "tiif"
        elif args.eval_set_dir:
            args.bench = "genai"
        elif args.prompts_file:
            args.bench = "geneval2"
        else:
            parser.error("cannot infer the benchmark: pass --prompts_file (GenEval2), "
                         "--eval_set_dir (GenAI-Bench) or --prompts_dir (TIIF), or --bench")
    if args.n_iters < 1:
        parser.error("--n_iters must be positive")
    if args.repeat != 1:
        parser.error("only --repeat 1 is supported (the scorers read sample 0 only)")
    if args.bench in ("genai", "tiif") and not args.gen_model:
        parser.error("--gen_model is required for GenAI-Bench / TIIF")
    args.model_path = args.model_path or backend.DEFAULT_MODEL
    args.gen_batch = args.gen_batch or backend.DEFAULT_BATCH
    if args.gen_batch < 1:
        parser.error("--gen_batch must be positive")
    return args, backend


class DryRun:
    def __init__(self, args, size):
        self.args, self.size = args, size

    def run(self, prompts, seeds):
        return [([Image.new("RGB", (self.size, self.size), tuple((hash(p) >> s) & 255 for s in (0, 8, 16)))], [])
                for p in prompts]


def main():
    args, backend = parse_args()
    rank, world_size, device = init_distributed(args)
    if rank == 0:
        shown = {k: v for k, v in vars(args).items() if k not in ("ignored",) and v not in (None, False, [])}
        print(f"[{args.family}] " + " ".join(f"{k}={v!r}" for k, v in sorted(shown.items())) + f" world={world_size}", flush=True)
        if args.ignored:
            print(f"[{args.family}] ignoring Tar-only arguments: {' '.join(args.ignored)}")

    n = args.n_iters
    if args.bench == "geneval2":
        save_dir = os.path.abspath(args.out_dir)
        iter_dirs = [os.path.join(save_dir, str(k)) for k in range(1, n + 1)]
        reset_output_dirs(iter_dirs, rank, world_size)
        records = load_geneval2(args.prompts_file, args.strip_photo_prefix)
        keys = list(range(len(records)))
        prompts = [r[0] for r in records]
    elif args.bench == "genai":
        save_dir = os.path.abspath(os.path.join(args.output_dir, args.gen_model))
        iter_dirs = [os.path.join(save_dir, str(k)) for k in range(1, n + 1)]
        if args.skip_existing:
            for d in iter_dirs:
                os.makedirs(d, exist_ok=True)
        else:
            reset_output_dirs(iter_dirs, rank, world_size)
        records = load_genai(args, iter_dirs)
        keys = [r[0] for r in records]
        prompts = [r[1] for r in records]
    else:
        save_dir = os.path.abspath(os.path.join(args.output_dir, args.gen_model))
        images_dir = os.path.join(save_dir, "images")
        model_names = [f"{args.eval_model_name}_iter{k}" for k in range(1, n + 1)]
        if not args.skip_existing:
            reset_output_dirs([images_dir], rank, world_size)
        records = load_tiif(args, images_dir, model_names)
        keys = [r[:3] for r in records]
        prompts = [r[3] for r in records]
        if rank == 0:
            for dimension, desc, idx in keys:
                for m in model_names:
                    os.makedirs(os.path.dirname(tiif_path(images_dir, dimension, m, desc, idx)), exist_ok=True)
        barrier(world_size)
    if args.limit_prompts:
        keys, prompts = keys[:args.limit_prompts], prompts[:args.limit_prompts]
    if rank == 0:
        print(f"[{args.family}] {len(prompts)} prompts, writing to {save_dir}", flush=True)

    gen = DryRun(args, getattr(backend, "DEFAULT_SIZE", 512)) if args.dry_run else backend.Backend(args, device)
    jobs = list(range(rank, len(prompts), world_size))
    local_results, local_geneval2 = {}, [{} for _ in range(n)]
    t0 = time.perf_counter()
    for b, batch in enumerate(chunks(jobs, args.gen_batch)):
        outputs = gen.run([prompts[i] for i in batch], [args.seed * 100003 + i for i in batch])
        for i, (images, rounds) in zip(batch, outputs):
            per_iter = iteration_images(images, n)
            if args.save_size:
                per_iter = [im.resize((args.save_size, args.save_size), Image.BICUBIC) for im in per_iter]
            if args.bench == "geneval2":
                for k, im in enumerate(per_iter):
                    prompt_dir = os.path.join(iter_dirs[k], f"{i:05d}")
                    sample_dir = os.path.join(prompt_dir, "samples")
                    os.makedirs(sample_dir, exist_ok=True)
                    path = os.path.join(sample_dir, "00000.png")
                    im.save(path)
                    with open(os.path.join(prompt_dir, "metadata.jsonl"), "w") as f:
                        f.write(records[i][1] + "\n")
                    local_geneval2[k][prompts[i]] = os.path.abspath(path)
                    if k >= 1:
                        reason = rounds[k - 1]["reason"] if k - 1 < len(rounds) else ""
                        with open(os.path.join(prompt_dir, "self_reflect.txt"), "w") as f:
                            f.write(f"Prompt: {prompts[i]}\n\nSample: 00000\nSelf-reflect: {reason}\n")
                key = f"{i:05d}"
            elif args.bench == "genai":
                for k, im in enumerate(per_iter):
                    im.convert("RGB").save(genai_path(iter_dirs[k], keys[i]))
                key = keys[i]
            else:
                dimension, desc, idx = keys[i]
                for k, im in enumerate(per_iter):
                    im.save(tiif_path(images_dir, dimension, model_names[k], desc, idx))
                key = f"{dimension}/{desc}/{idx}"
            local_results[key] = {"prompt": prompts[i], "seed": args.seed * 100003 + i, "n_images": len(images),
                                  "accepted_first": bool(rounds and rounds[0]["accepted"]), "rounds": rounds}
            if args.verbose:
                print(f"[rank {rank}] {key}: {prompts[i]!r} -> {len(images)} image(s)", flush=True)
        print(f"[rank {rank}] batch {b + 1}: {min((b + 1) * args.gen_batch, len(jobs))}/{len(jobs)} prompts, "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    all_results = gather(local_results, world_size)
    all_geneval2 = gather(local_geneval2, world_size) if args.bench == "geneval2" else None
    if rank == 0:
        merged = {}
        for part in all_results:
            merged.update(part)
        with open(os.path.join(save_dir, f"{args.family}_gen.json"), "w") as f:
            json.dump({"args": {k: v for k, v in vars(args).items() if k != "ignored"}, "prompts": merged},
                      f, indent=2, ensure_ascii=False, default=str)
        if args.bench == "geneval2":
            for k, d in enumerate(iter_dirs):
                result = {}
                for part in all_geneval2:
                    result.update(part[k])
                with open(os.path.join(d, "geneval2_results.json"), "w") as f:
                    json.dump(result, f, indent=2)
        else:   # same schema as jeet's self_reflect.json
            with open(os.path.join(save_dir, "self_reflect.json"), "w") as f:
                json.dump({k: {"prompt": v["prompt"],
                               "reflections": [v["rounds"][0]["reason"] if v["rounds"] else None],
                               "reflections_per_iter": [[r["reason"] for r in v["rounds"]]]}
                           for k, v in sorted(merged.items())}, f, indent=2, ensure_ascii=False)
        n_multi = sum(v["n_images"] > 1 for v in merged.values())
        print(f"[{args.family}] done: {len(merged)} prompts ({n_multi} with more than one image), "
              f"{time.perf_counter() - t0:.0f}s; {save_dir}", flush=True)
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
