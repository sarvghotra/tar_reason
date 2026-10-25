"""Single-pass diffusion baselines (FLUX.1-dev, SD3.5-Large, Qwen-Image) for scripts/eval/bench_jeet_mila.sh.

Drop-in GEN_SCRIPT like eval/baselines/janus_pro_r1_gen.py: same command line as jeet's iterative
generators (Tar-only flags are accepted and ignored), the same prompt sharding over torchrun ranks
(prompt i -> rank i % world_size) and the same per-iteration output layout, so the launcher's
completeness checks and the three scorers read the results unchanged (see the Janus module's
docstring for the layouts). These models have no critique step: every iteration k = 1..--n_iters
gets the *same* single-pass image, so iteration 1 = final = the model's one image.

Presets (--diff_model), each with its model card's defaults; override any of them with the flags:
    flux1-dev   black-forest-labs/FLUX.1-dev            1024 px, 50 steps, guidance 3.5 (distilled guidance, no CFG batch),
                                                        T5 max 512 tokens.  Gated: accept the licence on HF + HF_TOKEN.
    sd35-large  stabilityai/stable-diffusion-3.5-large  1024 px, 28 steps, CFG 3.5, T5 max 512 (card default 256; 512
                                                        keeps TIIF long prompts).  Gated: licence + HF_TOKEN.
    qwen-image  Qwen/Qwen-Image                         1328 px, 50 steps, true CFG 4.0, negative prompt " ".
                                                        Apache-2.0, ~57 GB; needs diffusers >= 0.35.
Prompts are passed verbatim (no "magic" suffixes such as Qwen-Image's "Ultra HD, 4K, cinematic
composition."; --prompt_suffix adds one if ever wanted). Per-prompt seeds: seed * 100003 + prompt
index, so a prompt's image does not depend on the number of ranks.

Usage (normally via the launcher: GEN_FAMILY=diffusers DIFF_MODEL=flux1-dev ... scripts/eval/bench_jeet_mila.sh):
    torchrun --standalone --nproc_per_node=4 eval/baselines/diffusers_gen.py --diff_model flux1-dev \\
        --n_iters 2 --prompts_file $BENCH/GenEval2/geneval2_data.jsonl --geneval2 --out_dir OUT --seed 713
CPU check of arguments / layout without loading a model (dummy images):
    python eval/baselines/diffusers_gen.py --dry_run --limit_prompts 3 <same args>
"""

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from janus_pro_r1_gen import (  # noqa: E402  (shared prompt loading, sharding and output layout)
    TIIF_DESCRIPTIONS, barrier, chunks, gather, genai_path, init_distributed, load_geneval2,
    load_genai, load_tiif, reset_output_dirs, tiif_path)

PRESETS = {
    "flux1-dev": dict(model_path="black-forest-labs/FLUX.1-dev", pipeline="FluxPipeline", size=1024,
                      steps=50, guidance=3.5, max_seq_len=512, negative=None, true_cfg=None),
    "sd35-large": dict(model_path="stabilityai/stable-diffusion-3.5-large", pipeline="StableDiffusion3Pipeline",
                       size=1024, steps=28, guidance=3.5, max_seq_len=512, negative="", true_cfg=None),
    "qwen-image": dict(model_path="Qwen/Qwen-Image", pipeline="QwenImagePipeline", size=1328,
                       steps=50, guidance=1.0, max_seq_len=None, negative=" ", true_cfg=4.0),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--diff_model", choices=sorted(PRESETS), required=True)
    parser.add_argument("--diff_model_path", default=None, help="HF id or local dir (default: the preset's).")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None, help="guidance_scale (FLUX: distilled guidance).")
    parser.add_argument("--true_cfg", type=float, default=None, help="Qwen-Image true_cfg_scale.")
    parser.add_argument("--max_seq_len", type=int, default=None, help="T5 max_sequence_length (FLUX / SD3).")
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--prompt_suffix", default="", help="Appended to every prompt (default none).")
    parser.add_argument("--diff_batch_size", type=int, default=None, help="Prompts per call (default 4; qwen-image 2).")
    parser.add_argument("--cpu_offload", action="store_true", help="enable_model_cpu_offload (if a GPU is too small).")
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
    p = PRESETS[args.diff_model]
    args.diff_model_path = args.diff_model_path or p["model_path"]
    args.height = args.height or p["size"]
    args.width = args.width or p["size"]
    args.steps = args.steps or p["steps"]
    args.guidance = p["guidance"] if args.guidance is None else args.guidance
    args.true_cfg = p["true_cfg"] if args.true_cfg is None else args.true_cfg
    args.max_seq_len = p["max_seq_len"] if args.max_seq_len is None else args.max_seq_len
    args.negative_prompt = p["negative"] if args.negative_prompt is None else args.negative_prompt
    args.diff_batch_size = args.diff_batch_size or (2 if args.diff_model == "qwen-image" else 4)
    if args.diff_batch_size < 1:
        parser.error("--diff_batch_size must be positive")
    return args


class DiffusionGen:
    """One diffusers pipeline per rank; run(prompts, seeds) -> [PIL image]."""

    def __init__(self, args, device):
        import diffusers
        self.args, self.device = args, device
        cls = getattr(diffusers, PRESETS[args.diff_model]["pipeline"])
        self.pipe = cls.from_pretrained(args.diff_model_path, torch_dtype=torch.bfloat16)
        if args.cpu_offload:
            self.pipe.enable_model_cpu_offload(device=device)
        else:
            self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def run(self, prompts, seeds):
        a = self.args
        gens = [torch.Generator(device=self.device).manual_seed(s) for s in seeds]
        kw = dict(prompt=prompts, height=a.height, width=a.width, num_inference_steps=a.steps,
                  guidance_scale=a.guidance, generator=gens)
        if a.max_seq_len:
            kw["max_sequence_length"] = a.max_seq_len
        if a.diff_model == "qwen-image":
            kw["true_cfg_scale"] = a.true_cfg
            kw["negative_prompt"] = [a.negative_prompt] * len(prompts)
        elif a.negative_prompt is not None:
            kw["negative_prompt"] = [a.negative_prompt] * len(prompts)
        return list(self.pipe(**kw).images)


class DryRun:
    def __init__(self, args):
        self.args = args

    def run(self, prompts, seeds):
        return [Image.new("RGB", (self.args.width, self.args.height), tuple((hash(p) >> s) & 255 for s in (0, 8, 16)))
                for p in prompts]


def main():
    args = parse_args()
    rank, world_size, device = init_distributed(args)
    if rank == 0:
        print(f"[diffusers] bench={args.bench} model={args.diff_model} ({args.diff_model_path}) "
              f"{args.width}x{args.height} steps={args.steps} guidance={args.guidance} true_cfg={args.true_cfg} "
              f"max_seq_len={args.max_seq_len} negative={args.negative_prompt!r} suffix={args.prompt_suffix!r} "
              f"n_iters={args.n_iters} seed={args.seed} batch={args.diff_batch_size} world={world_size} "
              f"dry_run={args.dry_run}", flush=True)
        if args.ignored:
            print(f"[diffusers] ignoring Tar-only arguments: {' '.join(args.ignored)}")

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
        print(f"[diffusers] {len(prompts)} prompts, writing to {save_dir}", flush=True)

    gen = DryRun(args) if args.dry_run else DiffusionGen(args, device)
    jobs = list(range(rank, len(prompts), world_size))
    local_results, local_geneval2 = {}, [{} for _ in range(n)]
    t0 = time.perf_counter()
    for b, batch in enumerate(chunks(jobs, args.diff_batch_size)):
        images = gen.run([prompts[i] + args.prompt_suffix for i in batch], [args.seed * 100003 + i for i in batch])
        for i, im in zip(batch, images):
            if args.save_size:
                im = im.resize((args.save_size, args.save_size), Image.BICUBIC)
            if args.bench == "geneval2":
                for k in range(n):
                    prompt_dir = os.path.join(iter_dirs[k], f"{i:05d}")
                    sample_dir = os.path.join(prompt_dir, "samples")
                    os.makedirs(sample_dir, exist_ok=True)
                    path = os.path.join(sample_dir, "00000.png")
                    im.save(path)
                    with open(os.path.join(prompt_dir, "metadata.jsonl"), "w") as f:
                        f.write(records[i][1] + "\n")
                    local_geneval2[k][prompts[i]] = os.path.abspath(path)
                    if k >= 1:
                        with open(os.path.join(prompt_dir, "self_reflect.txt"), "w") as f:
                            f.write(f"Prompt: {prompts[i]}\n\nSample: 00000\nSelf-reflect: \n")
                key = f"{i:05d}"
            elif args.bench == "genai":
                for k in range(n):
                    im.convert("RGB").save(genai_path(iter_dirs[k], keys[i]))
                key = keys[i]
            else:
                dimension, desc, idx = keys[i]
                for k in range(n):
                    im.save(tiif_path(images_dir, dimension, model_names[k], desc, idx))
                key = f"{dimension}/{desc}/{idx}"
            local_results[key] = {"prompt": prompts[i], "seed": args.seed * 100003 + i}
            if args.verbose:
                print(f"[rank {rank}] {key}: {prompts[i]!r}", flush=True)
        print(f"[rank {rank}] batch {b + 1}: {min((b + 1) * args.diff_batch_size, len(jobs))}/{len(jobs)} prompts, "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    all_results = gather(local_results, world_size)
    all_geneval2 = gather(local_geneval2, world_size) if args.bench == "geneval2" else None
    if rank == 0:
        merged = {}
        for part in all_results:
            merged.update(part)
        with open(os.path.join(save_dir, "diffusers_gen.json"), "w") as f:
            json.dump({"args": {k: v for k, v in vars(args).items() if k != "ignored"}, "prompts": merged},
                      f, indent=2, ensure_ascii=False)
        if args.bench == "geneval2":
            for k, d in enumerate(iter_dirs):
                result = {}
                for part in all_geneval2:
                    result.update(part[k])
                with open(os.path.join(d, "geneval2_results.json"), "w") as f:
                    json.dump(result, f, indent=2)
        else:   # same schema as jeet's self_reflect.json (no critiques: empty)
            with open(os.path.join(save_dir, "self_reflect.json"), "w") as f:
                json.dump({k: {"prompt": v["prompt"], "reflections": [None], "reflections_per_iter": [[]]}
                           for k, v in sorted(merged.items())}, f, indent=2, ensure_ascii=False)
        print(f"[diffusers] done: {len(merged)} prompts, {time.perf_counter() - t0:.0f}s; {save_dir}", flush=True)
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
