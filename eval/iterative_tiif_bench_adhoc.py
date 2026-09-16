"""Batched, multi-GPU iterative image generation for TIIF-Bench (forced schedule).

Same relationship to TIIF-Bench as eval/iterative_genai_bench_adhoc.py has to
GenAI-Bench: the three phases of the iterative model are *forced* rather than
left to the model's own emission schedule:

    Phase 1: <prefix><im_start><S{draft_img_scale}>  -> draft image codes
    Phase 2: ... <im_end>\\nSelf-reflect:             -> reflection text
    Phase 3: ... {reflection}\\n<im_start><S{scale}>  -> final image codes

Prompts are read from TIIF-Bench's generation prompt directory (one
``<dimension>_prompts.jsonl`` per dimension, each line holding ``type``,
``short_description`` and ``long_description``), e.g.

    ~/scratch/git/TIIF-Bench/data/testmini_prompts

Images are written in the layout TIIF-Bench's ``eval/eval_with_vlm.py``
expects, with the *iteration* encoded in the model-name level so that a single
scoring run covers both iterations and a single summary table compares them:

    <output_dir>/<gen_model>/images/<dimension>/<eval_model_name>_iter1/short_description/<idx>.png
    <output_dir>/<gen_model>/images/<dimension>/<eval_model_name>_iter1/long_description/<idx>.png
    <output_dir>/<gen_model>/images/<dimension>/<eval_model_name>_iter2/...

``<idx>`` is the 0-based line number in the prompt file, which is how
eval_with_vlm.py pairs an image with its yes/no question list (the eval JSONL
files are line-aligned with the generation ones). Score them with, from the
TIIF-Bench repo:

    python eval/eval_with_vlm.py --jsonl_dir data/testmini_eval_prompts \\
        --image_dir <output_dir>/<gen_model>/images \\
        --eval_model <eval_model_name>_iter1 --output_dir <...>/eval_results \\
        --base_url http://127.0.0.1:8000/v1 --model qwen2.5-vl

See scripts/eval/iter_tiif_bench_adhoc_e2e.sh for the end-to-end pipeline.
"""

import argparse
import glob
import json
import os
import sys

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.iterative_generation_adhoc import (  # noqa: E402
    SCALE_SEQ_LEN, even_chunks, generate_batch, get_image_token_map,
    init_distributed, load_models, reset_output_dirs)

# TIIF-Bench ships every prompt in two rhetorical registers and scores them
# separately; eval_with_vlm.py looks for a directory named after each field.
DESCRIPTIONS = ("short_description", "long_description")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HF model ID or local path (Qwen2-based Tar checkpoint).")
    parser.add_argument("--lora_path",
                        help="LoRA adapter checkpoint to merge on top of --model.")
    parser.add_argument(
        "--prompts_dir", required=True,
        help="TIIF-Bench generation prompt directory, e.g. "
             "<TIIF-Bench>/data/testmini_prompts.")
    parser.add_argument(
        "--dimensions", nargs="*",
        help="Restrict to these TIIF-Bench dimensions (default: all found).")
    parser.add_argument(
        "--descriptions", nargs="*", choices=DESCRIPTIONS, default=list(DESCRIPTIONS),
        help="Prompt registers to generate (default: both).")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument(
        "--gen_model", required=True,
        help="Name of the generating model; becomes the <output_dir> subfolder.")
    parser.add_argument(
        "--eval_model_name", default="tar",
        help="Base name for the TIIF-Bench model-name directory level; the "
             "iteration is appended as '_iter1' / '_iter2'.")
    parser.add_argument("--ar_path")
    parser.add_argument("--encoder_path")
    parser.add_argument("--decoder_path")
    parser.add_argument("--gen_seq_len", type=int, default=729,
                        help="Number of image tokens for the final (phase-3) image.")
    parser.add_argument("--draft_gen_seq_len", type=int, default=729,
                        help="Number of image tokens for the draft (phase-1) image.")
    parser.add_argument("--scale", type=int, default=0, choices=[0, 1, 2],
                        help="<S{scale}> used for the final (phase-3) image.")
    parser.add_argument("--draft_img_scale", type=int, default=0, choices=[0, 1, 2],
                        help="<S{scale}> used for the draft (phase-1) image.")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Samples per prompt. eval_with_vlm.py only scores the first one; "
             "extras are saved as <idx>_<k>.png.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Prompt/sample pairs per language-model call.")
    parser.add_argument("--decode_batch_size", type=int, default=4,
                        help="Images per visual-tokenizer call.")
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip prompts already on disk instead of wiping the image tree.")
    parser.add_argument("--reflect_tokens", type=int, default=256,
                        help="Max new tokens for the self-reflection text.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=1200)
    parser.add_argument("--reflect_temperature", type=float, default=0.85)
    parser.add_argument("--reflect_top_p", type=float, default=0.95)
    parser.add_argument("--reflect_top_k", type=int, default=50)
    parser.add_argument("--system_prompt", default="You are a helpful assistant.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true",
                        help="Print the generated sequences (image runs elided).")
    return parser.parse_args()


def iter_model_name(args, iteration):
    """TIIF-Bench model-name level, with the iteration folded into the name."""
    return f"{args.eval_model_name}_iter{iteration}"


def sample_path(images_dir, dimension, model_name, desc, idx, sample_idx):
    """eval_with_vlm.py globs '<idx>.*', so sample 0 keeps the bare name."""
    suffix = "" if sample_idx == 0 else f"_{sample_idx:02d}"
    return os.path.join(images_dir, dimension, model_name, desc,
                        f"{idx}{suffix}.png")


def load_prompt_files(args):
    """Return {dimension: [line_json, ...]} from <prompts_dir>/*_prompts.jsonl."""
    files = sorted(glob.glob(os.path.join(args.prompts_dir, "*_prompts.jsonl")))
    if not files:
        raise FileNotFoundError(
            f"No '*_prompts.jsonl' found in {args.prompts_dir}; point "
            "--prompts_dir at TIIF-Bench's data/testmini_prompts (or "
            "data/test_prompts).")
    wanted = set(args.dimensions) if args.dimensions else None
    dimensions = {}
    for path in files:
        lines = []
        with open(path, encoding="utf-8") as prompt_file:
            for line_number, raw_line in enumerate(prompt_file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "type" not in record:
                    raise ValueError(f"{path}:{line_number}: missing 'type'")
                lines.append(record)
        if not lines:
            continue
        # The dimension name comes from the records, not the filename: it is
        # what eval_with_vlm.py uses to build the image directory.
        dimension = lines[0]["type"]
        if wanted is not None and dimension not in wanted:
            continue
        dimensions[dimension] = lines
    if wanted is not None:
        missing = wanted - set(dimensions)
        if missing:
            raise ValueError(f"--dimensions not found in {args.prompts_dir}: "
                             f"{sorted(missing)}")
    return dimensions


def collect_records(args, images_dir, model_names):
    """Return [(dimension, desc, idx, prompt)] over every dimension/register."""
    records = []
    for dimension, lines in sorted(load_prompt_files(args).items()):
        for idx, record in enumerate(lines):
            for desc in args.descriptions:
                prompt = record.get(desc)
                if not prompt:
                    raise ValueError(
                        f"{dimension}[{idx}]: missing or empty '{desc}'")
                if args.skip_existing and all(
                        os.path.exists(sample_path(images_dir, dimension,
                                                   model_name, desc, idx, 0))
                        for model_name in model_names):
                    continue
                records.append((dimension, desc, idx, prompt))
    return records


def main():
    args = parse_args()
    if args.batch_size < 1 or args.decode_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.repeat < 1:
        raise ValueError("repeat must be positive")
    for name, scale, seq_len in (
            ("draft (phase 1)", args.draft_img_scale, args.draft_gen_seq_len),
            ("final (phase 3)", args.scale, args.gen_seq_len)):
        expected = SCALE_SEQ_LEN[scale]
        if seq_len != expected:
            raise ValueError(
                f"{name} image: scale {scale} requires {expected} image tokens, "
                f"got {seq_len}")

    rank, world_size, device = init_distributed(args)
    torch.manual_seed(args.seed + rank)

    save_dir = os.path.abspath(os.path.join(args.output_dir, args.gen_model))
    images_dir = os.path.join(save_dir, "images")
    model_names = (iter_model_name(args, 1), iter_model_name(args, 2))
    if not args.skip_existing:
        reset_output_dirs([images_dir], rank, world_size)

    records = collect_records(args, images_dir, model_names)
    if rank == 0:
        print(f"Loaded {len(records)} TIIF-Bench prompts x {args.repeat} "
              f"samples; using {world_size} GPU process(es); writing to "
              f"{images_dir}")
        leaf_dirs = {
            os.path.dirname(sample_path(images_dir, dimension, model_name,
                                        desc, idx, 0))
            for dimension, desc, idx, _ in records
            for model_name in model_names}
        for leaf_dir in sorted(leaf_dirs):
            os.makedirs(leaf_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    tokenizer, model, visual_tok = load_models(args, device)
    image_token_map = get_image_token_map(tokenizer)

    jobs = [
        records[index] + (sample,)
        for index in range(rank, len(records), world_size)
        for sample in range(args.repeat)
    ]
    local_reflections = {}

    done = 0
    for batch_jobs in even_chunks(jobs, args.batch_size):
        print(f"[rank {rank}] jobs {done + 1}-{done + len(batch_jobs)}/{len(jobs)}")
        done += len(batch_jobs)
        results = generate_batch(
            model, tokenizer, visual_tok, [job[3] for job in batch_jobs],
            image_token_map, args, device)
        for (dimension, desc, idx, prompt, sample), (img1, img2, reflection) in zip(
                batch_jobs, results):
            for model_name, image in zip(model_names, (img1, img2)):
                image.save(sample_path(images_dir, dimension, model_name, desc,
                                       idx, sample))
            entry = local_reflections.setdefault(
                f"{dimension}/{desc}/{idx}",
                {"prompt": prompt, "reflections": [None] * args.repeat})
            entry["reflections"][sample] = reflection

    gathered = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered, local_reflections)
    else:
        gathered[0] = local_reflections
    if rank == 0:
        merged = {}
        for rank_result in gathered:
            merged.update(rank_result)
        with open(os.path.join(save_dir, "self_reflect.json"), "w") as f:
            json.dump({k: merged[k] for k in sorted(merged)}, f, indent=2)
        print(f"Done, saved to {images_dir}")
        print("Score both iterations with (from the TIIF-Bench repo):")
        for model_name in model_names:
            print(f"  python eval/eval_with_vlm.py"
                  f" --jsonl_dir data/testmini_eval_prompts"
                  f" --image_dir {images_dir}"
                  f" --eval_model {model_name}"
                  f" --output_dir {os.path.join(save_dir, 'eval_results')}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
