"""Batched, multi-GPU iterative image generation for GenAI-Bench (forced schedule).

Same relationship to eval/iterative_genai_bench.py as
eval/iterative_generation_adhoc.py has to eval/iterative_generation.py: instead
of letting the model emit the whole image / reflection / image sequence on its
own, the three phases are *forced*:

    Phase 1: <prefix><im_start><S{draft_img_scale}>  -> draft image codes
    Phase 2: ... <im_end>\\nSelf-reflect:             -> reflection text
    Phase 3: ... {reflection}\\n<im_start><S{scale}>  -> final image codes

Output layout is the one expected by t2v_metrics/genai_bench/evaluate.py, one
directory per iteration:

    <output_dir>/<gen_model>/1/<prompt_idx>.jpeg   (draft)
    <output_dir>/<gen_model>/2/<prompt_idx>.jpeg   (after self-reflection)

Each iteration is scored separately (from the t2v_metrics repo):

    python -m genai_bench.evaluate --model clip-flant5-xxl \
        --root_dir <root_dir> --output_dir <output_dir> \
        --gen_model <gen_model>/1

Prompts are read from <root_dir>/<eval_set_dir>/genai_image.json, which must
already exist on disk (nothing is downloaded).
"""

import argparse
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
    init_distributed, load_models, reset_output_dirs, strip_photo_prefix)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HF model ID or local path (Qwen2-based Tar checkpoint).")
    parser.add_argument("--lora_path",
                        help="LoRA adapter checkpoint to merge on top of --model.")
    parser.add_argument(
        "--root_dir", default="./datasets",
        help="Dataset root; prompts live in <root_dir>/<eval_set_dir>/.")
    parser.add_argument(
        "--eval_set_dir", required=True,
        help="Directory name under --root_dir holding genai_image.json "
             "(e.g. 'GenAI-Image-1600').")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument(
        "--gen_model", required=True,
        help="Name of the generating model; becomes the <output_dir> subfolder.")
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
        help="Samples per prompt. evaluate.py only scores the first one; extras "
             "are saved as <prompt_idx>_<k>.jpeg.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Prompt/sample pairs per language-model call.")
    parser.add_argument("--decode_batch_size", type=int, default=4,
                        help="Images per visual-tokenizer call.")
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip prompts already on disk instead of wiping the output dirs.")
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
    parser.add_argument("--strip_photo_prefix",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Remove a leading 'a photo of ' from each prompt.")
    return parser.parse_args()


def sample_path(iter_dir, prompt_idx, sample_idx):
    """evaluate.py reads '<prompt_idx>.jpeg', so sample 0 keeps the bare name."""
    suffix = "" if sample_idx == 0 else f"_{sample_idx:02d}"
    return os.path.join(iter_dir, f"{prompt_idx}{suffix}.jpeg")


def load_prompts(args, output_dirs):
    """Return [(prompt_idx, prompt)] from genai_image.json, oldest key first."""
    prompt_file = os.path.join(args.root_dir, args.eval_set_dir, "genai_image.json")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(
            f"{prompt_file} not found; --root_dir/--eval_set_dir must point at a "
            "directory containing genai_image.json (and genai_skills.json for "
            "scoring).")
    with open(prompt_file) as f:
        dataset = json.load(f)

    records = []
    for prompt_idx in sorted(dataset.keys()):
        if args.skip_existing and all(
                os.path.exists(sample_path(d, prompt_idx, 0))
                for d in output_dirs):
            continue
        prompt = dataset[prompt_idx]["prompt"]
        if args.strip_photo_prefix:
            prompt = strip_photo_prefix(prompt)
        records.append((prompt_idx, prompt))
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
    output_dirs = (os.path.join(save_dir, "1"), os.path.join(save_dir, "2"))
    if args.skip_existing:
        for output_dir in output_dirs:
            os.makedirs(output_dir, exist_ok=True)
    else:
        reset_output_dirs(output_dirs, rank, world_size)

    records = load_prompts(args, output_dirs)
    if rank == 0:
        print(f"Loaded {len(records)} prompts x {args.repeat} samples; "
              f"using {world_size} GPU process(es); writing to {save_dir}")
    tokenizer, model, visual_tok = load_models(args, device)
    image_token_map = get_image_token_map(tokenizer)

    jobs = [
        (records[index][0], sample, records[index][1])
        for index in range(rank, len(records), world_size)
        for sample in range(args.repeat)
    ]
    local_reflections = {}

    done = 0
    for batch_jobs in even_chunks(jobs, args.batch_size):
        print(f"[rank {rank}] jobs {done + 1}-{done + len(batch_jobs)}/{len(jobs)}")
        done += len(batch_jobs)
        results = generate_batch(
            model, tokenizer, visual_tok, [job[2] for job in batch_jobs],
            image_token_map, args, device)
        for (prompt_idx, sample, prompt), (img1, img2, reflection) in zip(
                batch_jobs, results):
            img1.save(sample_path(output_dirs[0], prompt_idx, sample))
            img2.save(sample_path(output_dirs[1], prompt_idx, sample))
            entry = local_reflections.setdefault(
                prompt_idx, {"prompt": prompt,
                             "reflections": [None] * args.repeat})
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
        print(f"Done, saved to {save_dir}")
        print("Score each iteration with (from the t2v_metrics repo):")
        # for iteration in (1, 2):
        #     print(f"  python -m genai_bench.evaluate --model clip-flant5-xxl"
        #           f" --root_dir {os.path.abspath(args.root_dir)}"
        #           f" --output_dir {os.path.abspath(args.output_dir)}"
        #           f" --gen_model {args.gen_model}/{iteration}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
