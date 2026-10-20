import argparse
import json
import math
import multiprocessing as mp
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# DEFAULT_MODEL = "/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/Qwen3-VL-8B-Instruct"
DEFAULT_MODEL = "/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/Qwen3-VL-30B-A3B-Instruct"

# LazyParquetDataset loads a WHOLE row group into RAM per DataLoader worker.
# Keep row groups small so each read is a few MB, and write enough shards that
# workers get distinct files. See t2i_jouneyDB.py for the full rationale.
ROW_GROUP_SIZE = 128        # rows; bounds per-worker RAM, fixed regardless of dataset size
TARGET_ROWS_PER_SHARD = 2000
MIN_SHARDS = 64

# SYSTEM_PROMPT = (
#     "You are a careful image critic. "
#     "Given a generated image and the text prompt used to generate it, "
#     "identify what is wrong or missing in the image with respect to the prompt.\n\n"
#     "Respond strictly in the following format and nothing else:\n"
#     "Self-reflection: <What is wrong or missing in the generated image with respect to the input prompt>\n"
#     "Edit: <A concise editing instruction that describes exactly what changes to make in the image to fix the issues>"
# )

# SYSTEM_PROMPT = (
#     "You are a careful image generation critic. "
#     "Given a generated image and the text prompt used to generate it, "
#     "identify what is wrong or missing in the image with respect to the prompt.\n\n"
#     "It's okay to have some extra (insignificant) details in the image that are not mentioned in the prompt."
#     "If there are no major discrepancies, just say 'Self-reflection: no issues, it matches the prompt.' and 'Edit: looks good.'.\n"
#     "Be very concise and respond in exactly this format (no extra text before or after):\n"
#     "Self-reflection: <what is wrong or missing in the image with respect to the prompt. Major problems only, ignore small issues>\n"
#     "Edit: <a concise instruction to edit the image to fix the identified problem>"
# )

# USER_TEMPLATE = "Image generation prompt: {prompt}\n\nAnalyze the image above against this prompt."

# OLD_PROMPT = (
#     "Given the image and the text prompt that was used to generate it, carefully analyze "
#     "the image and identify all the MAJOR discrepancies between the image and the prompt."
#     "It's okay to have some extra (insignificant) details in the image that are not mentioned in the prompt."
#     "If there are no major discrepancies, just say 'Self-reflection: no issues, it matches the prompt.' and 'Edit: looks good.'.\n\n"
#     "Be very concise and respond in exactly this format (no extra text before or after):\n"
#     "Self-reflection: <what is wrong or missing in the image with respect to the prompt. Major problems only, ignore small issues>\n"
#     "Edit: <a concise instruction to edit the image to fix the identified problems>"
# )

PROMPT = (
    "Given the image and the text prompt used to generate it, carefully analyze the image to identify the major issues where it's not faithful to the prompt.\n"
    "Be very concise and respond in exactly this format (no extra text before or after):\n"
    "Self-reflection: <what is wrong or missing in the image mentioned in the prompt>\n"
    "Edit: <a concise instruction to edit the image (not the prompt) to fix the identified problems>\n\n"
    "Avoid subjective critiques. Instead, focus on objective criteria such as counts, colors, entities, etc. Precision of the criticize is more important than recall in this task, so make sure to criticize only when it's very sure and objective.\n"
    "It's okay to have some extra details in the image that are not mentioned in the prompt. The image generation model is free to add details matching the scene described in the prompt.\n"
    "It's sampled from a high-quality text-to-image dataset, so the majority of cases will not have major issues. In those cases, just say 'Self-reflection: no issues, it matches the prompt.' and 'Edit: looks good.'\n\n"
)


def build_messages(image_path: str, prompt: str, enable_thinking: bool,
                   max_pixels: int | None) -> list[dict]:
    # system_content = SYSTEM_PROMPT
    # if not enable_thinking:
    #     # Qwen3 no-think directive must be at the start of the system turn
    #     system_content = "/no_think\n\n" + system_content
    image_content = {"type": "image", "image": f"file://{image_path}"}
    if max_pixels is not None:
        # Cap the vision-token count per image; the single biggest throughput
        # lever for VL bulk generation. process_vision_info honours this key.
        image_content["max_pixels"] = max_pixels
    return [
        # {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": PROMPT + f'Image generation prompt: "{prompt}"\n\n'},
                # {"type": "text", "text": USER_TEMPLATE.format(prompt=prompt)},
            ],
        },
    ]


def prepare_inputs_for_vllm(sample: dict, processor, enable_thinking: bool,
                            max_pixels: int | None) -> dict:
    """Decode image + render chat template for one sample. Image-only path,
    so we skip the video-kwargs machinery entirely."""
    from qwen_vl_utils import process_vision_info

    messages = build_messages(
        sample["image_path"], sample["prompt"], enable_thinking, max_pixels
    )
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(text)
    print("-------------")

    image_inputs, _ = process_vision_info(messages)
    return {
        "prompt": text,
        "multi_modal_data": {"image": image_inputs},
    }


def parse_output(text: str) -> tuple[str, str]:
    """Return (self_reflection, edit) extracted from model output text."""
    reflection_match = re.search(
        r"Self-reflection:\s*(.+?)(?=\nEdit:|\Z)", text, re.DOTALL | re.IGNORECASE
    )
    edit_match = re.search(r"Edit:\s*(.+)\Z", text, re.DOTALL | re.IGNORECASE)
    reflection = reflection_match.group(1).strip() if reflection_match else ""
    edit = edit_match.group(1).strip() if edit_match else ""
    return reflection, edit


def load_done_paths(output_path: str) -> set[str]:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["image_path"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def collect_samples(images_dir: str, done: set[str]) -> list[dict]:
    samples = []
    for subdir in sorted(Path(images_dir).iterdir()):
        if not subdir.is_dir():
            continue
        image_path = subdir / "image.png"
        prompt_path = subdir / "prompt.txt"
        if not image_path.exists() or not prompt_path.exists():
            continue
        abs_image = str(image_path.resolve())
        if abs_image in done:
            continue
        samples.append({
            "image_path": abs_image,
            "prompt": prompt_path.read_text().strip(),
        })
    return samples


def shard_output_path(output_jsonl: str, rank: int) -> str:
    return f"{output_jsonl}-rank{rank}"


def run_worker(rank: int, world_size: int, gpu_ids: list[int],
               shard: list[dict], args) -> None:
    """One vLLM replica pinned to one GPU group, processing its shard.

    A GPU group can be a single GPU (data-parallel replicas) or multiple GPUs
    (tensor-parallel replica).

    Generation and image preprocessing are pipelined: while the GPU runs
    chunk N, a thread pool decodes the images for chunk N+1.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    # Heavy imports happen here (in the child) so CUDA_VISIBLE_DEVICES is
    # already set before any CUDA context is created.
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    out_path = shard_output_path(args.output_jsonl, rank)
    tag = f"[rank{rank}/gpus{','.join(str(g) for g in gpu_ids)}]"
    if not shard:
        print(f"{tag} nothing to do")
        return
    print(f"{tag} {len(shard)} samples, tensor_parallel_size={len(gpu_ids)}")

    processor = AutoProcessor.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        tensor_parallel_size=len(gpu_ids),
        max_num_seqs=args.max_num_seqs,     # let vLLM's scheduler batch widely
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=True,         # shared system prompt is cached
        disable_log_stats=True,
        seed=0,
        max_model_len=2048,
        safetensors_load_strategy=args.safetensors_load_strategy,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_tokens,
        stop_token_ids=[],
    )

    pool = ThreadPoolExecutor(max_workers=args.preprocess_workers)

    def preprocess(chunk: list[dict]) -> list[dict]:
        return list(pool.map(
            lambda s: prepare_inputs_for_vllm(
                s, processor, args.enable_thinking, args.max_pixels),
            chunk,
        ))

    chunks = [shard[i:i + args.chunk_size]
              for i in range(0, len(shard), args.chunk_size)]

    done = 0
    with open(out_path, "a") as out_f:
        # Prime the pipeline: kick off preprocessing of the first chunk.
        next_inputs = pool.submit(preprocess, chunks[0])
        for ci, chunk in enumerate(chunks):
            inputs = next_inputs.result()
            # Start decoding the *next* chunk's images while the GPU works.
            if ci + 1 < len(chunks):
                next_inputs = pool.submit(preprocess, chunks[ci + 1])

            outputs = llm.generate(inputs, sampling_params=sampling_params,
                                   use_tqdm=(rank == 0))

            lines = []
            for sample, output in zip(chunk, outputs):
                raw = output.outputs[0].text
                reflection, edit = parse_output(raw)
                lines.append(json.dumps({
                    "image_path": sample["image_path"],
                    "prompt": sample["prompt"],
                    "self_reflection": reflection,
                    "edit": edit,
                    "raw_output": raw,
                }))
            out_f.write("\n".join(lines) + "\n")
            out_f.flush()          # one flush per chunk, not per record
            os.fsync(out_f.fileno())
            done += len(chunk)
            print(f"{tag} {done}/{len(shard)}")

    pool.shutdown()
    print(f"{tag} saved to {out_path}")


def _make_conversation(prompt: str, self_reflection: str, edit: str) -> np.ndarray:
    gpt_value = f"<image>\nSelf-reflect: {self_reflection}\nSelf-correction: {edit}"
    return np.array(
        [{"from": "human", "value": prompt}, {"from": "gpt", "value": gpt_value}],
        dtype=object,
    )


def _build_shard_df(shard_records: list[dict]) -> pd.DataFrame | None:
    rows = []
    for rec in shard_records:
        try:
            with open(rec["image_path"], "rb") as img_f:
                img_bytes = img_f.read()
        except FileNotFoundError:
            print(f"Warning: image not found: {rec['image_path']}, skipping")
            continue
        conv = _make_conversation(
            rec["prompt"],
            rec.get("self_reflection", ""),
            rec.get("edit", ""),
        )
        rows.append({
            "conversations": conv,
            "conversations_short": conv.copy(),
            "image": {"bytes": img_bytes},
        })
    return pd.DataFrame(rows) if rows else None


def merge_and_convert_to_parquet(
    output_jsonl: str,
    world_size: int,
    output_parquet_dir: str,
) -> None:
    os.makedirs(output_parquet_dir, exist_ok=True)

    records = []
    for rank in range(world_size):
        shard_path = shard_output_path(output_jsonl, rank)
        if not os.path.exists(shard_path):
            continue
        with open(shard_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    n = len(records)
    num_shards = max(MIN_SHARDS, math.ceil(n / TARGET_ROWS_PER_SHARD))
    shard_len = math.ceil(n / num_shards)
    print(
        f"Converting {n} records → {output_parquet_dir} "
        f"({num_shards} shards, ~{shard_len} rows/shard, row_group_size={ROW_GROUP_SIZE})",
        flush=True,
    )

    written = 0
    for shard_idx in range(num_shards):
        lo = shard_idx * shard_len
        hi = min(lo + shard_len, n)
        if lo >= n:
            break
        df = _build_shard_df(records[lo:hi])
        if df is None:
            continue
        out_path = os.path.join(output_parquet_dir, f"shard_{shard_idx:05d}.parquet")
        df.to_parquet(out_path, index=False, row_group_size=ROW_GROUP_SIZE)
        written += len(df)
        print(f"  wrote {out_path}: {len(df)} rows", flush=True)

    print(f"Done. Saved {written} rows to {output_parquet_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate self-reflection critic data using Qwen3-VL"
    )
    parser.add_argument("--images_dir", required=True,
                        help="Root directory containing per-image subdirectories")
    parser.add_argument("--output_jsonl", required=True,
                        help="Output JSONL prefix; each GPU writes <output>.rank<N> "
                             "(append + resume supported)")
    parser.add_argument("--output_parquet", default=None,
                        help="Output directory for parquet files "
                             "(default: <output_jsonl>_parquet/)")
    parser.add_argument("--model_path", default=DEFAULT_MODEL,
                        help="Qwen3-VL model checkpoint path")
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="Total GPUs to use (0 = all visible GPUs)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="GPUs per vLLM replica. Use 4 for Qwen3-VL-30B on "
                             "a 4-GPU node; use 1 for one replica per GPU.")
    parser.add_argument("--chunk_size", type=int, default=2048,
                        help="Samples per generate() call / incremental-write unit")
    parser.add_argument("--max_num_seqs", type=int, default=256,
                        help="vLLM in-flight sequence cap (continuous batching)")
    parser.add_argument("--preprocess_workers", type=int, default=16,
                        help="Threads decoding images per replica")
    parser.add_argument("--max_pixels", type=int, default=None,
                        help="Cap pixels/image (e.g. 1003520 = 1280*784) to slash "
                             "vision tokens; None keeps the model default")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--presence_penalty", type=float, default=1.5)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--safetensors_load_strategy", default="eager",
                        choices=["lazy", "eager"],
                        help="vLLM safetensors loading strategy")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable Qwen3 chain-of-thought thinking (disabled by default)")
    args = parser.parse_args()

    import torch
    total_gpus = args.num_gpus or torch.cuda.device_count()
    if total_gpus < 1:
        raise RuntimeError("No GPUs visible")
    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor_parallel_size must be >= 1")
    if args.tensor_parallel_size > total_gpus:
        raise ValueError(
            f"--tensor_parallel_size ({args.tensor_parallel_size}) cannot exceed "
            f"total GPUs ({total_gpus})"
        )
    if total_gpus % args.tensor_parallel_size != 0:
        raise ValueError(
            f"--num_gpus/visible GPUs ({total_gpus}) must be divisible by "
            f"--tensor_parallel_size ({args.tensor_parallel_size})"
        )
    gpu_groups = [
        list(range(start, start + args.tensor_parallel_size))
        for start in range(0, total_gpus, args.tensor_parallel_size)
    ]
    world_size = len(gpu_groups)

    # Resume: skip anything already recorded in any rank's shard file.
    done = set()
    for rank in range(world_size):
        done |= load_done_paths(shard_output_path(args.output_jsonl, rank))
    samples = collect_samples(args.images_dir, done)
    print(f"Samples to process: {len(samples)}  (already done: {len(done)})  "
          f"replicas: {world_size}, total_gpus: {total_gpus}, "
          f"tensor_parallel_size: {args.tensor_parallel_size}")
    if not samples:
        print("No samples found for processing")
        parquet_dir = args.output_parquet or (args.output_jsonl + "_parquet")
        merge_and_convert_to_parquet(args.output_jsonl, world_size, parquet_dir)
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)

    if world_size == 1:
        run_worker(0, 1, gpu_groups[0], samples, args)
    else:
        # Round-robin shard so load stays balanced even if costs vary by index.
        shards = [samples[r::world_size] for r in range(world_size)]
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=run_worker,
                        args=(r, world_size, gpu_groups[r], shards[r], args))
            for r in range(world_size)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        failed = [p.exitcode for p in procs if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"Worker(s) exited with {failed}")

    print(f"Done. Shards: {args.output_jsonl}.rank0..{world_size - 1}")

    parquet_dir = args.output_parquet or (args.output_jsonl + "_parquet")
    merge_and_convert_to_parquet(args.output_jsonl, world_size, parquet_dir)


if __name__ == "__main__":
    main()
