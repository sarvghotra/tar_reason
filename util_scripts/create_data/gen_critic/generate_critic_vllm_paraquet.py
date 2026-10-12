'''
vLLM version of `generate_critic_hf_parquet.py` (same task, same I/O contract).

Use the HF script to debug on a few rows and this one to launch large jobs:
everything except the generation backend -- the prompt, the sampling defaults,
the shard/row sampling, the resume logic and the parquet writing -- is imported
from the HF script, so a change made there is picked up here automatically.

The only backend-specific pieces below are:
  - `build_vllm_generate_fn`: loads an `LLM` and returns `messages -> texts`
  - `add_vllm_arguments`: vLLM engine knobs, plus larger batch/flush defaults
    (vLLM does its own continuous batching, so a "batch" here is just how many
    rows are handed to the engine at once).
'''

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

# Import the HF script as the shared library. It lives next to this file, which
# is not necessarily on sys.path when the script is run by absolute path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_critic_hf_parquet import main, parse_args  # noqa: E402


def extract_images(messages: list[dict]) -> list[Image.Image]:
    """Pull the PIL images out of a `build_messages()` conversation.

    vLLM takes the prompt text and the images separately, unlike the HF
    processor which consumes the message list as a whole.
    """
    images = []
    for message in messages:
        for part in message["content"]:
            if part.get("type") == "image":
                images.append(part["image"])
    return images


def build_vllm_generate_fn(args: argparse.Namespace, sampling: dict, gpu_ids: list[int]):
    """Backend hook for `run_worker`: load the engine, return `messages -> texts`."""
    # Imported lazily: `run_worker` has already pinned CUDA_VISIBLE_DEVICES for
    # this replica, and vLLM reads it at import/engine-construction time.
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    if args.gpus_per_replica > 1:
        # Under the "spawn" start method used for data-parallel replicas, vLLM's
        # own fork-based worker launcher deadlocks.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    processor = AutoProcessor.from_pretrained(args.model)

    engine_kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.gpus_per_replica,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        seed=args.seed,
        disable_log_stats=True,
    )
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    if args.swap_space is not None:
        engine_kwargs["swap_space"] = args.swap_space

    llm = LLM(**engine_kwargs)

    sampling_params = SamplingParams(
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        top_k=sampling["top_k"],
        min_p=sampling["min_p"],
        presence_penalty=sampling["presence_penalty"],
        repetition_penalty=sampling["repetition_penalty"],
        max_tokens=sampling["max_new_tokens"],
    )

    def generate_fn(batch_messages: list[list[dict]]) -> list[str]:
        requests = []
        for messages in batch_messages:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                # Qwen3.6 thinks by default; the template opens and immediately
                # closes `<think>` for us when thinking is disabled.
                enable_thinking=(args.mode == "thinking"),
            )
            requests.append({
                "prompt": text,
                "multi_modal_data": {"image": extract_images(messages)},
            })
        outputs = llm.generate(requests, sampling_params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]

    return generate_fn


def add_vllm_arguments(parser: argparse.ArgumentParser) -> None:
    # vLLM batches continuously inside the engine, so feeding it more rows per
    # call is strictly better than the small HF batch; the flush interval grows
    # with it so part files stay a reasonable size.
    parser.set_defaults(batch_size=64, flush_rows=256)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="Fraction of each GPU vLLM may claim for weights + KV cache.")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Context length cap (default: the model config's). Lowering it "
                             "frees KV cache for more concurrent sequences.")
    parser.add_argument("--max_num_seqs", type=int, default=64,
                        help="Sequences the engine runs concurrently.")
    parser.add_argument("--swap_space", type=float, default=None,
                        help="CPU swap space in GiB per GPU (vLLM default: 4).")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Skip CUDA graph capture: slower, but saves memory and startup time.")


if __name__ == "__main__":
    main(
        parse_args(
            sys.argv[1:],
            add_arguments=add_vllm_arguments,
            description="Generate self-reflection/correction critique data with Qwen3.6 (vLLM).",
        ),
        build_generate_fn=build_vllm_generate_fn,
    )
