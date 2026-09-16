"""Batched, multi-GPU iterative image generation for GenEval."""

import argparse
import json
import os
import re
import shutil
import sys

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoTokenizer, Qwen2ForCausalLM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ITERATIVE_PROMPT_PREFIX = (
    "Generate an image iteratively by self-reflecting and correcting.\n"
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora_path")
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--out_dir", default="results/iterative")
    parser.add_argument("--ar_path")
    parser.add_argument("--encoder_path")
    parser.add_argument("--decoder_path")
    parser.add_argument("--gen_seq_len", type=int, default=729)
    parser.add_argument("--scale", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Prompt/sample pairs per language-model call.")
    parser.add_argument("--decode_batch_size", type=int, default=4,
                        help="Images per visual-tokenizer call.")
    parser.add_argument("--geneval2", action="store_true")
    parser.add_argument("--reflect_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=1200)
    parser.add_argument("--min_p", type=int, default=0.05)
    parser.add_argument("--system_prompt", default="You are a helpful assistant.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true",
                        help="Print full generated token sequences.")
    return parser.parse_args()


def load_prompts(path):
    records = []
    with open(path) as prompt_file:
        for line_number, raw_line in enumerate(prompt_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                prompt = line
                metadata = json.dumps({"prompt": prompt})
            else:
                if not isinstance(obj, dict) or "prompt" not in obj:
                    raise ValueError(
                        f"{path}:{line_number}: JSON record must contain a "
                        "'prompt' field"
                    )
                prompt = obj["prompt"]
                if not isinstance(prompt, str):
                    raise ValueError(
                        f"{path}:{line_number}: 'prompt' must be a string"
                    )
                metadata = line
            records.append((prompt, metadata))
    return records


def reset_output_dirs(output_dirs, rank, world_size):
    """Remove prior outputs once, then recreate clean directories on all ranks."""
    if rank == 0:
        for output_dir in output_dirs:
            if os.path.lexists(output_dir):
                if os.path.isdir(output_dir) and not os.path.islink(output_dir):
                    shutil.rmtree(output_dir)
                else:
                    os.unlink(output_dir)
    if world_size > 1:
        dist.barrier()
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)


def load_models(args, device):
    from huggingface_hub import hf_hub_download
    from tok.mm_autoencoder import MMAutoEncoder

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"[{device}] Loading LM from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = Qwen2ForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, args.lora_path, torch_dtype=dtype
        ).merge_and_unload()

    ar_path = args.ar_path or hf_hub_download(
        "csuhan/TA-Tok", "ar_dtok_lp_256px.pth")
    encoder_path = args.encoder_path or hf_hub_download(
        "csuhan/TA-Tok", "ta_tok.pth")
    decoder_path = args.decoder_path or hf_hub_download(
        "peizesun/llamagen_t2i", "vq_ds16_t2i.pt")
    visual_tok = MMAutoEncoder(
        ar_path=ar_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        encoder_args={"input_type": "rec"},
        decoder_args={},
    ).eval().to(dtype=dtype, device=device)
    visual_tok.ar_model.cls_token_num = args.gen_seq_len
    visual_tok.encoder.pool_scale = args.scale + 1
    return tokenizer, model, visual_tok


def make_prefix(tokenizer, prompt, args):
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": ITERATIVE_PROMPT_PREFIX + prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + f"<im_start><S{args.scale}>"


def get_image_token_map(tokenizer):
    """Build this once instead of regex-parsing every full decoded output."""
    result = {}
    for token, token_id in tokenizer.get_vocab().items():
        match = re.fullmatch(r"<I(\d+)>", token)
        if match:
            result[token_id] = int(match.group(1))
    if not result:
        raise ValueError("Tokenizer contains no <I...> image tokens")
    return result


def extract_iterations(token_ids, image_token_map, tokenizer, seq_len):
    """Extract two contiguous image-code runs and their intervening reflection."""
    groups, starts, ends, current = [], [], [], []
    for position, token_id in enumerate(token_ids):
        code = image_token_map.get(int(token_id))
        if code is not None:
            if not current:
                starts.append(position)
            current.append(code)
        elif current:
            groups.append(current)
            ends.append(position)
            current = []
    if current:
        groups.append(current)
        ends.append(len(token_ids))
    if not groups:
        raise ValueError("Model output contains no <I...> image tokens")

    # When the model decides the first image already matches the prompt it
    # stops after the reflection, so the text runs to the end of the sequence.
    reflection_end = starts[1] if len(groups) > 1 else len(token_ids)
    reflection = tokenizer.decode(
        token_ids[ends[0]:reflection_end], skip_special_tokens=True
    )
    reflection = re.sub(r"<\/?im_(?:start|end)>|<S\d+>", "", reflection)
    reflection = re.sub(r"^\s*Self-reflect:\s*", "", reflection).strip()

    groups = [(group + [0] * seq_len)[:seq_len] for group in groups[:2]]
    return groups, reflection


def summarize_model_output(text, head=2, tail=3):
    """Collapse long image-token runs while retaining useful boundaries."""
    pattern = re.compile(r"(?:<I\d+>){" + str(head + tail + 1) + r",}")

    def summarize(match):
        tokens = re.findall(r"<I\d+>", match.group(0))
        omitted = len(tokens) - head - tail
        return (
            "".join(tokens[:head])
            + f"<... {omitted} image tokens omitted ...>"
            + "".join(tokens[-tail:])
        )

    return pattern.sub(summarize, text)


@torch.inference_mode()
def generate_batch(model, tokenizer, visual_tok, jobs, image_token_map,
                   args, device):
    prefixes = [make_prefix(tokenizer, prompt, args) for _, _, prompt in jobs]
    inputs = tokenizer(prefixes, return_tensors="pt", padding=True)
    input_length = inputs.input_ids.shape[1]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    generated = model.generate(
        inputs.input_ids.to(device, non_blocking=True),
        attention_mask=inputs.attention_mask.to(device, non_blocking=True),
        max_new_tokens=2 * args.gen_seq_len + args.reflect_tokens + 16, # + 16 for special tokens
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        # min_p=args.min_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=im_end_id,
    )[:, input_length:].cpu()

    # generate() pads completed rows while the rest of the batch is still
    # sampling. Remove that padding so each row ends at its actual EOS token.
    generated_rows = []
    for row in generated.tolist():
        if im_end_id in row:
            row = row[:row.index(im_end_id) + 1]
        generated_rows.append(row)

    parsed = [
        extract_iterations(row, image_token_map, tokenizer,
                           args.gen_seq_len)
        for row in generated_rows
    ]
    # Rows that stop after one image contribute a single code run; decode it
    # once and reuse the image so iteration two is byte-identical (the visual
    # decoder samples, so decoding the same codes twice yields different pixels).
    slices, flat_codes = [], []
    for groups, _ in parsed:
        slices.append((len(flat_codes), len(groups)))
        flat_codes.extend(groups)
    codes = torch.tensor(flat_codes, dtype=torch.long)
    decoded = []
    for start in range(0, len(codes), args.decode_batch_size):
        code_batch = codes[start:start + args.decode_batch_size].to(
            device, non_blocking=True)
        decoded.extend(visual_tok.decode_from_encoder_indices(
            code_batch, {"cfg_scale": args.cfg_scale}))

    results = []
    for (offset, count), (_, reflection) in zip(slices, parsed):
        first = Image.fromarray(decoded[offset].numpy())
        second = (Image.fromarray(decoded[offset + 1].numpy())
                  if count > 1 else first)
        results.append((first, second, reflection))
    if args.verbose:
        for prefix, row in zip(prefixes, generated_rows):
            print(f"\n--- Input ---\n{prefix}\n--- Output ---")
            output = tokenizer.decode(row, skip_special_tokens=False)
            print(summarize_model_output(output))
    return results


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    return rank, world_size, device


def main():
    args = parse_args()
    if args.batch_size < 1 or args.decode_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.geneval2 and args.repeat != 1:
        raise ValueError("--geneval2 requires --repeat 1")

    rank, world_size, device = init_distributed(args)
    torch.manual_seed(args.seed + rank)
    records = load_prompts(args.prompts_file)
    if rank == 0:
        print(f"Loaded {len(records)} prompts; using {world_size} GPU process(es)")
    tokenizer, model, visual_tok = load_models(args, device)
    image_token_map = get_image_token_map(tokenizer)

    output_dirs = (f"{args.out_dir}/1", f"{args.out_dir}/2")
    reset_output_dirs(output_dirs, rank, world_size)

    local_indices = range(rank, len(records), world_size)
    jobs = [
        (index, sample, records[index][0])
        for index in local_indices
        for sample in range(args.repeat)
    ]
    reflections = {}
    local_geneval2 = ({}, {})

    for start in range(0, len(jobs), args.batch_size):
        batch_jobs = jobs[start:start + args.batch_size]
        print(f"[rank {rank}] jobs {start + 1}-{start + len(batch_jobs)}/{len(jobs)}")
        results = generate_batch(
            model, tokenizer, visual_tok, batch_jobs, image_token_map,
            args, device)
        for (index, sample, prompt), (img1, img2, reflection) in zip(
                batch_jobs, results):
            metadata = records[index][1]
            prompt_dirs = [os.path.join(path, f"{index:05d}")
                           for path in output_dirs]
            sample_dirs = [os.path.join(path, "samples")
                           for path in prompt_dirs]
            for sample_dir in sample_dirs:
                os.makedirs(sample_dir, exist_ok=True)
            name = f"{sample:05d}.png"
            img1.save(os.path.join(sample_dirs[0], name))
            img2.save(os.path.join(sample_dirs[1], name))
            reflections.setdefault(index, [None] * args.repeat)[sample] = reflection

            if sample == 0:
                for prompt_dir in prompt_dirs:
                    with open(os.path.join(prompt_dir, "metadata.jsonl"), "w") as f:
                        f.write(metadata + "\n")
                if args.geneval2:
                    local_geneval2[0][prompt] = os.path.abspath(
                        os.path.join(sample_dirs[0], name))
                    local_geneval2[1][prompt] = os.path.abspath(
                        os.path.join(sample_dirs[1], name))

    for index, values in reflections.items():
        path = os.path.join(
            output_dirs[1], f"{index:05d}", "self_reflect.txt")
        with open(path, "w") as reflect_file:
            reflect_file.write(f"Prompt: {records[index][0]}\n")
            for sample, reflection in enumerate(values):
                reflect_file.write(
                    f"\nSample: {sample:05d}\n"
                    f"Self-reflect: {reflection or ''}\n")

    if args.geneval2:
        gathered = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(gathered, local_geneval2)
        else:
            gathered[0] = local_geneval2
        if rank == 0:
            merged = ({}, {})
            for rank_result in gathered:
                merged[0].update(rank_result[0])
                merged[1].update(rank_result[1])
            for output_dir, result in zip(output_dirs, merged):
                with open(os.path.join(output_dir, "geneval2_results.json"), "w") as f:
                    json.dump(result, f, indent=2)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
