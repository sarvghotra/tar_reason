"""Batched, multi-GPU iterative image generation with a forced 3-phase schedule.

Unlike ``iterative_generation.py`` (which lets the model emit the whole
image / reflection / image sequence on its own), this script *forces* the
structure:

    Phase 1: <prefix><im_start><S{scale}>            -> first image codes
    Phase 2: ... <im_end>\\nSelf-reflect:             -> reflection text
    Phase 3: ... {reflection}\\n<im_start><S{scale}>  -> second image codes

Sampling is a two-level tree, mirroring the GRPO rollout in
``llava/train/rl/rollout.py``: every prompt gets ``--repeat`` independent
drafts, and every draft is branched ``--correct_repeat`` ways, each branch
re-sampling its own reflection and correction. Round-1 siblings therefore
share a parent image, which is the group GRPO normalises over.

Directory layout (D = draft index, C = branch index):
    {out_dir}/1/{prompt_index:05d}/samples/{D:05d}.png
    {out_dir}/1/{prompt_index:05d}/metadata.jsonl
    {out_dir}/2/{prompt_index:05d}/samples/{D:05d}_{C:05d}.png
    {out_dir}/2/{prompt_index:05d}/metadata.jsonl
    {out_dir}/2/{prompt_index:05d}/self_reflect.txt
    {out_dir}/tree.json                 (slot names and child -> parent map)

The iteration-2 file name drops the ``_{C:05d}`` part when --correct_repeat is
1, so a single-branch run keeps the old flat layout.

With --geneval2, one image per prompt is written per tree slot, since the
GenEval2 scorer keys its input by prompt and takes a single image for each:
    {out_dir}/1/geneval2_results_d{D}.json      prompt -> draft png
    {out_dir}/1/image_tokens_d{D}.json          prompt -> draft image codes
    {out_dir}/2/geneval2_results_d{D}c{C}.json  prompt -> correction png
    {out_dir}/2/image_tokens_d{D}c{C}.json      prompt -> correction codes
A run with --repeat 1 --correct_repeat 1 has one slot per iteration and writes
the unsuffixed ``geneval2_results.json`` / ``image_tokens.json`` instead.

Input file formats accepted:
  - JSONL  — one JSON object per line with a "prompt" key
  - Plain text — one prompt per line

Usage:
    torchrun --standalone --nproc_per_node=4 eval/iterative_generation_adhoc.py \\
        --model <path-or-hf-id> [--lora_path <adapter-checkpoint>] \\
        --prompts_file <path> --out_dir results/iterative \\
        --batch_size 32 --decode_batch_size 32 --repeat 4 --correct_repeat 2
"""

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

# TA-Tok pools a 27x27 grid by (scale + 1), so each <S{scale}> implies an exact
# image-token count: floor(27 / (scale + 1)) ** 2.
SCALE_SEQ_LEN = {0: 729, 1: 169, 2: 81}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HF model ID or local path (Qwen2-based Tar checkpoint).")
    parser.add_argument("--lora_path",
                        help="LoRA adapter checkpoint to merge on top of --model.")
    parser.add_argument("--prompts_file", required=True,
                        help="JSONL (with 'prompt' key) or plain-text file of prompts.")
    parser.add_argument("--out_dir", default="results/iterative")
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
    parser.add_argument("--repeat", type=int, default=4,
                        help="Drafts sampled per prompt (the round-0 group; "
                             "branch[0] in train_grpo.py --branch).")
    parser.add_argument("--correct_repeat", type=int, default=1,
                        help="Reflection/correction branches sampled per draft "
                             "(the round-1 group; branch[1] in train_grpo.py "
                             "--branch). 1 reproduces the old flat layout.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Prompt/sample pairs per language-model call.")
    parser.add_argument("--decode_batch_size", type=int, default=4,
                        help="Images per visual-tokenizer call.")
    parser.add_argument("--geneval2", action="store_true",
                        help="Also write a {prompt: image_path} JSON per stage "
                             "and per tree slot, for GenEval2 scoring.")
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
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true",
                        help="Print the generated sequences (image runs elided).")
    parser.add_argument("--strip_photo_prefix",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Remove a leading 'a photo of ' from each prompt.")
    return parser.parse_args()


PHOTO_PREFIX = "a photo of "


def strip_photo_prefix(prompt):
    """Drop a leading 'a photo of ' (case-insensitive) from a prompt."""
    if prompt.lower().startswith(PHOTO_PREFIX):
        return prompt[len(PHOTO_PREFIX):].lstrip()
    return prompt


def load_prompts(path, strip_prefix=False):
    """Load prompts together with GenEval-compatible JSONL metadata."""
    records = []
    with open(path) as prompt_file:
        for line_number, raw_line in enumerate(prompt_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                prompt = strip_photo_prefix(line) if strip_prefix else line
                metadata = json.dumps({"prompt": prompt})
            else:
                if not isinstance(obj, dict) or "prompt" not in obj:
                    raise ValueError(
                        f"{path}:{line_number}: JSON record must contain a "
                        "'prompt' field")
                prompt = obj["prompt"]
                if not isinstance(prompt, str):
                    raise ValueError(f"{path}:{line_number}: 'prompt' must be a string")
                if strip_prefix:
                    prompt = strip_photo_prefix(prompt)
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
        print(f"[{device}] Merging LoRA adapter from {args.lora_path}")
        model = PeftModel.from_pretrained(
            model, args.lora_path, torch_dtype=dtype).merge_and_unload()

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
    # cls_token_num / pool_scale are re-set per phase before each decode call,
    # since the draft and the final image may use different scales.
    visual_tok.ar_model.cls_token_num = args.gen_seq_len
    visual_tok.encoder.pool_scale = args.scale + 1
    return tokenizer, model, visual_tok


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


def summarize_model_output(text, head=2, tail=3):
    """Collapse long image-token runs while retaining useful boundaries."""
    pattern = re.compile(r"(?:<I\d+>){" + str(head + tail + 1) + r",}")

    def summarize(match):
        tokens = re.findall(r"<I\d+>", match.group(0))
        omitted = len(tokens) - head - tail
        return ("".join(tokens[:head])
                + f"<... {omitted} image tokens omitted ...>"
                + "".join(tokens[-tail:]))

    return pattern.sub(summarize, text)


def make_prefix(tokenizer, prompt, args):
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": ITERATIVE_PROMPT_PREFIX + prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + f"<im_start><S{args.draft_img_scale}>"


def left_pad(rows, pad_id, device):
    """Left-pad ragged id lists into a batch, matching padding_side='left'."""
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        input_ids[index, width - len(row):] = torch.tensor(row, dtype=torch.long)
        attention_mask[index, width - len(row):] = 1
    return (input_ids.to(device, non_blocking=True),
            attention_mask.to(device, non_blocking=True))


def even_chunks(items, max_size):
    """Split into equal-sized chunks of at most `max_size`.

    Both the LM and the visual detokenizer decode autoregressively, so a chunk
    costs roughly the same wall clock regardless of how full it is. Greedy
    fixed-size slicing leaves a tiny tail (50 items at 48 -> 48 + 2) that costs
    a near-full extra pass, so spread the items evenly instead.
    """
    if not len(items):
        return []
    count = -(-len(items) // max_size)
    size, extra = divmod(len(items), count)
    chunks, start = [], 0
    for index in range(count):
        stop = start + size + (1 if index < extra else 0)
        chunks.append(items[start:stop])
        start = stop
    return chunks


def draft_slot(draft):
    """Name of an iteration-1 tree slot (one draft of a prompt)."""
    return f"d{draft}"


def child_slot(draft, child):
    """Name of an iteration-2 tree slot (one correction branch of a draft)."""
    return f"d{draft}c{child}"


def trim_at(row, stop_ids):
    """Cut a generated row at the first stop token (generate right-pads rows
    that finished before the rest of the batch)."""
    for position, token_id in enumerate(row):
        if token_id in stop_ids:
            return row[:position]
    return row


@torch.inference_mode()
def generate_batch(model, tokenizer, visual_tok, prompts, image_token_map,
                   args, device):
    """Run the forced three-phase schedule for a batch of prompts.

    Each prompt gets one draft (phase 1), and that draft is branched
    ``args.correct_repeat`` ways: every branch re-samples its own reflection
    (phase 2) and correction (phase 3), so siblings share a parent image and
    differ only in how they criticise and fix it.

    Returns one entry per prompt: ``(draft_image, draft_codes, children)``,
    where ``children`` holds one ``(image, reflection, codes)`` per branch. The
    codes are the image token ids the LM actually emitted, before the
    decode/re-encode round trip.
    """
    pad_id = tokenizer.pad_token_id
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eof_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    im_start_tok_id = tokenizer.convert_tokens_to_ids("<im_start>")
    text_stop_ids = list({tokenizer.eos_token_id, im_end_id, eof_id,
                          im_start_tok_id, pad_id} - {None})
    # Image phases have no explicit stop; anything but an image token ends the run.
    image_stop_ids = set(text_stop_ids)

    def encode(text):
        return tokenizer(text, add_special_tokens=False).input_ids

    def generate_rows(rows, **kwargs):
        """Sample continuations for `rows`, at most --batch_size rows at a time.

        Branching multiplies the phase-2 and phase-3 row count by
        --correct_repeat, so chunk here to keep the live batch the size the
        caller asked for.
        """
        generated = []
        for chunk in even_chunks(rows, args.batch_size):
            input_ids, attention_mask = left_pad(chunk, pad_id, device)
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                repetition_penalty=1.0,
                pad_token_id=pad_id,
                **kwargs,
            )
            generated.extend(out[:, input_ids.shape[1]:].tolist())
        return generated

    def generate_image_codes(rows, gen_seq_len):
        """Sample `gen_seq_len` image tokens per row; return (codes, new_ids)."""
        generated = generate_rows(
            rows,
            max_new_tokens=gen_seq_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        codes, new_ids = [], []
        for row in generated:
            row = trim_at(row, image_stop_ids)
            row_codes = [image_token_map[t] for t in row if t in image_token_map]
            row_codes = (row_codes + [0] * gen_seq_len)[:gen_seq_len]
            codes.append(row_codes)
            new_ids.append(row)
        return codes, new_ids

    # ── Phase 1: draft image, one per prompt ─────────────────────────────────
    prefixes = [make_prefix(tokenizer, prompt, args) for prompt in prompts]
    phase1_rows = tokenizer(prefixes, add_special_tokens=False).input_ids
    codes1, image1_ids = generate_image_codes(phase1_rows, gen_seq_len=args.draft_gen_seq_len)

    # ── Phase 2: one self-reflection per branch ──────────────────────────────
    # Rows below are indexed by branch; `parents[branch]` is the draft it grew
    # from, so branches [i * width, (i + 1) * width) all belong to prompt i.
    width = args.correct_repeat
    parents = [index for index in range(len(prompts)) for _ in range(width)]
    reflect_prefix_ids = encode("<im_end>\nSelf-reflect:")
    phase2_rows = [phase1_rows[index] + image1_ids[index] + reflect_prefix_ids
                   for index in parents]
    reflect_out = generate_rows(
        phase2_rows,
        max_new_tokens=args.reflect_tokens,
        temperature=args.reflect_temperature,
        top_p=args.reflect_top_p,
        top_k=args.reflect_top_k,
        eos_token_id=text_stop_ids,
    )

    reflections = [
        tokenizer.decode(trim_at(row, set(text_stop_ids)),
                         skip_special_tokens=True).strip()
        for row in reflect_out
    ]

    # ── Phase 3: one corrected image per branch ──────────────────────────────
    # A reflection saying the draft "looks good" means no correction is needed,
    # so those branches skip phase 3 and reuse the phase-1 image (the visual
    # decoder samples, so re-decoding the same codes would not be byte-identical).
    image_start_ids = encode(f"\n<im_start><S{args.scale}>")
    pending = [branch for branch, reflection in enumerate(reflections)
               if "looks good" not in reflection.lower()]
    image2_ids = [[] for _ in parents]
    codes2 = []
    if pending:
        phase3_rows = [phase2_rows[branch] + encode(" " + reflections[branch])
                       + image_start_ids for branch in pending]
        codes2, pending_ids = generate_image_codes(phase3_rows, args.gen_seq_len)
        for slot, branch in enumerate(pending):
            image2_ids[branch] = pending_ids[slot]

    # ── Decode each phase separately: they may use different scales/lengths ──
    def decode_codes(codes, gen_seq_len, scale):
        visual_tok.ar_model.cls_token_num = gen_seq_len
        visual_tok.encoder.pool_scale = scale + 1
        all_codes = torch.tensor(codes, dtype=torch.long)
        images = []
        for code_batch in even_chunks(all_codes, args.decode_batch_size):
            images.extend(visual_tok.decode_from_encoder_indices(
                code_batch.to(device, non_blocking=True),
                {"cfg_scale": args.cfg_scale}))
        return images

    decoded1 = decode_codes(codes1, args.draft_gen_seq_len, args.draft_img_scale)
    decoded2 = (decode_codes(codes2, args.gen_seq_len, args.scale)
                if pending else [])

    if args.verbose:
        for branch, index in enumerate(parents):
            full = (tokenizer.decode(image1_ids[index], skip_special_tokens=False)
                    + "<im_end>\nSelf-reflect: " + reflections[branch])
            if image2_ids[branch]:
                full += (tokenizer.decode(image_start_ids, skip_special_tokens=False)
                         + tokenizer.decode(image2_ids[branch],
                                            skip_special_tokens=False))
            print(f"\n--- Input ---\n{prefixes[index]}\n"
                  f"--- Output (branch {branch % width}) ---")
            print(summarize_model_output(full))

    second_slots = {branch: slot for slot, branch in enumerate(pending)}
    results = []
    for index in range(len(prompts)):
        first = Image.fromarray(decoded1[index].numpy())
        children = []
        for child in range(width):
            branch = index * width + child
            slot = second_slots.get(branch)
            second = (Image.fromarray(decoded2[slot].numpy())
                      if slot is not None else first)
            # Branches that skipped phase 3 reuse the draft, so their codes
            # match the draft's too.
            second_codes = codes2[slot] if slot is not None else codes1[index]
            children.append((second, reflections[branch], second_codes))
        results.append((first, codes1[index], children))
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
    if args.repeat < 1 or args.correct_repeat < 1:
        raise ValueError("--repeat and --correct_repeat must be positive")
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
    records = load_prompts(args.prompts_file, args.strip_photo_prefix)
    if rank == 0:
        print(f"Loaded {len(records)} prompts; using {world_size} GPU process(es)")
    tokenizer, model, visual_tok = load_models(args, device)
    image_token_map = get_image_token_map(tokenizer)

    output_dirs = (f"{args.out_dir}/1", f"{args.out_dir}/2")
    reset_output_dirs(output_dirs, rank, world_size)

    jobs = [
        (index, draft, records[index][0])
        for index in range(rank, len(records), world_size)
        for draft in range(args.repeat)
    ]
    # prompt index -> {(draft, child): reflection}
    reflections = {}
    # One dict per iteration, each mapping a tree slot to {prompt: value}.
    local_geneval2 = ({}, {})
    local_tokens = ({}, {})

    done = 0
    for batch_jobs in even_chunks(jobs, args.batch_size):
        print(f"[rank {rank}] jobs {done + 1}-{done + len(batch_jobs)}/{len(jobs)}")
        done += len(batch_jobs)
        results = generate_batch(
            model, tokenizer, visual_tok, [job[2] for job in batch_jobs],
            image_token_map, args, device)
        for (index, draft, prompt), (draft_img, draft_codes,
                                     children) in zip(batch_jobs, results):
            prompt_dirs = [os.path.join(path, f"{index:05d}") for path in output_dirs]
            sample_dirs = [os.path.join(path, "samples") for path in prompt_dirs]
            for sample_dir in sample_dirs:
                os.makedirs(sample_dir, exist_ok=True)

            draft_path = os.path.join(sample_dirs[0], f"{draft:05d}.png")
            draft_img.save(draft_path)
            if args.geneval2:
                slot = draft_slot(draft)
                local_geneval2[0].setdefault(slot, {})[prompt] = \
                    os.path.abspath(draft_path)
                local_tokens[0].setdefault(slot, {})[prompt] = draft_codes

            for child, (image, reflection, codes) in enumerate(children):
                # Keep the flat name when nothing branched, so a single-branch
                # run stays readable by the old per-sample tooling.
                name = (f"{draft:05d}.png" if args.correct_repeat == 1
                        else f"{draft:05d}_{child:05d}.png")
                child_path = os.path.join(sample_dirs[1], name)
                image.save(child_path)
                reflections.setdefault(index, {})[(draft, child)] = reflection
                if args.geneval2:
                    slot = child_slot(draft, child)
                    local_geneval2[1].setdefault(slot, {})[prompt] = \
                        os.path.abspath(child_path)
                    local_tokens[1].setdefault(slot, {})[prompt] = codes

            if draft == 0:
                for prompt_dir in prompt_dirs:
                    with open(os.path.join(prompt_dir, "metadata.jsonl"), "w") as f:
                        f.write(records[index][1] + "\n")

    for index, values in reflections.items():
        path = os.path.join(output_dirs[1], f"{index:05d}", "self_reflect.txt")
        with open(path, "w") as reflect_file:
            reflect_file.write(f"Prompt: {records[index][0]}\n")
            for (draft, child), reflection in sorted(values.items()):
                reflect_file.write(f"\nSample: {child_slot(draft, child)}\n"
                                   f"Self-reflect: {reflection or ''}\n")

    # One slot per iteration means the tree is degenerate, so keep the plain
    # file names the existing GenEval2 scripts already point at.
    single_slot = args.repeat == 1 and args.correct_repeat == 1

    if args.geneval2:
        gathered = [None] * world_size
        local = (local_geneval2, local_tokens)
        if world_size > 1:
            dist.all_gather_object(gathered, local)
        else:
            gathered[0] = local
        if rank == 0:
            merged = (({}, {}), ({}, {}))
            for rank_result in gathered:
                for merged_pair, rank_pair in zip(merged, rank_result):
                    for merged_slots, rank_slots in zip(merged_pair, rank_pair):
                        for slot, mapping in rank_slots.items():
                            merged_slots.setdefault(slot, {}).update(mapping)
            for stem, merged_pair in zip(
                    ("geneval2_results", "image_tokens"), merged):
                for output_dir, slots in zip(output_dirs, merged_pair):
                    for slot, mapping in slots.items():
                        name = (f"{stem}.json" if single_slot
                                else f"{stem}_{slot}.json")
                        with open(os.path.join(output_dir, name), "w") as f:
                            json.dump(mapping, f, indent=2)

    if rank == 0:
        # The group structure the reward has to discriminate within: drafts of
        # one prompt (round 0), and branches of one draft (round 1).
        parent_of = {child_slot(draft, child): draft_slot(draft)
                     for draft in range(args.repeat)
                     for child in range(args.correct_repeat)}
        with open(os.path.join(args.out_dir, "tree.json"), "w") as f:
            json.dump({"drafts": args.repeat,
                       "children": args.correct_repeat,
                       "single_slot": single_slot,
                       "slots": {"1": [draft_slot(d) for d in range(args.repeat)],
                                 "2": list(parent_of)},
                       "parent": parent_of}, f, indent=2)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
