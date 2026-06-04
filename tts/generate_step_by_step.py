"""Step-by-step image generation: user-supplied sub-prompts → generate → iteratively edit.

The first sub-prompt is the full prompt (used only to name the output directory).
The remaining sub-prompts are the sequential generation steps:

  Phase 1: generate image from sub_prompts[1]
  Phase k: edit image_(k-1) to include sub_prompts[k]

Usage:
    python tts/generate_step_by_step.py \
        --model <path-or-hf-id> \
        --sub_prompts "a red cube on a blue table with a cat sitting next to it" \
                      "a red cube" \
                      "a red cube on a table" \
                      "a red cube on a blue table" \
                      "a red cube on a blue table with a cat sitting next to it" \
        --out_dir results/step_by_step \
        [--gen_seq_len 729] [--scale 0] \
        [--temperature 1.0] [--top_p 0.95] [--top_k 1200] [--cfg_scale 4.0] \
        [--ar_path ...] [--encoder_path ...] [--decoder_path ...]
"""

import argparse
import os
import re
import sys
import textwrap

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, LogitsProcessorList, Qwen2ForCausalLM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from huggingface_hub import hf_hub_download
from llava.train.rl.grpo import (
    ImageVocabLogitsProcessor,
    NonImageLogitsProcessor,
    build_t2i_prefix_ids,
    encode_splice,
)
from tok.mm_autoencoder import MMAutoEncoder


# ── decomposition constants (unused — decomposition is disabled) ───────────────
# DECOMPOSE_SYSTEM = (
#     "You are a helpful assistant that decomposes image generation prompts into "
#     "sequential, cumulative sub-steps. Each step adds one new element or attribute "
#     "to the scene."
# )
# DECOMPOSE_USER_TMPL = (
#     "Break the following image generation prompt into a short numbered list of "
#     "sequential sub-steps. Each step should describe ONE new element or attribute "
#     "to add (building on all previous steps). Output ONLY the numbered list, one "
#     "step per line, no extra commentary.\n\n"
#     "Prompt: {prompt}"
# )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="HF model ID or local path.")
    p.add_argument("--sub_prompts", type=str, nargs="+", required=True,
                   help="First element: full prompt (names the output dir). "
                        "Remaining elements: sequential sub-prompts for each phase.")
    p.add_argument("--out_dir", type=str, default="results/step_by_step")
    p.add_argument("--ar_path", type=str, default=None)
    p.add_argument("--encoder_path", type=str, default=None)
    p.add_argument("--decoder_path", type=str, default=None)
    p.add_argument("--scale", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--gen_seq_len", type=int, default=729)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=1200)
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--num_image_tokens", type=int, default=65536)
    p.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


# ── model loading ──────────────────────────────────────────────────────────────

def load_models(args):
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = torch.device(args.device)

    print(f"Loading LM from {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen2ForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()

    ar_path  = args.ar_path      or hf_hub_download("csuhan/TA-Tok",         "ar_dtok_lp_256px.pth")
    enc_path = args.encoder_path or hf_hub_download("csuhan/TA-Tok",         "ta_tok.pth")
    dec_path = args.decoder_path or hf_hub_download("peizesun/llamagen_t2i", "vq_ds16_t2i.pt")

    print("Loading visual tokenizer …")
    visual_tok = MMAutoEncoder(
        ar_path=ar_path,
        encoder_path=enc_path,
        decoder_path=dec_path,
        encoder_args={"input_type": "rec"},
        decoder_args={},
    ).eval().to(dtype=dtype, device=device)
    visual_tok.ar_model.cls_token_num = args.gen_seq_len
    visual_tok.encoder.pool_scale = args.scale + 1

    return tokenizer, model, visual_tok, device


# ── sampling helpers ───────────────────────────────────────────────────────────

def _readable_input(tokenizer, input_ids, image_start_id, num_image_tokens):
    """Decode input_ids to a string, replacing image token runs with a placeholder."""
    ids = input_ids[0].tolist()
    parts = []
    i = 0
    while i < len(ids):
        if image_start_id <= ids[i] < image_start_id + num_image_tokens:
            j = i
            while j < len(ids) and image_start_id <= ids[j] < image_start_id + num_image_tokens:
                j += 1
            parts.append(f"[IMAGE: {j - i} tokens]")
            i = j
        else:
            j = i
            while j < len(ids) and not (image_start_id <= ids[j] < image_start_id + num_image_tokens):
                j += 1
            parts.append(tokenizer.decode(ids[i:j], skip_special_tokens=False))
            i = j
    return "".join(parts)


@torch.no_grad()
def _sample_image_tokens(model, tokenizer, input_ids, attn_mask, n_tokens, args, proc):
    """Generate exactly n_tokens image tokens."""
    return model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        min_new_tokens=n_tokens,
        max_new_tokens=n_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        logits_processor=proc,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )


@torch.no_grad()
def _sample_text_tokens(model, tokenizer, input_ids, attn_mask, n_tokens, args, proc):
    """Generate up to n_tokens text tokens, stopping at EOS / im_end."""
    device = input_ids.device
    B = input_ids.shape[0]
    prefix_len = input_ids.shape[1]

    _im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    _eof_id    = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    stop_ids   = list({tokenizer.eos_token_id, _im_end_id, _eof_id} - {None})

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=n_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        logits_processor=proc,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_ids,
        use_cache=True,
    )

    new_tokens = output[:, prefix_len:]
    actual_len = new_tokens.shape[1]
    if actual_len < n_tokens:
        pad = torch.full(
            (B, n_tokens - actual_len), tokenizer.pad_token_id,
            dtype=torch.long, device=device,
        )
        new_tokens = torch.cat([new_tokens, pad], dim=1)

    stop_ids_t = torch.tensor(stop_ids, dtype=torch.long, device=device)
    is_stop    = torch.isin(new_tokens, stop_ids_t)
    has_stop   = is_stop.any(dim=1)
    first_stop = is_stop.long().argmax(dim=1)
    positions  = torch.arange(n_tokens, device=device).unsqueeze(0).expand(B, n_tokens)
    cutoff     = torch.where(has_stop, first_stop + 1, torch.full_like(first_stop, n_tokens))
    mask       = (positions < cutoff.unsqueeze(1)).long()

    ids_list   = new_tokens[0].tolist()
    first_stop_idx = min(
        (ids_list.index(s) for s in stop_ids if s in ids_list),
        default=len(ids_list),
    )
    text = tokenizer.decode(ids_list[:first_stop_idx], skip_special_tokens=True).strip()
    return text


# ── Phase 0: decompose the prompt into steps (disabled — user supplies steps) ───

# def _parse_steps(text):
#     steps = []
#     for line in text.splitlines():
#         line = line.strip()
#         if not line:
#             continue
#         line = re.sub(r"^\d+[\.\)]\s*", "", line)
#         line = re.sub(r"^[-\*•]\s*", "", line)
#         if line:
#             steps.append(line)
#     return steps
#
# def decompose_prompt(model, tokenizer, prompt, args, image_start_id, text_proc, device):
#     user_text  = DECOMPOSE_USER_TMPL.format(prompt=prompt)
#     input_text = (
#         f"<|im_start|>system\n{DECOMPOSE_SYSTEM}<|im_end|>\n"
#         f"<|im_start|>user\n{user_text}<|im_end|>\n"
#         f"<|im_start|>assistant\n"
#     )
#     input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
#     attn_mask = torch.ones_like(input_ids)
#     print("\n─── Phase 0 input (decompose prompt) ───")
#     print(_readable_input(tokenizer, input_ids, image_start_id, args.num_image_tokens))
#     raw = _sample_text_tokens(
#         model, tokenizer, input_ids, attn_mask, args.max_decomp_tokens, args, text_proc,
#     )
#     print(f"Decomposition:\n{raw}\n")
#     return _parse_steps(raw)


# ── Phase 1: generate the first image from s1 ─────────────────────────────────

def generate_first_image(model, tokenizer, step, args, image_start_id, image_proc, device):
    """Phase 1 — generate an image conditioned on the first sub-step only.

    Input format (via build_t2i_prefix_ids):
      <|im_start|>system\\n{system_prompt}<|im_end|>\\n
      <|im_start|>user\\n{step}<|im_end|>\\n
      <|im_start|>assistant\\n<im_start><S{scale}>
    """
    prefix_ids, prefix_mask, _ = build_t2i_prefix_ids(
        tokenizer, [step], device, scale=args.scale, system_prompt=args.system_prompt,
    )
    print(f"\n─── Phase 1 input (generate '{step}') ───")
    print(_readable_input(tokenizer, prefix_ids, image_start_id, args.num_image_tokens))

    D   = args.gen_seq_len
    ids = _sample_image_tokens(model, tokenizer, prefix_ids, prefix_mask, D, args, image_proc)
    start     = prefix_ids.shape[1]
    image_ids = ids[:, start : start + D] - image_start_id
    return image_ids


# ── Phase k: edit image to include the next step ──────────────────────────────

def edit_image(model, tokenizer, next_step, image_ids, args,
               image_start_id, image_proc, device, phase_num):
    """Phase k — generate the next image conditioned on the previous image + next sub-prompt.

    Input format:
      <|im_start|>system\\n{system_prompt}<|im_end|>\\n
      <|im_start|>user\\n
      <im_start><S{scale}>{image_tokens}<im_end> {next_step}<|im_end|>\\n
      <|im_start|>assistant\\n
      <im_start><S{scale}>
    """
    D             = args.gen_seq_len
    image_tok_ids = image_ids + image_start_id   # shift back to vocab space, (1, D)

    prefix_text = (
        f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"<im_start><S{args.scale}>"
    )
    suffix_text = (
        f"<im_end> {next_step}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<im_start><S{args.scale}>"
    )

    prefix_tok = encode_splice(tokenizer, prefix_text, device).unsqueeze(0)
    suffix_tok = encode_splice(tokenizer, suffix_text, device).unsqueeze(0)

    def ones(n):
        return torch.ones(1, n, dtype=torch.long, device=device)

    input_ids = torch.cat([prefix_tok, image_tok_ids, suffix_tok], dim=1)
    attn_mask = torch.cat([ones(prefix_tok.shape[1]), ones(D), ones(suffix_tok.shape[1])], dim=1)

    print(f"\n─── Phase {phase_num} input (edit: add '{next_step}') ───")
    print(_readable_input(tokenizer, input_ids, image_start_id, args.num_image_tokens))

    ids   = _sample_image_tokens(model, tokenizer, input_ids, attn_mask, D, args, image_proc)
    start = input_ids.shape[1]
    return ids[:, start : start + D] - image_start_id


# ── image decoding ─────────────────────────────────────────────────────────────

def decode_to_pil(visual_tok, image_ids, cfg_scale):
    """Decode (1, D) unshifted image token indices to a PIL Image."""
    imgs = visual_tok.decode_from_encoder_indices(image_ids, {"cfg_scale": cfg_scale})
    return Image.fromarray(imgs[0].numpy())


# ── composite strip ────────────────────────────────────────────────────────────

_FONT = None

def _font(size=14):
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            _FONT = ImageFont.load_default()
    return _FONT


def make_strip(prompt, steps, images, font_size=13):
    """Build a horizontal strip: header row + one image panel per step."""
    assert images, "need at least one image"
    img_w, img_h = images[0].size
    header_h = 36
    label_h  = 28
    pad      = 8

    n       = len(images)
    total_w = n * img_w + (n - 1) * pad
    total_h = header_h + img_h + label_h

    canvas = Image.new("RGB", (total_w, total_h), (240, 240, 240))
    draw   = ImageDraw.Draw(canvas)
    font   = _font(font_size)

    draw.text((8, 8), f'Prompt: "{prompt}"', fill=(10, 10, 10), font=font)

    for i, (img, step) in enumerate(zip(images, steps)):
        x = i * (img_w + pad)
        canvas.paste(img, (x, header_h))
        label = f"Step {i + 1}: {step}"
        if len(label) > 52:
            label = label[:49] + "…"
        draw.text((x + 4, header_h + img_h + 4), label, fill=(40, 40, 40), font=font)

    return canvas


# ── main ───────────────────────────────────────────────────────────────────────

def _save_step(run_dir, phase_num, step_text, img):
    """Save one step's image and sub-prompt text into run_dir."""
    safe = re.sub(r"[^\w\s-]", "_", step_text).strip()[:40]
    base = os.path.join(run_dir, f"step_{phase_num:02d}_{safe}")
    img.save(f"{base}.png")
    with open(f"{base}.txt", "w") as f:
        f.write(step_text + "\n")
    print(f"  saved → {base}.png  +  .txt")


def main():
    args = parse_args()

    if len(args.sub_prompts) < 2:
        raise ValueError("--sub_prompts needs at least 2 entries: the full prompt and one sub-step.")

    full_prompt = args.sub_prompts[0]   # used only to name the output directory
    steps       = args.sub_prompts[1:]  # sequential generation sub-prompts

    tokenizer, model, visual_tok, device = load_models(args)

    image_start_id = tokenizer.convert_tokens_to_ids("<I0>")
    if image_start_id is None or image_start_id == tokenizer.unk_token_id:
        raise RuntimeError("<I0> not in tokenizer — checkpoint must include image vocab.")

    image_proc = LogitsProcessorList([ImageVocabLogitsProcessor(image_start_id, args.num_image_tokens)])

    print(f"\nFull prompt : {full_prompt!r}")
    print(f"Steps ({len(steps)}):")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")

    # Create a sub-directory named after the full prompt
    safe_prompt = re.sub(r"[^\w\s-]", "_", full_prompt).strip()[:60]
    run_dir = os.path.join(args.out_dir, safe_prompt)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nSaving to {run_dir}/")

    # Phase 1: first image from s1
    image_ids = generate_first_image(
        model, tokenizer, steps[0], args, image_start_id, image_proc, device,
    )
    img = decode_to_pil(visual_tok, image_ids, args.cfg_scale)
    _save_step(run_dir, 1, steps[0], img)
    images = [img]

    # Phase 2+: iterative edits
    for k in range(1, len(steps)):
        image_ids = edit_image(
            model, tokenizer,
            next_step=steps[k],
            image_ids=image_ids,
            args=args,
            image_start_id=image_start_id,
            image_proc=image_proc,
            device=device,
            phase_num=k + 1,
        )
        img = decode_to_pil(visual_tok, image_ids, args.cfg_scale)
        _save_step(run_dir, k + 1, steps[k], img)
        images.append(img)

    # Save composite strip into the same sub-directory
    strip      = make_strip(full_prompt, steps, images)
    strip_path = os.path.join(run_dir, "strip.png")
    strip.save(strip_path)
    print(f"\nStrip saved → {strip_path}")


if __name__ == "__main__":
    main()
