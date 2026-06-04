"""Generate images in one shot from prompts using Tar — no decomposition.

Each prompt is passed directly to the image-generation model without any
step-by-step decomposition.

Directory layout (--generate_images):
    {out_dir}/
        {first_8_words_of_prompt}/
            image.png
            prompt.txt

Input file formats accepted:
  - JSONL  — one JSON object per line with a "prompt" key
  - Plain text — one prompt per line

Usage:
    python tts/eval/1shot_gen.py \\
        --model <path-or-hf-id> \\
        --prompts_file <path-to-prompts> \\
        [--generate_images] \\
        [--out_dir results/1shot] \\
        [--ar_path ...] [--encoder_path ...] [--decoder_path ...] \\
        [--gen_seq_len 729] [--scale 0] [--cfg_scale 4.0] \\
        [--num_image_tokens 65536] \\
        [--temperature 1.0] [--top_p 0.95] [--top_k 1200] \\
        [--device cuda:0] [--bf16]
"""

import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="HF model ID or local path (Qwen2-based Tar checkpoint).")
    p.add_argument("--prompts_file", type=str, required=True,
                   help="JSONL (with 'prompt' key) or plain-text file of prompts.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--bf16", action="store_true", default=True)
    # image generation
    p.add_argument("--generate_images", action="store_true", default=False,
                   help="Generate an image for each prompt.")
    p.add_argument("--out_dir", type=str, default="results/1shot",
                   help="Root directory for image output.")
    p.add_argument("--ar_path", type=str, default=None)
    p.add_argument("--encoder_path", type=str, default=None)
    p.add_argument("--decoder_path", type=str, default=None)
    p.add_argument("--gen_seq_len", type=int, default=729,
                   help="Number of image tokens to generate.")
    p.add_argument("--scale", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--num_image_tokens", type=int, default=65536)
    p.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    return p.parse_args()


def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompts.append(obj["prompt"])
            except (json.JSONDecodeError, KeyError):
                prompts.append(line)
    return prompts


def load_model(args):
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = torch.device(args.device)
    print(f"Loading model from {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen2ForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    return tokenizer, model, device


def _make_safe_name(text: str, max_words: int = None, max_chars: int = 60) -> str:
    if max_words:
        text = " ".join(text.split()[:max_words])
    return re.sub(r"[^\w\s-]", "_", text).strip()[:max_chars]


def load_visual_tokenizer(args, dtype, device):
    from huggingface_hub import hf_hub_download
    from tok.mm_autoencoder import MMAutoEncoder

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
    return visual_tok


@torch.no_grad()
def generate_image(model, tokenizer, visual_tok, prompt, args, device):
    from transformers import LogitsProcessorList
    from llava.train.rl.grpo import ImageVocabLogitsProcessor
    from tts.generate_step_by_step import generate_first_image, decode_to_pil

    image_start_id = tokenizer.convert_tokens_to_ids("<I0>")
    if image_start_id is None or image_start_id == tokenizer.unk_token_id:
        raise RuntimeError("<I0> not in tokenizer — checkpoint must include image vocab.")

    image_proc = LogitsProcessorList([ImageVocabLogitsProcessor(image_start_id, args.num_image_tokens)])

    out_dir = os.path.join(args.out_dir, _make_safe_name(prompt, max_words=8))
    os.makedirs(out_dir, exist_ok=True)

    image_ids = generate_first_image(
        model, tokenizer, prompt, args, image_start_id, image_proc, device)

    img = decode_to_pil(visual_tok, image_ids, args.cfg_scale)
    img.save(os.path.join(out_dir, "image.png"))
    with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
        f.write(prompt + "\n")
    print(f"  Saved → {out_dir}/image.png")


def main():
    args = parse_args()

    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    tokenizer, model, device = load_model(args)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    visual_tok = load_visual_tokenizer(args, dtype, device) if args.generate_images else None

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt!r}")
        if visual_tok is not None:
            generate_image(model, tokenizer, visual_tok, prompt, args, device)


if __name__ == "__main__":
    main()
