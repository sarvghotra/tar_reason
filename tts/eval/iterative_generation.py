"""Generate images iteratively with self-reflection.

The model first generates an image for the prompt, then self-reflects on it
in text, then generates a refined second image conditioned on the full context.

Conversation format in the assistant turn:
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    <image>
    Self-reflect: {text}
    <image>

where <image> = <im_start><S{scale}>{tokens}<im_end>.

Directory layout:
    {out_dir}/
        {first_8_words_of_prompt}/
            image_1.png        — first generated image
            image_2.png        — second generated image (after self-reflection)
            self_reflect.txt   — text generated after "Self-reflect: "

Input file formats accepted:
  - JSONL  — one JSON object per line with a "prompt" key
  - Plain text — one prompt per line

Usage:
    python tts/eval/iterative_generation.py \\
        --model <path-or-hf-id> \\
        --prompts_file <path-to-prompts> \\
        [--out_dir results/iterative] \\
        [--ar_path ...] [--encoder_path ...] [--decoder_path ...] \\
        [--gen_seq_len 729] [--scale 0] [--cfg_scale 4.0] \\
        [--num_image_tokens 65536] \\
        [--reflect_tokens 256] \\
        [--temperature 1.0] [--top_p 0.95] [--top_k 1200] \\
        [--device cuda:0] [--bf16]
"""

import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoTokenizer, LogitsProcessorList, Qwen2ForCausalLM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="HF model ID or local path (Qwen2-based Tar checkpoint).")
    p.add_argument("--prompts_file", type=str, required=True,
                   help="JSONL (with 'prompt' key) or plain-text file of prompts.")
    p.add_argument("--out_dir", type=str, default="results/iterative",
                   help="Root directory for output.")
    p.add_argument("--ar_path", type=str, default=None)
    p.add_argument("--encoder_path", type=str, default=None)
    p.add_argument("--decoder_path", type=str, default=None)
    p.add_argument("--gen_seq_len", type=int, default=729,
                   help="Number of image tokens to generate per image.")
    p.add_argument("--scale", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--num_image_tokens", type=int, default=65536)
    p.add_argument("--reflect_tokens", type=int, default=256,
                   help="Max new tokens for the self-reflection text.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=1200)
    p.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--bf16", action="store_true", default=True)
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


def load_models(args):
    from huggingface_hub import hf_hub_download
    from tok.mm_autoencoder import MMAutoEncoder

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


def _make_safe_name(text: str, max_words: int = None, max_chars: int = 60) -> str:
    if max_words:
        text = " ".join(text.split()[:max_words])
    return re.sub(r"[^\w\s-]", "_", text).strip()[:max_chars]


@torch.no_grad()
def generate_iterative(model, tokenizer, visual_tok, prompt, args, device):
    """Run the three-phase iterative generation for one prompt.

    Phase 1: generate first image from prompt.
    Phase 2: append <im_end>\\nSelf-reflect:  and generate text.
    Phase 3: append \\n<im_start><S{scale}> and generate second image.

    Returns:
        img1 (PIL.Image), img2 (PIL.Image), reflect_text (str)
    """
    from PIL import Image
    from llava.train.rl.grpo import (
        ImageVocabLogitsProcessor,
        NonImageLogitsProcessor,
        build_t2i_prefix_ids,
        encode_splice,
    )

    D = args.gen_seq_len
    image_start_id = tokenizer.convert_tokens_to_ids("<I0>")
    if image_start_id is None or image_start_id == tokenizer.unk_token_id:
        raise RuntimeError("<I0> not in tokenizer — checkpoint must include image vocab.")

    image_proc = LogitsProcessorList([ImageVocabLogitsProcessor(image_start_id, args.num_image_tokens)])
    text_proc  = LogitsProcessorList([NonImageLogitsProcessor(image_start_id, args.num_image_tokens)])

    # stop ids for text generation
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eof_id    = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    im_start_tok_id = tokenizer.convert_tokens_to_ids("<im_start>")
    stop_ids  = list({tokenizer.eos_token_id, im_end_id, eof_id, im_start_tok_id} - {None})

    # ── Phase 1: generate first image ─────────────────────────────────────────
    # Prefix ends with: ...<|im_start|>assistant\n<im_start><S{scale}>
    prefix_ids, prefix_mask, _ = build_t2i_prefix_ids(
        tokenizer, [prompt], device, scale=args.scale, system_prompt=args.system_prompt,
    )

    print(f"\n─── Phase 1: generating first image ───")
    input_ph1 = tokenizer.decode(prefix_ids[0])
    print(input_ph1)
    print()

    out1 = model.generate(
        input_ids=prefix_ids,
        attention_mask=prefix_mask,
        min_new_tokens=D,
        max_new_tokens=D,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        logits_processor=image_proc,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    p1_len = prefix_ids.shape[1]
    image1_ids = out1[:, p1_len : p1_len + D] - image_start_id  # unshifted, (1, D)

    img1 = Image.fromarray(
        visual_tok.decode_from_encoder_indices(image1_ids, {"cfg_scale": args.cfg_scale})[0].numpy()
    )

    # ── Phase 2: generate self-reflection text ────────────────────────────────
    # Full context: ...\n<|im_start|>assistant\n<im_start><S{scale}>{image1}<im_end>\nSelf-reflect:
    image1_vocab = image1_ids + image_start_id  # vocab-space, (1, D)

    asst_prefix_text = (
        f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<im_start><S{args.scale}>"
    )
    reflect_splice_text = f"<im_end>\nSelf-reflect: "

    asst_prefix_tok  = encode_splice(tokenizer, asst_prefix_text, device).unsqueeze(0)
    reflect_splice_tok = encode_splice(tokenizer, reflect_splice_text, device).unsqueeze(0)

    context2 = torch.cat([asst_prefix_tok, image1_vocab, reflect_splice_tok], dim=1)
    attn2    = torch.ones_like(context2)
    p2_len   = context2.shape[1]

    print(f"\n─── Phase 2: generating self-reflection text ───")
    input_ph2 = tokenizer.decode(context2[0])
    print(input_ph2)
    print()
    out2 = model.generate(
        input_ids=context2,
        attention_mask=attn2,
        max_new_tokens=args.reflect_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        logits_processor=text_proc,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_ids,
        use_cache=True,
    )

    new_ids = out2[0, p2_len:].tolist()
    cutoff  = min((new_ids.index(s) for s in stop_ids if s in new_ids), default=len(new_ids))
    reflect_text = tokenizer.decode(new_ids[:cutoff], skip_special_tokens=True).strip()
    print(f"Self-reflect: {reflect_text}")

    reflect_text_ids = out2[:, p2_len : p2_len + cutoff]  # (1, cutoff), excludes stop token

    # ── Phase 3: generate second image ────────────────────────────────────────
    # Append \n<im_start><S{scale}> to start the second image generation
    img2_splice_tok = encode_splice(tokenizer, f"\n<im_start><S{args.scale}>", device).unsqueeze(0)

    context3 = torch.cat([context2, reflect_text_ids, img2_splice_tok], dim=1)
    attn3    = torch.ones_like(context3)
    p3_len   = context3.shape[1]

    print(f"\n─── Phase 3: generating second image ───")
    input_ph3 = tokenizer.decode(context3[0])
    print(input_ph3)
    print()
    out3 = model.generate(
        input_ids=context3,
        attention_mask=attn3,
        min_new_tokens=D,
        max_new_tokens=D,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        logits_processor=image_proc,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    image2_ids = out3[:, p3_len : p3_len + D] - image_start_id

    img2 = Image.fromarray(
        visual_tok.decode_from_encoder_indices(image2_ids, {"cfg_scale": args.cfg_scale})[0].numpy()
    )

    return img1, img2, reflect_text


def main():
    args = parse_args()

    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    tokenizer, model, visual_tok, device = load_models(args)

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt!r}")

        out_dir = os.path.join(args.out_dir, _make_safe_name(prompt, max_words=8))
        os.makedirs(out_dir, exist_ok=True)

        img1, img2, reflect_text = generate_iterative(
            model, tokenizer, visual_tok, prompt, args, device,
        )

        img1.save(os.path.join(out_dir, "image_1.png"))
        img2.save(os.path.join(out_dir, "image_2.png"))
        with open(os.path.join(out_dir, "self_reflect.txt"), "w") as f:
            f.write(f"Prompt: {prompt}\n\nSelf-reflect: {reflect_text}\n")

        print(f"  Saved → {out_dir}/image_1.png, image_2.png, self_reflect.txt")


if __name__ == "__main__":
    main()
