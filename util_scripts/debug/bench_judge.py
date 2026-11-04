"""Throughput of a pixel-reward judge as the reward server uses it (llava/train/rl/pixel_reward_server.py's Judge):
requests of --images_per_request images, each with its real GenEval2-style question list from the training data,
scored by Judge.score (shared image prefix when the profile allows it and --share_image_prefix is on, else one
forward per (image, question) row). Images are synthetic 512x512 (Tar's decode size), so only timing is meaningful;
use test_pixel_judge.py for answer sanity. Prints s per request, images/s, questions/s and peak GPU memory, plus the
exact prompt text of one row (to check the chat template).
    <judge env>/bin/python util_scripts/debug/bench_judge.py <model_dir> [--answer_id_mode strict|geneval2]
        [--image_soft_tokens 280] [--share_image_prefix] [--n_images 120] [--data <jsonl with vqa_list>]
"""
import argparse
import json
import random
import sys
import time
sys.path.insert(0, ".")
import torch                                                     # noqa: E402
from PIL import Image, ImageDraw                                 # noqa: E402
from llava.train.rl.pixel_reward_server import Judge             # noqa: E402


def synthetic(seed):
    rng = random.Random(seed)
    im = Image.new("RGB", (512, 512), tuple(rng.randint(150, 255) for _ in range(3)))
    d = ImageDraw.Draw(im)
    for _ in range(rng.randint(2, 6)):
        x, y, r = rng.randint(0, 440), rng.randint(0, 440), rng.randint(30, 140)
        col = tuple(rng.randint(0, 255) for _ in range(3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([x, y, x + r, y + r], fill=col)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--data", default="data/promptmix/train_17K.jsonl")
    p.add_argument("--n_images", type=int, default=120)
    p.add_argument("--images_per_request", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--image_soft_tokens", type=int, default=None)
    p.add_argument("--answer_id_mode", default="strict", choices=["strict", "geneval2"])
    p.add_argument("--answer_suffix", default="geneval2")
    p.add_argument("--share_image_prefix", action="store_true")
    p.add_argument("--force_share", action="store_true",
                   help="use score_shared even if the profile has prefix_share=False (to time it before switching it on)")
    p.add_argument("--attn", default="sdpa")
    args = p.parse_args()
    rows = [json.loads(l) for _, l in zip(range(args.n_images), open(args.data))]
    vqas = [[tuple(q) for q in r["vqa_list"]] for r in rows]
    images = [synthetic(i) for i in range(len(vqas))]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    judge = Judge(args.model, args.answer_suffix, args.batch_size, torch.device("cuda"), args.attn,
                  image_soft_tokens=args.image_soft_tokens, answer_id_mode=args.answer_id_mode,
                  share_image_prefix=args.share_image_prefix)
    load_s = time.time() - t0
    if args.force_share:
        judge.share_image_prefix = True
    msgs = [[{"role": "user", "content": [{"type": "image", "image": images[0]},
                                          {"type": "text", "text": judge.question_text(vqas[0][0][0])}]}]]
    ids = judge._template_inputs(msgs)["input_ids"][0]
    text = judge.processor.tokenizer.decode(ids)
    n_img_tok = int((ids == getattr(judge.model.config, "image_token_id", -1)).sum())
    print(f"model {args.model}\n  type {judge.model_type} ({type(judge.model).__name__}), loaded in {load_s:.0f}s, "
          f"shared prefix {judge.share_image_prefix}, row length {len(ids)} tokens ({n_img_tok} image tokens)")
    import re
    print("  prompt:", repr(re.sub(r"(<\|image_pad\|>|<\|image\|>)+", lambda m: f"<image x{len(m.group(0)) // len(m.group(1))}>", text)))
    print("  answer ids for", vqas[0][0], "->", judge.answer_ids(*vqas[0][0]),
          [judge.processor.tokenizer.decode([i]) for i in judge.answer_ids(*vqas[0][0])])
    req = args.images_per_request
    judge.score(images[:req], vqas[:req])                        # warm-up
    torch.cuda.synchronize()
    t0, n_q = time.time(), 0
    probs = []
    for s in range(0, len(images), req):
        out = judge.score(images[s:s + req], vqas[s:s + req])
        probs += [x for o in out for x in o]
        n_q += sum(len(v) for v in vqas[s:s + req])
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_req = (len(images) + req - 1) // req
    print(f"  RESULT {judge.model_type}: {len(images)} images / {n_q} questions in {dt:.1f}s -> "
          f"{dt / n_req:.2f} s per {req}-image request, {len(images) / dt:.2f} images/s, {n_q / dt:.1f} questions/s; "
          f"peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; mean P {sum(probs) / len(probs):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
