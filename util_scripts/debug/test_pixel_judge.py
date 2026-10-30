"""GPU smoke test for a pixel-reward judge (llava/train/rl/pixel_reward_server.py's Judge class).

Draws two synthetic images (a red square, two blue circles) and scores GenEval2-style questions on
them, printing P(answer). A judge is usable when the right answers come out well above the wrong ones.

    <judge env>/bin/python util_scripts/debug/test_pixel_judge.py <model_dir> [--image_soft_tokens 280]
        [--answer_id_mode strict|geneval2] [--attn sdpa]
"""

import argparse
import sys
import time

sys.path.insert(0, ".")
import torch                                                     # noqa: E402
from PIL import Image, ImageDraw                                 # noqa: E402

from llava.train.rl.pixel_reward_server import Judge             # noqa: E402


def red_square():
    im = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(im).rectangle([128, 128, 384, 384], fill="red")
    return im


def two_blue_circles():
    im = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(im)
    d.ellipse([60, 180, 220, 340], fill="blue")
    d.ellipse([290, 180, 450, 340], fill="blue")
    return im


CASES = [
    (red_square, [("Is there a square in the image?", "Yes"), ("Is the square red?", "Yes"),
                  ("Is the square blue?", "Yes"), ("Is there a cat in the image?", "Yes"),
                  ("How many squares are in the image?", "one"), ("How many squares are in the image?", "three"),
                  ("What color is the square?", "red"), ("What color is the square?", "green")]),
    (two_blue_circles, [("How many circles are in the image?", "two"), ("How many circles are in the image?", "one"),
                        ("Are the circles blue?", "Yes"), ("Are the circles red?", "Yes"),
                        ("Is the left circle larger than the right one?", "Yes"),
                        ("Is there a dog in the image?", "No")]),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--image_soft_tokens", type=int, default=None)
    p.add_argument("--answer_id_mode", default="strict", choices=["strict", "geneval2"])
    p.add_argument("--answer_suffix", default="geneval2")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    t0 = time.time()
    judge = Judge(args.model, args.answer_suffix, args.batch_size, torch.device("cuda"), args.attn,
                  image_soft_tokens=args.image_soft_tokens, answer_id_mode=args.answer_id_mode)
    print(f"loaded {args.model} ({judge.model_type}, {type(judge.model).__name__}) in {time.time() - t0:.0f}s; "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    images = [make() for make, _ in CASES]
    vqa = [qas for _, qas in CASES]
    t0 = time.time()
    scores = judge.score(images, vqa)
    print(f"scored {sum(len(v) for v in vqa)} questions in {time.time() - t0:.1f}s", flush=True)
    for (make, qas), sc in zip(CASES, scores):
        print(f"--- {make.__name__}")
        for (q, a), s in zip(qas, sc):
            print(f"  P({a!r:8}| {q}) = {s:.3f}")


if __name__ == "__main__":
    main()
