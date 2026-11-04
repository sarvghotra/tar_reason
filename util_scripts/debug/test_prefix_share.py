"""GPU check that Judge.score_shared (image prefix once per image, questions from its KV cache) gives the
same probabilities as Judge.score_rows (one full forward per (image, question) row) for a judge, and how
much faster it is. Forces the shared path on even when the profile has prefix_share=False, so a new
architecture can be validated before its profile is switched on.
    <judge env>/bin/python util_scripts/debug/test_prefix_share.py <model_dir> [--image_soft_tokens 280]
        [--answer_id_mode strict|geneval2] [--n_images 8] [--questions_per_image 8]
Prints max |shared - rows| over all probabilities (expect ~1e-3 or less in bf16) and the two timings.
"""
import argparse
import random
import sys
import time
sys.path.insert(0, ".")
import torch                                                     # noqa: E402
from PIL import Image, ImageDraw                                 # noqa: E402
from llava.train.rl.pixel_reward_server import Judge             # noqa: E402

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "pink"]
SHAPES = ["square", "circle", "triangle"]


def draw(seed):
    rng = random.Random(seed)
    im = Image.new("RGB", (512, 512), rng.choice(["white", "lightgray", "beige"]))
    d = ImageDraw.Draw(im)
    objs = []
    for _ in range(rng.randint(1, 3)):
        shape, color = rng.choice(SHAPES), rng.choice(COLORS)
        x, y, r = rng.randint(60, 400), rng.randint(60, 400), rng.randint(40, 100)
        if shape == "square":
            d.rectangle([x, y, x + r, y + r], fill=color)
        elif shape == "circle":
            d.ellipse([x, y, x + r, y + r], fill=color)
        else:
            d.polygon([(x, y + r), (x + r, y + r), (x + r // 2, y)], fill=color)
        objs.append((shape, color))
    return im, objs


def questions(objs, k, rng):
    qs = []
    for shape, color in objs:
        qs += [(f"Is there a {shape} in the image?", "Yes"), (f"Is the {shape} {color}?", "Yes"),
               (f"Is the {shape} {rng.choice(COLORS)}?", "Yes"),
               (f"How many {shape}s are in the image?", rng.choice(["one", "two", "three"])),
               (f"Is there a {rng.choice(['cat', 'dog', 'tree', 'car'])} in the image?", "No")]
    rng.shuffle(qs)
    return qs[:k]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--image_soft_tokens", type=int, default=None)
    p.add_argument("--answer_id_mode", default="strict", choices=["strict", "geneval2"])
    p.add_argument("--answer_suffix", default="geneval2")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_images", type=int, default=8)
    p.add_argument("--questions_per_image", type=int, default=8)
    args = p.parse_args()
    t0 = time.time()
    judge = Judge(args.model, args.answer_suffix, args.batch_size, torch.device("cuda"), args.attn,
                  image_soft_tokens=args.image_soft_tokens, answer_id_mode=args.answer_id_mode,
                  share_image_prefix=True)
    print(f"loaded {judge.model_type} in {time.time() - t0:.0f}s; profile prefix_share={judge.profile['prefix_share']} "
          f"prefix_end={judge.profile['prefix_end']!r} mrope={judge.profile['mrope']}", flush=True)
    judge.share_image_prefix = True          # force the shared path for the check
    rng = random.Random(0)
    images, vqas = [], []
    for i in range(args.n_images):
        im, objs = draw(i)
        images.append(im)
        vqas.append(questions(objs, args.questions_per_image, rng))
    n_rows = sum(len(v) for v in vqas)
    for name in ("score_rows", "score_shared"):       # warm-up, then timed
        getattr(judge, name)(images[:2], vqas[:2])
    torch.cuda.synchronize(); t0 = time.time()
    rows = judge.score_rows(images, vqas)
    torch.cuda.synchronize(); t_rows = time.time() - t0; t0 = time.time()
    shared = judge.score_shared(images, vqas)
    torch.cuda.synchronize(); t_shared = time.time() - t0
    assert shared is not None, "score_shared returned None (ragged prefixes)"
    diffs = [abs(a - b) for ra, rb in zip(rows, shared) for a, b in zip(ra, rb)]
    print(f"{len(images)} images, {n_rows} rows: score_rows {t_rows:.1f}s, score_shared {t_shared:.1f}s "
          f"(x{t_rows / max(t_shared, 1e-6):.1f}); max |diff| {max(diffs):.5f}, mean {sum(diffs) / len(diffs):.6f}")
    for i in range(min(2, len(images))):
        for (q, a), pr, ps in zip(vqas[i], rows[i], shared[i]):
            print(f"  img{i} {q!r:52} {a!r:8} rows {pr:.4f} shared {ps:.4f}")
    # bf16 reference: jeet's Qwen3-VL-8B shared path (what E-450 / run 1 trained with) gives mean 0.0015, max 0.034
    # against score_rows on this test (Rorqual 2026-11-03), so accept that level of rounding noise.
    ok = sum(diffs) / len(diffs) < 0.005 and max(diffs) < 0.06
    print("PREFIX SHARE OK" if ok else "PREFIX SHARE MISMATCH", flush=True)


if __name__ == "__main__":
    main()
