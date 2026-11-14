"""Self-judging for rollback: score every image of an iterative TIIF-Bench run with the policy itself.

For each image of each chain (``<run>/seed_S/images/<dim>/<tag>_iter<k>/<desc>/<idx>.png``) the policy (base +
LoRA, HF) gets two teacher-forced scores, both from the image re-encoded into TA-Tok indices (scale 0, 729 tokens;
the run kept only pixels, so these are not the exact sampled codes):

* ``approve_lp``: log-probability of its own approval critique, in the MDP refinement context it was trained on,
      <prefix><im_start><S0>{image}<im_end>\\nSelf-reflect: + " no issues, it matches the prompt.\\nCorrection: looks good."
  (444 / 500 of t4's approvals on TIIF use exactly this text; the rest add a blank line). ``approve_lp_mean`` is
  the per-token mean.
* ``vqa_yes``: P(Yes) / (P(Yes) + P(No)) for a plain understanding turn (Tar's i2t format, bare image tokens):
  "{image}\\nDoes this image match the following description? {prompt}\\nAnswer Yes or No."

Identical image files (chains stopped by "looks good" repeat their last image) are scored once.
Writes ``<out_dir>/seed_S.jsonl``: {"key": "<dim>/<desc>/<idx>", "iter": k, "md5", "approve_lp", "approve_lp_mean",
"vqa_yes"}. Resumes: keys already in the file are skipped.

Run in the tar env (torch 2.1, transformers 4.50, peft) on one GPU.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from tok.ta_tok import TextAlignedTokenizer  # noqa: E402

ITERATIVE_PROMPT_PREFIX = "Generate an image iteratively by self-reflecting and correcting.\n"
SYSTEM_PROMPT = "You are a helpful assistant."
APPROVAL = " no issues, it matches the prompt.\nCorrection: looks good."
VQA_QUESTION = "Does this image match the following description? {prompt}\nAnswer Yes or No."


def load(args, device):
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen2ForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=torch.bfloat16).merge_and_unload()
    model = model.to(device).eval()
    ta_tok = TextAlignedTokenizer.from_checkpoint(args.ta_tok, load_teacher=False, input_type="indices",
                                                  teacher=args.siglip2).to(device).eval()
    return tokenizer, model, ta_tok


@torch.no_grad()
def image_text(ta_tok, paths, device):
    x = torch.stack([to_tensor(Image.open(p).convert("RGB")) for p in paths]).to(device)
    codes = ta_tok.encode(x, pool_scale=1)["bottleneck_rep"]
    assert codes.shape[1] == 729, codes.shape
    return ["".join(f"<I{c}>" for c in row) for row in codes.cpu().tolist()]


@torch.no_grad()
def target_logprobs(model, tokenizer, contexts, target_ids, device):
    """Sum and mean log p(target | context) per row (right padding, one target per row)."""
    rows = [tokenizer(c, add_special_tokens=False).input_ids + t for c, t in zip(contexts, target_ids)]
    n = max(map(len, rows))
    pad = tokenizer.pad_token_id or 0
    ids = torch.tensor([r + [pad] * (n - len(r)) for r in rows], device=device)
    mask = torch.tensor([[1] * len(r) + [0] * (n - len(r)) for r in rows], device=device)
    logits = model(input_ids=ids, attention_mask=mask).logits.float()
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    out = []
    for b, (r, t) in enumerate(zip(rows, target_ids)):
        start = len(r) - len(t)
        lp = logp[b, start - 1:len(r) - 1].gather(-1, torch.tensor(t, device=device)[:, None]).squeeze(-1)
        out.append((lp.sum().item(), lp.mean().item()))
    return out


@torch.no_grad()
def yes_prob(model, tokenizer, contexts, yes_id, no_id, device):
    tokenizer.padding_side = "left"
    enc = tokenizer(contexts, add_special_tokens=False, padding=True, return_tensors="pt")
    logits = model(input_ids=enc.input_ids.to(device), attention_mask=enc.attention_mask.to(device)).logits[:, -1]
    two = torch.log_softmax(logits[:, [yes_id, no_id]].float(), dim=-1)
    return two[:, 0].exp().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="TIIF run dir holding seed_*/")
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 17, 91])
    ap.add_argument("--iter_tag", default="E-450_ORC", help="images/<dim>/<iter_tag>_iter<k>/")
    ap.add_argument("--n_iters", type=int, default=4)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora_path", required=True)
    ap.add_argument("--ta_tok", required=True)
    ap.add_argument("--siglip2", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N images per seed")
    args = ap.parse_args()

    device = torch.device("cuda")
    tokenizer, model, ta_tok = load(args, device)
    approval_ids = tokenizer(APPROVAL, add_special_tokens=False).input_ids
    yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
    no_id = tokenizer("No", add_special_tokens=False).input_ids[0]
    os.makedirs(args.out_dir, exist_ok=True)

    for seed in args.seeds:
        seed_dir = Path(args.run_dir) / f"seed_{seed}"
        prompts = {k: v["prompt"] for k, v in json.load(open(seed_dir / "self_reflect.json")).items()}
        out_path = Path(args.out_dir) / f"seed_{seed}.jsonl"
        done = set()
        if out_path.exists():
            done = {(r["key"], r["iter"]) for r in map(json.loads, open(out_path))}
        items = [(key, k, seed_dir / "images" / key.split("/")[0] / f"{args.iter_tag}_iter{k}" /
                  key.split("/")[1] / f"{key.split('/')[2]}.png")
                 for key in sorted(prompts) for k in range(1, args.n_iters + 1) if (key, k) not in done]
        items = items[:args.limit] if args.limit else items
        print(f"seed {seed}: {len(items)} images to score", flush=True)
        cache = {}
        with open(out_path, "a") as fout:
            for s in range(0, len(items), args.batch_size):
                batch = items[s:s + args.batch_size]
                md5s = [hashlib.md5(open(p, "rb").read()).hexdigest() for _, _, p in batch]
                todo = [i for i, (b, m) in enumerate(zip(batch, md5s)) if (b[0], m) not in cache]
                if todo:
                    texts = image_text(ta_tok, [batch[i][2] for i in todo], device)
                    ctx = [tokenizer.apply_chat_template(
                               [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": ITERATIVE_PROMPT_PREFIX + prompts[batch[i][0]]}],
                               tokenize=False, add_generation_prompt=True)
                           + f"<im_start><S0>{t}<im_end>\nSelf-reflect:" for i, t in zip(todo, texts)]
                    lps = target_logprobs(model, tokenizer, ctx, [approval_ids] * len(todo), device)
                    vqa_ctx = [tokenizer.apply_chat_template(
                                   [{"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": f"{t}\n" + VQA_QUESTION.format(prompt=prompts[batch[i][0]])}],
                                   tokenize=False, add_generation_prompt=True) for i, t in zip(todo, texts)]
                    ys = yes_prob(model, tokenizer, vqa_ctx, yes_id, no_id, device)
                    for i, (lp, y) in zip(todo, zip(lps, ys)):
                        cache[(batch[i][0], md5s[i])] = {"approve_lp": lp[0], "approve_lp_mean": lp[1], "vqa_yes": y}
                for (key, k, _), m in zip(batch, md5s):
                    fout.write(json.dumps({"key": key, "iter": k, "md5": m, **cache[(key, m)]}) + "\n")
                fout.flush()
                if (s // args.batch_size) % 20 == 0:
                    print(f"seed {seed}: {s + len(batch)}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
