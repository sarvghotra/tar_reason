"""Self-judging on an iterative GenEval2 run: the policy scores its own images with GenEval2's own questions.

GenEval2's questions (``vqa_list`` in each prompt's metadata.jsonl) are used, their expected answers never are.
Many questions are not yes/no ("How many backpacks are in the image?" -> "one"), so the two variants are:

* ``am_ans`` (predicted answer): from the prompt text alone, the policy first writes the one-word answer a matching
  image should give (greedy); then for each image the score of a question is P(that answer) at the next token after
  "{image}\\n{question} Answer in one word." (summed over its surface forms: case, leading space, digit for number
  words), as GenEval2's soft-TIFA scores the expected answer. ``am_ans`` = mean over the questions.
* ``am_rephr`` (rewritten questions): from the prompt text alone, the policy rewrites each question as a yes/no
  question whose answer is Yes exactly when the prompt is satisfied ("Is there exactly one backpack in the image?");
  score = P(Yes) / (P(Yes) + P(No)); ``am_rephr`` = mean over the questions. Outputs that are not questions fall back
  to the original question.

Images: ``<run>/seed_S/<k>/<NNNNN>/samples/00000.png``. Writes ``<out_dir>/seed_S.jsonl``:
{"key": NNNNN, "iter": k, "md5", "am_ans", "am_rephr", "p_ans", "p_rephr", "answers", "rephrased"}; resumes.
Run in the tar env on one GPU.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from eval.tiif_self_judge import SYSTEM_PROMPT, image_text, load, yes_prob  # noqa: E402

ANSWER_Q = ("Text-to-image prompt: {prompt}\nQuestion about an image: {question}\n"
            "If an image matched the prompt exactly, what would the answer to the question be? Answer in one word.")
REPHRASE_Q = ("Text-to-image prompt: {prompt}\nQuestion about an image: {question}\n"
              "Rewrite the question as a yes/no question whose answer is Yes exactly when the image matches the prompt "
              "(for example, turn a 'how many' question into 'Is there exactly ...?'). Output only the question.")
NUMBERS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}


def chat(tokenizer, text):
    return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": text}],
                                         tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tokenizer, texts, max_new_tokens, device):
    tokenizer.padding_side = "left"
    enc = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt").to(device)
    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen[:, enc.input_ids.shape[1]:]]


def answer_token_ids(tokenizer, answer):
    a = answer.strip().strip(".").split()[0] if answer.strip() else "Yes"
    forms = {a, a.lower(), a.capitalize()}
    if a.lower() in NUMBERS:
        forms.add(NUMBERS[a.lower()])
    forms |= {" " + f for f in forms}
    return sorted({tokenizer(f, add_special_tokens=False).input_ids[0] for f in forms})


@torch.no_grad()
def answer_prob(model, tokenizer, contexts, id_lists, device):
    tokenizer.padding_side = "left"
    enc = tokenizer(contexts, add_special_tokens=False, padding=True, return_tensors="pt")
    logits = model(input_ids=enc.input_ids.to(device), attention_mask=enc.attention_mask.to(device)).logits[:, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    return [probs[i, ids].sum().item() for i, ids in enumerate(id_lists)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="GenEval2 run dir holding seed_*/")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2213, 713, 13])
    ap.add_argument("--n_iters", type=int, default=4)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora_path", default="")
    ap.add_argument("--ta_tok", required=True)
    ap.add_argument("--siglip2", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    tokenizer, model, ta_tok = load(args, device)
    yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
    no_id = tokenizer("No", add_special_tokens=False).input_ids[0]
    os.makedirs(args.out_dir, exist_ok=True)

    meta = None
    for seed in args.seeds:
        seed_dir = Path(args.run_dir) / f"seed_{seed}"
        keys = sorted(d.name for d in (seed_dir / "1").iterdir() if d.is_dir() and d.name.isdigit())
        if meta is None:   # prompts and questions are the same for every seed
            meta = {k: json.loads(open(seed_dir / "1" / k / "metadata.jsonl").readline()) for k in keys}
            rows = [(k, q) for k in keys for q, _ in meta[k]["vqa_list"]]
            answers, rephr = {}, {}
            for s in range(0, len(rows), args.batch_size * 4):
                batch = rows[s:s + args.batch_size * 4]
                outs = generate(model, tokenizer, [chat(tokenizer, ANSWER_Q.format(prompt=meta[k]["prompt"], question=q))
                                                   for k, q in batch], 4, device)
                for (k, _), o in zip(batch, outs):
                    answers.setdefault(k, []).append(o.split("\n")[0].strip().strip(".") or "Yes")
                outs = generate(model, tokenizer, [chat(tokenizer, REPHRASE_Q.format(prompt=meta[k]["prompt"], question=q))
                                                   for k, q in batch], 48, device)
                for (k, q), o in zip(batch, outs):
                    o = o.split("\n")[0].strip()
                    rephr.setdefault(k, []).append(o if o.endswith("?") and len(o.split()) >= 3 else q)
            print(f"answers / rewrites ready for {len(rows)} questions", flush=True)
        out_path = Path(args.out_dir) / f"seed_{seed}.jsonl"
        done = set()
        if out_path.exists():
            done = {(r["key"], r["iter"]) for r in map(json.loads, open(out_path))}
        items = [(k, it, seed_dir / str(it) / k / "samples" / "00000.png")
                 for k in keys for it in range(1, args.n_iters + 1) if (k, it) not in done]
        items = items[:args.limit] if args.limit else items
        print(f"seed {seed}: {len(items)} images", flush=True)
        cache = {}
        with open(out_path, "a") as fout:
            for s in range(0, len(items), args.batch_size):
                batch = items[s:s + args.batch_size]
                md5s = [hashlib.md5(open(p, "rb").read()).hexdigest() for _, _, p in batch]
                todo = [i for i, (b, m) in enumerate(zip(batch, md5s)) if (b[0], m) not in cache]
                if todo:
                    texts = image_text(ta_tok, [batch[i][2] for i in todo], device)
                    ans_rows = [(i, chat(tokenizer, f"{t}\n{q} Answer in one word."),
                                 answer_token_ids(tokenizer, a))
                                for i, t in zip(todo, texts)
                                for (q, _), a in zip(meta[batch[i][0]]["vqa_list"], answers[batch[i][0]])]
                    yn_rows = [(i, chat(tokenizer, f"{t}\n{q} Answer Yes or No."))
                               for i, t in zip(todo, texts) for q in rephr[batch[i][0]]]
                    p_ans, p_yn = [], []
                    for r in range(0, len(ans_rows), args.batch_size):
                        chunk = ans_rows[r:r + args.batch_size]
                        p_ans += answer_prob(model, tokenizer, [c for _, c, _ in chunk], [x for _, _, x in chunk], device)
                    for r in range(0, len(yn_rows), args.batch_size):
                        p_yn += yes_prob(model, tokenizer, [c for _, c in yn_rows[r:r + args.batch_size]],
                                         yes_id, no_id, device)
                    per_a, per_y = {}, {}
                    for (i, _, _), p in zip(ans_rows, p_ans):
                        per_a.setdefault(i, []).append(p)
                    for (i, _), p in zip(yn_rows, p_yn):
                        per_y.setdefault(i, []).append(p)
                    for i in todo:
                        k = batch[i][0]
                        cache[(k, md5s[i])] = {"am_ans": sum(per_a[i]) / len(per_a[i]),
                                               "am_rephr": sum(per_y[i]) / len(per_y[i]),
                                               "p_ans": per_a[i], "p_rephr": per_y[i],
                                               "answers": answers[k], "rephrased": rephr[k]}
                for (k, it, _), m in zip(batch, md5s):
                    fout.write(json.dumps({"key": k, "iter": it, "md5": m, **cache[(k, m)]}) + "\n")
                fout.flush()
                if (s // args.batch_size) % 20 == 0:
                    print(f"seed {seed}: {s + len(batch)}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
