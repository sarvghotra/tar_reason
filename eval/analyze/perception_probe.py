"""Critique plan C0: can the model perceive its own errors? (eval only, no training)

For every image in train_grpo val_dumps written with ``--val_dump_codes`` (the critiqued image of each line,
``parent_codes``, with the judge's ``parent_per_question``), ask the model each of the image's VQA questions in
plain understanding mode (no iterative prefix) and compare its answer with the judge's verdict:

    system + user "<im_start><S0><I...codes...><im_end>\\n{question} Answer with a single word or number."
    -> greedy short answer (image vocabulary masked)

The image is the same token sequence the critic reads (``rollout.TreeRollout._state``: ``<im_start><S0>`` +
codes), closed by ``<im_end>``. Judge verdict: the expected answer holds if per_question >= 0.5.

    # answers (GPU, sharded over ranks):
    torchrun --nproc_per_node=4 eval/analyze/perception_probe.py answer --model <dir> \\
        --dumps <dump_dir> [...] --out <answers_dir> [--limit N]
    # report (CPU):
    python eval/analyze/perception_probe.py report <answers_dir> [...]
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

NUM = {"zero": 0, "no": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
       "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
SUFFIX = " Answer with a single word or number."


def load_images(dump_dirs, limit=0):
    """Unique critiqued images (prompt, image round) with codes, questions and judge scores."""
    seen, out = set(), []
    for d in dump_dirs:
        for path in sorted(glob.glob(os.path.join(d, "val_step*_rank*.jsonl"))):
            for raw in open(path):
                r = json.loads(raw)
                if r.get("prefix", "val/") != "val/" or r.get("parent_codes") is None:
                    continue
                key = (r["prompt"], r["round"] - 1)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"prompt": r["prompt"], "round": r["round"] - 1, "codes": r["parent_codes"],
                            "questions": r["questions"], "answers": r["answers"], "skills": r["skills"],
                            "per_question": r["parent_per_question"]})
    out.sort(key=lambda e: (e["prompt"], e["round"]))
    return out[:limit] if limit else out


def norm(text):
    """First word of an answer, lower-case, numbers as ints."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    if not words:
        return ""
    w = words[0]
    if w.isdigit():
        return int(w)
    return NUM.get(w, w) if w not in ("yes", "no") else w


def expected(question, answer):
    if question.startswith("How many"):
        return NUM.get(answer.strip().lower(), answer.strip().lower())
    return answer.strip().lower()


def matches(question, answer, model_answer):
    m, e = norm(model_answer), expected(question, answer)
    if question.startswith("How many") and m == "no":   # "no X" means zero
        m = 0
    return m == e


# ----------------------------------------------------------------------------- answers (GPU)

def build_rows(tok, codes, question, img0):
    """Token ids: chat template with the image tokens spliced into the user turn."""
    ph = "IMG"
    text = tok.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": f"<im_start><S0>{ph}<im_end>\n{question}{SUFFIX}"}],
        tokenize=False, add_generation_prompt=True)
    a, b = text.split(ph)
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    return enc(a) + [img0 + c for c in codes] + enc(b)


def answer(args):
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer, LogitsProcessorList, Qwen2ForCausalLM
    from llava.train.rl.rollout import NoImageVocab

    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    img0 = tok.convert_tokens_to_ids("<I0>")
    n_img = sum(1 for t in tok.get_vocab() if t.startswith("<I") and t[2:-1].isdigit())
    model = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2").to(device).eval()
    # Text answers only: mask the image vocabulary and the image-start / scale tokens (the model sometimes
    # starts an image with <S0> when asked about one).
    blocked = [tok.convert_tokens_to_ids(t) for t in ("<im_start>", "<S0>", "<S1>", "<S2>")]

    class BlockTokens:
        def __call__(self, input_ids, scores):
            scores[:, blocked] = float("-inf")
            return scores
    proc = LogitsProcessorList([NoImageVocab(img0, img0 + n_img), BlockTokens()])
    stop = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                   tok.convert_tokens_to_ids("<|endoftext|>")} - {None})

    images = load_images(args.dumps, args.limit)
    items = [(i, q, a, s, p) for i, e in enumerate(images)
             for q, a, s, p in zip(e["questions"], e["answers"], e["skills"], e["per_question"])]
    mine = items[rank::world]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"answers_rank{rank}.jsonl")
    if rank == 0:
        print(f"{len(images)} images, {len(items)} questions, {world} ranks", flush=True)
        print("example prompt:", tok.decode(build_rows(tok, images[0]["codes"][:3], items[0][1], img0)), flush=True)
    with open(path, "w") as f, torch.no_grad():
        for start in range(0, len(mine), args.batch_size):
            chunk = mine[start:start + args.batch_size]
            rows = [build_rows(tok, images[i]["codes"], q, img0) for i, q, _, _, _ in chunk]
            width = max(map(len, rows))
            ids = torch.full((len(rows), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, r in enumerate(rows):
                ids[j, width - len(r):] = torch.tensor(r)
                mask[j, width - len(r):] = 1
            gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=6,
                                 do_sample=False, logits_processor=proc, eos_token_id=stop,
                                 pad_token_id=tok.pad_token_id)[:, width:]
            for (i, q, a, s, p), g in zip(chunk, gen.tolist()):
                ans = tok.decode(g, skip_special_tokens=True).strip()
                e = images[i]
                f.write(json.dumps({"prompt": e["prompt"], "round": e["round"], "question": q, "answer": a,
                                    "skill": s, "judge": p, "model_answer": ans}) + "\n")
            if rank == 0 and (start // args.batch_size) % 50 == 0:
                print(f"{start + len(chunk)}/{len(mine)}", flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        print("Done.", flush=True)


# ----------------------------------------------------------------------------- report (CPU)

def rates(pairs):
    """pairs of (model_says_ok, judge_ok) -> dict."""
    n = len(pairs)
    tp = sum(m and j for m, j in pairs); tn = sum((not m) and (not j) for m, j in pairs)
    jp = sum(j for _, j in pairs); jf = n - jp
    return {"n": n, "agree": (tp + tn) / n if n else None,
            "recall_pass": tp / jp if jp else None,           # model confirms what the judge passes
            "spec_fail": tn / jf if jf else None,             # model rejects what the judge fails
            "judge_pass_rate": jp / n if n else None}


def report(args, out=sys.stdout):
    rows = [json.loads(l) for d in args.answers for p in glob.glob(os.path.join(d, "answers_rank*.jsonl"))
            for l in open(p)]
    for r in rows:
        r["m_ok"] = matches(r["question"], r["answer"], r["model_answer"])
        r["j_ok"] = r["judge"] >= 0.5
    pf = lambda x: "  -  " if x is None else f"{100 * x:5.1f}"
    print(f"{len(rows)} answers over {len({(r['prompt'], r['round']) for r in rows})} images", file=out)
    print(f"{'group':22s} {'n':>6s} {'agree':>6s} {'rec(pass)':>9s} {'spec(fail)':>10s} {'judge pass':>10s}", file=out)
    groups = collections.defaultdict(list)
    for r in rows:
        kind = "count" if r["question"].startswith("How many") else "yes/no"
        for g in ("all", f"skill:{r['skill']}", f"type:{kind}", f"round:{r['round']}"):
            groups[g].append((r["m_ok"], r["j_ok"]))
    for g in sorted(groups, key=lambda g: (g != "all", g)):
        s = rates(groups[g])
        print(f"{g:22s} {s['n']:6d} {pf(s['agree'])} {pf(s['recall_pass']):>9s} {pf(s['spec_fail']):>10s} "
              f"{pf(s['judge_pass_rate']):>10s}", file=out)
    yn = [r for r in rows if not r["question"].startswith("How many")]
    says_yes = sum(norm(r["model_answer"]) == "yes" for r in yn) / max(len(yn), 1)
    print(f"yes/no: model says 'yes' on {100 * says_yes:.1f}% of {len(yn)} questions", file=out)
    imgs = collections.defaultdict(list)
    for r in rows:
        imgs[(r["prompt"], r["round"])].append((r["m_ok"], r["j_ok"]))
    per_img = [(all(m for m, _ in v), all(j for _, j in v)) for v in imgs.values()]
    s = rates(per_img)
    print(f"per image ('all model answers as expected' vs 'judge: all pass'): n={s['n']} judge all-pass "
          f"{pf(s['judge_pass_rate'])}%  recall {pf(s['recall_pass'])}%  false 'all ok' on failing images "
          f"{pf(1 - s['spec_fail'] if s['spec_fail'] is not None else None)}%  agree {pf(s['agree'])}%", file=out)
    unk = collections.Counter(r["model_answer"] for r in rows if norm(r["model_answer"]) not in
                              ("yes", "no") and not isinstance(norm(r["model_answer"]), int))
    print("unparsed answers (top 8):", unk.most_common(8), file=out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("answer")
    a.add_argument("--model", required=True)
    a.add_argument("--dumps", nargs="+", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--batch_size", type=int, default=48)
    a.add_argument("--limit", type=int, default=0, help="first N images only")
    r = sub.add_parser("report")
    r.add_argument("answers", nargs="+")
    args = p.parse_args()
    answer(args) if args.cmd == "answer" else report(args)


if __name__ == "__main__":
    main()
