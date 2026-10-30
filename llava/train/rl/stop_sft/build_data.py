"""Stop-SFT data (task 3a): label the images of a policy rollout with the critique the
critic *should* have written, so SFT can teach it when to say "looks good".

Input: val_dump JSONL files written by train_grpo.py with ``--val_dump_dir --val_dump_codes``
(and ``--eval_sample_critiques``), one line per refinement node. A line holds the parent
image (``parent_codes``), its per-question judge scores and the critique the policy wrote
on it. Each line becomes one candidate example (image -> critique):

* image passes every question (all per_question >= threshold):
  target = the policy's own critique if it said "looks good", else the SFT no-issue text
  (``NO_ISSUE_CRITIQUE``);
* image fails a question: target = the policy's own critique if it asked for a fix, else
  (it wrongly said "looks good") the oracle critique naming the failed questions.

Pass and fail examples are then balanced (the larger side is subsampled). Images after the
last round have no critique in the dump and are not used.

``--targets oracle`` (critique distillation from the judge, jeet's run 2, 2026-10-24): the policy's
own critique is ignored; every passing image gets ``NO_ISSUE_CRITIQUE`` and every failing image the
oracle critique naming its failed questions. ``--ratio 0`` keeps the rollout's natural pass/fail ratio.

``--draft_replay R`` (2026-10-30, after run 2's draft collapse): also emit ``R`` x as many *draft* examples
(``kind: draft``) as critique examples: the prompt as context and the policy's own round-0 image as the
target, CE on the image tokens. Replaying its own drafts anchors the model's drafting while the LoRA
learns the critique (critique-only CE drifted the draft am_0 from 0.80 to 0.21 over one epoch). Drafts
are sampled from the unique round-0 images (with replacement once they run out), every draft regardless
of its judge score, so the drafting distribution is held where it was rather than pushed.

    python -m llava.train.rl.stop_sft.build_data --dumps <dir> [<dir> ...] --out data.jsonl
"""

import argparse
import collections
import glob
import json
import os
import random

from llava.train.rl.oracle_critique import NO_ISSUE_CRITIQUE, oracle_critique
from llava.train.rl.rollout import LOOKS_GOOD_RE


def label(line, threshold=0.5, targets="mixed"):
    """One dump line -> example dict (or None if the line has no image codes)."""
    codes, pq = line.get("parent_codes"), line.get("parent_per_question")
    if codes is None or pq is None:
        return None
    vqa_list = list(zip(line["questions"], line["answers"]))
    passed = all(p >= threshold for p in pq)
    own = line["critique"]
    own_lg = LOOKS_GOOD_RE.search(own) is not None
    if targets == "oracle":
        target, source = ((NO_ISSUE_CRITIQUE, "template") if passed else
                          (oracle_critique(vqa_list, line["skills"], pq, threshold), "oracle"))
    elif passed:
        target, source = (own, "own") if own_lg else (NO_ISSUE_CRITIQUE, "template")
    else:
        target, source = ((own, "own") if not own_lg else
                          (oracle_critique(vqa_list, line["skills"], pq, threshold), "oracle"))
    return {"prompt": line["prompt"], "round": line["round"] - 1, "codes": codes, "kind": "critique",
            "target": target, "passed": passed, "source": source, "own_looks_good": own_lg}


def draft_examples(examples, n, seed=0):
    """``n`` draft-replay examples drawn from the unique round-0 images in ``examples``
    (without replacement while they last, then with). Target = the image itself."""
    drafts = {}
    for e in examples:
        if e["round"] == 0 and e["prompt"] not in drafts:
            drafts[e["prompt"]] = e
    pool = list(drafts.values())
    if not pool or n <= 0:
        return []
    rng = random.Random(seed + 1)
    picked = []
    while len(picked) < n:
        chunk = pool[:]
        rng.shuffle(chunk)
        picked += chunk[:n - len(picked)]
    return [{"prompt": e["prompt"], "round": 0, "codes": e["codes"], "kind": "draft", "target": "",
             "passed": e["passed"], "source": "draft", "own_looks_good": False} for e in picked]


def read_dumps(dirs, prefix="val"):
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, f"{prefix}_step*_rank*.jsonl"))):
            with open(path) as f:
                for raw in f:
                    if raw.strip():
                        yield json.loads(raw)


def balance(examples, seed=0, ratio=1.0):
    """Keep every example of the smaller class and ``ratio`` x as many of the larger."""
    rng = random.Random(seed)
    if ratio <= 0:                      # natural ratio: keep everything
        out = list(examples)
        rng.shuffle(out)
        return out
    pos = [e for e in examples if e["passed"]]
    neg = [e for e in examples if not e["passed"]]
    small, large = (pos, neg) if len(pos) <= len(neg) else (neg, pos)
    keep = rng.sample(large, min(len(large), int(round(ratio * len(small)))))
    out = small + keep
    rng.shuffle(out)
    return out


def counts(examples):
    c = collections.Counter((e["round"], "pass" if e["passed"] else "fail", e["source"]) for e in examples)
    lines = []
    for r in sorted({k[0] for k in c}):
        cells = [f"{cls}/{src}={c[(r, cls, src)]}" for cls in ("pass", "fail")
                 for src in ("own", "template", "oracle", "draft") if c[(r, cls, src)]]
        lines.append(f"  image round {r}: " + "  ".join(cells))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dumps", nargs="+", required=True, help="val_dump dirs")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--ratio", type=float, default=1.0, help="larger class / smaller class after balancing (0 = no balancing)")
    p.add_argument("--targets", choices=["mixed", "oracle"], default="mixed",
                   help="mixed: the policy's own critique where right (task 3a); oracle: judge-derived critique for every image")
    p.add_argument("--draft_replay", type=float, default=0.0,
                   help="also emit this many draft-replay examples per critique example (0 = none)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seen, examples = set(), []
    for line in read_dumps(args.dumps):
        ex = label(line, args.threshold, args.targets)
        if ex is None:
            continue
        key = (ex["prompt"], ex["round"])   # a re-run shard must not add a chain twice
        if key in seen:
            raise SystemExit(f"duplicate (prompt, round) {key}: a dump dir holds two runs")
        seen.add(key)
        examples.append(ex)
    print(f"{len(examples)} labelled images from {len({e['prompt'] for e in examples})} prompts")
    print(counts(examples))
    out = balance(examples, args.seed, args.ratio)
    print(f"balanced: {len(out)} ({sum(e['passed'] for e in out)} pass / {sum(not e['passed'] for e in out)} fail)")
    print(counts(out))
    if args.draft_replay > 0:
        drafts = draft_examples(examples, int(round(args.draft_replay * len(out))), args.seed)
        n_uniq = len({e["prompt"] for e in drafts})
        print(f"draft replay: {len(drafts)} examples from {n_uniq} unique drafts "
              f"({sum(e['passed'] for e in drafts)} pass)")
        out = out + drafts
        random.Random(args.seed + 2).shuffle(out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")


if __name__ == "__main__":
    main()
