"""Critique-sensitivity probe: does the refine image depend on the critique?

For every prompt one draft is sampled and scored, then refined under several
critiques while everything else is held fixed (same draft tokens, same token
layout as ``eval/iterative_generation_adhoc.py``):

    <prefix><im_start><S0> draft <im_end>\\nSelf-reflect: {critique}\\n<im_start><S0> -> refine

Conditions (``--conditions``):
    real      K critiques sampled from the policy for this draft
    shuffled  a real, non-"looks good" critique written for a *different* prompt
    generic   a fixed, content-free critique (``--generic_critique``)
    oracle    built from the judge's failed VQA questions on this draft
    fresh     no loop at all: a new draft sampled from the prompt
    redecode  the draft's own codes decoded again (de-tokenizer noise floor)

Every critique gets ``--images_per_critique`` (M) refine images, and every
critique is followed by an image even if it says "looks good", so the
conditions differ only in the critique text. All images are scored with the
GenEval2 oracle judge (``llava/train/rl/pixel_reward_server.py``, Qwen3-VL-8B
with ``--answer_id_mode geneval2``, one judge per rank).

How to read the summary (subset "wrong" = drafts failing >= 1 question):
    oracle ~ generic ~ real  -> the refiner ignores the critique text; RL on
                                reflection tokens has nothing to learn from.
    oracle >> generic ~ real -> the refiner follows instructions, the critic is
                                the bottleneck: critic RL is worth it.
    real   >  generic        -> both work; the loop is limited by RL mechanics.
    shuffled vs real         -> is the real critique draft-specific?
    generic vs fresh         -> does having the draft in context matter at all?
    fix / break rates        -> is the change *targeted* at the failed atoms?
    variance split (real)    -> between-critique vs within-critique variance of
                                the child score, i.e. how much of the GRPO
                                signal on reflection tokens is image-sampling
                                noise; redecode gives the decode-noise share.

Outputs under --out_dir:
    records/rank{r}.jsonl     one line per prompt, all scores and critiques
    images/{idx:05d}/*.png    draft + refine images (see --save_images)
    summary.txt, summary.json the analysis (rank 0, after all ranks finish)

Rerunning with the same --out_dir resumes: prompts already in records/ are
skipped. ``--analyze_only`` just re-reads records/ and rewrites the summary.

Usage (see critique_sensitivity.sh, which also starts the judges):
    torchrun --standalone --nproc_per_node=4 eval/analyze/critique_sensitivity.py \\
        --model <ckpt> --prompts_file geneval2_data.jsonl --out_dir <dir> \\
        --reward_server_url http://127.0.0.1:8850,http://127.0.0.1:8851,...
"""

import argparse
import glob
import json
import os
import random
import re
import sys
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.iterative_generation_adhoc import (  # noqa: E402
    SCALE_SEQ_LEN, even_chunks, get_image_token_map, left_pad, load_models,
    make_prefix, trim_at)
from llava.train.rl.oracle_critique import (  # noqa: E402
    SKILL_ORDER, oracle_critique)

ALL_CONDITIONS = ("real", "shuffled", "generic", "oracle", "fresh", "redecode")
# Conditions whose "critique" is text spliced after the draft.
TEXT_CONDITIONS = ("real", "shuffled", "generic", "oracle")
LOOKS_GOOD_RE = re.compile(r"looks good", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora_path")
    parser.add_argument("--prompts_file", required=True,
                        help="GenEval2-style JSONL: prompt, vqa_list, skills.")
    parser.add_argument("--max_prompts", type=int, default=0,
                        help="Keep a random (--subset_seed) subset of N prompts (0 = all).")
    parser.add_argument("--subset_seed", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--reward_server_url", default="",
                        help="Comma-separated judge URLs; rank r uses URL r % n.")
    parser.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    parser.add_argument("--num_real", type=int, default=4,
                        help="K: real critiques sampled per draft.")
    parser.add_argument("--images_per_critique", type=int, default=4,
                        help="M: refine images per critique (also used for "
                             "fresh and redecode).")
    parser.add_argument("--fail_threshold", type=float, default=0.5,
                        help="A question 'fails' when the judge's answer "
                             "probability is below this.")
    parser.add_argument("--generic_critique",
                        default="the image does not fully match the prompt\n"
                                "Correction: regenerate the image so that it matches the prompt")
    parser.add_argument("--prompt_chunk", type=int, default=16,
                        help="Prompts processed (and checkpointed) together.")
    parser.add_argument("--save_images", default="first", choices=["none", "first", "all"],
                        help="'first' keeps the draft and image 0 of every critique.")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--ar_path")
    parser.add_argument("--encoder_path")
    parser.add_argument("--decoder_path")
    parser.add_argument("--gen_seq_len", type=int, default=729)
    parser.add_argument("--scale", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Rows per language-model call.")
    parser.add_argument("--decode_batch_size", type=int, default=64)
    parser.add_argument("--reflect_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=1200)
    parser.add_argument("--reflect_temperature", type=float, default=1.0)
    parser.add_argument("--reflect_top_p", type=float, default=0.95)
    parser.add_argument("--reflect_top_k", type=int, default=1200)
    parser.add_argument("--system_prompt", default="You are a helpful assistant.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # make_prefix / load_models read these; this probe uses one scale throughout.
    args.draft_img_scale = args.scale
    args.draft_gen_seq_len = args.gen_seq_len
    args.conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = set(args.conditions) - set(ALL_CONDITIONS)
    if unknown:
        parser.error(f"unknown conditions: {sorted(unknown)}")
    if SCALE_SEQ_LEN[args.scale] != args.gen_seq_len:
        parser.error(f"scale {args.scale} requires {SCALE_SEQ_LEN[args.scale]} image tokens")
    return args


# ---------------------------------------------------------------------------
# Generation, decoding, scoring
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, args, device, reward):
        self.args = args
        self.device = device
        self.reward = reward
        self.tokenizer, self.model, self.visual_tok = load_models(args, device)
        self.image_token_map = get_image_token_map(self.tokenizer)
        tok = self.tokenizer
        self.pad_id = tok.pad_token_id
        self.text_stop_ids = list({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                                   tok.convert_tokens_to_ids("<|endoftext|>"),
                                   tok.convert_tokens_to_ids("<im_start>"),
                                   self.pad_id} - {None})
        self.reflect_prefix_ids = self.encode("<im_end>\nSelf-reflect:")
        self.image_start_ids = self.encode(f"\n<im_start><S{args.scale}>")

    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    @torch.inference_mode()
    def _generate(self, rows, **kwargs):
        out = []
        for chunk in even_chunks(rows, self.args.batch_size):
            input_ids, attention_mask = left_pad(chunk, self.pad_id, self.device)
            out.extend(self.model.generate(
                input_ids, attention_mask=attention_mask, do_sample=True,
                repetition_penalty=1.0, pad_token_id=self.pad_id, **kwargs,
            )[:, input_ids.shape[1]:].tolist())
        return out

    def sample_images(self, rows):
        """Returns (codes, image_token_ids) per row."""
        a = self.args
        codes, ids = [], []
        for row in self._generate(rows, max_new_tokens=a.gen_seq_len, temperature=a.temperature,
                                  top_p=a.top_p, top_k=a.top_k):
            row = trim_at(row, set(self.text_stop_ids))
            row_codes = [self.image_token_map[t] for t in row if t in self.image_token_map]
            codes.append((row_codes + [0] * a.gen_seq_len)[:a.gen_seq_len])
            ids.append(row)
        return codes, ids

    def sample_reflections(self, rows):
        a = self.args
        out = self._generate(rows, max_new_tokens=a.reflect_tokens,
                             temperature=a.reflect_temperature, top_p=a.reflect_top_p,
                             top_k=a.reflect_top_k, eos_token_id=self.text_stop_ids)
        return [self.tokenizer.decode(trim_at(r, set(self.text_stop_ids)),
                                      skip_special_tokens=True).strip() for r in out]

    @torch.inference_mode()
    def decode(self, codes):
        images = []
        for batch in even_chunks(torch.tensor(codes, dtype=torch.long),
                                 self.args.decode_batch_size):
            images.extend(Image.fromarray(im.numpy()) for im in
                          self.visual_tok.decode_from_encoder_indices(
                              batch.to(self.device), {"cfg_scale": self.args.cfg_scale}))
        return images

    def score(self, images, vqa_lists):
        am, gm, per_question = self.reward.score_images(images, vqa_lists)
        return [{"am": a, "gm": g, "pq": q} for a, g, q in zip(am, gm, per_question)]

    def run_chunk(self, prompts, donor_pool, rng):
        """prompts: list of dict(idx, prompt, vqa_list, skills). Returns records."""
        a = self.args
        M, K = a.images_per_critique, a.num_real
        conds = a.conditions
        prefix_rows = self.tokenizer([make_prefix(self.tokenizer, p["prompt"], a)
                                      for p in prompts], add_special_tokens=False).input_ids

        # Drafts.
        draft_codes, draft_ids = self.sample_images(prefix_rows)
        draft_images = self.decode(draft_codes)
        draft_scores = self.score(draft_images, [p["vqa_list"] for p in prompts])
        reflect_rows = [row + ids + self.reflect_prefix_ids
                        for row, ids in zip(prefix_rows, draft_ids)]

        # Critique texts per prompt and condition.
        critiques = [{} for _ in prompts]
        if "real" in conds or "shuffled" in conds:
            texts = self.sample_reflections([row for row in reflect_rows for _ in range(K)])
            for i in range(len(prompts)):
                critiques[i]["real"] = [{"text": t} for t in texts[i * K:(i + 1) * K]]
        if "shuffled" in conds:
            for i, p in enumerate(prompts):
                for c in critiques[i]["real"]:
                    if not LOOKS_GOOD_RE.search(c["text"]):
                        donor_pool.append((p["idx"], p["prompt"], c["text"]))
            for i, p in enumerate(prompts):
                donors = [d for d in donor_pool if d[1] != p["prompt"]]
                critiques[i]["shuffled"] = []
                if donors:
                    donor_idx, _, text = rng.choice(donors)
                    critiques[i]["shuffled"] = [{"text": text, "donor_idx": donor_idx}]
            if "real" not in conds:
                for c in critiques:
                    c.pop("real")
        for i, p in enumerate(prompts):
            if "generic" in conds:
                critiques[i]["generic"] = [{"text": a.generic_critique}]
            if "oracle" in conds:
                critiques[i]["oracle"] = [{"text": oracle_critique(
                    p["vqa_list"], p["skills"], draft_scores[i]["pq"], a.fail_threshold)}]
            for c in critiques[i].values():
                for crit in c:
                    crit["looks_good"] = bool(LOOKS_GOOD_RE.search(crit["text"]))

        # Every refine row: (prompt i, condition, critique k, image m).
        slots, rows = [], []
        for i in range(len(prompts)):
            for cond, crits in critiques[i].items():
                for k, crit in enumerate(crits):
                    row = reflect_rows[i] + self.encode(" " + crit["text"]) + self.image_start_ids
                    for m in range(M):
                        slots.append((i, cond, k, m))
                        rows.append(row)
            if "fresh" in conds:
                critiques[i]["fresh"] = [{"text": None, "looks_good": False}]
                for m in range(M):
                    slots.append((i, "fresh", 0, m))
                    rows.append(prefix_rows[i])
        codes = self.sample_images(rows)[0] if rows else []
        if "redecode" in conds:
            for i in range(len(prompts)):
                critiques[i]["redecode"] = [{"text": None, "looks_good": False}]
                for m in range(M):
                    slots.append((i, "redecode", 0, m))
                    codes.append(draft_codes[i])
        images = self.decode(codes) if codes else []
        scores = self.score(images, [prompts[s[0]]["vqa_list"] for s in slots])

        for crits in critiques:
            for c in crits.values():
                for crit in c:
                    crit["children"] = [None] * M
        for (i, cond, k, m), image, score in zip(slots, images, scores):
            if self.save(m):
                score["path"] = self.save_image(prompts[i]["idx"], f"{cond}_{k}_{m}", image)
            critiques[i][cond][k]["children"][m] = score

        records = []
        for i, p in enumerate(prompts):
            draft = dict(draft_scores[i])
            if a.save_images != "none":
                draft["path"] = self.save_image(p["idx"], "draft", draft_images[i])
            records.append({**p, "draft": draft, "conditions": critiques[i]})
        return records

    def save(self, m):
        return self.args.save_images == "all" or (self.args.save_images == "first" and m == 0)

    def save_image(self, idx, name, image):
        d = os.path.join(self.args.out_dir, "images", f"{idx:05d}")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{name}.png")
        image.save(path)
        return path


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_records(out_dir):
    by_idx = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "records", "rank*.jsonl"))):
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    by_idx[rec["idx"]] = rec
    return [by_idx[i] for i in sorted(by_idx)]


def mean_se(values):
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan")
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return float(v.mean()), float(se)


def children_am(rec, cond, policy=False):
    """All child AMs of a condition. policy=True: a 'looks good' critique keeps
    the draft (what the eval pipeline does) instead of the forced refine."""
    out = []
    for crit in rec["conditions"].get(cond, []):
        for child in crit["children"]:
            out.append(rec["draft"]["am"] if policy and crit["looks_good"] else child["am"])
    return out


def per_draft_means(records, thr):
    """{condition: {idx: mean child AM}}, plus the derived 'real_policy'."""
    means = {}
    for rec in records:
        for cond in rec["conditions"]:
            ams = children_am(rec, cond)
            if ams:
                means.setdefault(cond, {})[rec["idx"]] = float(np.mean(ams))
        if "real" in rec["conditions"]:
            means.setdefault("real_policy", {})[rec["idx"]] = float(
                np.mean(children_am(rec, "real", policy=True)))
    return means


def is_wrong(rec, thr):
    return any(p < thr for p in rec["draft"]["pq"])


def fix_break(records, cond, thr):
    """Per skill: fix = P(child passes | draft failed q), break = P(child fails | draft passed q)."""
    stats = {}
    for rec in records:
        for crit in rec["conditions"].get(cond, []):
            for child in crit["children"]:
                for slot, (dp, cp) in enumerate(zip(rec["draft"]["pq"], child["pq"])):
                    skill = rec["skills"][slot]
                    for key in (skill, "all"):
                        s = stats.setdefault(key, [0, 0, 0, 0])   # fixed, failed, broken, passed
                        if dp < thr:
                            s[0] += cp >= thr
                            s[1] += 1
                        else:
                            s[2] += cp < thr
                            s[3] += 1
    return {k: {"fix": s[0] / s[1] if s[1] else float("nan"), "n_failed": s[1],
                "break": s[2] / s[3] if s[3] else float("nan"), "n_passed": s[3]}
            for k, s in stats.items()}


def variance_split(records, cond):
    """One-way random effects over critiques within each draft (needs K, M >= 2)."""
    within, between_obs, m_sizes = [], [], []
    for rec in records:
        crits = [[c["am"] for c in crit["children"]] for crit in rec["conditions"].get(cond, [])]
        crits = [c for c in crits if len(c) >= 2]
        if len(crits) < 2:
            continue
        within.append(np.mean([np.var(c, ddof=1) for c in crits]))
        between_obs.append(np.var([np.mean(c) for c in crits], ddof=1))
        m_sizes.append(np.mean([len(c) for c in crits]))
    if not within:
        return None
    s_w = float(np.mean(within))
    s_b = max(0.0, float(np.mean(between_obs)) - s_w / float(np.mean(m_sizes)))
    return {"n_drafts": len(within), "var_within": s_w, "var_between": s_b,
            "icc": s_b / (s_b + s_w) if s_b + s_w > 0 else float("nan"),
            "m_for_equal_noise": s_w / s_b if s_b > 0 else float("inf")}


def within_var(records, cond):
    v = [np.var([c["am"] for c in crit["children"]], ddof=1)
         for rec in records for crit in rec["conditions"].get(cond, [])
         if len(crit["children"]) >= 2]
    return float(np.mean(v)) if v else float("nan")


def analyze(records, thr):
    lines, summary = [], {"n_prompts": len(records), "fail_threshold": thr, "subsets": {}}
    out = lines.append
    subsets = {"all": records,
               "wrong": [r for r in records if is_wrong(r, thr)],
               "correct": [r for r in records if not is_wrong(r, thr)]}
    order = [c for c in ("real", "real_policy", "shuffled", "generic", "oracle", "fresh",
                         "redecode") if any(c in r["conditions"] or c == "real_policy"
                                            and "real" in r["conditions"] for r in records)]
    pairs = [("real", "generic"), ("oracle", "generic"), ("shuffled", "generic"),
             ("real", "shuffled"), ("oracle", "real"), ("generic", "fresh"),
             ("real_policy", "generic")]

    out(f"prompts: {len(records)}  wrong drafts: {len(subsets['wrong'])}  "
        f"correct drafts: {len(subsets['correct'])}  (fail threshold {thr})")
    for name, recs in subsets.items():
        if not recs:
            continue
        means = per_draft_means(recs, thr)
        draft = {r["idx"]: r["draft"]["am"] for r in recs}
        sub = {"n": len(recs), "draft_am": mean_se(list(draft.values())),
               "conditions": {}, "paired": {}}
        out("")
        out(f"=== subset: {name} (n={len(recs)}, draft AM {sub['draft_am'][0]:.4f}) ===")
        out(f"{'condition':<12} {'child AM':>9} {'±se':>7} {'Δ vs draft':>11} {'±se':>7} "
            f"{'improved':>9} {'degraded':>9} {'looks_good':>10}")
        for cond in order:
            m = means.get(cond, {})
            if not m:
                continue
            deltas = [m[i] - draft[i] for i in m]
            per_image = [(am - r["draft"]["am"]) for r in recs if r["idx"] in m
                         for am in children_am(r, "real" if cond == "real_policy" else cond,
                                               policy=cond == "real_policy")]
            crits = [c for r in recs for c in r["conditions"].get(
                "real" if cond == "real_policy" else cond, [])]
            row = {"child_am": mean_se(list(m.values())), "delta": mean_se(deltas),
                   "improved": float(np.mean([d > 0.05 for d in per_image])),
                   "degraded": float(np.mean([d < -0.05 for d in per_image])),
                   "looks_good": float(np.mean([c["looks_good"] for c in crits]))}
            sub["conditions"][cond] = row
            out(f"{cond:<12} {row['child_am'][0]:>9.4f} {row['child_am'][1]:>7.4f} "
                f"{row['delta'][0]:>+11.4f} {row['delta'][1]:>7.4f} {row['improved']:>9.3f} "
                f"{row['degraded']:>9.3f} {row['looks_good']:>10.3f}")
        out("paired differences over drafts (mean child AM, a - b):")
        for a_, b_ in pairs:
            if a_ not in means or b_ not in means:
                continue
            common = sorted(set(means[a_]) & set(means[b_]))
            d = [means[a_][i] - means[b_][i] for i in common]
            mu, se = mean_se(d)
            sub["paired"][f"{a_}-{b_}"] = {"mean": mu, "se": se, "n": len(d)}
            out(f"  {a_ + ' - ' + b_:<24} {mu:+.4f} ± {1.96 * se:.4f} (95% CI, n={len(d)})")
        draft_ams = [draft[i] for i in means.get("generic", {})]
        if len(draft_ams) > 2:
            gen = [means["generic"][i] - draft[i] for i in means["generic"]]
            sub["corr_draft_delta_generic"] = float(np.corrcoef(draft_ams, gen)[0, 1])
            out(f"corr(draft AM, generic Δ) = {sub['corr_draft_delta_generic']:+.3f} "
                "(regression to the mean)")
        summary["subsets"][name] = sub

    wrong = subsets["wrong"]
    if wrong:
        out("")
        out("=== targeted change on wrong drafts: fix = P(child passes | draft failed q), "
            "break = P(child fails | draft passed q) ===")
        text_conds = [c for c in TEXT_CONDITIONS + ("fresh", "redecode") if c in order]
        fb = {c: fix_break(wrong, c, thr) for c in text_conds}
        summary["fix_break_wrong"] = fb
        skills = ["all"] + [s for s in SKILL_ORDER if any(s in fb[c] for c in fb)]
        out(f"{'skill':<10} " + " ".join(f"{c + ' fix':>13} {c + ' brk':>13}" for c in text_conds))
        for s in skills:
            cells = []
            for c in text_conds:
                v = fb[c].get(s)
                cells.append(f"{v['fix']:>13.3f} {v['break']:>13.3f}" if v else f"{'-':>13} {'-':>13}")
            n = fb[text_conds[0]].get(s, {}).get("n_failed", 0)
            out(f"{s:<10} " + " ".join(cells) + f"   (failed q x children in {text_conds[0]}: {n})")

    out("")
    out("=== variance of child AM (all drafts) ===")
    vs = variance_split(records, "real")
    summary["variance_split_real"] = vs
    if vs:
        out(f"real: between-critique var {vs['var_between']:.4f}, within-critique var "
            f"{vs['var_within']:.4f}, ICC {vs['icc']:.3f} (n={vs['n_drafts']} drafts)")
        out(f"      images per critique for critique-mean noise = between var: "
            f"{vs['m_for_equal_noise']:.1f}")
    for cond in ("oracle", "generic", "fresh", "redecode"):
        if cond in order:
            v = within_var(records, cond)
            summary[f"within_var_{cond}"] = v
            out(f"{cond}: within-critique var {v:.4f}"
                + ("  <- decode-noise floor" if cond == "redecode" else ""))
    return "\n".join(lines), summary


def write_summary(args):
    records = load_records(args.out_dir)
    if not records:
        print("no records to analyze")
        return
    text, summary = analyze(records, args.fail_threshold)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_prompts(path, max_prompts, subset_seed=0):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows.append({"idx": len(rows), "prompt": d["prompt"],
                             "vqa_list": d["vqa_list"], "skills": d["skills"]})
    if max_prompts and max_prompts < len(rows):
        # GenEval2 is sorted by atom count, so a head slice would keep only the
        # easy prompts. A fixed seed keeps the subset stable across resumes.
        keep = sorted(random.Random(subset_seed).sample(range(len(rows)), max_prompts))
        rows = [rows[i] for i in keep]
    return rows


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        # Ranks finish at different times; the default 30 min would kill the
        # final barrier.
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    return rank, world_size, device


def main():
    args = parse_args()
    if args.analyze_only:
        write_summary(args)
        return

    from llava.train.rl.pixel_reward import PixelVQAReward
    from llava.train.rl.reward import RewardConfig

    rank, world_size, device = init_distributed(args)
    urls = [u.strip() for u in args.reward_server_url.split(",") if u.strip()]
    if not urls:
        raise ValueError("--reward_server_url is required")
    url = urls[rank % len(urls)]
    # Images are decoded here, so the reward's decode step is the identity.
    reward = PixelVQAReward(lambda images: images, url,
                            RewardConfig(answer_suffix="geneval2"), images_per_request=8)
    info = reward.health()
    print(f"[rank {rank}] judge {info.get('model')} at {url} "
          f"(answer_suffix={info.get('answer_suffix')}, "
          f"answer_id_mode={info.get('answer_id_mode')})", flush=True)

    os.makedirs(os.path.join(args.out_dir, "records"), exist_ok=True)
    if rank == 0:
        with open(os.path.join(args.out_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    prompts = load_prompts(args.prompts_file, args.max_prompts, args.subset_seed)
    done = {r["idx"] for r in load_records(args.out_dir)}
    mine = [p for p in prompts[rank::world_size] if p["idx"] not in done]
    print(f"[rank {rank}] {len(mine)} prompts to run ({len(done)} already done overall)",
          flush=True)

    runner = Runner(args, device, reward)
    donor_pool = []
    records_path = os.path.join(args.out_dir, "records", f"rank{rank}.jsonl")
    for start in range(0, len(mine), args.prompt_chunk):
        chunk = mine[start:start + args.prompt_chunk]
        torch.manual_seed(args.seed * 1_000_003 + chunk[0]["idx"])
        rng = random.Random(args.seed * 1_000_003 + chunk[0]["idx"])
        records = runner.run_chunk(chunk, donor_pool, rng)
        with open(records_path, "a") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"[rank {rank}] {start + len(chunk)}/{len(mine)} prompts", flush=True)

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        write_summary(args)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
