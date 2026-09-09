"""Sanity-check the latent VQA reward against GenEval2's Qwen3-VL soft-TIFA.

Takes a finished ``llava/train/rl/iterative_generation_adhoc_img_tokens.py``
results directory, re-encodes every PNG with TA-Tok (pool_scale 1 -> 729
tokens, the same path ``TATokVisionTower`` uses for understanding), scores it
with ``TarLatentVQAReward`` and compares that reward to the judge.

Tree layout
-----------
The generator samples ``--repeat`` drafts per prompt and ``--correct_repeat``
correction branches per draft, and writes one file per tree *slot*
(``d{D}`` for iteration 1, ``d{D}c{C}`` for iteration 2), described by
``{results_dir}/tree.json``. Per slot this script reads::

    {results_dir}/{it}/geneval2_results_{slot}.json   prompt -> png
    {results_dir}/{it}/image_tokens_{slot}.json       prompt -> emitted codes
    {results_dir}/{it}/results_geneval2_{slot}.jsonl  judge probabilities

A run with one slot per iteration (``--repeat 1 --correct_repeat 1``) uses the
unsuffixed names, and a directory with no ``tree.json`` is read that way too,
so pre-tree runs still work.

What it reports
---------------
1. Per iteration, pooled over slots: mean AM / GM and the *cross-prompt*
   correlation with the judge. High here mostly means the reward tracks prompt
   difficulty, which is the easy part.
2. Per group, the numbers GRPO actually depends on. Advantages are normalised
   within a group, so only *within-group* ordering moves the policy:
     - round 0: the drafts of one prompt        (needs --repeat > 1)
     - round 1: the branches of one draft       (needs --correct_repeat > 1)
   For each, the pairwise ordering accuracy against the judge, the correlation
   of group-normalised advantages (computed exactly as ``normalize_groups``
   does), and the judge score of a best-of-N pick by the reward against a
   random pick and an oracle pick.
3. The correction delta: each child against its own parent draft, so it is
   visible whether the reward can tell an improvement from a regression.
4. The cost of the decode -> re-encode round trip, scoring the codes the LM
   emitted next to the re-encoded PNGs.

Usage:
    python llava/train/rl/check_reward.py \
        --model <tar-checkpoint> --encoder_path /tmp/ta_tok.pth \
        --benchmark_data ~/scratch/git/GenEval2/geneval2_data.jsonl \
        --results_dir results/GenEval2/iter_adhoc_..._repeat4 \
        [--answer_suffix llava|geneval2] [--slots d0,d1]
"""

import argparse
import itertools
import json
import math
import os
import sys

import torch
import torch.distributed as dist
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, Qwen2ForCausalLM
from transformers.models.siglip.image_processing_siglip import SiglipImageProcessor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from llava.train.rl.reward import ANSWER_SUFFIXES, RewardConfig, TarLatentVQAReward, geometric_mean
from tok.ta_tok import TextAlignedTokenizer
from tok.utils import ScalingLayer

# Mirrors GRPOConfig defaults, so the advantages here match the ones training
# would compute from the same rewards.
ADV_EPS = 1e-4
TIE_EPS = 1e-6
SIGN_EPS = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--encoder_path", required=True, help="ta_tok.pth")
    p.add_argument("--benchmark_data", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--iterations", default="1,2")
    p.add_argument("--slots", default=None,
                   help="Comma-separated slot names to restrict to (e.g. "
                        "d0,d1,d0c0). Default: every slot in tree.json.")
    p.add_argument("--answer_suffix", default="llava", choices=sorted(ANSWER_SUFFIXES))
    p.add_argument("--token_source", default="both",
                   choices=["both", "tokens", "roundtrip"],
                   help="Score the LM's original image tokens (image_tokens.json), "
                        "the decoded->re-encoded PNGs, or both.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None, help="Optional JSON dump of per-prompt scores")
    return p.parse_args()


def init_distributed(args):
    """torchrun-aware device setup; falls back to --device without torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    return rank, world_size, device


class Encoder:
    def __init__(self, path, device):
        self.vt = TextAlignedTokenizer.from_checkpoint(path, load_teacher=False).to(device)
        self.vt.bottleneck.regularizer.set_eval_deterministic(deterministic=True)
        self.vt.input_type = "rec"
        # The Siglip processor already maps pixels to [-1, 1].
        self.vt.scale_layer = ScalingLayer(mean=[0., 0., 0.], std=[1., 1., 1.]).to(device)
        self.vt.eval()
        size = self.vt.input_size
        self.proc = SiglipImageProcessor()
        self.proc.size = (size, size)
        self.proc.crop_size = {"height": size, "width": size}
        self.device = device

    @torch.no_grad()
    def __call__(self, paths):
        imgs = [Image.open(p).convert("RGB") for p in paths]
        px = self.proc.preprocess(imgs, return_tensors="pt")["pixel_values"].to(self.device)
        out = self.vt.encode(px, pool_scale=1)
        return out["bottleneck_rep"].long().cpu().tolist()


SRC_LABELS = {"tokens": "orig image tokens", "roundtrip": "re-encoded images"}


def print_table(title, headers, rows):
    """Print rows (lists of equal length) as a padded ASCII table."""
    cols = [list(map(str, headers))] + [list(map(str, r)) for r in rows]
    widths = [max(len(c[j]) for c in cols) for j in range(len(headers))]

    def line(ch="-", junc="+"):
        return junc + junc.join(ch * (w + 2) for w in widths) + junc

    def fmt(cells):
        out = [f" {c:<{widths[j]}} " if j == 0 else f" {c:>{widths[j]}} "
               for j, c in enumerate(map(str, cells))]
        return "|" + "|".join(out) + "|"

    if title:
        print(f"\n{title}")
    print(line())
    print(fmt(headers))
    print(line("="))
    for r in rows:
        print(fmt(r))
    print(line())


def corr(a, b):
    if len(a) < 3:
        return float("nan"), float("nan")
    return pearsonr(a, b)[0], spearmanr(a, b)[0]


# ---------------------------------------------------------------------------
# Tree layout
# ---------------------------------------------------------------------------

def load_tree(results_dir, iterations):
    """Slot names per iteration, plus the child -> parent-draft map.

    A directory without ``tree.json`` predates branching: it holds one image
    per prompt per iteration under the unsuffixed file names.
    """
    path = os.path.join(results_dir, "tree.json")
    if not os.path.exists(path):
        print(f"[layout] no {path}; reading it as a single-slot (pre-tree) run")
        return {"drafts": 1, "children": 1, "single_slot": True,
                "slots": {"1": ["d0"], "2": ["d0c0"]}, "parent": {"d0c0": "d0"}}
    tree = json.load(open(path))
    if "slots" not in tree:  # first version of the generator
        tree["slots"] = {"1": tree.get("iter1_slots", []),
                         "2": tree.get("iter2_slots", [])}
    missing = [it for it in iterations if not tree["slots"].get(it)]
    if missing:
        raise ValueError(f"{path} lists no slots for iteration(s) {missing}")
    return tree


def slot_path(it_dir, stem, slot, single_slot, ext="json"):
    name = f"{stem}.{ext}" if single_slot else f"{stem}_{slot}.{ext}"
    return os.path.join(it_dir, name)


# ---------------------------------------------------------------------------
# Group metrics: what GRPO actually sees
# ---------------------------------------------------------------------------

def group_advantages(values):
    """Group-relative advantage, as ``grpo.normalize_groups`` computes it.

    Returns None for a group whose rewards carry no signal (size < 2 or zero
    variance), which is exactly when training gets a zero advantage.
    """
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    centered = [v - mean for v in values]
    std = math.sqrt(sum(c * c for c in centered) / len(values))
    if std < 1e-8:
        return None
    return [c / (std + ADV_EPS) for c in centered]


def pairwise_accuracy(groups):
    """Fraction of same-group pairs the reward orders the way the judge does.

    Pairs the judge calls a tie carry no training signal and are skipped; a
    tie in the reward counts as half credit. 0.5 is chance.
    """
    total = agree = 0.0
    for members in groups:
        for (lat_a, ref_a), (lat_b, ref_b) in itertools.combinations(members, 2):
            if abs(ref_a - ref_b) <= TIE_EPS:
                continue
            total += 1
            delta = lat_a - lat_b
            if abs(delta) <= TIE_EPS:
                agree += 0.5
            elif (delta > 0) == (ref_a - ref_b > 0):
                agree += 1
    return (agree / total if total else float("nan")), int(total)


def advantage_correlation(groups):
    """Correlate group-normalised reward advantages with the judge's."""
    lat, ref, live = [], [], 0
    for members in groups:
        adv_lat = group_advantages([m[0] for m in members])
        adv_ref = group_advantages([m[1] for m in members])
        if adv_lat is None or adv_ref is None:
            continue
        live += 1
        lat.extend(adv_lat)
        ref.extend(adv_ref)
    return corr(lat, ref), live


def selection_scores(groups):
    """Judge score of the reward's best-of-N pick, a random pick, the oracle."""
    picked = random = oracle = 0.0
    for members in groups:
        picked += max(members, key=lambda m: m[0])[1]
        random += sum(m[1] for m in members) / len(members)
        oracle += max(m[1] for m in members)
    n = len(groups) or 1
    return picked / n, random / n, oracle / n


def print_group_table(title, sources, groups_by_src):
    """One table per group definition; the judge-only columns go under Qwen3VL."""
    any_src = sources[0]
    groups = groups_by_src[any_src]
    if not groups:
        print(f"\n{title}\n  (no groups with more than one member; skipped)")
        return
    _, random, oracle = selection_scores(groups)
    _, pairs = pairwise_accuracy(groups)
    rows = [["groups", str(len(groups))] + ["-"] * len(sources),
            ["judged pairs", str(pairs)] + ["-"] * len(sources)]

    stats = {}
    for src in sources:
        acc, _ = pairwise_accuracy(groups_by_src[src])
        (adv_p, adv_s), live = advantage_correlation(groups_by_src[src])
        picked, _, _ = selection_scores(groups_by_src[src])
        stats[src] = dict(acc=acc, adv_p=adv_p, adv_s=adv_s, live=live, picked=picked)

    for name, key, fmt in (("pairwise accuracy", "acc", "{:.3f}"),
                           ("advantage pearson", "adv_p", "{:.3f}"),
                           ("advantage spearman", "adv_s", "{:.3f}"),
                           ("groups with signal", "live", "{:d}")):
        rows.append([name, "-"] + [fmt.format(stats[src][key]) for src in sources])
    rows.append(["judge AM, random pick", f"{100 * random:.2f}"] + ["-"] * len(sources))
    rows.append(["judge AM, reward pick", "-"]
                + [f"{100 * stats[src]['picked']:.2f}" for src in sources])
    rows.append(["judge AM, oracle pick", f"{100 * oracle:.2f}"] + ["-"] * len(sources))
    print_table(title, ["metric", "Qwen3VL"] + [SRC_LABELS[s] for s in sources], rows)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_slot(it_dir, slot, single_slot, bench, prompt_to_idx, reward, encoder,
               args, label, rank=0, world_size=1):
    """Score this rank's share of one tree slot.

    Returns (scores_by_source, ref_am, ref_gm, ref_q); every one of them is
    keyed by the global benchmark index, so the shards merge with ``update``.
    """
    paths = json.load(open(slot_path(it_dir, "geneval2_results", slot, single_slot)))
    judge_file = slot_path(it_dir, "results_geneval2", slot, single_slot, ext="jsonl")
    if not os.path.exists(judge_file):
        raise FileNotFoundError(
            f"{judge_file} is missing: score this slot with GenEval2 first "
            f"(evaluation_optz.py --image_filepath_data "
            f"{slot_path(it_dir, 'geneval2_results', slot, single_slot)} "
            f"--output_file {judge_file})")
    ref_lists = json.load(open(judge_file))

    tokens_file = slot_path(it_dir, "image_tokens", slot, single_slot)
    saved_tokens = None
    if args.token_source in ("both", "tokens"):
        if os.path.exists(tokens_file):
            saved_tokens = json.load(open(tokens_file))
        elif args.token_source == "tokens":
            raise FileNotFoundError(tokens_file)
        else:
            print(f"[{label}] no {tokens_file}; scoring re-encoded images only")
    sources = []
    if saved_tokens is not None:
        sources.append("tokens")
    if args.token_source in ("both", "roundtrip") or not sources:
        sources.append("roundtrip")

    idxs = [prompt_to_idx[p] for p in paths if p in prompt_to_idx]
    if saved_tokens is not None:
        idxs = [i for i in idxs if bench[i]["prompt"] in saved_tokens]
    idxs.sort()
    # Every rank scores a stride of the prompts, so even a single slot keeps
    # all GPUs busy and the shards stay evenly sized.
    idxs = idxs[rank::world_size]

    scores = {src: dict(am={}, gm={}, per_q={}) for src in sources}
    ref_am, ref_gm, ref_q = {}, {}, {}
    for start in range(0, len(idxs), args.batch_size):
        chunk = idxs[start:start + args.batch_size]
        vqa_lists = [bench[i]["vqa_list"] for i in chunk]
        for src in sources:
            if src == "tokens":
                codes = [saved_tokens[bench[i]["prompt"]] for i in chunk]
            else:
                codes = encoder([paths[bench[i]["prompt"]] for i in chunk])
            am, gm, per_q = reward.score_images(codes, vqa_lists)
            for i, a, g, q in zip(chunk, am, gm, per_q):
                scores[src]["am"][i] = a
                scores[src]["gm"][i] = g
                assert len(q) == len(ref_lists[i]), \
                    f"question count mismatch for prompt {i}"
                scores[src]["per_q"][i] = list(q)
        for i in chunk:
            ref = ref_lists[i]
            ref_am[i] = sum(ref) / len(ref)
            ref_gm[i] = geometric_mean(ref)
            ref_q[i] = list(ref)
        if rank == 0:
            done = min(start + args.batch_size, len(idxs))
            print(f"[{label}] {done}/{len(idxs)} (rank 0 of {world_size})",
                  flush=True)
    return scores, ref_am, ref_gm, ref_q


def print_iteration_table(it, slots, sources, lat, ref):
    """Cross-prompt agreement for one iteration, pooled over its slots."""
    keys = [(slot, i) for slot in slots for i in sorted(ref[slot]["am"])]
    ref_am = [ref[slot]["am"][i] for slot, i in keys]
    ref_gm = [ref[slot]["gm"][i] for slot, i in keys]
    ref_q = [p for slot in slots for p in ref[slot]["per_q"]]

    stats = {}
    for src in sources:
        lat_am = [lat[slot][src]["am"][i] for slot, i in keys]
        lat_gm = [lat[slot][src]["gm"][i] for slot, i in keys]
        lat_q = [p for slot in slots for p in lat[slot][src]["per_q"]]
        stats[src] = dict(am=100 * sum(lat_am) / len(keys),
                          gm=100 * sum(lat_gm) / len(keys),
                          am_corr=corr(lat_am, ref_am),
                          gm_corr=corr(lat_gm, ref_gm),
                          q_corr=corr(lat_q, ref_q))

    rows = [["mean AM", f"{100 * sum(ref_am) / len(keys):.2f}"]
            + [f"{stats[src]['am']:.2f}" for src in sources],
            ["mean GM", f"{100 * sum(ref_gm) / len(keys):.2f}"]
            + [f"{stats[src]['gm']:.2f}" for src in sources]]
    for name, key in (("AM", "am_corr"), ("GM", "gm_corr"), ("per-question", "q_corr")):
        for ci, cname in ((0, "pearson"), (1, "spearman")):
            rows.append([f"{name} {cname}", "-"]
                        + [f"{stats[src][key][ci]:.3f}" for src in sources])
    print_table(f"=== iteration {it}: {len(keys)} images "
                f"({len(slots)} slot(s) x {len(keys) // len(slots)} prompts) ===",
                ["metric", "Qwen3VL"] + [SRC_LABELS[s] for s in sources], rows)

    if len(sources) == 2:
        tok = [lat[slot]["tokens"]["am"][i] for slot, i in keys]
        rt = [lat[slot]["roundtrip"]["am"][i] for slot, i in keys]
        p_rt, s_rt = corr(tok, rt)
        gap = sum(t - r for t, r in zip(tok, rt)) / len(keys)
        print_table(f"=== iteration {it}: round-trip cost (tokens vs re-encoded) ===",
                    ["metric", "value"],
                    [["mean AM gap", f"{100 * gap:+.2f}"],
                     ["pearson", f"{p_rt:.3f}"],
                     ["spearman", f"{s_rt:.3f}"]])


def build_groups(slot_groups, sources, lat, ref):
    """Turn lists of co-grouped slots into per-source (reward, judge) groups.

    A prompt missing from any slot of a group is dropped, so every group is
    a complete set of siblings.
    """
    out = {src: [] for src in sources}
    for slots in slot_groups:
        shared = set.intersection(*(set(ref[s]["am"]) for s in slots))
        for i in sorted(shared):
            for src in sources:
                out[src].append([(lat[s][src]["am"][i], ref[s]["am"][i])
                                 for s in slots])
    return out


def print_delta_table(tree, slots, sources, lat, ref, its):
    """Each correction against its own parent draft.

    Only *scored* slots are paired, so --slots restricting the run to part of
    the tree drops the pairs whose other half was not scored.
    """
    first, second = its[0], its[1]
    pairs = [(child, tree["parent"][child]) for child in slots[second]
             if tree["parent"].get(child) in slots[first]]
    if not pairs:
        print("\n=== correction delta AM (parent -> child) ===\n"
              "  (no child slot maps onto a scored draft; skipped)")
        return
    rows_ref, cols = None, {}
    for src in sources:
        d_lat, d_ref = [], []
        for child, parent in pairs:
            shared = sorted(set(ref[second][child]["am"]) & set(ref[first][parent]["am"]))
            for i in shared:
                d_lat.append(lat[second][child][src]["am"][i]
                             - lat[first][parent][src]["am"][i])
                d_ref.append(ref[second][child]["am"][i] - ref[first][parent]["am"][i])
        p, s = corr(d_lat, d_ref)
        agree = sum((x > SIGN_EPS) == (y > SIGN_EPS)
                    and (x < -SIGN_EPS) == (y < -SIGN_EPS)
                    for x, y in zip(d_lat, d_ref)) / len(d_lat)
        cols[src] = [f"{100 * sum(d_lat) / len(d_lat):+.2f}", f"{p:.3f}",
                     f"{s:.3f}", f"{agree:.3f}"]
        rows_ref = f"{100 * sum(d_ref) / len(d_ref):+.2f}"
    names = ["mean delta AM", "delta pearson", "delta spearman",
             f"sign agreement (|d|>{SIGN_EPS})"]
    refs = [rows_ref, "-", "-", "-"]
    rows = [[n, r] + [cols[src][i] for src in sources]
            for i, (n, r) in enumerate(zip(names, refs))]
    print_table(f"=== correction delta AM (parent -> child, {len(pairs)} pair(s)) ===",
                ["metric", "Qwen3VL"] + [SRC_LABELS[s] for s in sources], rows)


def merge_shards(gathered):
    """Merge one slot's per-rank shards back into the single-process layout.

    The shards hold disjoint prompt indices, so the index-keyed dicts merge
    with ``update``; ``per_q`` is flattened in sorted index order, which is the
    order the single-GPU run produced it in, keeping the reward and judge
    per-question lists aligned.
    """
    sources = list(gathered[0][0])
    scores = {src: dict(am={}, gm={}, per_q={}) for src in sources}
    ref_am, ref_gm, ref_q = {}, {}, {}
    for shard_scores, shard_am, shard_gm, shard_q in gathered:
        for src in sources:
            for key in ("am", "gm", "per_q"):
                scores[src][key].update(shard_scores[src][key])
        ref_am.update(shard_am)
        ref_gm.update(shard_gm)
        ref_q.update(shard_q)

    def flatten(by_idx):
        return [p for i in sorted(by_idx) for p in by_idx[i]]

    for src in sources:
        scores[src]["per_q"] = flatten(scores[src]["per_q"])
    return scores, dict(am=ref_am, gm=ref_gm, per_q=flatten(ref_q))


def main():
    args = parse_args()
    rank, world_size, device = init_distributed(args)
    bench = [json.loads(l) for l in open(os.path.expanduser(args.benchmark_data)) if l.strip()]
    if args.max_prompts:
        bench = bench[:args.max_prompts]
    prompt_to_idx = {d["prompt"]: i for i, d in enumerate(bench)}

    its = args.iterations.split(",")
    tree = load_tree(args.results_dir, its)
    keep = set(args.slots.split(",")) if args.slots else None
    slots = {it: [s for s in tree["slots"][it] if keep is None or s in keep]
             for it in its}
    for it in its:
        if not slots[it]:
            raise ValueError(f"--slots left iteration {it} with nothing to score")
    if rank == 0:
        print(f"[layout] {tree['drafts']} draft(s) x {tree['children']} branch(es); "
              + "; ".join(f"iteration {it}: {' '.join(slots[it])}" for it in its))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen2ForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn_implementation
    ).to(device).eval()
    image_start_id = tokenizer.convert_tokens_to_ids("<I0>")

    def forward_hidden(input_ids, attention_mask):
        return model.model(input_ids=input_ids, attention_mask=attention_mask,
                           use_cache=False, return_dict=True).last_hidden_state

    reward = TarLatentVQAReward(
        tokenizer, forward_hidden, model.lm_head, device,
        RewardConfig(batch_size=args.batch_size, answer_suffix=args.answer_suffix), image_start_id)
    encoder = Encoder(os.path.expanduser(args.encoder_path), device)

    lat, ref = {}, {}
    for it in its:
        it_dir = os.path.join(args.results_dir, it)
        lat[it], ref[it] = {}, {}
        for slot in slots[it]:
            shard = score_slot(
                it_dir, slot, tree["single_slot"], bench, prompt_to_idx,
                reward, encoder, args, label=f"iter {it} slot {slot}",
                rank=rank, world_size=world_size)
            gathered = [None] * world_size
            if world_size > 1:
                dist.all_gather_object(gathered, shard)
            else:
                gathered[0] = shard
            lat[it][slot], ref[it][slot] = merge_shards(gathered)

    # Only compare sources every scored slot actually has.
    sources = [src for src in ("tokens", "roundtrip")
               if all(src in lat[it][s] for it in its for s in slots[it])]
    if not sources:
        raise RuntimeError("no image source is available across every slot")

    if rank != 0:
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    for it in its:
        print_iteration_table(it, slots[it], sources, lat[it], ref[it])

    # Round 0: the drafts of one prompt compete with each other.
    if len(slots[its[0]]) > 1:
        print_group_table(
            f"=== round-0 groups: {len(slots[its[0]])} drafts per prompt ===",
            sources, build_groups([slots[its[0]]], sources, lat[its[0]], ref[its[0]]))
    else:
        print("\n=== round-0 groups ===\n"
              "  (only one draft per prompt; re-run generation with --repeat > 1)")

    # Round 1: the branches of one draft compete with each other.
    if len(its) >= 2:
        by_parent = {}
        for child in slots[its[1]]:
            by_parent.setdefault(tree["parent"].get(child, child), []).append(child)
        sibling_sets = [v for v in by_parent.values() if len(v) > 1]
        if sibling_sets:
            print_group_table(
                f"=== round-1 groups: {max(len(v) for v in sibling_sets)} branches "
                f"per draft ===",
                sources, build_groups(sibling_sets, sources, lat[its[1]], ref[its[1]]))
        else:
            print("\n=== round-1 groups ===\n"
                  "  (only one branch per draft; re-run generation with "
                  "--correct_repeat > 1)")
        print_delta_table(tree, slots, sources, lat, ref, its)

    if args.out:
        dump = {it: {slot: {"ref_am": ref[it][slot]["am"],
                            "ref_gm": ref[it][slot]["gm"],
                            **{f"{src}_{key}": lat[it][slot][src][key]
                               for src in sources for key in ("am", "gm")}}
                     for slot in slots[it]} for it in its}
        with open(args.out, "w") as f:
            json.dump(dump, f)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
