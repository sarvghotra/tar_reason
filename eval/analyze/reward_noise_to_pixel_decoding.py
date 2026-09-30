"""How much of the GRPO pixel reward is de-tokenizer noise?

The pixel reward (``llava/train/rl/pixel_reward.py``) decodes image codes to a
PNG and asks the judge the prompt's VQA questions. The AR de-tokenizer
*samples* (``MMAutoEncoder.ar_sample``: temperature 1, no top-k, CFG 4), so the
same 729 codes decode to different pixels, and a different score, on every
call. Training scores every node exactly once, so that noise goes straight
into the advantages.

This script samples GRPO-style groups with the policy, then decodes and scores
every image ``--decode_repeats`` times through the *same* code path as
training (``PixelVQAReward.score_images``), and splits the within-group
reward variance GRPO sees into

    V_obs  = within-group variance of a single-decode reward  (what GRPO sees)
    V_dec  = variance across re-decodes of the same codes      (pure noise)
    V_sig  = V_obs - V_dec                                     (the part that
                                                                depends on the codes)

``noise_frac = V_dec / V_obs`` is the share of the advantage variance that is
decoding noise; ``reliability_k = V_sig / (V_sig + V_dec / k)`` is the squared
correlation between the true and the observed advantage when the reward
averages ``k`` decodes. A test-retest correlation and a sign-agreement rate
check the same thing without the variance algebra.

Groups follow ``--branch`` like ``train_grpo.py``:
  * ``--branch 8``   -> 8 drafts per prompt; a group = the drafts of a prompt.
  * ``--branch 1,8`` -> 1 draft + 8 refine children; a group = the refined
    siblings of one parent. "Looks good" children are dropped: they are not
    decoded in training (they inherit the parent's score). The parent's own
    decode noise is common to the whole group and cancels in
    ``score(child) - score(parent)``, so only the children are scored here.

A judge check re-scores identical PNGs twice (the second time in reversed
batch order) to separate judge/batching non-determinism from decoder noise.

Outputs in ``--output_dir``: ``codes_shard*.jsonl`` (sampled groups, reused
on restart), ``scores_shard*.jsonl`` (per-question probabilities per repeat),
``judge_recheck_shard*.jsonl``, ``strips/`` (the repeats of a few images side
by side), and after ``--analyze_only`` (or a single-shard run)
``summary.json`` + ``per_node.jsonl``.

Needs a running judge (``llava/train/rl/pixel_reward_server.py``); see
``reward_noise_to_pixel_decoding.sh``.
"""

import argparse
import glob
import itertools
import json
import os
import sys
import time
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # Policy
    p.add_argument("--model_name_or_path", default=None)
    p.add_argument("--adapter_dir", default=None,
                   help="Optional LoRA checkpoint (e.g. output_dir/<run>/checkpoint-500), merged in.")
    p.add_argument("--attn_implementation", default="flash_attention_2")
    # Prompts
    p.add_argument("--prompts", default=None,
                   help="GenEval2 .jsonl, or a data .yaml as used by train_grpo.py")
    p.add_argument("--num_prompts", type=int, default=64)
    p.add_argument("--prompt_offset", type=int, default=0)
    p.add_argument("--prompts_per_rollout", type=int, default=2,
                   help="Prompts per TreeRollout.run call (images are still batched by --gen_batch_size).")
    # Rollout (defaults = rl_ft_oracle_t4)
    p.add_argument("--branch", default="8")
    p.add_argument("--scale", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--gen_seq_len", type=int, default=729)
    p.add_argument("--img_temperature", type=float, default=1.0)
    p.add_argument("--img_top_k", type=int, default=1200)
    p.add_argument("--img_top_p", type=float, default=0.95)
    p.add_argument("--reflect_tokens", type=int, default=128)
    p.add_argument("--reflect_temperature", type=float, default=1.0)
    p.add_argument("--reflect_top_k", type=int, default=0)
    p.add_argument("--reflect_top_p", type=float, default=0.95)
    p.add_argument("--min_refines", type=int, default=2)
    p.add_argument("--refine_resample_tries", type=int, default=3)
    p.add_argument("--gen_batch_size", type=int, default=16)
    # De-tokenizer (defaults = train_grpo.py::ImageDecoder)
    p.add_argument("--ar_path", default=None)
    p.add_argument("--encoder_path", default=None)
    p.add_argument("--decoder_path", default=None)
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--decode_temperature", type=float, default=1.0,
                   help="AR de-tokenizer sampling temperature (training uses the default 1.0).")
    p.add_argument("--decode_top_k", type=int, default=0)
    p.add_argument("--decode_top_p", type=float, default=1.0)
    p.add_argument("--decode_batch", type=int, default=24)
    p.add_argument("--decode_repeats", type=int, default=8)
    # Judge
    p.add_argument("--reward_server_url", default=None)
    p.add_argument("--images_per_request", type=int, default=8)
    p.add_argument("--answer_suffix", default="geneval2")
    p.add_argument("--alpha", type=float, default=1.0, help="reward = a*AM + (1-a)*GM")
    p.add_argument("--judge_recheck", type=int, default=32,
                   help="Images per shard whose identical PNG is scored twice.")
    # IO / sharding
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--save_strips", type=int, default=6,
                   help="Images per shard saved as a strip of all their decodes.")
    p.add_argument("--seed", type=int, default=421)
    p.add_argument("--analyze_only", action="store_true",
                   help="Skip sampling/scoring; merge all shards in --output_dir and analyze.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_jsonl(path) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_prompts(path) -> List[Dict]:
    if path.endswith((".yaml", ".yml")):
        from llava.train.rl.dataset import GenEval2PromptDataset
        return GenEval2PromptDataset._load_rows(path)
    rows = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                vqa = [tuple(qa) for qa in obj["vqa_list"]]
                if vqa:
                    rows.append({"prompt": str(obj["prompt"]).strip(), "vqa_list": vqa})
    return rows


# ---------------------------------------------------------------------------
# Sampling + scoring (GPU)
# ---------------------------------------------------------------------------

class Decoder:
    """``train_grpo.py::ImageDecoder`` with the de-tokenizer's sampling knobs exposed."""

    def __init__(self, args, device):
        import torch
        from tok.mm_autoencoder import MMAutoEncoder
        self.torch = torch
        self.tok = MMAutoEncoder(
            ar_path=args.ar_path, encoder_path=args.encoder_path, decoder_path=args.decoder_path,
            encoder_args={"input_type": "rec"}, decoder_args={},
        ).eval().to(dtype=torch.bfloat16, device=device)
        self.tok.ar_model.cls_token_num = args.gen_seq_len
        self.tok.encoder.pool_scale = args.scale + 1
        self.device = device
        self.sample_args = {"cfg_scale": args.cfg_scale, "temperature": args.decode_temperature,
                            "top_k": args.decode_top_k, "top_p": args.decode_top_p}
        self.decode_batch = max(1, args.decode_batch)
        # The first `capture` decoded images of a pass are kept for the strips.
        self.capture = 0
        self.captured = []

    def __call__(self, codes_list):
        torch = self.torch
        from PIL import Image
        out = []
        with torch.inference_mode():
            for start in range(0, len(codes_list), self.decode_batch):
                chunk = codes_list[start:start + self.decode_batch]
                codes = torch.tensor([list(c) for c in chunk], dtype=torch.long, device=self.device)
                imgs = self.tok.decode_from_encoder_indices(codes, self.sample_args)
                out.extend(Image.fromarray(im.numpy()) for im in imgs)
        room = self.capture - len(self.captured)
        if room > 0:
            self.captured.extend(out[:room])
        return out


def sample_groups(args, device, prompts, prompt_ids) -> List[Dict]:
    """Roll out the policy and return one record per scorable node of the last round."""
    import torch
    from transformers import AutoTokenizer, Qwen2ForCausalLM
    from llava.train.rl.rollout import RolloutConfig, TreeRollout

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    image_start_id = tokenizer.convert_tokens_to_ids("<I0>")
    num_image_tokens = sum(1 for t in tokenizer.get_vocab() if t.startswith("<I") and t[2:-1].isdigit())

    model = Qwen2ForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation)
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir).merge_and_unload()
    model.to(device).eval()

    branch = [int(x) for x in args.branch.split(",") if x.strip()]
    rcfg = RolloutConfig(
        branch=branch, scale=args.scale, gen_seq_len=args.gen_seq_len,
        img_temperature=args.img_temperature, img_top_k=args.img_top_k, img_top_p=args.img_top_p,
        reflect_tokens=args.reflect_tokens, reflect_sample=True,
        reflect_temperature=args.reflect_temperature, reflect_top_k=args.reflect_top_k,
        reflect_top_p=args.reflect_top_p, gen_batch_size=args.gen_batch_size,
        min_refines=args.min_refines, refine_resample_tries=args.refine_resample_tries)
    rollout = TreeRollout(tokenizer, rcfg, image_start_id, num_image_tokens, device)

    torch.manual_seed(args.seed * 1_000_003 + args.shard_id)
    records = []
    step = max(1, args.prompts_per_rollout)
    for start in range(0, len(prompts), step):
        chunk, chunk_ids = prompts[start:start + step], prompt_ids[start:start + step]
        t0 = time.time()
        batch = rollout.run(model, chunk, branch=branch)
        last = batch.nodes_by_round[-1]
        if len(branch) == 1:
            nodes = last
            keys = [f"p{chunk_ids[n.prompt_idx]}" for n in nodes]
        else:
            parent_pos = {id(n): i for i, n in enumerate(batch.nodes_by_round[-2])}
            nodes = [n for n in last if not n.looks_good]
            keys = [f"p{chunk_ids[n.prompt_idx]}/n{parent_pos[id(n.parent)]}" for n in nodes]
        for n, key in zip(nodes, keys):
            records.append({
                "node": f"s{args.shard_id}_{len(records)}",
                "group": key,
                "prompt_idx": chunk_ids[n.prompt_idx],
                "round": n.round,
                "reflection": n.reflection,
                "forced_refine": n.forced_refine,
                "codes": n.codes,
            })
        print(f"[shard {args.shard_id}] rollout prompts {start + len(chunk)}/{len(prompts)} "
              f"-> {len(records)} nodes ({time.time() - t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return records


def run_shard(args):
    import torch
    from llava.train.rl.pixel_reward import PixelVQAReward, _png_b64
    from llava.train.rl.reward import RewardConfig

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    sid = args.shard_id
    codes_path = os.path.join(args.output_dir, f"codes_shard{sid}.jsonl")
    scores_path = os.path.join(args.output_dir, f"scores_shard{sid}.jsonl")
    recheck_path = os.path.join(args.output_dir, f"judge_recheck_shard{sid}.jsonl")

    all_prompts = load_prompts(args.prompts)
    sel = list(range(args.prompt_offset, min(len(all_prompts), args.prompt_offset + args.num_prompts)))
    my_ids = sel[sid::args.num_shards]
    my_prompts = [all_prompts[i] for i in my_ids]
    if sid == 0:
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    records = read_jsonl(codes_path)
    if records:
        print(f"[shard {sid}] reusing {len(records)} sampled nodes from {codes_path}", flush=True)
    else:
        records = sample_groups(args, device, my_prompts, my_ids)
        tmp = codes_path + ".tmp"
        with open(tmp, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, codes_path)
    if not records:
        print(f"[shard {sid}] no scorable nodes", flush=True)
        return

    decoder = Decoder(args, device)
    reward = PixelVQAReward(
        decoder, args.reward_server_url,
        RewardConfig(answer_suffix=args.answer_suffix, scale=args.scale, alpha=args.alpha),
        images_per_request=args.images_per_request)
    info = reward.health()
    print(f"[shard {sid}] judge {info.get('model')} ({info.get('answer_id_mode')})", flush=True)

    codes = [r["codes"] for r in records]
    vqa = [all_prompts[r["prompt_idx"]]["vqa_list"] for r in records]
    done = {row["repeat"] for row in read_jsonl(scores_path)}
    n_strip = min(args.save_strips, len(records))
    strips = [[] for _ in range(n_strip)]
    strip_am = [[] for _ in range(n_strip)]
    for rep in range(args.decode_repeats):
        if rep in done:
            continue
        torch.manual_seed(args.seed * 7919 + 1000 * rep + sid)
        decoder.capture, decoder.captured = n_strip, []
        t0 = time.time()
        am, gm, per_q = reward.score_images(codes, vqa)
        append_jsonl(scores_path, [{"repeat": rep, "node": r["node"], "am": a, "gm": g, "per_question": q}
                                   for r, a, g, q in zip(records, am, gm, per_q)])
        for k, im in enumerate(decoder.captured):
            strips[k].append(im)
            strip_am[k].append(am[k])
        print(f"[shard {sid}] repeat {rep + 1}/{args.decode_repeats}: {len(codes)} images "
              f"in {time.time() - t0:.0f}s", flush=True)
    decoder.capture = 0

    # Strips: every decode of the same codes, side by side (only for repeats run now).
    if n_strip and strips[0]:
        from PIL import Image
        sdir = os.path.join(args.output_dir, "strips")
        os.makedirs(sdir, exist_ok=True)
        for k in range(n_strip):
            w, h = strips[k][0].size
            canvas = Image.new("RGB", (w * len(strips[k]), h))
            for i, im in enumerate(strips[k]):
                canvas.paste(im, (i * w, 0))
            name = records[k]["node"]
            canvas.save(os.path.join(sdir, f"{name}.png"))
            with open(os.path.join(sdir, f"{name}.json"), "w") as f:
                json.dump({"prompt": all_prompts[records[k]["prompt_idx"]]["prompt"],
                           "reflection": records[k]["reflection"],
                           "am_per_decode": strip_am[k]}, f, indent=2)

    # Judge check: the identical PNGs scored twice, second time in reversed batch order.
    if args.judge_recheck > 0 and not os.path.exists(recheck_path):
        k = min(args.judge_recheck, len(records))
        torch.manual_seed(args.seed * 7919 + 999_999 + sid)
        items = [{"png_b64": _png_b64(im), "vqa": [list(qa) for qa in v]}
                 for im, v in zip(decoder(codes[:k]), vqa[:k])]

        def score_all(order):
            out = [None] * len(order)
            for s in range(0, len(order), args.images_per_request):
                idx = order[s:s + args.images_per_request]
                for i, pq in zip(idx, reward._post_score([items[i] for i in idx])):
                    out[i] = pq
            return out

        first = score_all(list(range(k)))
        second = score_all(list(range(k))[::-1])
        append_jsonl(recheck_path, [{"node": records[i]["node"], "per_question_a": first[i],
                                     "per_question_b": second[i]} for i in range(k)])
        print(f"[shard {sid}] judge recheck on {k} images done", flush=True)
    print(f"[shard {sid}] done", flush=True)


# ---------------------------------------------------------------------------
# Analysis (CPU)
# ---------------------------------------------------------------------------

def analyze(out_dir, alpha: float):
    import numpy as np

    records = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "codes_shard*.jsonl"))):
        for r in read_jsonl(path):
            records[r["node"]] = r
    reps: Dict[str, Dict[int, Dict]] = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "scores_shard*.jsonl"))):
        for row in read_jsonl(path):
            reps.setdefault(row["node"], {})[row["repeat"]] = row
    if not reps:
        raise SystemExit(f"no scores in {out_dir}")

    # Use only the repeats every node has, so all nodes are compared on equal footing.
    common = sorted(set.intersection(*(set(v) for v in reps.values())))
    R = len(common)
    if R < 2:
        raise SystemExit(f"need >= 2 decode repeats per node, have {R}")

    def score(row):
        return alpha * row["am"] + (1.0 - alpha) * row["gm"]

    X = {n: np.array([score(reps[n][r]) for r in common]) for n in reps}          # (R,)
    PQ = {n: np.array([reps[n][r]["per_question"] for r in common]) for n in reps}  # (R, Q)

    groups: Dict[str, List[str]] = {}
    for n in reps:
        groups.setdefault(records[n]["group"], []).append(n)
    n_singleton = sum(len(v) < 2 for v in groups.values())
    groups = {g: v for g, v in groups.items() if len(v) >= 2}
    G = {g: np.stack([X[n] for n in v]) for g, v in groups.items()}                # (n_g, R)

    # --- variance decomposition (per group, then averaged over groups) ---
    v_obs, v_dec, grpo_std, zero_var = [], [], [], 0
    for M in G.values():
        v_obs.append(M.var(axis=0, ddof=1).mean())          # single-decode spread, averaged over repeats
        v_dec.append(M.var(axis=1, ddof=1).mean())          # re-decode spread of the same codes
        grpo_std.append(M.std(axis=0, ddof=0).mean())       # the std normalize_groups() divides by
        zero_var += int((M.std(axis=0) < 1e-8).sum())
    V_obs, V_dec = float(np.mean(v_obs)), float(np.mean(v_dec))
    V_sig = V_obs - V_dec
    noise_frac = V_dec / V_obs if V_obs > 0 else float("nan")
    rel = {k: (max(V_sig, 0.0) / (max(V_sig, 0.0) + V_dec / k)) if V_obs > 0 else float("nan")
           for k in (1, 2, 4, 8, 16)}

    # --- test-retest: advantages from two independent decodes of the same codes ---
    corrs = []
    for r1, r2 in itertools.combinations(range(R), 2):
        a = np.concatenate([M[:, r1] - M[:, r1].mean() for M in G.values()])
        b = np.concatenate([M[:, r2] - M[:, r2].mean() for M in G.values()])
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 0:
            corrs.append(float((a * b).sum() / den))

    # --- sign agreement / pair flips vs the leave-one-out mean of the other decodes ---
    agree, n_agree, flips, n_pairs = 0, 0, 0, 0
    for M in G.values():
        for r in range(R):
            single = M[:, r] - M[:, r].mean()
            ref = np.delete(M, r, axis=1).mean(axis=1)
            ref = ref - ref.mean()
            ok = np.abs(ref) > 1e-6
            agree += int((np.sign(single[ok]) == np.sign(ref[ok])).sum())
            n_agree += int(ok.sum())
            for i, j in itertools.combinations(range(M.shape[0]), 2):
                d_ref = ref[i] - ref[j]
                if abs(d_ref) > 1e-6:
                    n_pairs += 1
                    flips += int(np.sign(single[i] - single[j]) != np.sign(d_ref))

    # --- per-node noise, bucketed by the node's mean score ---
    nodes = [n for v in groups.values() for n in v]
    means = np.array([X[n].mean() for n in nodes])
    sds = np.array([X[n].std(ddof=1) for n in nodes])
    ranges = np.array([X[n].max() - X[n].min() for n in nodes])
    buckets = {}
    for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        m = (means >= lo) & (means < hi)
        if m.any():
            buckets[f"[{lo:.2f},{min(hi, 1.0):.2f}]"] = {"n": int(m.sum()),
                                                         "decode_sd": float(np.sqrt((sds[m] ** 2).mean()))}

    # --- per-question: how often a question's answer crosses 0.5 across decodes ---
    q_sd, q_unstable, q_total = [], 0, 0
    for n in nodes:
        P = PQ[n]
        q_sd.append(P.std(axis=0, ddof=1))
        q_unstable += int(((P.min(axis=0) < 0.5) & (P.max(axis=0) > 0.5)).sum())
        q_total += P.shape[1]
    q_sd = np.concatenate(q_sd)

    # --- judge determinism on identical PNGs ---
    judge = None
    rows = [r for p in sorted(glob.glob(os.path.join(out_dir, "judge_recheck_shard*.jsonl")))
            for r in read_jsonl(p)]
    if rows:
        am_diff = np.array([abs(np.mean(r["per_question_a"]) - np.mean(r["per_question_b"])) for r in rows])
        q_diff = np.concatenate([np.abs(np.array(r["per_question_a"]) - np.array(r["per_question_b"]))
                                 for r in rows])
        judge = {"n_images": len(rows), "am_abs_diff_mean": float(am_diff.mean()),
                 "am_abs_diff_max": float(am_diff.max()), "question_abs_diff_max": float(q_diff.max())}

    summary = {
        "output_dir": out_dir, "alpha": alpha, "decode_repeats": R,
        "n_groups": len(G), "n_nodes": len(nodes), "dropped_singleton_groups": n_singleton,
        "mean_group_size": float(np.mean([M.shape[0] for M in G.values()])),
        "mean_score": float(means.mean()),
        "V_obs_single_decode": V_obs, "V_dec": V_dec, "V_sig": V_sig,
        "sd_obs": float(np.sqrt(V_obs)), "sd_dec": float(np.sqrt(V_dec)),
        "sd_sig": float(np.sqrt(max(V_sig, 0.0))),
        "grpo_within_group_std": float(np.mean(grpo_std)),
        "noise_frac": noise_frac,
        "reliability_by_decodes_averaged": {str(k): v for k, v in rel.items()},
        "test_retest_adv_corr": float(np.mean(corrs)) if corrs else float("nan"),
        "adv_sign_agreement_loo": agree / max(n_agree, 1),
        "pair_order_flip_rate": flips / max(n_pairs, 1),
        "zero_var_group_frac_single_decode": zero_var / (len(G) * R),
        "node_decode_sd_percentiles": {f"p{q}": float(np.percentile(sds, q)) for q in (50, 90, 99)},
        "node_decode_range_mean": float(ranges.mean()),
        "decode_sd_by_mean_score": buckets,
        "question_prob_sd_mean": float(q_sd.mean()),
        "question_crosses_0.5_frac": q_unstable / max(q_total, 1),
        "judge_recheck": judge,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "per_node.jsonl"), "w") as f:
        for n, m, s in zip(nodes, means, sds):
            r = records[n]
            f.write(json.dumps({"node": n, "group": r["group"], "prompt_idx": r["prompt_idx"],
                                "mean": float(m), "sd": float(s), "scores": X[n].tolist(),
                                "reflection": r["reflection"]}) + "\n")

    print(f"\n=== reward noise from pixel decoding: {out_dir} ===")
    print(f"groups {len(G)}  nodes {len(nodes)}  mean group size {summary['mean_group_size']:.1f}  "
          f"decodes/node {R}  mean score {summary['mean_score']:.3f}")
    print(f"within-group sd, single decode (what GRPO sees)  {summary['sd_obs']:.4f}")
    print(f"re-decode sd of identical codes (noise)          {summary['sd_dec']:.4f}")
    print(f"code-dependent sd (signal)                       {summary['sd_sig']:.4f}")
    print(f"noise_frac = V_dec / V_obs                       {noise_frac:.3f}")
    print("reliability (true-vs-observed adv corr^2) when averaging k decodes: " +
          "  ".join(f"k={k}: {v:.3f}" for k, v in rel.items()))
    print(f"test-retest advantage corr (1 decode vs 1 decode) {summary['test_retest_adv_corr']:.3f}")
    print(f"advantage sign agreement vs other decodes        {summary['adv_sign_agreement_loo']:.3f}")
    print(f"sibling pair order flip rate                     {summary['pair_order_flip_rate']:.3f}")
    print(f"questions whose prob crosses 0.5 across decodes  {summary['question_crosses_0.5_frac']:.3f}")
    print("decode sd by mean score: " +
          "  ".join(f"{k}: {v['decode_sd']:.3f} (n={v['n']})" for k, v in buckets.items()))
    if judge:
        print(f"judge on identical PNGs: mean |dAM| {judge['am_abs_diff_mean']:.2e}, "
              f"max |dAM| {judge['am_abs_diff_max']:.2e} over {judge['n_images']} images")
    print(f"wrote {os.path.join(out_dir, 'summary.json')}")
    return summary


def main():
    args = parse_args()
    if not args.analyze_only:
        missing = [k for k in ("model_name_or_path", "prompts", "ar_path", "encoder_path",
                               "decoder_path", "reward_server_url") if not getattr(args, k)]
        if missing:
            raise SystemExit(f"missing required args: {', '.join('--' + m for m in missing)}")
        run_shard(args)
    if args.analyze_only or args.num_shards == 1:
        analyze(args.output_dir, args.alpha)


if __name__ == "__main__":
    main()
