"""Build the frozen draft bank for the in-loop generation eval.

One draft per val prompt (first ``--max_prompts`` rows of every set in the
gen-eval YAML), sampled from ``--model_name_or_path`` with the benchmark
settings, scored by the GenEval2 judge, written as JSONL::

    {"set": "geneval2", "idx": 17, "prompt": ..., "codes": [729 ints],
     "per_question": [...], "am": 0.6, "gm": 0.2}

The SFT trainer reads it through ``--gen_eval_bank`` (llava/train/gen_eval.py)
and evaluates every checkpoint's critic and refiner on these same drafts.
Launch with torchrun (multi-node is fine); the judge is started per node the
same way as in training. Skipped by the launch script when the file exists.
"""

import argparse
import json
import os
from typing import Dict, List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, Qwen2ForCausalLM

from llava.train.gen_eval import GenEvalConfig, GenEvaluator, PlainGenerate
from llava.train.rl.pixel_reward import PixelVQAReward
from llava.train.rl.reward import RewardConfig
from llava.train.rl.train_grpo import ImageDecoder, score_nodes
from llava.utils import rank0_print


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--gen_eval_data_path", required=True)
    p.add_argument("--out", required=True, help="bank JSONL to write")
    p.add_argument("--max_prompts", type=int, default=400)
    p.add_argument("--judge_python", required=True)
    p.add_argument("--judge_model", required=True)
    p.add_argument("--judge_port", type=int, default=8765)
    p.add_argument("--ar_path", required=True)
    p.add_argument("--encoder_path", required=True)
    p.add_argument("--decoder_path", required=True)
    p.add_argument("--siglip2_path", default=None)
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--gen_batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn_implementation", default="flash_attention_2")
    return p.parse_args()


def main():
    args = parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Plain Qwen2 load, as the RL trainer does: the checkpoint's vision-tower
    # keys are not needed (drafts are sampled from token ids only).
    model = Qwen2ForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation).to(device).eval()

    cfg = GenEvalConfig(
        data_path=args.gen_eval_data_path, max_prompts=args.max_prompts, bank=None,
        judge_python=args.judge_python, judge_model=args.judge_model, judge_port=args.judge_port,
        ar_path=args.ar_path, encoder_path=args.encoder_path, decoder_path=args.decoder_path,
        siglip2_path=args.siglip2_path, cfg_scale=args.cfg_scale,
        gen_batch_size=args.gen_batch_size, log_images=0, seed=args.seed)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    ev = GenEvaluator(cfg, tokenizer, out_dir, device)
    gen = PlainGenerate(model)

    rows_out: List[Dict] = []
    ev.judge.start()
    try:
        decoder = ImageDecoder(ev.decoder_args, device)
        reward = PixelVQAReward(
            decoder.decode_pils, ev.judge.url,
            RewardConfig(batch_size=16, answer_suffix=cfg.answer_suffix, scale=cfg.scale, alpha=1.0),
            images_per_request=cfg.images_per_request, decode_chunk=cfg.decode_batch)
        rank0_print(f"[bank] judge: {reward.health().get('model')}")
        torch.manual_seed(args.seed * 7919 + 17)   # same draw as the in-loop eval
        for name, rows in ev.val_sets.items():
            bs = cfg.gen_batch_size
            for s in range(0, len(rows), bs):
                chunk = rows[s:s + bs]
                batch = ev.rollout.run(gen, chunk, branch=[1])
                score_nodes(batch.roots, chunk, reward)
                for n in batch.roots:
                    r = chunk[n.prompt_idx]
                    rows_out.append({"set": name, "idx": r["idx"], "prompt": r["prompt"],
                                     "codes": n.codes, "per_question": n.per_question,
                                     "am": n.am, "gm": n.gm})
            rank0_print(f"[bank] {name}: {len(rows)} drafts on rank {rank}")
    finally:
        ev.judge.stop()

    if world > 1:
        gathered: List[List[Dict]] = [None] * world
        dist.all_gather_object(gathered, rows_out)
        rows_out = [r for g in gathered for r in g]
    if rank == 0:
        rows_out.sort(key=lambda r: (r["set"], r["idx"]))
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            for r in rows_out:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, args.out)
        by_set: Dict[str, List[float]] = {}
        for r in rows_out:
            by_set.setdefault(r["set"], []).append(r["am"])
        for name, ams in by_set.items():
            fails = sum(any(q < cfg.fail_threshold for q in r["per_question"])
                        for r in rows_out if r["set"] == name)
            print(f"[bank] {name}: {len(ams)} drafts, am_0 {sum(ams) / len(ams):.4f}, "
                  f"{fails} fail at least one question", flush=True)
        print(f"[bank] wrote {len(rows_out)} rows to {args.out}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
