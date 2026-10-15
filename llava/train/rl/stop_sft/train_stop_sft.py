"""Stop-SFT (task 3a): supervised fine-tuning of the critique on labelled images.

One example = system + iterative prefix + prompt + one image, in the MDP layout a rollout
critiques from (``TreeRollout._state`` of a round-k node: the prompt prefix, then the
image as a draft), then ``<im_end>\\nSelf-reflect:`` and the target critique. The target is
spliced exactly like a frozen critique in the rollout (``" " + text``) and closed by the token
the critic stops on: ``<|im_end|>`` after "looks good" (the SFT turn ends there), ``\\n<im_start>``
after a fix (the next image follows). Cross-entropy on the target tokens only (the image
vocabulary is masked as in generation); a fresh LoRA on the base, which stays the KL reference
of later stages. Data: ``build_data.py``.

    torchrun --nproc_per_node=4 -m llava.train.rl.stop_sft.train_stop_sft \\
        --model_name_or_path <F-merged> --data data.jsonl --output_dir output_dir/<run>

The saved adapter (``<output_dir>/final``, plus ``checkpoint-<step>``) loads with train_grpo.py
``--init_adapter`` for the depth-4 eval.
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from llava.train.rl.grpo import KIND_NONE, KIND_TXT, masked_token_logprobs
from llava.train.rl.rollout import LOOKS_GOOD_RE, RolloutConfig, TreeRollout
from llava.train.rl.train_grpo import is_main, load_policy, rank0_print


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--data", required=True, help="build_data.py output (JSONL)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=16, help="examples per optimizer step, all ranks")
    p.add_argument("--micro_batch_size", type=int, default=4, help="examples per forward, per rank")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report_to", default="none", choices=["none", "wandb"])
    p.add_argument("--run_name", default=None)
    return p.parse_args(argv)


class Encoder:
    """Token rows for stop-SFT examples, built with the rollout's own helpers."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.img_start = tokenizer.convert_tokens_to_ids("<I0>")
        n_img = sum(1 for t in tokenizer.get_vocab() if t.startswith("<I") and t[2:-1].isdigit())
        self.img_end = self.img_start + n_img
        # Only the token helpers are used: no model, no generation.
        self.rollout = TreeRollout(tokenizer, RolloutConfig(), self.img_start, n_img, device="cpu")
        self.stop_lg = [tokenizer.convert_tokens_to_ids("<|im_end|>")]
        self.stop_fix = self.rollout._encode("\n<im_start>")

    def encode(self, ex):
        """-> (context ids, target ids). Context = MDP state + reflect splice."""
        ctx = (self.rollout.make_prefix(ex["prompt"]) + [self.img_start + c for c in ex["codes"]]
               + self.rollout.reflect_splice)
        lg = LOOKS_GOOD_RE.search(ex["target"]) is not None
        tgt = self.rollout._encode(" " + ex["target"]) + (self.stop_lg if lg else self.stop_fix)
        return ctx, tgt

    def batch(self, examples, pad_id):
        """Right-padded tensors with train_mask on the target tokens."""
        rows = [self.encode(e) for e in examples]
        width = max(len(c) + len(t) for c, t in rows)
        B = len(rows)
        input_ids = torch.full((B, width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, width), dtype=torch.long)
        train_mask = torch.zeros((B, width), dtype=torch.bool)
        pos_kind = torch.full((B, width), KIND_NONE, dtype=torch.long)
        for i, (c, t) in enumerate(rows):
            L = len(c) + len(t)
            input_ids[i, :L] = torch.tensor(c + t)
            attention_mask[i, :L] = 1
            train_mask[i, len(c):L] = True
            pos_kind[i, len(c):L] = KIND_TXT
        return dict(input_ids=input_ids, attention_mask=attention_mask,
                    train_mask=train_mask, pos_kind=pos_kind)


def main():
    args = parse_args()
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    else:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = Encoder(tokenizer)
    data = [json.loads(l) for l in open(args.data) if l.strip()]
    rank0_print(f"{len(data)} examples ({sum(e['passed'] for e in data)} pass)")

    model = load_policy(args, device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    assert args.batch_size % (world * args.micro_batch_size) == 0, \
        "--batch_size must be a multiple of world_size * --micro_batch_size"
    accum = args.batch_size // (world * args.micro_batch_size)
    steps = max(1, math.floor(args.epochs * len(data) / args.batch_size))
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * steps), steps)
    rank0_print(f"{steps} steps of {args.batch_size} examples ({accum} micro-batches of "
                f"{args.micro_batch_size} per rank), lr {args.learning_rate}")

    if args.report_to == "wandb" and is_main():
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT", "tar_reasoning"), name=args.run_name,
                   id=args.run_name, resume="allow", config=vars(args))

    # Same shuffled order on every rank; each rank takes its slice of every step.
    order = []
    rng = random.Random(args.seed)
    while len(order) < steps * args.batch_size:
        epoch = list(range(len(data)))
        rng.shuffle(epoch)
        order += epoch
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = order[(step - 1) * args.batch_size: step * args.batch_size]
        mine = idx[rank::world]
        tot_lp, tot_n, tot_lg_ok = 0.0, 0, 0
        for m in range(accum):
            ex = [data[i] for i in mine[m * args.micro_batch_size:(m + 1) * args.micro_batch_size]]
            b = {k: v.to(device) for k, v in enc.batch(ex, tokenizer.pad_token_id).items()}
            lp = masked_token_logprobs(model, b["input_ids"], b["attention_mask"], b["train_mask"],
                                       b["pos_kind"], enc.img_start, enc.img_end)
            n_tok = int(b["train_mask"].sum())
            # Token mean over the whole optimizer batch (all ranks): scale by the expected count.
            loss = -lp.sum() / max(n_tok, 1) / accum
            loss.backward()
            tot_lp += float(lp.sum())
            tot_n += n_tok
        if world > 1:
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world)
            t = torch.tensor([tot_lp, tot_n], device=device, dtype=torch.float64)
            dist.all_reduce(t)
            tot_lp, tot_n = float(t[0]), int(t[1])
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        stats = {"train/nll": -tot_lp / max(tot_n, 1), "train/grad_norm": float(gnorm),
                 "train/lr": scheduler.get_last_lr()[0]}
        if is_main():
            print(f"step={step}/{steps} " + json.dumps({k: round(v, 5) for k, v in stats.items()})
                  + f" {time.time() - t0:.0f}s", flush=True)
            if args.report_to == "wandb":
                wandb.log(stats, step=step)
        if step % args.save_steps == 0 or step == steps:
            name = "final" if step == steps else f"checkpoint-{step}"
            if is_main():
                ck = os.path.join(args.output_dir, name)
                model.save_pretrained(ck)
                with open(os.path.join(ck, "state.json"), "w") as f:
                    json.dump({"step": step}, f)
                rank0_print(f"Saved {ck}")
            if world > 1:
                dist.barrier()
    if args.report_to == "wandb" and is_main():
        wandb.finish()
    rank0_print("Done.")


if __name__ == "__main__":
    main()
