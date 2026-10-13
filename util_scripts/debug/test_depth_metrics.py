"""CPU test for the fair depth curve in train_grpo.tree_stats (val/am_upto_d,
val/allpass_upto_d): every prompt counts at every depth with its image after d
rounds, or its last image if the chain ended earlier. Fakes the sampler and the
judge; only the tokenizer is real.

    python util_scripts/debug/test_depth_metrics.py [tokenizer_dir]
"""

import random
import sys
import types

import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from llava.train.rl import train_grpo as tg  # noqa: E402
from llava.train.rl.rollout import RolloutConfig, TreeRollout  # noqa: E402

TOK = sys.argv[1] if len(sys.argv) > 1 else "output_dir/slf_ref_edit_t21/checkpoint-17000"
D = 4


class FakeReward:
    cfg = types.SimpleNamespace(alpha=1.0)

    def score_images(self, codes, vqas):
        # ~30% of images pass every question, the rest fail at least one.
        pq = [[random.uniform(0.6, 1.0) if random.random() < 0.3 else random.uniform(0.0, 0.45)]
              + [random.uniform(0.6, 1.0) for _ in v[1:]] for v in vqas]
        return [sum(q) / len(q) for q in pq], [0.0] * len(pq), pq

    def combine(self, am, gm):
        return am


def make_rollout(tok):
    img0 = tok.convert_tokens_to_ids("<I0>")
    ro = TreeRollout(tok, RolloutConfig(branch=[1] * (D + 1), gen_batch_size=8), img0, 65536, torch.device("cpu"))
    ro._sample_images = lambda model, rows: [[img0 + random.randrange(65536) for _ in range(729)] for _ in rows]
    good = tok("no issues, it matches the prompt.\nCorrection: looks good.", add_special_tokens=False).input_ids
    fix = tok("the cat is not red\nCorrection: make the cat red", add_special_tokens=False).input_ids
    ro._sample_reflections = lambda model, rows: [good if random.random() < 0.2 else fix for _ in rows]
    return ro


def manual_upto(batch, d):
    vals = []
    for leaf in batch.leaves:
        chain = leaf.ancestors()
        vals.append(chain[min(d, len(chain) - 1)].am)
    return sum(vals) / len(vals)


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    tok.padding_side = "left"
    prompts = [{"prompt": f"p{i}", "vqa_list": [("Is the cat red?", "Yes"), ("Is the cat running?", "Yes")],
                "skills": ["attribute", "verb"]} for i in range(40)]
    model = types.SimpleNamespace(training=False, eval=lambda: None, train=lambda: None)
    rw = FakeReward()
    random.seed(11)
    torch.manual_seed(11)
    ro = make_rollout(tok)

    for name, crit in (("own", None), ("oracle", tg.make_critique_fn(rw, 0.5, "oracle"))):
        b = ro.run(model, prompts, branch=[1] * (D + 1), critique_fn=crit)
        tg.score_tree(b, rw, "own", 0.5, 0.8, 0.5, 1.0)
        out = tg.finalize_stats(tg.tree_stats(b, "val/", D, 0.5), "val/")
        assert len(b.leaves) == len(prompts), (name, len(b.leaves))
        assert abs(out["val/am_upto_0"] - out["val/am_0"]) < 1e-9, name
        assert abs(out[f"val/am_upto_{D}"] - out["val/am_final"]) < 1e-9, name
        for d in range(D + 1):
            assert abs(out[f"val/am_upto_{d}"] - manual_upto(b, d)) < 1e-9, (name, d)
        stopped = sum(1 for lf in b.leaves if lf.round < D or lf.looks_good)
        curve = [round(out[f"val/am_upto_{d}"], 3) for d in range(D + 1)]
        am_k = [round(out.get(f"val/am_{k}", float("nan")), 3) for k in range(D + 1)]
        print(f"{name:6s} am_upto {curve}  am_k {am_k}  allpass_upto "
              f"{[round(out[f'val/allpass_upto_{d}'], 2) for d in range(D + 1)]}  chains ended early: {stopped}")
        if name == "own":
            # Every draft gets a round-1 node (a 'looks good' node carries the draft's score).
            assert abs(out["val/am_upto_1"] - out["val/am_1"]) < 1e-9
        else:
            # Passing drafts get no oracle critique: am_1 covers only the failing ones.
            n_pass = sum(1 for r in b.roots if all(p >= 0.5 for p in r.per_question))
            assert n_pass > 0 and abs(out["val/am_upto_1"] - out["val/am_1"]) > 1e-6
    print("OK")


if __name__ == "__main__":
    main()
