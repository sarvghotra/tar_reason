"""CPU test for --stop_reward: in K critiques x M images rollouts a "looks good"
critique is rewarded +R when its parent passes every question and -R otherwise,
refining critiques keep the refine reward, and the critique-level advantage then
favours stopping on passing parents and refining on failing ones. Fakes the
sampler and the judge; only the tokenizer is real.

    python util_scripts/debug/test_stop_reward.py [tokenizer_dir]
"""

import random
import sys
import types

import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from llava.train.rl import train_grpo as tg  # noqa: E402
from llava.train.rl.grpo import GRPOConfig  # noqa: E402
from llava.train.rl.rollout import RolloutConfig, TreeRollout  # noqa: E402

TOK = sys.argv[1] if len(sys.argv) > 1 else "output_dir/slf_ref_edit_t21/checkpoint-17000"
K, M, R = 3, 4, 0.2


class FakeReward:
    """About half the images pass every question; the rest fail the first one."""
    cfg = types.SimpleNamespace(alpha=1.0)

    def score_images(self, codes, vqas):
        pq = [[random.uniform(0.6, 1.0) if random.random() < 0.5 else random.uniform(0.0, 0.45)]
              + [random.uniform(0.6, 1.0) for _ in v[1:]] for v in vqas]
        return [sum(q) / len(q) for q in pq], [0.0] * len(pq), pq

    def combine(self, am, gm):
        return am


def make_rollout(tok):
    img0 = tok.convert_tokens_to_ids("<I0>")
    ro = TreeRollout(tok, RolloutConfig(branch=[1, K], gen_batch_size=8, images_per_critique=M,
                                        context_mode="mdp"), img0, 65536, torch.device("cpu"))
    ro._sample_images = lambda model, rows: [[img0 + random.randrange(65536) for _ in range(729)] for _ in rows]
    good = tok("no issues, it matches the prompt.\nCorrection: looks good.", add_special_tokens=False).input_ids
    fix = tok("the cat is not red\nCorrection: make the cat red", add_special_tokens=False).input_ids
    ro._sample_reflections = lambda model, rows: [good if random.random() < 0.4 else fix for _ in rows]
    return ro


def passes(n):
    return all(q >= 0.5 for q in n.per_question)


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    tok.padding_side = "left"
    prompts = [{"prompt": f"p{i}", "vqa_list": [("Is the cat red?", "Yes"), ("Is the cat running?", "Yes")],
                "skills": ["attribute", "verb"]} for i in range(40)]
    model = types.SimpleNamespace(training=True, eval=lambda: None, train=lambda: None)
    args = types.SimpleNamespace(group="parent", images_per_critique=M)
    gcfg = GRPOConfig(adv_norm="std", adv_std_floor=0.04)
    rw = FakeReward()
    random.seed(7)
    torch.manual_seed(7)

    ro = make_rollout(tok)
    b = ro.run(model, prompts)
    tg.score_tree(b, rw, "own", 0.0, 1.0, 0.5, 2.0, stop_reward=R)
    tg.assign_advantages(b, args, gcfg)

    kids = b.nodes_by_round[1]
    n_lg = {True: 0, False: 0}
    for n in kids:
        if n.looks_good:
            ok = passes(n.parent)
            assert n.reward == (R if ok else -R), (n.reward, ok)
            n_lg[ok] += 1
        else:
            assert n.reward == tg.refine_reward(n, rw, 0.5, 2.0)
    assert n_lg[True] > 0 and n_lg[False] > 0, n_lg

    # Critique advantage, in parent groups with both a looks-good and a refining critique:
    # on a failing parent refining always wins (refine rewards are > -R); on a passing parent
    # stopping wins unless the refinement really raised the score by more than R, so check
    # the mean advantage gap there.
    checked = {True: [], False: 0}
    for root in b.roots:
        lg = [c for c in root.children if c.looks_good]
        rf = [c for c in root.children if not c.looks_good]
        if not lg or not rf:
            continue
        if passes(root):
            gap = sum(c.adv_txt for c in lg) / len(lg) - sum(c.adv_txt for c in rf) / len(rf)
            checked[True].append(gap)
        else:
            assert all(a.adv_txt < b_.adv_txt for a in lg for b_ in rf), "refine should beat stop on a failing image"
            checked[False] += 1
    assert checked[True] and checked[False] > 0, checked
    mean_gap = sum(checked[True]) / len(checked[True])
    assert mean_gap > 0, mean_gap

    # stop_reward = 0 keeps the old stop_penalty behaviour.
    tg.score_tree(b, rw, "own", 0.5, 0.8, 0.5, 2.0)
    for n in kids:
        if n.looks_good:
            ps = rw.combine(n.parent.am, n.parent.gm)
            assert n.reward == -0.5 * max(0.0, 0.8 - ps)
    print(f"looks-good critiques on passing / failing parents: {n_lg[True]} / {n_lg[False]}; "
          f"mixed groups (pass / fail parent): {len(checked[True])} / {checked[False]}; "
          f"passing parents: stop beats refine in {sum(g > 0 for g in checked[True])}/{len(checked[True])}, "
          f"mean critique-adv gap {mean_gap:+.2f}")
    print("OK")


if __name__ == "__main__":
    main()
