"""CPU test of llava/train/rl/critique_reward.py (stage-3 critique / stop reward) on a hand-built tree.

    PYTHONPATH=. ~/envs/tar/bin/python util_scripts/debug/test_critique_reward.py
"""

from types import SimpleNamespace

from llava.train.rl.critique_reward import (add_critique_advantages, assign_critique_rewards,
                                            finalize_critique_stats)
from llava.train.rl.grpo import GRPOConfig
from llava.train.rl.rollout import Node, RolloutBatch

VQA = [("Are there any dogs in the image?", "Yes"), ("Is the dog red?", "Yes"), ("Is the cat blue?", "Yes")]


def node(round_, parent, looks_good=False, refl="", pq=None, teacher_forced=False):
    n = Node(prompt_idx=0, round=round_, parent=parent, first_born=False, seq=[], kinds=[],
             looks_good=looks_good, reflection=refl, teacher_forced=teacher_forced)
    n.per_question = pq
    if parent is not None:
        parent.children.append(n)
    return n


class FakeJudge:
    """Names a check when its keyword appears in the critique."""
    KEY = {VQA[0][0]: "add a dog", VQA[1][0]: "red", VQA[2][0]: "cat"}

    def critique_mentions(self, items):
        return [[1.0 if self.KEY[q] in it["critique"] else 0.0 for q, _ in it["vqa"]] for it in items]


def build():
    passing = node(0, None, pq=[0.9, 0.9, 0.9])
    failing = node(0, None, pq=[0.9, 0.2, 0.9])          # fails "Is the dog red?"
    kids_p = [node(1, passing, looks_good=True, refl="looks good"),
              node(1, passing, refl="make the cat bigger"),
              node(1, passing, looks_good=True, refl="looks good")]
    kids_f = [node(1, failing, looks_good=True, refl="looks good"),
              node(1, failing, refl="make the dog red"),               # names exactly the failed check
              node(1, failing, refl="make the dog red and the cat bigger")]   # + a false alarm on the cat
    lone = node(1, node(0, None, pq=[0.2, 0.2, 0.2]), refl="add a dog")       # alone in its group: no reward
    forced = node(1, failing, refl="oracle text", teacher_forced=True)        # not an own critique: skipped
    for n in kids_p + kids_f + [lone]:
        n.adv = n.adv_txt = 0.1
    forced.adv = forced.adv_txt = 0.1
    roots = [passing, failing, lone.parent]
    level1 = kids_p + kids_f + [lone, forced]
    batch = RolloutBatch(prompts=[{"prompt": "a red dog and a blue cat", "vqa_list": VQA}],
                         roots=roots, leaves=level1, nodes_by_round=[roots, level1])
    return kids_p, kids_f, lone, forced, batch


def main():
    gcfg = GRPOConfig(adv_norm="std", adv_std_floor=0.04)
    for mode in ("stop", "full"):
        kids_p, kids_f, lone, forced, batch = build()
        args = SimpleNamespace(critique_reward=mode, oracle_fail_threshold=0.5, critique_fa_beta=0.5,
                               critique_reward_coef=0.25)
        s = assign_critique_rewards(batch, args, FakeJudge())
        assert [n.crit_reward for n in kids_p] == [1.0, -1.0, 1.0], [n.crit_reward for n in kids_p]
        if mode == "stop":
            assert [n.crit_reward for n in kids_f] == [-1.0, 1.0, 1.0], [n.crit_reward for n in kids_f]
        else:
            got = [round(n.crit_reward, 3) for n in kids_f]
            # failed check named (recall 1); the cat check is passed: false alarm 1/2 -> 1 - 0.5 * 0.5
            assert got == [-1.0, 1.0, 0.75], got   # 2nd: exact; 3rd: + cat (1 of 2 passed checks)
        assert lone.crit_reward is None and forced.crit_reward is None
        a = add_critique_advantages(batch, args, gcfg)
        # the stop that was wrong is pushed down, the right ones up; image advantage kept underneath
        assert kids_f[0].adv_txt < 0.1 < kids_f[1].adv_txt, [n.adv_txt for n in kids_f]
        assert kids_p[1].adv_txt < 0.1 < kids_p[0].adv_txt
        assert lone.adv_txt == 0.1 and forced.adv_txt == 0.1
        assert all(n.adv == 0.1 for n in kids_p + kids_f), "image advantage must not change"
        stats = finalize_critique_stats({**s, **a})
        assert abs(stats["train/crit_stop_acc"] - 4 / 6) < 1e-9, stats
        print(mode, {k: round(v, 3) for k, v in stats.items()})
    print("OK")


if __name__ == "__main__":
    main()
