"""CPU check of quality_list rows (AlphaGRPO's DVReward) in train_grpo.score_nodes and the geomean
combine: semantic probs -> am / per_question, quality probs -> gm, reward = sqrt(am * gm).

    python util_scripts/debug/test_quality_reward.py
"""
import math
import sys
import types

sys.path.insert(0, ".")
from llava.train.rl import train_grpo as tg  # noqa: E402
from llava.train.rl.reward import RewardConfig, combine_scores  # noqa: E402


class FakeReward:
    def __init__(self):
        self.cfg = RewardConfig(alpha=1.0, combine="geomean")
        self.seen = []

    def score_images(self, codes, vqas):
        self.seen.append([len(v) for v in vqas])
        pq = [[0.1 * (k + 1) for k in range(len(v))] for v in vqas]
        return [sum(q) / len(q) for q in pq], [0.0] * len(pq), pq

    def combine(self, am, gm):
        return combine_scores(self.cfg, am, gm)


def main():
    rw = FakeReward()
    prompts = [{"prompt": "p0", "vqa_list": [("q1", "Yes"), ("q2", "Yes")], "skills": ["a", "b"],
                "quality_list": [("k1", "Yes"), ("k2", "Yes"), ("k3", "Yes")]},
               {"prompt": "p1", "vqa_list": [("q1", "Yes"), ("q2", "Yes"), ("q3", "Yes")], "skills": ["a", "b", "c"]}]
    nodes = [types.SimpleNamespace(codes=[1, 2], prompt_idx=0, per_question=None, am=0.0, gm=0.0),
             types.SimpleNamespace(codes=[3, 4], prompt_idx=1, per_question=None, am=0.0, gm=0.0)]
    tg.score_nodes(nodes, prompts, rw)
    assert rw.seen == [[5, 3]], rw.seen                    # one request, both lists concatenated
    n0, n1 = nodes
    close = lambda xs, ys: len(xs) == len(ys) and all(abs(x - y) < 1e-9 for x, y in zip(xs, ys))
    assert close(n0.per_question, [0.1, 0.2]) and abs(n0.am - 0.15) < 1e-9
    assert abs(n0.gm - (0.3 + 0.4 + 0.5) / 3) < 1e-9        # quality mean
    assert close(n1.per_question, [0.1, 0.2, 0.3]) and abs(n1.am - 0.2) < 1e-9 and n1.gm == 0.0
    assert abs(rw.combine(n0.am, n0.gm) - math.sqrt(0.15 * 0.4)) < 1e-9
    rw.cfg.combine = "linear"
    assert abs(rw.combine(n0.am, n0.gm) - 0.15) < 1e-9
    print("OK")


if __name__ == "__main__":
    main()
