"""Build the draft-anchor set for train_grpo.py --anchor_data (2026-11-12, stage 3 draft drift).

Reads the draft-replay rows (``kind: draft``) of a ``stop_sft/build_data.py --draft_replay`` file, keeps one
draft per prompt (the file repeats each draft), and writes a compact .npz:
``prompts`` (str, N), ``codes`` (int32, N x 729), ``passed`` (bool, N).

Default source: run 1 step 500's own drafts on all 17,259 promptmix train prompts (run 2's data, built on Fir).

    python util_scripts/build_anchor_drafts.py \
        --src /scratch/dars11/share/run2_distill/distill_run1_orc_replay.jsonl \
        --out /home/dars11/scratch/data/anchor/run1_500_drafts_promptmix.npz
"""

import argparse
import json

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    seen = {}
    with open(a.src) as f:
        for line in f:
            if '"kind": "draft"' not in line:
                continue
            j = json.loads(line)
            if j.get("kind") != "draft" or j["prompt"] in seen:
                continue
            seen[j["prompt"]] = (j["codes"], bool(j.get("passed", False)))
    prompts = sorted(seen)
    lens = {len(seen[q][0]) for q in prompts}
    assert len(lens) == 1, f"draft lengths differ: {lens}"
    codes = np.asarray([seen[q][0] for q in prompts], dtype=np.int32)
    passed = np.asarray([seen[q][1] for q in prompts], dtype=bool)
    np.savez(a.out, prompts=np.asarray(prompts, dtype=object), codes=codes, passed=passed)
    print(f"{len(prompts)} drafts x {codes.shape[1]} codes, {passed.mean():.3f} pass every check -> {a.out}")


if __name__ == "__main__":
    main()
