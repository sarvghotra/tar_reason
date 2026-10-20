"""Markov (MDP) variant of ``iterative_generation_adhoc.py``.

Same forced schedule, arguments and output layout, but each refinement step
is conditioned only on the prompt and the previous image: after a corrected
image is sampled, the context is reset to

    <prefix><im_start><S{scale}>{previous image}<im_end>\\nSelf-reflect:

so the next reflection and correction never see older drafts or critiques.
With ``--n_iters 2`` this is identical to the full-history script.

Usage:
    torchrun --standalone --nproc_per_node=4 eval/iterative_generation_adhoc_MDP.py \\
        --model <path-or-hf-id> [--lora_path <adapter-checkpoint>] \\
        --prompts_file <path> --out_dir results/iterative_mdp \\
        --n_iters 3 --batch_size 32 --decode_batch_size 32 --repeat 1 --geneval2
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.iterative_generation_adhoc import main  # noqa: E402


if __name__ == "__main__":
    main(markov=True)
