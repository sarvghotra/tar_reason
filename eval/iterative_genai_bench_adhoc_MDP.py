"""Markov (MDP) variant of ``iterative_genai_bench_adhoc.py``.

Same forced schedule, arguments and output layout, but each refinement step
is conditioned only on the prompt, the previous image and its critique: after
a corrected image is sampled, the context is reset to

    <prefix><im_start><S{scale}>{previous image}<im_end>\\nSelf-reflect:

so the next reflection and correction never see older drafts or critiques.
With ``--n_iters 2`` this is identical to the full-history script.

See scripts/eval/iter_genai_bench_adhoc_MDP.sh for the end-to-end pipeline.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.iterative_genai_bench_adhoc import main  # noqa: E402


if __name__ == "__main__":
    main(markov=True)
