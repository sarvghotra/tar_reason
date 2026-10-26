"""diffusers pipelines (FLUX.1-dev, SD3.5-Large, Qwen-Image) as a unified_gen.py backend.

Thin wrapper over eval/baselines/diffusers_gen.py (its PRESETS and DiffusionGen), which remains the GEN_SCRIPT
the launcher uses for GEN_FAMILY=diffusers; this backend exists so every family shares one driver.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusers_gen import PRESETS, DiffusionGen  # noqa: E402

DEFAULT_MODEL = None       # chosen by --diff_model
DEFAULT_BATCH = 4
DEFAULT_SIZE = 1024


def add_args(parser):
    parser.add_argument("--diff_model", choices=sorted(PRESETS), default="flux1-dev")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--true_cfg", type=float, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--prompt_suffix", default="")
    parser.add_argument("--cpu_offload", action="store_true")


class Backend:
    def __init__(self, args, device):
        p = PRESETS[args.diff_model]
        args.diff_model_path = args.model_path or p["model_path"]
        args.height = args.height or p["size"]
        args.width = args.width or p["size"]
        args.steps = args.steps or p["steps"]
        args.guidance = p["guidance"] if args.guidance is None else args.guidance
        args.true_cfg = p["true_cfg"] if args.true_cfg is None else args.true_cfg
        args.max_seq_len = p["max_seq_len"] if args.max_seq_len is None else args.max_seq_len
        args.negative_prompt = p["negative"] if args.negative_prompt is None else args.negative_prompt
        self.args = args
        self.gen = DiffusionGen(args, device)

    def run(self, prompts, seeds):
        images = self.gen.run([p + self.args.prompt_suffix for p in prompts], seeds)
        return [([im], []) for im in images]
