"""BLIP3-o 8B (BLIP3o/BLIP3o-Model-8B: Qwen2.5-VL-7B + CLIP-feature DiT + SDXL-style pixel decoder) as a
unified_gen.py backend.

Follows the authors' inference.py / eval/geneval/generate.py (JiuhaiChen/BLIP3o `main` @ 418fc44): the
`blip3o` package's load_pretrained_model (LLM + DiT in fp16) and the diffusion-decoder loaded as the custom
`pipeline_llava_gen` pipeline (UNet + VAE in bf16); the qwen chat template with "Please generate image based on
the following caption: <prompt>"; one prompt per call (the pipeline concatenates a list into one prompt);
feature DiT 30 steps / CFG 3 (hard-coded), decoder 50 steps / guidance 3.0, 1024x1024. There is no generator
argument: the global RNG is seeded per prompt as their GenEval script does. The library hard-codes cuda:0, so the
driver pins one visible GPU per rank for this family (see unified_gen.SINGLE_GPU_FAMILIES).

Env (Mila): the repo's pins (torch 2.3.0, transformers 4.51.3, the authors' diffusers fork
git+https://github.com/JiuhaiChen/diffusers.git, flash-attn 2.6.2, timm 0.6.13, qwen_vl_utils, datasets, peft;
`pip install -e .` from the `main` checkout). Importing `blip3o.model` snapshot-downloads the 17 GB
jiuhai/eva_clip_vision_tower once (unused for T2I). ~30 GB GPU at batch 1. --blip3o_weights must be a local
snapshot (the custom pipeline file is resolved from <weights>/diffusion-decoder).
"""

import os
import random
import sys

import numpy as np
import torch

DEFAULT_MODEL = "BLIP3o/BLIP3o-Model-8B"
DEFAULT_BATCH = 1
DEFAULT_SIZE = 1024
CAPTION_PREFIX = "Please generate image based on the following caption: "


def add_args(parser):
    parser.add_argument("--blip3o_repo", default=os.environ.get("BLIP3O_REPO", "/network/scratch/s/singhsd/baselines/BLIP3o"),
                        help="JiuhaiChen/BLIP3o clone on branch main (pip install -e .).")
    parser.add_argument("--blip3o_weights", default=None, help="local snapshot of BLIP3o/BLIP3o-Model-8B (default: snapshot_download)")
    parser.add_argument("--guidance", type=float, default=3.0, help="decoder CFG (their GenEval: 3.0)")
    parser.add_argument("--steps", type=int, default=50, help="decoder steps (pipeline default 50)")


class Backend:
    def __init__(self, args, device):
        if args.blip3o_repo not in sys.path:
            sys.path.insert(0, args.blip3o_repo)
        from diffusers import DiffusionPipeline
        from blip3o.conversation import conv_templates
        from blip3o.model.builder import load_pretrained_model
        from blip3o.utils import disable_torch_init
        self.args, self.device = args, device
        weights = args.blip3o_weights
        if weights is None:
            from huggingface_hub import snapshot_download
            weights = snapshot_download(repo_id=args.model_path)
        disable_torch_init()
        tokenizer, multi_model, _ = load_pretrained_model(weights, None, os.path.basename(weights))
        self.pipe = DiffusionPipeline.from_pretrained(
            os.path.join(weights, "diffusion-decoder"), custom_pipeline="pipeline_llava_gen", torch_dtype=torch.bfloat16,
            use_safetensors=True, variant="bf16", multimodal_encoder=multi_model, tokenizer=tokenizer, safety_checker=None)
        self.pipe.vae.to(device)
        self.pipe.unet.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.conv_templates = conv_templates

    def _template(self, prompt):
        conv = self.conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], CAPTION_PREFIX + prompt)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    @torch.no_grad()
    def run(self, prompts, seeds):
        out = []
        for prompt, seed in zip(prompts, seeds):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            res = self.pipe([self._template(prompt)], guidance_scale=self.args.guidance,
                            num_inference_steps=self.args.steps)
            out.append(([res.image], []))
        return out
