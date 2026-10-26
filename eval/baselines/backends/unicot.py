"""Uni-CoT (Fr0zencr4nE/UniCoT-7B-MoT, BAGEL-7B-MoT fine-tune; ICLR 2026) as a unified_gen.py backend, run as a
*self-correction* baseline: plan -> image -> self-reflect -> regenerate, their v0.1 "MDP self-reflection" loop.

Follows the authors' inference_mdp_self_reflection_v0.1.py (Fr0zenCrane/UniCoT @ 2a35249; the WISE setting):
generate_image_with_think (their system prompt, a sampled <think> plan at T 0.3, then the image: cfg 4.0,
cfg_interval [0.4, 1.0], shift 3.0, 50 steps, 1024 px), then refine_image_with_cot per round (the model looks at
the image, writes <sr> / <sugg>; if it does not say "Everything is good" / "No editing" it rewrites the suggestion and
regenerates image-conditioned with text CFG 4.0 / image CFG --cfg_img, up to 3 critique samples per round). Their
loop runs until the model is satisfied or 21 rounds; ours stops after --max_rounds (default 4 = draft + 4
corrections, our own budget) and reports image 1 and the final image like the Janus-Pro-R1 loop. Each round's
critique text is the round's "reason". One prompt at a time (their text generation is batch-1 only).

The script is not importable as a module (dotted file name, functions read module globals set under __main__),
so it is loaded with importlib from --unicot_repo and its globals are injected. Weights: a full BAGEL-format
snapshot (llm_config.json, vit_config.json, ae.safetensors, ema.safetensors, tokenizer) of the Uni-CoT repo,
passed as --unicot_weights (snapshot_download of --model_path by default). Env: same as BAGEL (torch 2.5.1,
transformers 4.49.0, flash-attn 2.5.8 mandatory); the clone vendors modeling/ and data/ (an older BAGEL
snapshot) and needs no BAGEL repo. ~29 GB bf16 + KV-cache copies per round: one process per 80 GB GPU.
"""

import importlib.util
import os
import sys

import torch

from .bagel import load_bagel

DEFAULT_MODEL = "Fr0zencr4nE/UniCoT-7B-MoT"
DEFAULT_BATCH = 1
DEFAULT_SIZE = 1024
SCRIPT = "inference_mdp_self_reflection_v0.1.py"


def add_args(parser):
    parser.add_argument("--unicot_repo", default=os.environ.get("UNICOT_REPO", "/network/scratch/s/singhsd/baselines/UniCoT"),
                        help="Fr0zenCrane/UniCoT clone (modeling/, data/, inference_mdp_self_reflection_v0.1.py).")
    parser.add_argument("--unicot_weights", default=None, help="local BAGEL-format snapshot (default: snapshot_download)")
    parser.add_argument("--max_rounds", type=int, default=4, help="self-reflection rounds after the first image (their loop: 21)")
    parser.add_argument("--cfg_text", type=float, default=4.0, help="their WISE run: 4")
    parser.add_argument("--cfg_img", type=float, default=1.5, help="their script default for the regeneration (argparse 1.5)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=8192, help="text budget per refine call (their main: 8192)")


class Backend:
    def __init__(self, args, device):
        self.args, self.device = args, device
        weights = args.unicot_weights
        if weights is None:
            from huggingface_hub import snapshot_download
            weights = snapshot_download(repo_id=args.model_path,
                                        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"])
        _, model, vae, tokenizer, new_token_ids, vae_transform, vit_transform = load_bagel(
            args.unicot_repo, weights, device, vit_transform_size=(980, 378, 14))   # their v0.1 ViT transform
        spec = importlib.util.spec_from_file_location("unicot_mdp_v01", os.path.join(args.unicot_repo, SCRIPT))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["unicot_mdp_v01"] = mod
        spec.loader.exec_module(mod)   # defines the functions only; its main is under __main__
        for name, value in dict(model=model, gen_model=model, tokenizer=tokenizer, new_token_ids=new_token_ids,
                                vae_model=vae, vae_transform=vae_transform, vit_transform=vit_transform,
                                device=str(device)).items():
            setattr(mod, name, value)
        self.mod = mod
        self.gen_hyper = dict(cfg_scale=args.cfg_text, cfg_interval=[0.4, 1.0], timestep_shift=3.0,
                              num_timesteps=args.steps, resolution=args.size)
        self.refine_hyper = dict(cfg_text_scale=args.cfg_text, cfg_img_scale=args.cfg_img, cfg_interval=[0, 1.0],
                                 cfg_renorm_min=0.0, timestep_shift=3.0, num_timesteps=args.steps,
                                 max_length=args.max_length)

    @torch.no_grad()
    def run(self, prompts, seeds):
        out = []
        for prompt, seed in zip(prompts, seeds):
            torch.manual_seed(seed)
            image, think = self.mod.generate_image_with_think(prompt=prompt, device=None, **self.gen_hyper)
            image = image.convert("RGB")
            images, rounds = [image], []
            for _ in range(self.args.max_rounds):
                new_image, text, modified = self.mod.refine_image_with_cot(
                    image=images[-1], prompt=prompt, think_text=think.strip(), device=str(self.device), **self.refine_hyper)
                rounds.append({"accepted": not modified, "reason": (text or "").strip()})
                if not modified:
                    break
                images.append(new_image.convert("RGB"))
            out.append((images, rounds))
        return out
