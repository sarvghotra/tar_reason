"""BAGEL-7B-MoT (ByteDance-Seed/BAGEL-7B-MoT) as a unified_gen.py backend; also the base of the Uni-CoT backend.

Follows the authors' inference.ipynb / eval/gen/gen_images_mp.py (ByteDance-Seed/Bagel @ a2fa77d): the vendored
Qwen2-MoT + SigLIP + FLUX-style VAE loaded with accelerate onto one GPU (~29 GB bf16), the InterleaveInferencer
(one prompt per call, as their official GenEval run with batch_size 1), 1024x1024, text CFG 4.0 over the whole
trajectory (their GenEval setting cfg_interval [0, 1]; the notebook demo uses [0.4, 1]), no image CFG
(cfg_img_scale 1.0), timestep shift 3.0, 50 Euler steps, global CFG renorm. Prompts verbatim (their 0.88 GenEval
used LLM-rewritten prompts; 0.82 is the raw-prompt number). --think prepends their planning system prompt and lets
the model write a plan before the image (WISE setting: sampled, T 0.3); the plan is kept as the round's "reason".
Seeds: torch.manual_seed per prompt (the latent noise is drawn from the global RNG).

Env (Mila): Python 3.10, torch == 2.5.1 (flex_attention import), transformers == 4.49.0, flash-attn 2.5.8 (hard
import, no SDPA fallback), accelerate, einops, opencv-python, safetensors. The clone's top-level packages are
`modeling` and `data`: the clone is put first on sys.path.
"""

import os
import sys

import torch

DEFAULT_MODEL = "ByteDance-Seed/BAGEL-7B-MoT"
DEFAULT_BATCH = 1
DEFAULT_SIZE = 1024


def add_args(parser):
    parser.add_argument("--bagel_repo", default=os.environ.get("BAGEL_REPO", "/network/scratch/s/singhsd/baselines/Bagel"),
                        help="ByteDance-Seed/Bagel clone (modeling/, data/, inferencer.py).")
    parser.add_argument("--bagel_weights", default=None,
                        help="Local snapshot dir with llm_config.json, vit_config.json, ae.safetensors, ema.safetensors "
                             "(default: snapshot_download of --model_path).")
    parser.add_argument("--cfg_text", type=float, default=4.0)
    parser.add_argument("--cfg_img", type=float, default=1.0)
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=[0.0, 1.0], help="their GenEval run; demo 0.4 1.0")
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--cfg_renorm_type", default="global")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--think", action="store_true", help="planning text before the image (their WISE setting)")
    parser.add_argument("--think_sample", action=argparse_bool_default_true(), help="sample the plan (T 0.3) as in WISE")
    parser.add_argument("--max_think_tokens", type=int, default=1024)


def argparse_bool_default_true():
    import argparse
    return argparse.BooleanOptionalAction


def load_bagel(repo, weights_dir, device, ema_path=None, vit_transform_size=(980, 224, 14)):
    """-> (inferencer, model, vae, tokenizer, new_token_ids, vae_transform, vit_transform) for BAGEL-style
    checkpoints (BAGEL, Uni-CoT). ema_path: weights file if not <weights_dir>/ema.safetensors."""
    if repo not in sys.path:
        sys.path.insert(0, repo)
    for name in ("modeling", "data"):
        if name in sys.modules and not getattr(sys.modules[name], "__file__", "").startswith(repo):
            del sys.modules[name]
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    try:
        from inferencer import InterleaveInferencer   # not in the Uni-CoT clone
    except ImportError:
        InterleaveInferencer = None
    from modeling.autoencoder import load_ae
    from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer

    llm_config = Qwen2Config.from_json_file(os.path.join(weights_dir, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(weights_dir, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    vae_model, vae_config = load_ae(local_path=os.path.join(weights_dir, "ae.safetensors"))
    config = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config,
                         vae_config=vae_config, vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
                         latent_patch_size=2, max_latent_size=64)
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(weights_dir)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(*vit_transform_size)
    model = load_checkpoint_and_dispatch(model, checkpoint=ema_path or os.path.join(weights_dir, "ema.safetensors"),
                                         device_map={"": str(device)}, dtype=torch.bfloat16, force_hooks=True).eval()
    vae_model = vae_model.to(device).eval()   # the notebook leaves the VAE on CPU (slow decode)
    inferencer = (InterleaveInferencer(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
                  if InterleaveInferencer else None)
    return inferencer, model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform


def resolve_weights(args):
    if args.bagel_weights:
        return args.bagel_weights
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=args.model_path,
                             allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"])


class Backend:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.inferencer, *_ = load_bagel(args.bagel_repo, resolve_weights(args), device)
        self.hyper = dict(cfg_text_scale=args.cfg_text, cfg_img_scale=args.cfg_img, cfg_interval=list(args.cfg_interval),
                          timestep_shift=args.timestep_shift, num_timesteps=args.steps,
                          cfg_renorm_min=args.cfg_renorm_min, cfg_renorm_type=args.cfg_renorm_type,
                          image_shapes=(args.size, args.size))
        if args.think:
            self.hyper.update(think=True, max_think_token_n=args.max_think_tokens,
                              do_sample=bool(args.think_sample), text_temperature=0.3)

    @torch.no_grad()
    def run(self, prompts, seeds):
        out = []
        for prompt, seed in zip(prompts, seeds):
            torch.manual_seed(seed)
            res = self.inferencer(text=prompt, **self.hyper)
            rounds = [{"accepted": False, "reason": res["text"]}] if res.get("text") else []
            out.append(([res["image"]], rounds))
        return out
