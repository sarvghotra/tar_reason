"""Show-o (showlab/show-o-512x512, 1.3B, MaskGIT-style discrete diffusion) as a unified_gen.py backend.

Follows the authors' inference_t2i.py (showlab/Show-o @ 45a5a2d, README t2i command): Phi-1.5 tokenizer,
UniversalPrompting 't2i_gen' rows (128 text tokens, left padded; longer prompts are truncated), MAGVIT-v2 VQ
(1024 tokens = 32x32 for 512 px), classifier-free guidance 5 (MUSE convention: (1+w)*cond - w*uncond with the
empty prompt), 50 generation timesteps, cosine mask schedule, Gumbel temperature 1.0, fp32. Batched over
prompts; the per-batch torch seed is the first prompt's.

Env (Mila): the repo's pins, transformers == 4.41.1, diffusers == 0.30.1 (vendored ModelMixin imports diffusers
internals), omegaconf, einops; no flash-attn needed. The repo's top-level packages are named `models` and
`training`: the clone is put first on sys.path so they shadow anything else of that name.
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "showlab/show-o-512x512"
DEFAULT_BATCH = 8
DEFAULT_SIZE = 512


def add_args(parser):
    parser.add_argument("--showo_repo", default=os.environ.get("SHOWO_REPO", "/network/scratch/s/singhsd/baselines/Show-o"),
                        help="showlab/Show-o clone (models/, training/, configs/).")
    parser.add_argument("--showo_config", default=None, help="default: <repo>/configs/showo_demo_512x512.yaml")
    parser.add_argument("--vq_model", default="showlab/magvitv2")
    parser.add_argument("--llm_model", default="microsoft/phi-1_5", help="tokenizer + config only")
    parser.add_argument("--guidance", type=float, default=5.0, help="README t2i: 5")
    parser.add_argument("--steps", type=int, default=50, help="generation_timesteps (README: 50)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Gumbel remasking temperature")
    parser.add_argument("--mask_schedule", default="cosine")


class Backend:
    def __init__(self, args, device):
        if args.showo_repo not in sys.path:
            sys.path.insert(0, args.showo_repo)
        for name in ("models", "training"):   # make sure the clone's packages win over same-named ones
            if name in sys.modules and not getattr(sys.modules[name], "__file__", "").startswith(args.showo_repo):
                del sys.modules[name]
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer
        from models import MAGVITv2, Showo, get_mask_chedule
        from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
        self.args, self.device = args, device
        self.create_mask = create_attention_mask_predict_next
        cfg = OmegaConf.load(args.showo_config or os.path.join(args.showo_repo, "configs", "showo_demo_512x512.yaml"))
        cfg.model.showo.pretrained_model_path = args.model_path
        cfg.model.vq_model.vq_model_name = args.vq_model
        cfg.model.showo.llm_model_path = args.llm_model
        cfg.training.guidance_scale = args.guidance
        cfg.training.generation_timesteps = args.steps
        cfg.training.generation_temperature = args.temperature
        cfg.training.batch_size = args.gen_batch
        self.cfg = cfg
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model, padding_side="left")
        self.prompting = UniversalPrompting(
            tokenizer, max_text_len=cfg.dataset.preprocessing.max_seq_length,
            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
            ignore_id=-100, cond_dropout_prob=cfg.training.cond_dropout_prob)
        self.vq = MAGVITv2.from_pretrained(args.vq_model).to(device).eval()
        self.vq.requires_grad_(False)
        self.model = Showo.from_pretrained(args.model_path).to(device).eval()
        self.mask_token_id = self.model.config.mask_token_id
        self.mask_schedule = get_mask_chedule(args.mask_schedule)

    @torch.no_grad()
    def run(self, prompts, seeds):
        a, cfg = self.args, self.cfg
        torch.manual_seed(seeds[0])
        gen = torch.Generator(device=self.device).manual_seed(seeds[0])
        n_tok = cfg.model.showo.num_vq_tokens
        image_tokens = torch.full((len(prompts), n_tok), self.mask_token_id, dtype=torch.long, device=self.device)
        input_ids, _ = self.prompting((prompts, image_tokens), "t2i_gen")
        sp = self.prompting.sptids_dict
        if a.guidance > 0:
            uncond_ids, _ = self.prompting(([""] * len(prompts), image_tokens), "t2i_gen")
            mask = self.create_mask(torch.cat([input_ids, uncond_ids], dim=0), pad_id=int(sp["<|pad|>"]),
                                    soi_id=int(sp["<|soi|>"]), eoi_id=int(sp["<|eoi|>"]), rm_pad_in_image=True)
        else:
            uncond_ids = None
            mask = self.create_mask(input_ids, pad_id=int(sp["<|pad|>"]), soi_id=int(sp["<|soi|>"]),
                                    eoi_id=int(sp["<|eoi|>"]), rm_pad_in_image=True)
        ids = self.model.t2i_generate(
            input_ids=input_ids, uncond_input_ids=uncond_ids, attention_mask=mask, guidance_scale=a.guidance,
            temperature=a.temperature, timesteps=a.steps, noise_schedule=self.mask_schedule, noise_type="mask",
            seq_len=n_tok, uni_prompting=self.prompting, config=cfg, generator=gen)
        ids = torch.clamp(ids, max=cfg.model.showo.codebook_size - 1, min=0)
        images = self.vq.decode_code(ids)
        images = (torch.clamp((images + 1.0) / 2.0, 0, 1) * 255.0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return [([Image.fromarray(im)], []) for im in images]
