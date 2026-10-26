"""Show-o2 (showlab/show-o2-7B, 432 px, flow matching in Wan2.1 VAE latents) as a unified_gen.py backend.

Follows the authors' show-o2/inference_t2i.py (showlab/Show-o @ 45a5a2d) with the 7B 432x432 config: Qwen2.5-7B
tokenizer with Show-o tokens, raw prompt (no chat template; truncated to 290 Qwen tokens), 729 + 1 (time
embedding) image tokens, CFG stacking (cond + empty prompt), the dense omni attention mask, Euler ODE from
the `transport` package (time shift 3.0), Wan2.1 VAE decode. Defaults follow their GenEval script
(evaluation/sample_geneval.sh): guidance 7.5, 100 steps (their run used LLM-rewritten prompts; ours are
verbatim). Batched over prompts; per-batch torch seed = the first prompt's. 1.5B: --model_path
showlab/show-o2-1.5B --llm_model Qwen/Qwen2.5-1.5B-Instruct --showo2_config configs/showo2_1.5b_demo_432x432.yaml.

Env (Mila): torch == 2.5.1, transformers == 4.47.0, diffusers == 0.31.0, omegaconf, torchdiffeq, timm, einops,
decord (imported by utils.py); flash-attn not needed. Needs Wan2.1_VAE.pth from Wan-AI/Wan2.1-T2V-14B
(--wan_vae), google/siglip-so400m-patch14-384 and the Qwen2.5 config/tokenizer reachable. The clone's
top-level packages `models`, `transport`, `utils`, `datasets` shadow same-named packages: it goes first on sys.path.
"""

import os
import sys

import torch
from PIL import Image

DEFAULT_MODEL = "showlab/show-o2-7B"
DEFAULT_BATCH = 8
DEFAULT_SIZE = 432


def add_args(parser):
    parser.add_argument("--showo2_repo", default=os.environ.get("SHOWO2_REPO", "/network/scratch/s/singhsd/baselines/Show-o/show-o2"),
                        help="showlab/Show-o clone's show-o2/ dir.")
    parser.add_argument("--showo2_config", default="configs/showo2_7b_demo_432x432.yaml", help="relative to the repo")
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-7B-Instruct", help="config + tokenizer (must match the checkpoint)")
    parser.add_argument("--wan_vae", default=os.environ.get("WAN_VAE", "/network/scratch/s/singhsd/baselines/Wan2.1_VAE.pth"))
    parser.add_argument("--guidance", type=float, default=7.5, help="their GenEval script: 7.5 (README demo 7.5, yaml 5)")
    parser.add_argument("--steps", type=int, default=100, help="their GenEval script: 100 (README demo 50)")


class Backend:
    def __init__(self, args, device):
        repo = args.showo2_repo
        if repo not in sys.path:
            sys.path.insert(0, repo)
        for name in ("models", "transport", "utils", "datasets"):
            if name in sys.modules and not getattr(sys.modules[name], "__file__", "").startswith(repo):
                del sys.modules[name]
        from omegaconf import OmegaConf
        from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
        from models.misc import get_text_tokenizer, prepare_gen_input
        from transport import Sampler, create_transport
        from utils import denorm, get_hyper_params, path_to_llm_name
        self.args, self.device = args, device
        self.omni_mask, self.prepare_gen_input, self.denorm = omni_attn_mask_naive, prepare_gen_input, denorm
        if args.guidance <= 0:
            raise ValueError("Show-o2's sampler keeps only the conditional half of the batch: --guidance must be > 0")
        cfg = OmegaConf.load(os.path.join(repo, args.showo2_config))
        cfg.model.showo.pretrained_model_path = args.model_path
        cfg.model.showo.llm_model_path = args.llm_model
        cfg.model.vae_model.pretrained_model_path = args.wan_vae
        if cfg.model.showo.add_time_embeds:   # once, as the script does
            cfg.dataset.preprocessing.num_t2i_image_tokens += 1
            cfg.dataset.preprocessing.num_mmu_image_tokens += 1
            cfg.dataset.preprocessing.num_video_tokens += 1
        cfg.transport.num_inference_steps = args.steps
        self.cfg = cfg
        self.dtype = torch.bfloat16 if cfg.model.weight_type == "bfloat16" else torch.float32
        self.vae = WanVAE(vae_pth=args.wan_vae, dtype=self.dtype, device=device)
        tokenizer, showo_token_ids = get_text_tokenizer(args.llm_model, add_showo_tokens=True, return_showo_token_ids=True,
                                                        llm_name=path_to_llm_name[args.llm_model])
        cfg.model.showo.llm_vocab_size = len(tokenizer)
        (self.n_img, _, _, self.max_seq_len, self.max_text_len, self.latent_dim, self.patch, self.lat_w, self.lat_h,
         self.pad_id, self.bos_id, self.eos_id, self.boi_id, self.eoi_id, _, _, self.img_pad_id, _, _) = \
            get_hyper_params(cfg, tokenizer, showo_token_ids)
        self.tokenizer = tokenizer
        self.model = Showo2Qwen2_5.from_pretrained(args.model_path, use_safetensors=False).to(device).to(self.dtype).eval()
        t = cfg.transport
        self.sampler = Sampler(create_transport(path_type=t.path_type, prediction=t.prediction, loss_weight=t.loss_weight,
                                                train_eps=t.train_eps, sample_eps=t.sample_eps, snr_type=t.snr_type,
                                                do_shift=t.do_shift, seq_len=self.n_img))
        self.sample_fn = self.sampler.sample_ode(sampling_method=t.sampling_method, num_steps=args.steps, atol=t.atol,
                                                 rtol=t.rtol, reverse=t.reverse, time_shifting_factor=t.time_shifting_factor)

    @torch.no_grad()
    def run(self, prompts, seeds):
        torch.manual_seed(seeds[0])
        tok, tok_null, pos, pos_null = self.prepare_gen_input(
            prompts, self.tokenizer, self.n_img, self.bos_id, self.eos_id, self.boi_id, self.eoi_id, self.pad_id,
            self.img_pad_id, self.max_text_len, self.device)
        z = torch.randn((len(prompts), self.latent_dim, self.lat_h * self.patch, self.lat_w * self.patch),
                        device=self.device).to(self.dtype)
        z = torch.cat([z, z], dim=0)
        text_tokens = torch.cat([tok, tok_null], dim=0)
        modality_positions = torch.cat([pos, pos_null], dim=0)
        mask = self.omni_mask(text_tokens.size(0), self.max_seq_len, modality_positions, self.device).to(self.dtype)
        kw = dict(text_tokens=text_tokens, attention_mask=mask, modality_positions=modality_positions,
                  output_hidden_states=True, max_seq_len=self.max_seq_len, guidance_scale=self.args.guidance)
        samples = self.sample_fn(z, self.model.t2i_generate, **kw)[-1]
        samples = torch.chunk(samples, 2)[0]
        images = self.vae.batch_decode(samples.unsqueeze(2)).squeeze(2)
        return [([Image.fromarray(im)], []) for im in self.denorm(images)]
