"""AlphaGRPO on BAGEL-7B-MoT (huangrh99/AlphaGRPO, ICML 2026; released LoRA adapters huangrh9/AlphaGRPO-rt2i and
huangrh9/AlphaGRPO-srr) as a unified_gen.py backend.

Follows the authors' eval scripts in their BAGEL fork (AlphaGRPO/Bagel/eval/gen/gen_images_mp.py --think and
gen_images_mp_reflect.py, scripts/eval/run_geneval_multickpt{,_reflect}.sh): BAGEL base weights + the LoRA adapter
(peft, rank 32 / alpha 64, merged), their planning system prompt, think text sampled at T 0.3, then the image
(text CFG 4, cfg_interval [0, 1], shift 3.0, 50 steps, 1024 px). Modes:
  --mode think    (adapter rt2i): plan -> image, single pass. Their "reasoning T2I" model.
  --mode reflect  (adapter srr): plan -> image, then --reflect_rounds (1, as their pipeline) self-reflections: the
                  model is shown its image with "The user requires to generate image of `<prompt>`. You tried ...
                  but you failed ... Reflect on what went wrong and write down the editing instruction ..." and
                  regenerates image-conditioned (text CFG 4, image CFG 2, cfg_interval [0.4, 1], shift 3.0, 50
                  steps, text_channel renorm, deterministic ODE), via their inferencer.batch_generate_images.
                  Their GenEval pipeline reflected only images the GenEval detector marked wrong (--only_wrong);
                  we reflect every image, since the model has no judge at test time.
Images are reported first -> final like the other self-correction baselines (iteration 1 = plan+image, last =
after reflection); the think and reflection texts are the rounds' "reasons". One prompt per call.

Env (Mila): the AlphaGRPO repo's requirements (uv; torch 2.5+, transformers 4.49-ish, flash-attn, accelerate,
peft, safetensors, einops, opencv-python). --agrpo_repo is the AlphaGRPO clone; its Bagel/ dir (a modified BAGEL)
goes first on sys.path. BAGEL weights: a local snapshot of ByteDance-Seed/BAGEL-7B-MoT (--bagel_weights).
"""

import importlib.util
import os
import sys

import torch

DEFAULT_MODEL = None   # the adapter is chosen by --mode / --lora
DEFAULT_BATCH = 1
DEFAULT_SIZE = 1024
ADAPTERS = {"think": "huangrh9/AlphaGRPO-rt2i", "reflect": "huangrh9/AlphaGRPO-srr"}
REFLECT_PROMPT = ("The user requires to generate image of `{}`. You tried to generate an image based on the user's request, "
                  "but you failed to create the correct image. \nReflect on what went wrong and write down the editing "
                  "instruction that will help you do better and refine the image to meet user's request based on your own reflection.")


def add_args(parser):
    parser.add_argument("--agrpo_repo", default=os.environ.get("AGRPO_REPO", "/network/scratch/s/singhsd/baselines/AlphaGRPO"),
                        help="huangrh99/AlphaGRPO clone (its Bagel/ dir is the modeling root).")
    parser.add_argument("--bagel_weights", default=None, help="local snapshot of ByteDance-Seed/BAGEL-7B-MoT (default: snapshot_download)")
    parser.add_argument("--mode", choices=["think", "reflect"], default="reflect")
    parser.add_argument("--lora", default=None, help="adapter HF id or dir (default by mode: rt2i / srr); 'none' = base BAGEL")
    parser.add_argument("--reflect_rounds", type=int, default=1, help="reflections after the first image (their pipeline: 1)")
    parser.add_argument("--cfg_text", type=float, default=4.0)
    parser.add_argument("--cfg_img", type=float, default=2.0, help="reflection stage (their default 2)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--max_think_tokens", type=int, default=2048)


class Backend:
    def __init__(self, args, device):
        self.args, self.device = args, device
        repo = os.path.join(args.agrpo_repo, "Bagel")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        for name in ("modeling", "data"):
            if name in sys.modules and not getattr(sys.modules[name], "__file__", "").startswith(repo):
                del sys.modules[name]
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from data.data_utils import add_special_tokens
        from data.transforms import ImageTransform
        from inferencer import InterleaveInferencer
        from modeling.autoencoder import load_ae
        from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
        from modeling.qwen2 import Qwen2Tokenizer
        from huggingface_hub import snapshot_download

        weights = args.bagel_weights or snapshot_download(
            repo_id="ByteDance-Seed/BAGEL-7B-MoT", allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"])
        lora = args.lora or ADAPTERS[args.mode]
        if lora != "none" and not os.path.isdir(lora):
            lora = snapshot_download(repo_id=lora)
        # --- their loading sequence (gen_images_mp_reflect.py __main__)
        llm_config = Qwen2Config.from_json_file(os.path.join(weights, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(weights, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1
        vae_model, vae_config = load_ae(local_path=os.path.join(weights, "ae.safetensors"))
        config = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config,
                             vae_config=vae_config, vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
                             latent_patch_size=2, max_latent_size=64)
        with init_empty_weights():
            model = Bagel(Qwen2ForCausalLM(llm_config), SiglipVisionModel(vit_config), config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
        tokenizer = Qwen2Tokenizer.from_pretrained(weights)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        state = load_file(os.path.join(weights, "ema.safetensors"), device="cpu")
        model.load_state_dict(state, strict=False, assign=True)
        del state
        model.language_model.resize_token_embeddings(len(tokenizer))
        if lora != "none":
            from peft import PeftModel
            gpu = device.index if device.type == "cuda" else 0
            model = PeftModel.from_pretrained(model, lora, device_map={"": gpu})
            model.set_adapter("default")
            model = model.merge_and_unload()
        self.model = model.to(device).eval()
        self.vae = vae_model.to(device).eval()
        vae_transform, vit_transform = ImageTransform(1024, 512, 16), ImageTransform(980, 378, 14)
        self.inferencer = InterleaveInferencer(self.model, self.vae, tokenizer, vae_transform, vit_transform, new_token_ids)
        # Their inferencer defaults to compiled flex_attention (written for torch 2.8); torch 2.5's flex kernel rejects
        # Qwen2-7B's 7 query heads per KV head ("must be power of 2"). The other branch is the same exact attention via
        # flash_attn_varlen (upstream BAGEL's path).
        self.inferencer.use_flex_attention = False
        # --- their think-then-generate function (module globals injected, as the Uni-CoT backend does)
        spec = importlib.util.spec_from_file_location("agrpo_gen_images_mp", os.path.join(repo, "eval", "gen", "gen_images_mp.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["agrpo_gen_images_mp"] = mod
        spec.loader.exec_module(mod)
        for name, value in dict(model=self.model, gen_model=self.model, tokenizer=tokenizer, new_token_ids=new_token_ids,
                                vae_model=self.vae, vae_transform=vae_transform, vit_transform=vit_transform,
                                inferencer=self.inferencer, device=str(device)).items():
            setattr(mod, name, value)
        self.mod = mod

    @torch.no_grad()
    def run(self, prompts, seeds):
        a = self.args
        out = []
        for prompt, seed in zip(prompts, seeds):
            torch.manual_seed(seed)
            image_list, think = self.mod.generate_image_with_think(
                prompt=prompt, num_timesteps=a.steps, cfg_scale=a.cfg_text, cfg_interval=[0, 1.0], cfg_renorm_min=0.0,
                timestep_shift=3.0, resolution=a.size, num_images=1, max_length=a.max_think_tokens,
                simple_think=False, device=str(self.device))
            images = [image_list[0].convert("RGB")]
            rounds = [{"accepted": False, "reason": (think[0] if isinstance(think, list) else think).strip()}]
            if a.mode == "reflect":
                for _ in range(a.reflect_rounds):
                    refined, texts = self.inferencer.batch_generate_images(
                        input_lists=[[images[-1], REFLECT_PROMPT.format(prompt)]], image_shape=(a.size, a.size),
                        cfg_text_scale=a.cfg_text, cfg_img_scale=a.cfg_img, num_timesteps=a.steps, cfg_interval=[0.4, 1.0],
                        timestep_shift=3.0, cfg_renorm_min=0.0, cfg_renorm_type="text_channel", deterministic=True,
                        think=True, text_temperature=0.3, max_think_token_n=a.max_think_tokens)
                    images.append(refined[0].convert("RGB"))
                    rounds.append({"accepted": False, "reason": (texts[0] or "").strip()})
            out.append((images, rounds))
        return out
