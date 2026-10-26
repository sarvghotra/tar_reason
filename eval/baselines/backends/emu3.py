"""Emu3-Gen (BAAI/Emu3-Gen, 8B autoregressive, 720 px) as a unified_gen.py backend.

Follows the authors' image_generation.py (baaivision/Emu3 @ dbbf985): the Hub model via trust_remote_code,
the Emu3Processor from the repo's emu3/ package (--emu3_repo), mode 'G' prompts "<prompt> masterpiece, film
grained, best quality." with their negative prompt under classifier-free guidance 3.0 (unbatched CFG logits
processor), top-k 2048 sampling, a prefix constraint that forces the 90x90 visual-token grid + eol/eof/eoi/eos
(8,193 new tokens per 1:1 image), decoded with the Emu3 VisionTokenizer. The positive suffix and the
negative prompt are the model card's inference recipe (--no_positive_suffix / --negative_prompt "" to drop them).
Sampling is batched over prompts (left padding), so one seed per batch (the first prompt's); images in a
batch are still deterministic given the batch composition.

Env (Mila): Python 3.10, torch >= 2.2, transformers == 4.44.0 (the repo's pin; the processor breaks on much
newer versions), tiktoken == 0.6.0, flash-attn optional (--attn sdpa otherwise). ~17 GB bf16 + KV cache
(~1.2 GB per row, x2 for CFG): batch 4-8 fits on 80 GB. Slow: ~8k tokens per image with 2 forwards per token.
"""

import os
import sys

import torch

DEFAULT_MODEL = "BAAI/Emu3-Gen"
DEFAULT_BATCH = 4
DEFAULT_SIZE = 720
VQ_MODEL = "BAAI/Emu3-VisionTokenizer"
POSITIVE_SUFFIX = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = ("lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
                   "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, "
                   "username, blurry.")


def add_args(parser):
    parser.add_argument("--emu3_repo", default=os.environ.get("EMU3_REPO", "/network/scratch/s/singhsd/baselines/Emu3"),
                        help="baaivision/Emu3 clone (for emu3.mllm.processing_emu3).")
    parser.add_argument("--vq_model", default=VQ_MODEL)
    parser.add_argument("--cfg", type=float, default=3.0, help="Classifier-free guidance scale (official 3.0).")
    parser.add_argument("--top_k", type=int, default=2048, help="official 2048")
    parser.add_argument("--ratio", default="1:1")
    parser.add_argument("--no_positive_suffix", action="store_true", help="Do not append the quality suffix.")
    parser.add_argument("--negative_prompt", default=NEGATIVE_PROMPT, help='"" disables CFG\'s negative prompt.')
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])


class Backend:
    def __init__(self, args, device):
        if args.emu3_repo not in sys.path:
            sys.path.insert(0, args.emu3_repo)
        from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from emu3.mllm.processing_emu3 import Emu3Processor  # noqa: E402
        self.args, self.device = args, device
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map={"": device}, torch_dtype=torch.bfloat16,
            attn_implementation=args.attn, trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
        image_processor = AutoImageProcessor.from_pretrained(args.vq_model, trust_remote_code=True)
        image_tokenizer = AutoModel.from_pretrained(args.vq_model, device_map={"": device}, trust_remote_code=True).eval()
        self.processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
        self.gen_config = GenerationConfig(
            use_cache=True, eos_token_id=self.model.config.eos_token_id, pad_token_id=self.model.config.pad_token_id,
            max_new_tokens=40960, do_sample=True, top_k=args.top_k)

    @torch.no_grad()
    def run(self, prompts, seeds):
        from transformers import LogitsProcessorList, PrefixConstrainedLogitsProcessor
        from transformers.generation import UnbatchedClassifierFreeGuidanceLogitsProcessor
        from PIL import Image
        a = self.args
        torch.manual_seed(seeds[0])
        texts = [p if a.no_positive_suffix else p + POSITIVE_SUFFIX for p in prompts]
        kw = dict(mode="G", ratio=[a.ratio] * len(prompts), image_area=self.model.config.image_area,
                  return_tensors="pt", padding="longest")
        pos = self.processor(text=texts, **kw)
        procs = []
        if a.cfg != 1.0 and a.negative_prompt:
            neg = self.processor(text=[a.negative_prompt] * len(prompts), **kw)
            procs.append(UnbatchedClassifierFreeGuidanceLogitsProcessor(
                a.cfg, self.model, unconditional_ids=neg.input_ids.to(self.device),
                unconditional_attention_mask=neg.attention_mask.to(self.device)))
        h, w = pos.image_size[:, 0], pos.image_size[:, 1]
        procs.append(PrefixConstrainedLogitsProcessor(self.processor.build_prefix_constrained_fn(h, w), num_beams=1))
        out = self.model.generate(pos.input_ids.to(self.device), self.gen_config,
                                  logits_processor=LogitsProcessorList(procs),
                                  attention_mask=pos.attention_mask.to(self.device))
        results = []
        for row in out:
            mm = self.processor.decode(row)
            if not isinstance(mm, list):
                mm = [mm]
            images = [m for m in mm if isinstance(m, Image.Image)]
            if not images:   # malformed grid: blank image rather than a crash (scored as a failure)
                images = [Image.new("RGB", (DEFAULT_SIZE, DEFAULT_SIZE))]
            results.append(([images[0]], []))
        return results
