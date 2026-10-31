"""Janus-Pro-R1 / Janus-Pro-7B baseline generator for scripts/eval/bench_jeet_mila.sh.

Drop-in GEN_SCRIPT for jeet's iterative generators (eval/iterative_generation_adhoc*.py,
eval/iterative_genai_bench_adhoc*.py, eval/iterative_tiif_bench_adhoc*.py): it takes the same
command line (Tar-only flags such as --model, --cfg_scale, --top_k, --ar_path ... are accepted and
ignored), shards prompts across torchrun ranks the same way (prompt index i goes to rank
i % world_size) and writes the same per-iteration layout, so the launcher's completeness checks and
the scorers read the results unchanged. The benchmark is inferred from the flags the launcher passes:

    GenEval2  : --prompts_file ... --out_dir D --geneval2
                D/{k}/{idx:05d}/samples/00000.png, D/{k}/{idx:05d}/metadata.jsonl,
                D/{k}/{idx:05d}/self_reflect.txt (k >= 2), D/{k}/geneval2_results.json
                ({prompt: absolute png path}), D/janus_selfcheck.jsonl
    GenAI     : --root_dir R --eval_set_dir E --output_dir O --gen_model G
                O/G/{k}/<prompt_idx>.jpeg, O/G/self_reflect.json, O/G/janus_selfcheck.jsonl
    TIIF      : --prompts_dir P --output_dir O --gen_model G --eval_model_name M
                O/G/images/<dimension>/M_iter{k}/<short|long>_description/<idx>.png,
                O/G/self_reflect.json, O/G/janus_selfcheck.jsonl

Iterations (k = 1..--n_iters, the launcher forces 2 for Janus):
    k = 1          the first generated image (plain Janus text-to-image, CFG 5);
    k = n_iters    the loop's final image: if the model's self-check says "Yes" (accept) it is
                   image 1, otherwise the regenerated image (after up to --max_images - 1 rounds);
    1 < k < n_iters the image after k - 1 rounds (only when n_iters > 2).
--mode single = plain Janus-Pro text-to-image (image 1 only; every iteration repeats it).

The R1 loop is copied from Janus-Pro-R1 (https://github.com/wendell0218/Janus-Pro-R1, commit
0e40b3a), inference/inference.py:generate_with_refine (lines 29-262; task 1 = text-to-image,
task 2 = self-check "Let me think Does this image match the prompt..." whose first token is
restricted to Yes/No, task 3 = regeneration conditioned on prompt + image + self-check text).
Changes, all deliberate:
  * batching over *different* prompts with left padding: the original only batches copies of one
    prompt and builds tasks 2/3 with an all-ones attention mask over the prompt; here the prompt's
    real padding mask is used in all three tasks (identical for unpadded rows);
  * prompts are not truncated (the original truncates at 200 tokens, which would cut the trailing
    <begin_of_image> off long TIIF prompts; the longest TIIF long_description is 522 tokens);
  * task 3 only runs for rows whose self-check said "No" (the original regenerates every row and
    then discards the regenerated image for "Yes" rows; same outputs, less compute);
  * the understanding embeddings of task 2 are computed batched as aligner(vision_model(pixels)),
    which is exactly what prepare_inputs_embeds(...)[:, 2:-1] puts at the 576 image positions;
  * --max_images > 2 runs further rounds Markov-style as inference/inference_scaling.py does
    (reflect on the latest image given the prompt only, stop on "Yes"), with inference.py's
    fixed CFG / sampling instead of inference_scaling.py's random CFG and yes-score threshold.
The vendored Janus model code is imported from the clone (inference/models, inference/utils); it is
not modified.

Usage (normally via the launcher: GEN_FAMILY=janus ... scripts/eval/bench_jeet_mila.sh):
    torchrun --standalone --nproc_per_node=4 eval/baselines/janus_pro_r1_gen.py \\
        --model_path midbee/Janus-Pro-R1-7B --mode r1_loop --n_iters 2 \\
        --prompts_file $BENCH/GenEval2/geneval2_data.jsonl --geneval2 --out_dir OUT --seed 713
CPU check of arguments / layout without loading a model (dummy 384 px images):
    python eval/baselines/janus_pro_r1_gen.py --dry_run --limit_prompts 3 <same args>
"""

import argparse
import datetime
import glob
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image

DEFAULT_JANUS_REPO = "/network/scratch/s/singhsd/baselines/janus_pro_r1/Janus-Pro-R1/inference"

# Janus-Pro-R1 inference/inference.py:generate_with_refine, the two text templates.
SELFCHECK_TEXT = "<end_of_image>\nLet me think Does this image match the prompt..."
REGEN_TEXT = "<｜end▁of▁sentence｜>\nNext, I will draw a new image<begin_of_image>"
EOS_TEXT = "<｜end▁of▁sentence｜>"
PAD_TEXT = "<｜▁pad▁｜>"
TIIF_DESCRIPTIONS = ("short_description", "long_description")


# ----------------------------------------------------------------------------- arguments
def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)  # no prefix matching: --model != --model_path
    parser.add_argument("--model_path", default="midbee/Janus-Pro-R1-7B",
                        help="HF id or local dir (midbee/Janus-Pro-R1-7B, deepseek-ai/Janus-Pro-7B).")
    parser.add_argument("--mode", choices=["r1_loop", "single"], default="r1_loop")
    parser.add_argument("--max_images", type=int, default=2,
                        help="Images per prompt including the first (inference.py: 2 = one "
                             "self-check + regeneration round).")
    parser.add_argument("--janus_cfg", type=float, default=5.0, help="CFG weight (inference.py: 5.0).")
    parser.add_argument("--janus_temperature", type=float, default=1.0,
                        help="Sampling temperature for image and text tokens (inference.py: 1.0).")
    parser.add_argument("--janus_img_top_k", type=int, default=None)
    parser.add_argument("--janus_img_top_p", type=float, default=None)
    parser.add_argument("--janus_txt_top_k", type=int, default=None)
    parser.add_argument("--janus_txt_top_p", type=float, default=None)
    parser.add_argument("--max_reflect_len", type=int, default=80, help="inference.py: 80.")
    parser.add_argument("--janus_batch_size", type=int, default=16,
                        help="Prompts per forward (x2 rows for CFG). The launcher's --batch_size "
                             "is sized for Tar and is ignored.")
    parser.add_argument("--janus_repo", default=os.environ.get("JANUS_R1_REPO", DEFAULT_JANUS_REPO),
                        help="Janus-Pro-R1 clone's inference/ dir (vendored models/ and utils/).")
    parser.add_argument("--save_size", type=int, default=0,
                        help="Resize saved images to this square size (0 = native 384 px).")
    parser.add_argument("--dry_run", action="store_true",
                        help="No model: write dummy images to check arguments and layout (CPU).")
    parser.add_argument("--limit_prompts", type=int, default=0,
                        help="Use only the first N prompts (0 = all).")
    parser.add_argument("--bench", choices=["geneval2", "genai", "tiif"],
                        help="Default: inferred from the flags.")
    # --- flags shared with jeet's generators (same names and defaults)
    parser.add_argument("--n_iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    # GenEval2
    parser.add_argument("--prompts_file")
    parser.add_argument("--out_dir", default="results/iterative")
    parser.add_argument("--geneval2", action="store_true")
    parser.add_argument("--strip_photo_prefix", action=argparse.BooleanOptionalAction, default=False)
    # GenAI-Bench / TIIF
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--gen_model")
    parser.add_argument("--root_dir", default="./datasets")
    parser.add_argument("--eval_set_dir")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--prompts_dir")
    parser.add_argument("--dimensions", nargs="*")
    parser.add_argument("--descriptions", nargs="*", choices=TIIF_DESCRIPTIONS,
                        default=list(TIIF_DESCRIPTIONS))
    parser.add_argument("--eval_model_name", default="tar")
    args, unknown = parser.parse_known_args(argv)
    args.ignored = unknown
    if args.bench is None:
        if args.prompts_dir:
            args.bench = "tiif"
        elif args.eval_set_dir:
            args.bench = "genai"
        elif args.prompts_file:
            args.bench = "geneval2"
        else:
            parser.error("cannot infer the benchmark: pass --prompts_file (GenEval2), "
                         "--eval_set_dir (GenAI-Bench) or --prompts_dir (TIIF), or --bench")
    if args.n_iters < 1 or args.max_images < 1:
        parser.error("--n_iters and --max_images must be positive")
    if args.repeat != 1:
        parser.error("only --repeat 1 is supported (the scorers read sample 0 only)")
    if args.bench in ("genai", "tiif") and not args.gen_model:
        parser.error("--gen_model is required for GenAI-Bench / TIIF")
    if args.mode == "single":
        args.max_images = 1
    return args


# ----------------------------------------------------------------------------- prompts
PHOTO_PREFIX = "a photo of "


def strip_photo_prefix(prompt):  # = eval/iterative_generation_adhoc.py
    if prompt.lower().startswith(PHOTO_PREFIX):
        return prompt[len(PHOTO_PREFIX):].lstrip()
    return prompt


def load_geneval2(path, strip_prefix=False):
    """[(prompt, metadata_line)], as eval/iterative_generation_adhoc.py:load_prompts."""
    records = []
    with open(path) as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                prompt = strip_photo_prefix(line) if strip_prefix else line
                metadata = json.dumps({"prompt": prompt})
            else:
                if not isinstance(obj, dict) or not isinstance(obj.get("prompt"), str):
                    raise ValueError(f"{path}:{line_number}: JSON record needs a 'prompt' string")
                prompt = strip_photo_prefix(obj["prompt"]) if strip_prefix else obj["prompt"]
                metadata = line
            records.append((prompt, metadata))
    return records


def genai_path(iter_dir, prompt_idx):
    return os.path.join(iter_dir, f"{prompt_idx}.jpeg")   # evaluate.py reads <prompt_idx>.jpeg


def load_genai(args, iter_dirs):
    """[(prompt_idx, prompt)] sorted by key, as eval/iterative_genai_bench_adhoc.py:load_prompts."""
    path = os.path.join(args.root_dir, args.eval_set_dir, "genai_image.json")
    with open(path) as f:
        dataset = json.load(f)
    records = []
    for prompt_idx in sorted(dataset):
        if args.skip_existing and all(os.path.exists(genai_path(d, prompt_idx)) for d in iter_dirs):
            continue
        prompt = dataset[prompt_idx]["prompt"]
        records.append((prompt_idx, strip_photo_prefix(prompt) if args.strip_photo_prefix else prompt))
    return records


def tiif_path(images_dir, dimension, model_name, desc, idx):
    return os.path.join(images_dir, dimension, model_name, desc, f"{idx}.png")


def load_tiif(args, images_dir, model_names):
    """[(dimension, desc, idx, prompt)], as eval/iterative_tiif_bench_adhoc.py:collect_records."""
    files = sorted(glob.glob(os.path.join(args.prompts_dir, "*_prompts.jsonl")))
    if not files:
        raise FileNotFoundError(f"No '*_prompts.jsonl' in {args.prompts_dir}")
    wanted = set(args.dimensions) if args.dimensions else None
    dims = {}
    for path in files:
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if not lines:
            continue
        dimension = lines[0]["type"]   # eval_with_vlm.py builds the image dir from "type"
        if wanted is None or dimension in wanted:
            dims[dimension] = lines
    records = []
    for dimension, lines in sorted(dims.items()):
        for idx, record in enumerate(lines):
            for desc in args.descriptions:
                prompt = record.get(desc)
                if not prompt:
                    raise ValueError(f"{dimension}[{idx}]: missing or empty '{desc}'")
                if args.skip_existing and all(
                        os.path.exists(tiif_path(images_dir, dimension, m, desc, idx))
                        for m in model_names):
                    continue
                records.append((dimension, desc, idx, prompt))
    return records


# ----------------------------------------------------------------------------- distributed / io
def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        if args.dry_run:
            dist.init_process_group("gloo")
            device = torch.device("cpu")
        else:
            local_rank = int(os.environ["LOCAL_RANK"])
            # Long timeout: ranks finish their shards at very different times (resumed jobs, self-correction loops
            # with variable rounds), and the final gather must not hit NCCL's 10-minute watchdog.
            dist.init_process_group("nccl", timeout=datetime.timedelta(hours=48))
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
    else:
        device = torch.device("cpu" if args.dry_run else args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    return rank, world_size, device


def barrier(world_size):
    if world_size > 1:
        dist.barrier()


def reset_output_dirs(dirs, rank, world_size):
    """= eval/iterative_generation_adhoc.py: rank 0 wipes, then every rank recreates."""
    if rank == 0:
        for d in dirs:
            if os.path.lexists(d):
                if os.path.isdir(d) and not os.path.islink(d):
                    shutil.rmtree(d)
                else:
                    os.unlink(d)
    barrier(world_size)
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def gather(obj, world_size):
    if world_size == 1:
        return [obj]
    out = [None] * world_size
    dist.all_gather_object(out, obj)
    return out


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def iteration_images(images, n_iters):
    """Map the loop's images (first, ..., final) to iterations 1..n_iters (see module doc)."""
    if n_iters == 1:
        return [images[0]]
    out = [images[min(k, len(images)) - 1] for k in range(1, n_iters + 1)]
    out[-1] = images[-1]
    return out


# ----------------------------------------------------------------------------- Janus model
def load_janus(args, device):
    if args.janus_repo not in sys.path:
        sys.path.insert(0, args.janus_repo)   # vendored `models` and `utils` packages
    from transformers import AutoModelForCausalLM
    from models import MultiModalityCausalLM, VLChatProcessor  # noqa: F401  (registers the config)
    processor = VLChatProcessor.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    return processor, model


def sample_token(logits, temperature, top_k=None, top_p=None):
    """inference.py's sampling (top-k, temperature, top-p, multinomial)."""
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        probs_sort[probs_sum - probs_sort > top_p] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        return torch.gather(probs_idx, -1, torch.multinomial(probs_sort, num_samples=1))
    return torch.multinomial(probs, num_samples=1)


def center_crop_arr(pil_image, image_size):  # inference.py
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size),
                                 resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class JanusR1:
    """Batched copy of inference.py:generate_with_refine (see module doc for the changes)."""

    IMG_TOKENS = 576
    IMG_SIZE = 384
    PATCH = 16

    def __init__(self, processor, model, args, device):
        self.p, self.m, self.args, self.device = processor, model, args, device
        tok = processor.tokenizer
        self.tok = tok
        self.pad_id = processor.pad_id
        self.eos_token = tok.encode(EOS_TEXT)[-1]
        self.padding_token = tok.encode(PAD_TEXT)[-1]
        self.yes_token = tok.encode("Yes")[-1]
        self.no_token = tok.encode("No")[-1]
        self.selfcheck_ids = torch.tensor(tok.encode(SELFCHECK_TEXT)[1:], device=device)  # drop BOS
        self.regen_ids = torch.tensor(tok.encode(REGEN_TEXT)[1:], device=device)
        self.embed = model.language_model.get_input_embeddings()

    # -- prompt tokens: inference.py __main__ (sft template, system prompt "", + <begin_of_image>)
    def prompt_tokens(self, prompts):
        texts = []
        for prompt in prompts:
            conv = [{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}]
            texts.append(self.p.apply_sft_template_for_multi_turn_prompts(
                conversations=conv, sft_format=self.p.sft_format, system_prompt="")
                + self.p.image_start_tag)
        enc = self.tok(texts, return_tensors="pt", padding="longest")  # left padding; no truncation
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def cfg_rows(self, ids, mask):
        """Interleave cond / uncond rows; uncond keeps BOS and <begin_of_image>, pads the rest."""
        tokens = torch.repeat_interleave(ids, 2, dim=0)
        for i in range(1, tokens.size(0), 2):
            st = int((mask[i // 2] == 0).sum().item()) + 1   # first position after BOS
            tokens[i, st:-1] = self.pad_id
        return tokens, torch.repeat_interleave(mask, 2, dim=0)

    def sample_image(self, inputs_embeds, attn_mask):
        """576 CFG-guided image tokens from a (2B, T, D) cond/uncond prefix."""
        a = self.args
        lm = self.m.language_model.model
        n = inputs_embeds.size(0) // 2
        out_tokens = torch.zeros((n, self.IMG_TOKENS), dtype=torch.int, device=self.device)
        past = None
        for i in range(self.IMG_TOKENS):
            outputs = lm(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                         use_cache=True, past_key_values=past)
            past = outputs.past_key_values
            logits = self.m.gen_head(outputs.last_hidden_state[:, -1, :])
            logits = logits[1::2] + a.janus_cfg * (logits[0::2] - logits[1::2])
            next_token = sample_token(logits, a.janus_temperature, a.janus_img_top_k, a.janus_img_top_p)
            out_tokens[:, i] = next_token.squeeze(-1)
            next_token = torch.cat([next_token, next_token], dim=1).view(-1)
            inputs_embeds = self.m.prepare_gen_img_embeds(next_token).unsqueeze(1)
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones(attn_mask.size(0), 1)], dim=1)
        return out_tokens

    def decode(self, tokens):
        s = self.IMG_SIZE // self.PATCH
        dec = self.m.gen_vision_model.decode_code(tokens.to(torch.int), shape=[tokens.size(0), 8, s, s])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return [Image.fromarray(x) for x in dec]

    # task 1
    def text_to_image(self, ids, mask):
        tokens, m2 = self.cfg_rows(ids, mask)
        return self.decode(self.sample_image(self.embed(tokens), m2))

    # task 2
    def self_check(self, ids, mask, images):
        a = self.args
        n = ids.size(0)
        pixel = self.p.image_processor(images, return_tensors="pt").pixel_values
        pixel = pixel.to(device=self.device, dtype=torch.bfloat16)
        under = self.m.aligner(self.m.vision_model(pixel))          # (n, 576, D)
        sc = self.embed(self.selfcheck_ids).unsqueeze(0).expand(n, -1, -1)
        inputs_embeds = torch.cat([self.embed(ids), under.to(sc.dtype), sc], dim=1)
        attn_mask = torch.cat([mask, mask.new_ones(n, under.size(1) + sc.size(1))], dim=1)
        reflect = torch.zeros((n, a.max_reflect_len), dtype=torch.long, device=self.device)
        eos_list = torch.zeros((n, 1), dtype=torch.bool, device=self.device)
        add_padding = torch.zeros((n, 1), dtype=torch.long, device=self.device)
        past, length = None, 0
        for i in range(a.max_reflect_len):
            outputs = self.m.language_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                                            use_cache=True, past_key_values=past)
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()
            if i == 0:   # first token restricted to Yes / No (sampled, not argmax)
                allowed = [self.yes_token, self.no_token]
                kept = logits[:, allowed]
                logits[:, :] = -math.inf
                logits[:, allowed] = kept
            next_token = sample_token(logits, a.janus_temperature, a.janus_txt_top_k, a.janus_txt_top_p)
            if i >= 1:
                add_padding = ((reflect[:, i - 1] == self.eos_token)
                               | (reflect[:, i - 1] == self.padding_token)).unsqueeze(1).long()
            next_token = add_padding * self.padding_token + (1 - add_padding) * next_token
            if i == 0:
                yes = (next_token.squeeze(1) == self.yes_token)
            reflect[:, i] = next_token.squeeze(1)
            is_eos = next_token == self.eos_token
            eos_list |= is_eos
            new_attn = ((1 - add_padding).bool() & ~is_eos).to(attn_mask.dtype)
            attn_mask = torch.cat([attn_mask, new_attn], dim=1)
            inputs_embeds = self.embed(next_token)
            length = i + 1
            if eos_list.all():
                break
        reflect = reflect[:, :length]
        txt_mask = torch.ones_like(reflect)
        txt_mask[(reflect == self.padding_token) | (reflect == self.eos_token)] = 0
        texts = [self.tok.decode(r.tolist(), skip_special_tokens=True) for r in reflect]
        return yes.cpu().tolist(), reflect, txt_mask, texts

    # task 3
    def regenerate(self, ids, mask, images, reflect, txt_mask):
        n = ids.size(0)
        tokens, m2 = self.cfg_rows(ids, mask)
        to_tensor = T.Compose([   # inference.py gen_transform
            T.Lambda(lambda im: center_crop_arr(im, self.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True)])
        pix = torch.stack([to_tensor(im) for im in images]).to(device=self.device, dtype=torch.bfloat16)
        _, _, info = self.m.gen_vision_model.encode(pix)            # VQ re-encode of the decoded image
        image_ids = info[2].view(n, -1)
        gen = self.m.gen_aligner(self.m.gen_embed(image_ids))       # (n, 576, D)
        gen = torch.repeat_interleave(gen, 2, dim=0)
        sc = self.embed(self.selfcheck_ids).unsqueeze(0).expand(2 * n, -1, -1)
        txt = self.embed(torch.repeat_interleave(reflect, 2, dim=0))
        rg = self.embed(self.regen_ids).unsqueeze(0).expand(2 * n, -1, -1)
        inputs_embeds = torch.cat([self.embed(tokens), gen.to(sc.dtype), sc, txt, rg], dim=1)
        ones = lambda k: m2.new_ones(2 * n, k)  # noqa: E731
        attn_mask = torch.cat([m2, ones(gen.size(1) + sc.size(1)),
                               torch.repeat_interleave(txt_mask, 2, dim=0).to(m2.dtype),
                               ones(rg.size(1))], dim=1)
        return self.decode(self.sample_image(inputs_embeds, attn_mask))

    @torch.no_grad()
    def run(self, prompts):
        """Per prompt: (images [first, ..., final], rounds [{accepted, reason}])."""
        ids, mask = self.prompt_tokens(prompts)
        first = self.text_to_image(ids, mask)
        images = [[im] for im in first]
        rounds = [[] for _ in prompts]
        active = list(range(len(prompts)))
        for _ in range(self.args.max_images - 1):
            if not active:
                break
            sel = torch.tensor(active, device=self.device)
            cur = [images[i][-1] for i in active]
            yes, reflect, txt_mask, texts = self.self_check(ids[sel], mask[sel], cur)
            for i, y, t in zip(active, yes, texts):
                rounds[i].append({"accepted": bool(y), "reason": t})
            redo = [j for j, y in enumerate(yes) if not y]
            if redo:
                rsel = torch.tensor(redo, device=self.device)
                new = self.regenerate(ids[sel][rsel], mask[sel][rsel], [cur[j] for j in redo],
                                      reflect[rsel], txt_mask[rsel])
                for j, im in zip(redo, new):
                    images[active[j]].append(im)
            active = [active[j] for j in redo]
        return list(zip(images, rounds))


class DryRun:
    """No model: dummy 384 px images; alternating accept / reject self-checks."""

    def __init__(self, args):
        self.args = args

    def run(self, prompts):
        out = []
        for n, prompt in enumerate(prompts):
            color = tuple((hash(prompt) >> s) & 255 for s in (0, 8, 16))
            images = [Image.new("RGB", (384, 384), color)]
            rounds = []
            if self.args.max_images > 1:
                accepted = n % 2 == 0
                rounds.append({"accepted": accepted, "reason": ("Yes" if accepted else "No")
                               + ", dry run"})
                if not accepted:
                    images.append(Image.new("RGB", (384, 384), tuple(255 - c for c in color)))
            out.append((images, rounds))
        return out


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    rank, world_size, device = init_distributed(args)
    if args.janus_batch_size < 1:
        raise ValueError("--janus_batch_size must be positive")
    torch.manual_seed(args.seed + rank)   # same convention as jeet's generators
    np.random.seed(args.seed + rank)
    if rank == 0:
        print(f"[janus] bench={args.bench} mode={args.mode} model={args.model_path} "
              f"max_images={args.max_images} n_iters={args.n_iters} cfg={args.janus_cfg} "
              f"T={args.janus_temperature} seed={args.seed} world={world_size} dry_run={args.dry_run}")
        if args.ignored:
            print(f"[janus] ignoring Tar-only arguments: {' '.join(args.ignored)}")

    n = args.n_iters
    # ---- output dirs and prompt records (same rules as jeet's generators)
    if args.bench == "geneval2":
        save_dir = os.path.abspath(args.out_dir)
        iter_dirs = [os.path.join(save_dir, str(k)) for k in range(1, n + 1)]
        reset_output_dirs(iter_dirs, rank, world_size)
        records = load_geneval2(args.prompts_file, args.strip_photo_prefix)
        keys = list(range(len(records)))
        prompts = [r[0] for r in records]
    elif args.bench == "genai":
        save_dir = os.path.abspath(os.path.join(args.output_dir, args.gen_model))
        iter_dirs = [os.path.join(save_dir, str(k)) for k in range(1, n + 1)]
        if args.skip_existing:
            for d in iter_dirs:
                os.makedirs(d, exist_ok=True)
        else:
            reset_output_dirs(iter_dirs, rank, world_size)
        records = load_genai(args, iter_dirs)
        keys = [r[0] for r in records]
        prompts = [r[1] for r in records]
    else:
        save_dir = os.path.abspath(os.path.join(args.output_dir, args.gen_model))
        images_dir = os.path.join(save_dir, "images")
        model_names = [f"{args.eval_model_name}_iter{k}" for k in range(1, n + 1)]
        if not args.skip_existing:
            reset_output_dirs([images_dir], rank, world_size)
        records = load_tiif(args, images_dir, model_names)
        keys = [r[:3] for r in records]
        prompts = [r[3] for r in records]
        if rank == 0:
            for dimension, desc, idx in keys:
                for m in model_names:
                    os.makedirs(os.path.dirname(tiif_path(images_dir, dimension, m, desc, idx)),
                                exist_ok=True)
        barrier(world_size)
    if args.limit_prompts:
        keys, prompts = keys[:args.limit_prompts], prompts[:args.limit_prompts]
    if rank == 0:
        print(f"[janus] {len(prompts)} prompts, writing to {save_dir}", flush=True)

    gen = DryRun(args) if args.dry_run else JanusR1(*load_janus(args, device), args, device)
    if rank == 0 and not args.dry_run:
        print(f"[janus] attention: {gen.m.language_model.config._attn_implementation}", flush=True)

    jobs = list(range(rank, len(prompts), world_size))   # prompt i -> rank i % world_size
    local_results, local_geneval2 = {}, [{} for _ in range(n)]
    t0 = time.perf_counter()
    for b, batch in enumerate(chunks(jobs, args.janus_batch_size)):
        outputs = gen.run([prompts[i] for i in batch])
        for i, (images, rounds) in zip(batch, outputs):
            per_iter = iteration_images(images, n)
            if args.save_size:
                per_iter = [im.resize((args.save_size, args.save_size), Image.BICUBIC) for im in per_iter]
            if args.bench == "geneval2":
                for k, im in enumerate(per_iter):
                    prompt_dir = os.path.join(iter_dirs[k], f"{i:05d}")
                    sample_dir = os.path.join(prompt_dir, "samples")
                    os.makedirs(sample_dir, exist_ok=True)
                    path = os.path.join(sample_dir, "00000.png")
                    im.save(path)
                    with open(os.path.join(prompt_dir, "metadata.jsonl"), "w") as f:
                        f.write(records[i][1] + "\n")
                    local_geneval2[k][prompts[i]] = os.path.abspath(path)
                    if k >= 1:   # critique that led to image k+1 (empty: no further round)
                        reason = rounds[k - 1]["reason"] if k - 1 < len(rounds) else ""
                        with open(os.path.join(prompt_dir, "self_reflect.txt"), "w") as f:
                            f.write(f"Prompt: {prompts[i]}\n\nSample: 00000\nSelf-reflect: {reason}\n")
                key = f"{i:05d}"
            elif args.bench == "genai":
                for k, im in enumerate(per_iter):
                    im.convert("RGB").save(genai_path(iter_dirs[k], keys[i]))
                key = keys[i]
            else:
                dimension, desc, idx = keys[i]
                for k, im in enumerate(per_iter):
                    im.save(tiif_path(images_dir, dimension, model_names[k], desc, idx))
                key = f"{dimension}/{desc}/{idx}"
            local_results[key] = {
                "prompt": prompts[i], "n_images": len(images),
                "accepted_first": bool(rounds and rounds[0]["accepted"]),
                "rounds": rounds}
            if args.verbose:
                print(f"[rank {rank}] {key}: {prompts[i]!r} -> {len(images)} image(s); "
                      f"{[r['reason'] for r in rounds]}", flush=True)
        print(f"[rank {rank}] batch {b + 1}: {min((b + 1) * args.janus_batch_size, len(jobs))}/"
              f"{len(jobs)} prompts, {time.perf_counter() - t0:.0f}s", flush=True)

    all_results = gather(local_results, world_size)
    all_geneval2 = gather(local_geneval2, world_size) if args.bench == "geneval2" else None
    if rank == 0:
        merged = {}
        for part in all_results:
            merged.update(part)
        with open(os.path.join(save_dir, "janus_selfcheck.jsonl"), "w") as f:
            for key in sorted(merged):
                f.write(json.dumps({"key": key, **merged[key]}, ensure_ascii=False) + "\n")
        if args.bench == "geneval2":
            for k, d in enumerate(iter_dirs):
                result = {}
                for part in all_geneval2:
                    result.update(part[k])
                with open(os.path.join(d, "geneval2_results.json"), "w") as f:
                    json.dump(result, f, indent=2)
        else:   # same schema as jeet's self_reflect.json
            with open(os.path.join(save_dir, "self_reflect.json"), "w") as f:
                json.dump({k: {"prompt": v["prompt"],
                               "reflections": [v["rounds"][0]["reason"] if v["rounds"] else None],
                               "reflections_per_iter": [[r["reason"] for r in v["rounds"]]]}
                           for k, v in sorted(merged.items())}, f, indent=2, ensure_ascii=False)
        n_acc = sum(v["accepted_first"] for v in merged.values())
        print(f"[janus] done: {len(merged)} prompts, first image accepted for {n_acc}, "
              f"regenerated for {sum(v['n_images'] > 1 for v in merged.values())}; {save_dir}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
