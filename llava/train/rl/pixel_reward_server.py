"""Pixel-space GenEval2 soft-TIFA reward server.

Decoded PNGs in, per-question answer probabilities out. Scoring mirrors
``GenEval2/evaluation_optz.py``: one forward pass per (image, question) row
with ``logits_to_keep=1``, softmax in fp32 at the last position, and the score
is the summed probability of the accepted answer variants
(``Yes``/``yes``/`` Yes``/`` yes`` for yes-no questions; the number word, its
capitalised form and the digit for "How many" questions). The image always
comes before the text.

Two judges are in use (``--model`` picks one, the prompt details follow the
architecture, see JUDGE_PROFILES):

* **Qwen3-VL-8B-Instruct** -- GenEval2's *own* judge, so with
  ``--answer_id_mode geneval2`` its scores are the benchmark's own numbers: the
  oracle reward. Template and pixel budget are left at their defaults for
  exactly that reason. Host it with the ``geneval2`` env's python
  (transformers 4.57), the env GenEval2 itself runs in.
* **Gemma 4 26B A4B it** -- a bigger, stricter judge. Thinking is disabled, so
  the generation prompt ends with an empty thought block
  (``<|channel>thought\\n<channel|>``) and the next token is the first token of
  the final answer. Needs transformers >= 5.5: use the ``vllm`` env's python.

Nothing is sampled, so neither model card's sampling parameters apply; the
probabilities are read straight off the next-token distribution (temperature 1).

This runs in a *separate process* from training on purpose: the judges need a
newer transformers than the Tar policy stack (pinned to 4.50), and a 26B judge
does not fit next to the policy on every rank. Launch it on its own GPU, then
point ``train_grpo.py --reward_kind pixel --reward_server_url http://...`` at
it; see output_dir/dbg_rl_ft/bash.sh.

    <env>/bin/python llava/train/rl/pixel_reward_server.py \
        --model /network/scratch/.../models/pre_train/Qwen3-VL-8B-Instruct \
        --port 8765 --batch_size 32 --answer_suffix geneval2

Protocol (JSON over HTTP, one connection per request):
    GET  /health -> {"ok": true, "model": ..., "model_type": ..., "answer_suffix": ...}
    POST /score  <- {"images": [{"png_b64": str, "vqa": [[question, answer], ...]}, ...]}
                 -> {"per_question": [[float, ...], ...]}
"""

import argparse
import base64
import inspect
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from llava.train.rl.reward import ANSWER_SUFFIXES, answer_variants

# Per-architecture prompt handling. Everything else (image before text, one
# forward pass, fp32 softmax at the last position) is shared.
#   template_kwargs: extra kwargs for apply_chat_template
#   soft_tokens:     whether the processor takes a visual token budget
JUDGE_PROFILES = {
    "gemma4": dict(template_kwargs={"enable_thinking": False}, soft_tokens=True),
    "qwen3_vl": dict(template_kwargs={}, soft_tokens=False),
}
DEFAULT_PROFILE = dict(template_kwargs={}, soft_tokens=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="VLM judge checkpoint")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--batch_size", type=int, default=32,
                   help="(image, question) rows scored per forward pass")
    p.add_argument("--answer_suffix", default="geneval2", choices=sorted(ANSWER_SUFFIXES),
                   help="Appended to every question. GenEval2's judge prompt is 'geneval2'.")
    p.add_argument("--image_soft_tokens", type=int, default=None,
                   choices=[70, 140, 280, 560, 1120],
                   help="Gemma 4 visual token budget per image (default: the model's "
                        "own, 280). Ignored by judges without a soft-token budget.")
    p.add_argument("--answer_id_mode", default="strict", choices=["strict", "geneval2"],
                   help="Which token ids count as the answer. 'geneval2' replicates "
                        "evaluation_optz.py exactly -- the first token of every variant, "
                        "duplicates summed twice, and the bare space of variants like ' 6' "
                        "credited -- so the score is the benchmark's own number (use it for "
                        "the oracle judge). 'strict' dedupes and drops bare-space variants.")
    p.add_argument("--attn_implementation", default="sdpa",
                   help="sdpa matches GenEval2's default.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ready_file", default=None,
                   help="Touched once the model is loaded and the socket is listening.")
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


class Judge:
    """Frozen VLM scoring (image, question) rows by answer-token probability."""

    def __init__(self, model_path, answer_suffix, batch_size, device, attn_implementation,
                 image_soft_tokens=None, answer_id_mode="strict"):
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        self.model_type = getattr(AutoConfig.from_pretrained(model_path), "model_type", "?")
        self.profile = JUDGE_PROFILES.get(self.model_type, DEFAULT_PROFILE)
        proc_kwargs = {}
        if image_soft_tokens is not None:
            if self.profile["soft_tokens"]:
                proc_kwargs = {"max_soft_tokens": image_soft_tokens,
                               "image_seq_length": image_soft_tokens}
            else:
                print(f"[reward] --image_soft_tokens ignored for {self.model_type}", flush=True)
                image_soft_tokens = None
        self.processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)
        # Scores are read at the last position, so every row's final real token
        # must sit at index -1.
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation=attn_implementation)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.batch_size = batch_size
        self.image_soft_tokens = image_soft_tokens
        self.answer_id_mode = answer_id_mode
        # transformers >= 5 wants processor arguments (padding) in processor_kwargs;
        # 4.x takes them as plain kwargs.
        self._nests_processor_kwargs = "processor_kwargs" in inspect.signature(
            type(self.processor).apply_chat_template).parameters
        suffix = ANSWER_SUFFIXES[answer_suffix]
        self._suffix_text = (" " + suffix) if suffix else ""
        self._answer_cache = {}
        space = self.processor.tokenizer.encode(" ", add_special_tokens=False)
        self._space_id = space[0] if len(space) == 1 else None

    def question_text(self, question: str) -> str:
        return f"{question}{self._suffix_text}"

    def answer_ids(self, question: str, answer: str):
        """Token ids whose probability counts as this answer.

        ``geneval2`` mode keeps the first token of every variant as
        evaluation_optz.py does, duplicates included. ``strict`` mode dedupes and
        drops variants whose first token is a bare space (`' 2'` tokenises to
        [space, '2'] on some tokenizers, and a bare space is not an answer).
        """
        key = (question.startswith("How many"), answer)
        ids = self._answer_cache.get(key)
        if ids is None:
            firsts = []
            for variant in answer_variants(question, answer):
                enc = self.processor.tokenizer.encode(variant, add_special_tokens=False)
                if enc:
                    firsts.append(enc[0])
            if self.answer_id_mode == "geneval2":
                ids = firsts
            else:
                ids = sorted({i for i in firsts if i != self._space_id})
            self._answer_cache[key] = ids
        return ids

    def _template_inputs(self, messages):
        kwargs = dict(self.profile["template_kwargs"])
        if self._nests_processor_kwargs:
            kwargs["processor_kwargs"] = {"padding": True}
        else:
            kwargs["padding"] = True
        return self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", **kwargs)

    @torch.inference_mode()
    def score(self, images, vqa_lists):
        """images: list[PIL.Image]; vqa_lists: list[list[(question, answer)]].

        Returns one list of probabilities per image, aligned with its vqa list.
        """
        rows = []
        for i, (image, vqa) in enumerate(zip(images, vqa_lists)):
            for slot, (question, answer) in enumerate(vqa):
                rows.append((i, slot, image, self.question_text(question),
                             self.answer_ids(question, answer)))
        out = [[0.0] * len(vqa) for vqa in vqa_lists]
        for start in range(0, len(rows), self.batch_size):
            chunk = rows[start:start + self.batch_size]
            messages = [[{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ]}] for _, _, image, text, _ in chunk]
            inputs = self._template_inputs(messages).to(self.device)
            logits = self.model(**inputs, logits_to_keep=1).logits
            # `generate` upcasts before the softmax; match it for identical numbers.
            probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
            for j, (i, slot, _, _, ids) in enumerate(chunk):
                out[i][slot] = float(probs[j, ids].sum()) if ids else 0.0
        return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            judge = self.server.judge
            self._send(200, {"ok": True, "model": self.server.model_path,
                             "model_type": judge.model_type,
                             "answer_suffix": self.server.answer_suffix,
                             "image_soft_tokens": judge.image_soft_tokens,
                             "answer_id_mode": judge.answer_id_mode})
        else:
            self._send(404, {"error": f"no such path: {self.path}"})

    def do_POST(self):
        if self.path.rstrip("/") != "/score":
            self._send(404, {"error": f"no such path: {self.path}"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            items = req["images"]
            images = [Image.open(io.BytesIO(base64.b64decode(it["png_b64"]))).convert("RGB")
                      for it in items]
            vqa_lists = [[tuple(qa) for qa in it["vqa"]] for it in items]
        except Exception as e:                                  # malformed request
            self._send(400, {"error": f"{type(e).__name__}: {e}"})
            return
        try:
            t0 = time.time()
            # One judge, many training ranks: serialise the GPU work.
            with self.server.lock:
                per_question = self.server.judge.score(images, vqa_lists)
            if self.server.verbose:
                nq = sum(len(v) for v in vqa_lists)
                print(f"[reward] {len(images)} images / {nq} questions "
                      f"in {time.time() - t0:.1f}s", flush=True)
            self._send(200, {"per_question": per_question})
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *a):     # quiet: one line per /score instead
        pass


def main():
    args = parse_args()
    print(f"[reward] loading {args.model} on {args.device} "
          f"(attn={args.attn_implementation})", flush=True)
    t0 = time.time()
    judge = Judge(args.model, args.answer_suffix, args.batch_size,
                  args.device, args.attn_implementation, args.image_soft_tokens,
                  args.answer_id_mode)
    print(f"[reward] loaded {judge.model_type} in {time.time() - t0:.0f}s", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.judge = judge
    server.lock = threading.Lock()
    server.model_path = args.model
    server.answer_suffix = args.answer_suffix
    server.verbose = args.verbose
    if args.ready_file:
        with open(args.ready_file, "w") as f:
            f.write(f"{args.host}:{args.port}\n")
    print(f"[reward] listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
