'''
vLLM version of decompose_reflection_correction.py for full-size runs.

Everything except the generation backend is imported from decompose_reflection_correction.py: the prompt and few-shot
examples, row parsing / junk filtering, reply validation and the retry message, the JSONL cache (same prompt version,
so a cache started with the HF script is reused here and vice versa), the rank barrier and the shard rewrite. Only
model loading and the generation loop (run_llm) are replaced.

Generation: one vLLM engine per process (tensor_parallel_size GPUs each), see decompose_reflection_correction_vllm.sh.
Requests are streamed into the engine (LLMEngine.add_request / step), keeping --max_in_flight of them queued, so the
batch never drains between chunks. Each finished reply is validated as soon as it arrives: a valid (or out of retries)
row is appended to the cache (a killed job resumes from there), and a rejected reply is resubmitted right away with the
rejected reply and reason added to the conversation, exactly as in the HF script.

Throughput notes (Qwen3.6-27B, A100-80G):
    - Prompts are ~3.7k tokens, 3.5k of them the shared system prompt + few-shot turns. That prefix is tokenized once
      and only the row's own turns are tokenized per request (token-identical to tokenizing the rendered chat template,
      checked on 600 hqedt / UnicEdit prompts with retries; 0.1 ms instead of 5 ms per row), and prefix caching skips
      its prefill.
    - Measured on 4x A100-80G (NVLink), hqedt small_sample (5.5k rows, ~14% of rows retried at least once), 2026-10-04:
      TP 1 x 4 engines (256 seqs each) 13.5 rows/s, TP 2 x 2 engines (512 seqs each) 12.7 rows/s, the same 97% ok
      either way. TP 1 is the default. One engine per GPU has 364 linear-attention state slots at 0.9 memory
      utilization, so --max_num_seqs above that fails at startup.

Length outliers: rows whose source text (caption + Self-reflect + Correction) is longer than --max_source_tokens
(default 512), or in the longest --drop_longest_pct percent of the rank's rows (off by default), are not generated and
are dropped from the output shards like junk rows. On small_sample, 512 tokens is above hqedt's p99.9 (471, max 567)
and above every UnicEdit row (max 161), so it only trims the extreme tail. They are cached as status "too_long" and
re-checked against the current cut on every run, so changing the cut needs no cache cleanup.

Arguments: every option of decompose_reflection_correction.py (its --batch_size, --device and --attn_implementation are
unused here) plus the vLLM options defined below.
'''

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
from tqdm import tqdm

import decompose_reflection_correction as base
from decompose_reflection_correction import (
    _resolve_no_think_kwargs,
    _user_message,
    build_messages,
    caption_of,
    find_edit_turn,
    job_key,
    logger,
    parse_reply,
    prompt_version,
)

TOO_LONG = "too_long"


def load_engine(args):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    speculative_config = None
    if args.num_speculative_tokens > 0:
        # Qwen3.6 ships a multi-token-prediction head; it drafts tokens that the main model verifies.
        speculative_config = {"method": "mtp", "num_speculative_tokens": args.num_speculative_tokens}
    llm = LLM(
        model=args.model_path,
        runner="generate",
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        # Qwen3.6 ships as an image-text model; the vision tower is not needed for text-only prompts.
        language_model_only=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        disable_log_stats=True,
        enforce_eager=args.enforce_eager,
        speculative_config=speculative_config,
        seed=args.seed + args.rank,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p if args.temperature > 0 else 1.0,
        top_k=args.top_k if args.temperature > 0 else -1,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        # Only the finished reply is needed; the default (cumulative) re-detokenizes every running request every step.
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    return llm, sampling_params


class PromptTokens:
    """Token ids of a job's chat prompt. The system prompt + few-shot turns are the same for every job, so they are
    tokenized once; per job only the text after them (its own user turn, a retry's turns, the generation prompt) is."""

    def __init__(self, tokenizer, no_think_kwargs: dict):
        self.tokenizer, self.no_think_kwargs = tokenizer, no_think_kwargs
        probes = [self.render({"caption": None, "self_reflect": None, "correction": text, "rewrite_caption": False})
                  for text in ("a", "b")]
        common = os.path.commonprefix(probes)
        # Cut before the special token that opens the job's user turn: tokenization never merges across it.
        self.prefix = common[:max(common.rfind("<|im_start|>"), 0)]
        self.prefix_ids = self.encode(self.prefix)
        logger.info("shared prompt prefix: %d tokens", len(self.prefix_ids))

    def render(self, job: dict) -> str:
        return self.tokenizer.apply_chat_template(build_messages(job), tokenize=False, add_generation_prompt=True,
                                                  **self.no_think_kwargs)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def __call__(self, job: dict) -> dict:
        text = self.render(job)
        if not self.prefix or not text.startswith(self.prefix):
            return {"prompt_token_ids": self.encode(text)}
        return {"prompt_token_ids": self.prefix_ids + self.encode(text[len(self.prefix):])}


def drop_too_long(jobs, pending, cache, cache_path: Path, tokenizer, args) -> list[dict]:
    """Cache the pending jobs over the length cut as "too_long" and return the rest."""
    if not jobs or (args.drop_longest_pct <= 0 and args.max_source_tokens is None):
        return pending
    texts = [_user_message(job["caption"], job["self_reflect"], job["correction"], job["rewrite_caption"]) for job in jobs]
    lengths = {job["key"]: len(ids) for job, ids in zip(jobs, tokenizer(texts)["input_ids"])}
    # The percentile is over all of the rank's jobs, cached or not, so it does not drift as the cache fills up.
    cut = float("inf")
    if args.drop_longest_pct > 0:
        cut = float(np.percentile(list(lengths.values()), 100 - args.drop_longest_pct))
    if args.max_source_tokens is not None:
        cut = min(cut, args.max_source_tokens)
    kept, dropped = [], 0
    with open(cache_path, "a") as handle:
        for job in pending:
            if lengths[job["key"]] <= cut:
                kept.append(job)
                continue
            record = {
                "key": job["key"], "status": TOO_LONG,
                "caption": job["caption"], "self_reflect": job["self_reflect"], "correction": job["correction"],
                "rewrite_caption": job["rewrite_caption"], "result": None, "attempts": 0,
                "raw": None, "reason": f"{lengths[job['key']]} source tokens > cut {cut:.0f}",
            }
            if cache.get(job["key"], {}).get("reason") != record["reason"]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            cache[job["key"]] = record
            dropped += 1
    logger.info("rank %d: length cut %.0f source tokens, %d rows dropped as too long", args.rank, cut, dropped)
    return kept


def run_llm(jobs, cache, cache_path: Path, engine, tokenizer, args) -> None:
    """vLLM drop-in for base.run_llm: generate every uncached job, retry invalid replies, append results to the cache."""
    llm, sampling_params = engine
    llm_engine = llm.llm_engine
    # vLLM forwards chat_template_kwargs to apply_chat_template, so the kwargs the HF probe found work unchanged.
    prompt_tokens = PromptTokens(tokenizer, _resolve_no_think_kwargs(tokenizer))
    # A "too_long" record is re-checked below against the current cut.
    pending = [job for job in jobs if job["key"] not in cache or cache[job["key"]]["status"] == TOO_LONG]
    logger.info("rank %d: %d jobs, %d cached, %d to generate", args.rank, len(jobs), len(jobs) - len(pending), len(pending))
    queue = deque(drop_too_long(jobs, pending, cache, cache_path, tokenizer, args))
    total = len(queue)
    max_in_flight = args.max_in_flight or 2 * args.max_num_seqs
    attempts = {job["key"]: 0 for job in queue}
    in_flight = {}
    started = last_log = last_flush = time.time()
    done = retried = 0
    # One bar per engine of this job, each on its own line (LOCAL_RANK is set by the launcher).
    bar = tqdm(total=total, desc=f"rank {args.rank}", unit="row", position=int(os.environ.get("LOCAL_RANK", 0)),
               dynamic_ncols=True, mininterval=5, smoothing=0.05)
    with bar, open(cache_path, "a") as handle:
        while queue or in_flight:
            # Keep the engine's waiting queue full so a new request takes every freed slot immediately.
            while queue and len(in_flight) < max_in_flight:
                job = queue.popleft()
                request_id = f"{job['key']}-{attempts[job['key']]}"
                llm_engine.add_request(request_id, prompt_tokens(job), sampling_params)
                in_flight[request_id] = job
            for output in llm_engine.step():
                if not output.finished:
                    continue
                job = in_flight.pop(output.request_id)
                reply = output.outputs[0].text
                result, reason = parse_reply(reply, job, args)
                attempts[job["key"]] += 1
                caption_failed = result is not None and job["rewrite_caption"] and result["caption"] is None
                if (result is None or caption_failed) and attempts[job["key"]] <= args.max_retries:
                    job["rejected"] = (reply, reason)
                    # Front of the queue: a retry goes in with the next free slot instead of after every new row.
                    queue.appendleft(job)
                    retried += 1
                    continue
                record = {
                    "key": job["key"], "status": "ok" if result else "failed",
                    "caption": job["caption"], "self_reflect": job["self_reflect"], "correction": job["correction"],
                    "rewrite_caption": job["rewrite_caption"],
                    "result": result, "attempts": attempts[job["key"]], "raw": reply if result is None or caption_failed else None,
                    "reason": reason,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                cache[job["key"]] = record
                done += 1
                bar.update(1)
            bar.set_postfix(retries=retried, in_flight=len(in_flight), refresh=False)
            now = time.time()
            if now - last_flush >= 10:
                handle.flush()
                last_flush = now
            if now - last_log >= args.log_interval or not (queue or in_flight):
                rate = done / max(now - started, 1e-6)
                logger.info("rank %d: %d/%d done, %d queued, %d in flight, %d retries, %.2f rows/s, eta %.1f h",
                            args.rank, done, total, len(queue), len(in_flight), retried, rate,
                            (total - done) / max(rate, 1e-6) / 3600)
                last_log = now


_base_new_conversations = base.new_conversations


def new_conversations(conversations, cache, version, keep_junk: bool):
    """base.new_conversations, except that rows cut as too long are dropped from the output shards."""
    turns, statuses = _base_new_conversations(conversations, cache, version, keep_junk)
    if statuses == ["failed"]:
        _, parts = find_edit_turn(turns)
        key = job_key(version, {"caption": caption_of(turns), "self_reflect": parts["self_reflect"], "correction": parts["correction"]})
        if cache[key]["status"] == TOO_LONG:
            return turns, [f"dropped_{TOO_LONG}"]
    return turns, statuses


def main(args) -> None:
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s rank{args.rank} %(levelname)s %(message)s")
    version = prompt_version(args)
    logger.info("prompt version %s", version)
    engine = tokenizer = None
    if args.stage != "rewrite":
        engine = load_engine(args)
        tokenizer = engine[0].get_tokenizer()
    # base.process_dir calls base.run_llm; point it at the vLLM loop so collection, caching and rewriting stay shared.
    base.run_llm = run_llm
    base.new_conversations = new_conversations
    for input_dir in args.input_dirs:
        base.process_dir(Path(input_dir).expanduser(), engine, tokenizer, args, version)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="vLLM options; all other options are those of decompose_reflection_correction.py.",
        add_help=False,
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="Longest prompt (system + few-shot + retry turn) is ~5k tokens, plus --max_new_tokens.")
    parser.add_argument("--max_num_seqs", type=int, default=None,
                        help="Running batch cap. Default 256 at TP 1 (one A100-80G has 364 linear-attention state slots at "
                             "0.9 memory utilization; vLLM refuses to start above that) and 512 at TP 2+.")
    parser.add_argument("--max_num_batched_tokens", type=int, default=None, help="Per-step token budget (vLLM default if unset).")
    parser.add_argument("--max_in_flight", type=int, default=None,
                        help="Requests handed to the engine at once (default 2 x --max_num_seqs); the rest wait here.")
    parser.add_argument("--num_speculative_tokens", type=int, default=0,
                        help="MTP speculative decoding with this many draft tokens; 0 disables. Helps most at small batch.")
    parser.add_argument("--log_interval", type=float, default=300,
                        help="Seconds between progress log lines (the tqdm bar updates every 5 s).")
    parser.add_argument("--drop_longest_pct", type=float, default=0.0,
                        help="Also drop the longest this-percent rows (source tokens) of each rank; 0 disables.")
    parser.add_argument("--max_source_tokens", type=int, default=512,
                        help="Drop rows whose source text (caption + Self-reflect + Correction) is longer than this "
                             "many tokens; pass a huge value to keep all.")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="vLLM seed; each rank adds its rank.")
    vllm_args, rest = parser.parse_known_args(argv)
    if "-h" in rest or "--help" in rest:
        parser.print_help()
        print()
    args = base.parse_args(rest)
    for name, value in vars(vllm_args).items():
        setattr(args, name, value)
    if args.max_num_seqs is None:
        args.max_num_seqs = 256 if args.tensor_parallel_size == 1 else 512
    return args


if __name__ == "__main__":
    main(parse_args())
