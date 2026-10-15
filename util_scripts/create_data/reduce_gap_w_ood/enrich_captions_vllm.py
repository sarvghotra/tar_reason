'''
vLLM version of enrich_captions.py for full-size runs (hqedt ~200k, UnicEdit ~1M rows).

Everything except the VLM backend and the perception batching is imported from enrich_captions.py: the prompts,
few-shot examples, fact derivation and selection, reply parsers and validators, the per-chunk pipeline
(process_jobs), the per-image perception logic (analyze), the record format and the shard rewrite.

What is different here:
    - VLLMClient replaces the HF VLM. It has the same run(jobs, build, parse, max_tokens, stage) contract, but sends
      every request of a stage concurrently to one or more `vllm serve` OpenAI endpoints (--vllm_urls); a reply
      that fails `parse` is re-asked, as in the HF loop. The worker never loads the 27B weights.
    - BatchedPerception runs Grounding DINO and Depth Anything over a whole chunk at once through the prepare_chunk
      hook of enrich_captions.process_jobs, and reuses one image encoding for all SAM 3 nouns of an image.
      analyze() then reads the precomputed outputs; its decisions are unchanged.
    - Stages for job arrays, as in decompose_reflection_correction_vllm.py: `generate` (GPU, appends to the cache)
      and `rewrite` (CPU, all shards from the cache, --rewrite_workers processes); `all` does both per rank.
    - The cache is read slim: generate only needs the finished keys, rewrite only {status, result} per key, so 1M
      cached rows do not load several GB of facts per process.

Launch: enrich_captions_vllm.sh (servers + workers on one node, job arrays across nodes).

Arguments: every option of enrich_captions.py (its --batch_size, --device and --attn_implementation are unused here)
plus the options defined below.
'''

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import io
import json
import logging
import multiprocessing
import os
import queue
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

import enrich_captions as ec
from enrich_captions import logger

# Last field of every record this script writes: a line that ends with it is complete (a killed run can leave a cut
# line), so the generate stage can read the finished keys without parsing the JSON.
END_FIELD = '"_end": 1}'
KEY_PREFIX = '{"key": "'


# ------------------------------------------------------------------------------------------------- the VLM client

def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def to_openai(messages: list[dict], images: list[Image.Image]) -> list[dict]:
    """enrich_captions' messages ({"type": "image"} placeholders + a parallel image list) in OpenAI chat format."""
    images = iter(images)
    out = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            content = [{"type": "image_url", "image_url": {"url": _data_url(next(images))}} if part.get("type") == "image"
                       else part for part in content]
        out.append({"role": message["role"], "content": content})
    return out


def _seed(*parts) -> int:
    return int(hashlib.sha1(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class VLLMClient:
    def __init__(self, args):
        import httpx

        self.args = args
        self.urls = [url.rstrip("/") for url in args.vllm_urls]
        deadline = time.time() + args.server_wait
        for url in self.urls:
            while True:
                try:
                    if httpx.get(f"{url}/health", timeout=10).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.time() > deadline:
                    raise RuntimeError(f"vLLM server {url} not healthy after {args.server_wait} s")
                time.sleep(10)
        self.model = args.served_model_name or httpx.get(f"{self.urls[0]}/v1/models", timeout=30).json()["data"][0]["id"]
        logger.info("vLLM servers %s, model %s", self.urls, self.model)
        self._next = 0

    def run(self, jobs: list[dict], build, parse, max_tokens: int, stage: str) -> dict[int, tuple]:
        """{job index: (parsed or None, raw reply, attempts)}; replies that fail `parse` are retried."""
        if not jobs:
            return {}
        return asyncio.run(self._run(jobs, build, parse, max_tokens, stage))

    async def _run(self, jobs, build, parse, max_tokens, stage):
        import httpx
        from openai import AsyncOpenAI

        limits = httpx.Limits(max_connections=self.args.max_concurrency, max_keepalive_connections=self.args.max_concurrency)
        clients = [AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY", max_retries=0, timeout=self.args.request_timeout,
                               http_client=httpx.AsyncClient(limits=limits, timeout=self.args.request_timeout))
                   for url in self.urls]
        semaphore = asyncio.Semaphore(self.args.max_concurrency)
        try:
            results = await asyncio.gather(*(self._job(clients, semaphore, job, build, parse, max_tokens, stage)
                                             for job in jobs))
        finally:
            for client in clients:
                await client.close()
        return dict(enumerate(results))

    async def _job(self, clients, semaphore, job, build, parse, max_tokens, stage):
        messages, images = build(job)
        messages = await asyncio.to_thread(to_openai, messages, images)  # JPEG encoding off the event loop
        reply, attempts = "", 0
        while attempts <= self.args.max_retries:
            attempts += 1
            reply = await self._request(clients, semaphore, messages, max_tokens,
                                        _seed(self.args.seed, job.get("key"), stage, attempts), stage)
            parsed = parse(reply, job) if reply is not None else None
            if parsed is not None:
                return parsed, reply, attempts
        return None, reply if reply is not None else "", attempts

    async def _request(self, clients, semaphore, messages, max_tokens, seed, stage) -> str | None:
        import openai

        args = self.args
        greedy = args.temperature <= 0
        extra_body = {"top_k": -1 if greedy else args.top_k, "min_p": args.min_p,
                      "repetition_penalty": args.repetition_penalty,
                      "chat_template_kwargs": {"enable_thinking": False}}
        for http_try in range(args.http_retries + 1):
            client = clients[self._next % len(clients)]
            self._next += 1
            try:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=self.model, messages=messages, max_tokens=max_tokens, temperature=max(args.temperature, 0.0),
                        top_p=1.0 if greedy else args.top_p, presence_penalty=args.presence_penalty, seed=seed,
                        extra_body=extra_body)
                return response.choices[0].message.content or ""
            except openai.BadRequestError as error:  # e.g. over max_model_len: asking again will not help
                logger.warning("%s: request rejected: %s", stage, str(error)[:300])
                return None
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError,
                    openai.RateLimitError) as error:
                wait = min(120, 5 * 2 ** http_try)
                logger.warning("%s: %s (try %d/%d), retrying in %d s", stage, type(error).__name__, http_try + 1,
                               args.http_retries + 1, wait)
                await asyncio.sleep(wait)
        return None


# ------------------------------------------------------------------------------------------ batched perception

def _expand(value, n: int):
    """A batch-1 model output field (tensor, or tuple of tensors) viewed as batch n."""
    if isinstance(value, torch.Tensor):
        return value.expand(n, *value.shape[1:]) if value.dim() and value.shape[0] == 1 else value
    if isinstance(value, (tuple, list)):
        return type(value)(_expand(v, n) for v in value)
    return value


class SharedImageBackbone(torch.nn.Module):
    """Grounding DINO's image backbone, run once per run of identical consecutive images in a batch. detect() asks one
    name per query, so a batch holds the same image several times; only the text fusion and the decoder differ."""

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values, **kwargs):
        starts = [0] + [i for i in range(1, len(pixel_values)) if not torch.equal(pixel_values[i], pixel_values[i - 1])]
        if len(starts) == len(pixel_values):
            return self.backbone(pixel_values, **kwargs)
        owner = torch.zeros(len(pixel_values), dtype=torch.long, device=pixel_values.device)
        for k, start in enumerate(starts):
            owner[start:] = k
        output = self.backbone(pixel_values[starts], **kwargs)
        output.feature_maps = tuple(feature.index_select(0, owner) for feature in output.feature_maps)
        return output


class BatchedPerception(ec.Perception):
    """ec.Perception whose model calls read outputs precomputed for the whole chunk (falling back to the per-image
    call when there are none). Precomputed outputs are keyed by id(image); the images are kept referenced until the
    next chunk so that no id is reused meanwhile."""

    def __init__(self, args, device="cuda"):
        super().__init__(args, device)
        encoder = self.detector.model.backbone.conv_encoder
        encoder.model = SharedImageBackbone(encoder.model)
        self._store: dict[tuple, object] = {}
        self._keep: list[Image.Image] = []
        self.timings: Counter = Counter()

    @contextlib.contextmanager
    def _timed(self, name: str):
        started = time.time()
        try:
            yield
        finally:
            self.timings[name] += time.time() - started

    def pop_timings(self) -> str:
        text = ", ".join(f"{name} {seconds:.0f}s" for name, seconds in self.timings.most_common())
        self.timings.clear()
        return text

    # --- chunk-level precompute

    def _in_batches(self, items: list, size_arg: str, run) -> list:
        """run(batch) over `items` in batches of args.<size_arg>; on CUDA OOM the size is halved (for the rest of the
        run too: the other worker and the server on this GPU leave less room than the defaults assume)."""
        out, start = [], 0
        while start < len(items):
            size = getattr(self.args, size_arg)
            batch = items[start:start + size]
            try:
                results = run(batch)
            except torch.OutOfMemoryError:
                if size == 1:
                    raise
                torch.cuda.empty_cache()
                setattr(self.args, size_arg, max(1, size // 2))
                logger.warning("CUDA OOM, %s -> %d", size_arg, getattr(self.args, size_arg))
                continue
            out.extend(results)
            start += len(batch)
        return out

    def prepare_chunk(self, jobs: list[dict]) -> None:
        self._store.clear()
        images = [job["final"] for job in jobs]
        self._keep = list(images)
        if not images:
            return
        # The Grounding DINO queries analyze() will make: the VLM's nouns (negation candidates go to SAM 3 only).
        queries = [[item["name"] for item in job["vocab"]["objects"]] for job in jobs]
        self._detect(images, queries)
        with self._timed("depth"):
            self._depth([image for image, names in zip(images, queries) if names])

    def _detect(self, images: list[Image.Image], queries: list[list[str]]) -> None:
        # Same-size images next to each other: a batch mixing sizes pads the smaller images, which shifts their scores.
        order = sorted(range(len(images)), key=lambda i: images[i].size)
        pairs = [(i, name) for i in order for name in queries[i]]
        if not pairs:
            return
        with self._timed("gdino"):
            found = self.detect_pairs(images, pairs)
        for key in [key for key in self._store if key[0] == "det_pixels"]:
            del self._store[key]
        for image, names, per_name in zip(images, queries, found):
            for name in names:
                self._store["det", id(image), name] = per_name[name]

    def _depth(self, images: list[Image.Image]) -> None:
        by_size: dict[tuple, list[Image.Image]] = {}
        for image in images:
            by_size.setdefault(image.size, []).append(image)

        @torch.inference_mode()
        def run(chunk):
            inputs = self.depth_processor(images=chunk, return_tensors="pt").to(self.device)
            outputs = self.depth_model(**inputs)
            results = self.depth_processor.post_process_depth_estimation(outputs, target_sizes=[chunk[0].size[::-1]] * len(chunk))
            for image, result in zip(chunk, results):
                self._store["depth", id(image)] = self.normalize_depth(result["predicted_depth"])
            return chunk

        for group in by_size.values():
            self._in_batches(group, "depth_batch_size", run)

    # --- the per-image calls analyze() makes

    def det_inputs(self, images, texts):
        """Grounding DINO inputs with each image resized and normalised once, not once per name (the same image
        fills most of a batch). A batch of differently sized images is padded jointly as before."""
        if len({image.size for image in images}) > 1:
            return super().det_inputs(images, texts)
        pixels = []
        for image in images:
            key = ("det_pixels", id(image))
            if key not in self._store:
                self._store[key] = self.det_processor(images=[image], return_tensors="pt")
            pixels.append(self._store[key])
        inputs = self.det_processor(text=texts, return_tensors="pt", padding=True)
        for name in ("pixel_values", "pixel_mask"):
            inputs[name] = torch.cat([p[name] for p in pixels])
        return inputs

    def depth(self, image):
        key = ("depth", id(image))
        if key in self._store:
            return self._store[key]
        with self._timed("depth(miss)"):
            return super().depth(image)

    def detect(self, images, names):
        keys = [[("det", id(image), name) for name in names] for image in images]
        if all(key in self._store for row in keys for key in row):
            return [{name: self._store[key] for name, key in zip(names, row)} for row in keys]
        with self._timed("gdino(miss)"):
            return super().detect(images, names)

    @torch.inference_mode()
    def detect_sam3(self, image, names):
        """ec.Perception.detect_sam3 with the image encoded once for all nouns."""
        with self._timed("sam3"):
            image_inputs = self.sam3_processor(images=image, return_tensors="pt").to(self.device)
            vision = self.sam3.get_vision_features(pixel_values=image_inputs["pixel_values"].to(torch.bfloat16))
            target_sizes = image_inputs["original_sizes"].tolist()
            out = {}

            # Several nouns per pass: the processor pads every text to 32 tokens, so a batch row is the same input as
            # a single noun; the image encoding is expanded (not copied) over the batch.
            def run(chunk):
                text_inputs = self.sam3_processor(text=chunk, return_tensors="pt").to(self.device)
                expanded = type(vision)(**{key: _expand(value, len(chunk)) for key, value in vision.items()})
                outputs = self.sam3(vision_embeds=expanded, input_ids=text_inputs["input_ids"],
                                    attention_mask=text_inputs.get("attention_mask"))
                results = self.sam3_processor.post_process_instance_segmentation(
                    outputs, threshold=self.args.sam3_low, mask_threshold=0.5, target_sizes=target_sizes * len(chunk))
                for name, result in zip(chunk, results):
                    boxes = result["boxes"].float().cpu().tolist()
                    scores = result["scores"].float().cpu().tolist()
                    masks = result["masks"].cpu().numpy().astype(bool) if len(boxes) else []
                    out[name] = [{"box": [round(v, 1) for v in boxes[k]], "score": round(scores[k], 3), "_mask": masks[k]}
                                 for k in ec.clean_boxes(boxes, scores)]
                return chunk

            self._in_batches(names, "sam3_batch_size", run)
            return out

    def attributes(self, image, noun, instances, masks):
        with self._timed("siglip"):
            return super().attributes(image, noun, instances, masks)


# ------------------------------------------------------------------------------------------------- cache

def _repair_tail(path: Path) -> None:
    """End a line a killed run cut short, so the next record does not get glued onto it."""
    if path.exists() and path.stat().st_size:
        with open(path, "rb+") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")


def load_done_keys(cache_dir: Path, version: str) -> set[str]:
    keys = set()
    for path in sorted(cache_dir.glob(f"{version}_rank*.jsonl")):
        with open(path) as handle:
            for line in handle:
                line = line.rstrip("\n")
                if line.startswith(KEY_PREFIX) and line.endswith(END_FIELD):
                    keys.add(line[len(KEY_PREFIX):line.index('"', len(KEY_PREFIX))])
    return keys


def _slim_records(path: Path) -> dict[str, dict]:
    out = {}
    with open(path) as handle:
        for line in handle:
            if not line.rstrip("\n").endswith(END_FIELD):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[record["key"]] = {"status": record["status"], "result": record.get("result")}
    return out


def load_slim_cache(cache_dir: Path, version: str, workers: int) -> dict[str, dict]:
    """key -> {status, result}: all that ec.rewrite_file reads."""
    paths = sorted(cache_dir.glob(f"{version}_rank*.jsonl"))
    cache = {}
    with ProcessPoolExecutor(max(1, min(workers, len(paths) or 1))) as pool:
        for part in pool.map(_slim_records, paths):
            cache.update(part)
    return cache


# ------------------------------------------------------------------------------------------------- generate

def count_rank_rows(files: list[Path], args) -> int:
    """Rows in this rank's row groups, from the parquet footers only (for the ETA)."""
    total = global_group = 0
    for path in files:
        metadata = pq.ParquetFile(path).metadata
        for group in range(metadata.num_row_groups):
            if global_group % args.world_size == args.rank:
                total += metadata.row_group(group).num_rows
            global_group += 1
    return total


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def iter_rank_jobs(files: list[Path], version: str, args):
    """(file name, row group, rows read, jobs) for this rank's row groups: global row group g goes to rank g % world_size."""
    global_group = -1
    for path in files:
        parquet_file = pq.ParquetFile(path)
        columns = ec.conv_columns(parquet_file, args.v1_prefix)
        read = [c for _, c in columns] + [n for n in (ec.IMAGE_COLUMN, ec.ID_COLUMN) if n in parquet_file.schema_arrow.names]
        for group in range(parquet_file.metadata.num_row_groups):
            global_group += 1
            if global_group % args.world_size != args.rank:
                continue
            rows = parquet_file.read_row_group(group, columns=read).to_pylist()
            jobs = [job for job in (ec.make_job(row, columns, version, args) for row in rows) if job is not None]
            if args.ids:
                jobs = [job for job in jobs if job["id"] in args.ids]
            yield path.name, group, len(rows), jobs


def prefetched(iterator, depth: int = 2):
    """Run `iterator` in a thread, `depth` items ahead (parquet reads overlap the GPU work)."""
    items: queue.Queue = queue.Queue(maxsize=depth)
    done = object()

    def worker():
        try:
            for item in iterator:
                items.put(item)
        except BaseException as error:  # re-raised in the consumer
            items.put(error)
        items.put(done)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = items.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def generate_dir(files, done_keys: set[str], cache_path: Path, vlm, perception, args, version) -> tuple[Counter, list[dict]]:
    """Enrich every unfinished row of this rank; returns status counts (and the records under --dry_run)."""
    status, kept = Counter(), []
    seen = cached = enriched = read = 0
    started = time.time()
    _repair_tail(cache_path)
    # Rows, not jobs: rows make_job skips count as done. Under --limit the total is the limit (in jobs).
    total = count_rank_rows(files, args)
    if args.limit is not None:
        total = min(total, args.limit)

    def flush(pending, handle):
        nonlocal enriched
        chunk_started = time.time()
        records = ec.process_jobs(pending, vlm, perception, args)
        for record in records:
            record["_end"] = 1
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            done_keys.add(record["key"])
            status[record["status"]] += 1
            if args.dry_run:
                kept.append(record)
        handle.flush()
        enriched += len(records)
        timings = perception.pop_timings() if isinstance(perception, BatchedPerception) else ""
        elapsed = time.time() - started
        # Done = rows read minus those still pending after this chunk. The ETA assumes the rest is uncached (resumed
        # runs hit the cache first), so cached rows are left out of the rate.
        done = min(total, (seen if args.limit is not None else read) - (len(pending) - len(records)))
        rate = max(done - cached, 1) / max(elapsed, 1e-6)
        logger.info("rank %d: %d rows in %.0f s; %d/%d (%.1f%%) done, %d enriched, %d cached, %.3f rows/s, "
                    "elapsed %s, ETA %s%s", args.rank, len(records), time.time() - chunk_started, done, total,
                    100 * done / max(total, 1), enriched, cached, enriched / max(elapsed, 1e-6), _hms(elapsed),
                    _hms((total - done) / rate), f" | perception: {timings}" if timings else "")

    pending = []
    with open(cache_path, "a") as handle:
        for _, _, num_rows, jobs in prefetched(iter_rank_jobs(files, version, args)):
            read += num_rows
            if args.limit is not None:
                jobs = jobs[:max(0, args.limit - seen)]
            seen += len(jobs)
            for job in jobs:
                if job["key"] in done_keys:
                    cached += 1
                    status["cached"] += 1
                else:
                    pending.append(job)
            while len(pending) >= args.chunk_size:
                flush(pending[:args.chunk_size], handle)
                pending = pending[args.chunk_size:]
            if args.limit is not None and seen >= args.limit:
                break
        if pending:
            flush(pending, handle)
    return status, kept


# ------------------------------------------------------------------------------------------------- rewrite

_REWRITE = {}


def _rewrite_one(task):
    path, output_path = task
    state = _REWRITE
    return path.name, ec.rewrite_file(path, output_path, state["cache"], state["version"], state["args"])


def rewrite_files(tasks: list[tuple[Path, Path]], cache: dict, version: str, args) -> None:
    _REWRITE.update(cache=cache, version=version, args=args)
    total = Counter()
    if args.rewrite_workers > 1 and len(tasks) > 1:
        # fork: the workers share the (read-only) cache instead of pickling it
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(args.rewrite_workers, mp_context=context) as pool:
            results = list(pool.map(_rewrite_one, tasks))
    else:
        results = [_rewrite_one(task) for task in tasks]
    for name, stats in results:
        logger.info("%s: %s", name, stats)
        total.update(stats)
    logger.info("rewrote %d shards: %s", len(tasks), dict(total))


# ----------------------------------------------------------------------------------------------------------- main

def process_dir(input_dir: Path, vlm, perception, args, version: str) -> None:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {input_dir}")
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else input_dir / ".enrich_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir).expanduser() / input_dir.name if args.output_dir else input_dir

    if args.stage in ("all", "generate"):
        # A marker left by an earlier run must not tell the other ranks this one is done before it has generated.
        (cache_dir / f"{version}_rank{args.rank}.done").unlink(missing_ok=True)
        done_keys = load_done_keys(cache_dir, version)
        logger.info("rank %d, %s: %d rows already in the cache", args.rank, input_dir.name, len(done_keys))
        status, records = generate_dir(files, done_keys, cache_dir / f"{version}_rank{args.rank}.jsonl", vlm,
                                       perception, args, version)
        logger.info("rank %d, %s: %s", args.rank, input_dir.name, dict(status))
        if args.dry_run:
            for record in records:
                ec.print_record(record)
            return
        if args.stage == "generate":
            (cache_dir / f"{version}_rank{args.rank}.done").touch()
            return
        if args.world_size > 1:
            ec.wait_for_ranks(cache_dir, version, args)
        tasks = [(path, output_dir / path.name) for index, path in enumerate(files) if index % args.world_size == args.rank]
    else:
        tasks = [(path, output_dir / path.name) for path in files]
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_slim_cache(cache_dir, version, args.rewrite_workers)
    logger.info("%s: %d cached rows, rewriting %d shards into %s", input_dir.name, len(cache), len(tasks), output_dir)
    rewrite_files(tasks, cache, version, args)


def prompt_version(args) -> str:
    """ec.prompt_version plus the backend, so HF and vLLM caches never mix."""
    return hashlib.sha1(f"{ec.prompt_version(args)}:vllm".encode()).hexdigest()[:12]


def main(args) -> None:
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s rank{args.rank} %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    version = prompt_version(args)
    logger.info("prompt version %s, stage %s, rank %d/%d", version, args.stage, args.rank, args.world_size)
    vlm = perception = None
    if args.stage != "rewrite":
        vlm = VLLMClient(args)
        perception = BatchedPerception(args) if args.batch_perception else ec.Perception(args)
    for input_dir in args.input_dirs:
        process_dir(Path(input_dir).expanduser(), vlm, perception, args, version)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="vLLM options; all other options are those of enrich_captions.py.",
                                     add_help=False)
    parser.add_argument("--vllm_urls", nargs="+", default=["http://localhost:8100"],
                        help="`vllm serve` endpoints; requests are spread round-robin over them.")
    parser.add_argument("--served_model_name", default=None, help="Default: the first model the first server lists.")
    parser.add_argument("--max_concurrency", type=int, default=64, help="Requests in flight per worker process.")
    parser.add_argument("--request_timeout", type=float, default=900)
    parser.add_argument("--http_retries", type=int, default=6, help="Retries of a request the server failed (with backoff).")
    parser.add_argument("--server_wait", type=float, default=3600, help="Seconds to wait for the servers to be healthy.")
    parser.add_argument("--stage", choices=("all", "generate", "rewrite"), default="all")
    parser.add_argument("--rewrite_workers", type=int, default=8)
    parser.add_argument("--no_batch_perception", dest="batch_perception", action="store_false",
                        help="Per-image perception calls exactly as in enrich_captions.py (for comparisons).")
    parser.add_argument("--depth_batch_size", type=int, default=8,
                        help="Images per Depth Anything batch. This and the SAM 3 batch size halve on CUDA OOM.")
    parser.add_argument("--sam3_batch_size", type=int, default=16, help="Nouns per SAM 3 pass over one image encoding.")
    vllm_args, rest = parser.parse_known_args(argv)
    if "-h" in rest or "--help" in rest:
        parser.print_help()
        print()
    args = ec.parse_args(rest)
    for name, value in vars(vllm_args).items():
        setattr(args, name, value)
    if args.dry_run and args.stage != "all":
        raise SystemExit("--dry_run needs --stage all")
    return args


if __name__ == "__main__":
    main(parse_args())
