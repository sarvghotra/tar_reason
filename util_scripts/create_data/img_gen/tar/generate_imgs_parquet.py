"""Generate images in one shot from prompts with Tar and write them as parquet.

Same generation path as `generate_imgs.py` next to this file (Qwen2-based Tar
checkpoint -> image tokens -> TA-Tok/LlamaGen decoder), but instead of one PNG
per prompt directory the output is a NEW dataset in the tar training format,
the same layout `../z_image/z_image_img_gen_hf.py` produces.

Input (--prompts_file), either
  * .txt  : one prompt per line (blank lines and '#' comments are ignored), or
  * .jsonl: one JSON object per line, the prompt read from --prompt_key
            (default "prompt").

Output (--output_dir): parquet shards with the tar-format columns

    conversations       [human: prompt, gpt: "<image>"]
    conversations_short copy of conversations
    image               {"bytes": <jpeg/png>, "path": None}
    image_tokens        the --gen_seq_len generated image token ids, as codebook
                        indices (i.e. with the <I0> offset removed)
    prompt              the text this image was rendered from
    prompt_index        index of the prompt in the input file
    sample_index        0..--images_per_prompt-1
    metadata            the input JSON line verbatim (null for .txt input)

plus EVERY column of a .jsonl input, carried through as a string column (values
that are not strings are stored JSON-encoded). A key that collides with a
tar-format column above is prefixed with `input_`; the input's own prompt key is
dropped, since the `prompt` column already holds it verbatim.

Shards are named shard_{NNNNN}.parquet (plus a .rank{NN} suffix under torchrun)
and written with a small row group size, since LazyParquetDataset loads a WHOLE
row group into RAM per DataLoader worker.

Work is split by strided row assignment: rank r renders rows r, r+W, r+2W, ...
of the (prompt x sample) grid. Each rank's slice is cut into fixed --chunk_rows
groups whose shard names depend only on (rank, group index), so a shard that
already exists is skipped and a preempted job resumes by being relaunched.

Everything that is not the GPU is kept off the critical path: prompts are
tokenized in DataLoader workers, JPEG/PNG encoding runs on a thread pool while
the GPU renders the next batch, and parquet writing has its own thread.

Usage (single GPU):
    python util_scripts/create_data/img_gen/tar/generate_imgs_parquet.py \\
        --model <path-or-hf-id> \\
        --prompts_file <path-to-prompts> \\
        --output_dir <out> \\
        --ar_path ... --encoder_path ... --decoder_path ...

Usage (multi-GPU, single node):
    torchrun --nproc_per_node=<N> \\
        util_scripts/create_data/img_gen/tar/generate_imgs_parquet.py ...
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2ForCausalLM

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# LazyParquetDataset loads a WHOLE row group into RAM per DataLoader worker, so
# keep row groups small.
ROW_GROUP_SIZE = 128
CHUNK_ROWS = 500

IMAGE_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
CONVERSATION_TYPE = pa.list_(pa.struct([("from", pa.string()), ("value", pa.string())]))
BASE_FIELDS = [
    ("conversations", CONVERSATION_TYPE),
    ("conversations_short", CONVERSATION_TYPE),
    ("image", IMAGE_TYPE),
    ("image_tokens", pa.list_(pa.int32())),
    ("prompt", pa.string()),
    ("prompt_index", pa.int64()),
    ("sample_index", pa.int32()),
    ("metadata", pa.string()),
]
BASE_COLUMNS = {name for name, _ in BASE_FIELDS}


def build_parser() -> argparse.ArgumentParser:
    """Arguments shared with the vLLM mirror (`generate_imgs_parquet_vllm.py`)."""
    p = argparse.ArgumentParser(
        description=(
            "Render a list of prompts with Tar and write the images out as a "
            "tar-format parquet dataset."
        )
    )
    p.add_argument("--model", type=str, required=True,
                   help="HF model ID or local path (Qwen2-based Tar checkpoint).")
    p.add_argument("--prompts_file", type=str, required=True,
                   help="JSONL (see --prompt_key) or plain-text file of prompts.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--prompt_key", default="prompt",
                   help="Field holding the prompt in a .jsonl input; ignored for .txt.")
    p.add_argument("--images_per_prompt", type=int, default=1,
                   help="Samples per prompt; each gets its own row and its own seed.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=1200)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Prompts to generate in parallel per GPU.")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers for async tokenization.")
    p.add_argument("--device", type=str, default=None,
                   help="Override device (e.g. cuda:1). Defaults to cuda:{LOCAL_RANK}.")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--ar_path", type=str, default=None)
    p.add_argument("--encoder_path", type=str, default=None)
    p.add_argument("--decoder_path", type=str, default=None)
    p.add_argument("--gen_seq_len", type=int, default=729,
                   help="Number of image tokens to generate.")
    p.add_argument("--scale", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    p.add_argument("--chunk_rows", type=int, default=CHUNK_ROWS,
                   help="Rows per output shard file (per rank).")
    p.add_argument("--write_rows", type=int, default=256,
                   help="Rows buffered before a write is queued inside a shard.")
    p.add_argument("--row_group_size", type=int, default=ROW_GROUP_SIZE)
    p.add_argument("--encode_workers", type=int, default=8,
                   help="CPU workers per GPU process for image encoding.")
    p.add_argument("--image_format", default="jpeg", choices=["jpeg", "png"])
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--png_compress_level", type=int, default=1,
                   choices=range(10), metavar="0-9",
                   help="Lower values encode faster but create larger parquet shards.")
    p.add_argument("--samples", type=int, default=None,
                   help="Run on N randomly-chosen prompts only (default: all); for "
                        "smoke tests. Changes which rows land in which shard, so "
                        "point --output_dir somewhere throwaway.")
    p.add_argument("--overwrite", action="store_true",
                   help="Rewrite output shards that already exist (default: skip them, "
                        "which is what makes a preempted large job resumable).")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; every batch derives a deterministic seed from it.")
    return p


def check_args(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("batch_size", "chunk_rows", "write_rows", "row_group_size",
                 "encode_workers", "images_per_prompt", "gen_seq_len"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be at least 1")
    if args.num_workers < 0:
        p.error("--num_workers must not be negative")
    if args.samples is not None and args.samples < 1:
        p.error("--samples must be at least 1")
    if not 1 <= args.jpeg_quality <= 100:
        p.error("--jpeg_quality must be in [1, 100]")


def parse_args() -> argparse.Namespace:
    p = build_parser()
    args = p.parse_args()
    check_args(p, args)
    return args


# ── input ──────────────────────────────────────────────────────────────────────

def read_prompts(path: Path, prompt_key: str) -> tuple[list[dict], list[str]]:
    """(rows, extra column names) for the input file, in file order.

    Every row carries the prompt, the raw JSON line (None for .txt) and the
    input record's own fields, so nothing the prompt file held is lost.
    """
    lines = path.read_text().splitlines()
    prompts: list[dict] = []
    extra_columns: list[str] = []
    seen_extra: set[str] = set()

    if path.suffix == ".jsonl":
        for number, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            prompt = record.get(prompt_key)
            if not prompt or not str(prompt).strip():
                raise ValueError(
                    f"{path}:{number} has no non-empty '{prompt_key}' field; "
                    "pass --prompt_key to point at the right one"
                )
            extra = {}
            for key, value in record.items():
                if key == prompt_key:
                    # Already carried verbatim in the `prompt` column.
                    continue
                column = f"input_{key}" if key in BASE_COLUMNS else key
                if column not in seen_extra:
                    seen_extra.add(column)
                    extra_columns.append(column)
                extra[column] = value if isinstance(value, str) else json.dumps(value)
            prompts.append({"prompt": str(prompt).strip(), "metadata": line, "extra": extra})
    else:
        for line in lines:
            line = line.strip()
            # '#' is a comment only at the start of a line, so prompts that
            # merely contain a '#' survive intact.
            if not line or line.startswith("#"):
                continue
            prompts.append({"prompt": line, "metadata": None, "extra": {}})

    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts, extra_columns


def build_schema(extra_columns: list[str]) -> pa.Schema:
    return pa.schema(BASE_FIELDS + [(name, pa.string()) for name in extra_columns])


def build_rows(prompts: list[dict], images_per_prompt: int) -> list[dict]:
    """The full (prompt x sample) grid, one dict per output row."""
    return [
        {
            "prompt": entry["prompt"],
            "metadata": entry["metadata"],
            "extra": entry["extra"],
            "prompt_index": prompt_index,
            "sample_index": sample_index,
        }
        for prompt_index, entry in enumerate(prompts)
        for sample_index in range(images_per_prompt)
    ]


def make_conversation(prompt: str) -> list[dict]:
    return [{"from": "human", "value": prompt}, {"from": "gpt", "value": "<image>"}]


def build_table(rows: list[dict], images: list[dict], image_tokens: list[list[int]],
                schema: pa.Schema, extra_columns: list[str]) -> pa.Table:
    conversations = [make_conversation(row["prompt"]) for row in rows]
    data = {
        "conversations": conversations,
        "conversations_short": conversations,
        "image": images,
        "image_tokens": image_tokens,
        "prompt": [row["prompt"] for row in rows],
        "prompt_index": [row["prompt_index"] for row in rows],
        "sample_index": [row["sample_index"] for row in rows],
        "metadata": [row["metadata"] for row in rows],
    }
    # A key missing from one input line but present in another becomes null for
    # that row rather than shifting the schema between shards.
    for name in extra_columns:
        data[name] = [row["extra"].get(name) for row in rows]
    return pa.Table.from_pydict(data, schema=schema)


# ── async preprocessing ────────────────────────────────────────────────────────

class RowDataset(Dataset):
    """Tokenizes prompts in DataLoader worker processes (CPU, async).

    Each worker calls __getitem__ independently, so tokenization of batch N+1
    overlaps with GPU generation of batch N.
    """

    def __init__(self, rows: list[dict], tokenizer, scale: int, system_prompt: str):
        self.rows = rows
        self.tokenizer = tokenizer
        self.scale = scale
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        prompt = self.rows[idx]["prompt"]
        text = (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<im_start><S{self.scale}>"
        )
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return idx, torch.tensor(ids, dtype=torch.long)


def collate_rows(batch: list, pad_id: int):
    """Left-pad a batch of variable-length token sequences (main process, cheap)."""
    idxs, id_list = zip(*batch)
    max_len = max(ids.shape[0] for ids in id_list)
    B = len(id_list)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, ids in enumerate(id_list):
        L = ids.shape[0]
        input_ids[i, max_len - L:] = ids
        attn_mask[i, max_len - L:] = 1
    return list(idxs), input_ids, attn_mask


# ── model loading ──────────────────────────────────────────────────────────────

def load_model(args, device):
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"[{device}] Loading LM from {args.model} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen2ForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    return tokenizer, model


def load_visual_tokenizer(args, dtype, device):
    from tok.mm_autoencoder import MMAutoEncoder

    ar_path  = args.ar_path      # or hf_hub_download("csuhan/TA-Tok",         "ar_dtok_lp_256px.pth")
    enc_path = args.encoder_path # or hf_hub_download("csuhan/TA-Tok",         "ta_tok.pth")
    dec_path = args.decoder_path # or hf_hub_download("peizesun/llamagen_t2i", "vq_ds16_t2i.pt")

    print(f"[{device}] Loading visual tokenizer …", flush=True)
    visual_tok = MMAutoEncoder(
        ar_path=ar_path,
        encoder_path=enc_path,
        decoder_path=dec_path,
        encoder_args={"input_type": "rec"},
        decoder_args={},
    ).eval().to(dtype=dtype, device=device)
    visual_tok.ar_model.cls_token_num = args.gen_seq_len
    visual_tok.encoder.pool_scale = args.scale + 1
    return visual_tok


# ── generation ─────────────────────────────────────────────────────────────────

def batch_seed(base_seed: int, rows: list[dict]) -> int:
    """Deterministic per-batch seed: keyed off the batch's first row so a rerun
    of the same shard reproduces the same images, and off the sample index so
    repeats of one prompt differ from each other."""
    row = rows[0]
    digest = hashlib.sha1(row["prompt"].strip().encode()).digest()[:4]
    return (base_seed + int.from_bytes(digest, "big") + row["sample_index"]) % (2**31)


def image_to_bytes(image: Image.Image, image_format: str, quality: int,
                   compress_level: int) -> bytes:
    buffer = BytesIO()
    if image_format == "png":
        image.save(buffer, format="PNG", compress_level=compress_level)
    else:
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def encode_image(img_arr, image_format: str, quality: int, compress_level: int) -> dict:
    """One decoded image (host-side numpy array) in the tar format's image-struct
    form. Runs in a thread; no CUDA call happens here, so the worker never syncs
    against the GPU stream running the next batch."""
    return {
        "bytes": image_to_bytes(Image.fromarray(img_arr), image_format,
                                quality, compress_level),
        "path": None,
    }


@torch.no_grad()
def generate_batch(model, visual_tok, rows: list[dict],
                   prefix_ids: torch.Tensor, prefix_mask: torch.Tensor,
                   image_start_id: int, args, device):
    """Render `rows`; returns (kept rows, host-side image arrays, image token ids)."""
    prefix_ids  = prefix_ids.to(device, non_blocking=True)
    prefix_mask = prefix_mask.to(device, non_blocking=True)

    torch.manual_seed(batch_seed(args.seed, rows))

    D = args.gen_seq_len
    ids = model.generate(
        input_ids=prefix_ids,
        attention_mask=prefix_mask,
        min_new_tokens=D,
        max_new_tokens=D,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        pad_token_id=prefix_ids[0, 0].item(),
        use_cache=True,
    )

    start     = prefix_ids.shape[1]
    image_ids = ids[:, start : start + D] - image_start_id  # (B, D), unshifted
    return decode_batch(visual_tok, rows, image_ids, args, device)


@torch.no_grad()
def decode_batch(visual_tok, rows: list[dict], image_ids: torch.Tensor, args, device):
    """Decode (B, D) unshifted image token ids on `device`; returns (kept rows,
    host-side image arrays, image token ids)."""
    # `model.generate` can sample tokens outside the image-token range (e.g. EOS or
    # text tokens), which makes some `image_ids` negative or >= codebook_size. Those
    # feed an out-of-bounds codebook lookup (F.embedding) inside decode_from_encoder_indices
    # and trigger an *async* CUDA device-side assert that corrupts the whole context —
    # a try/except around decode cannot recover from it. So filter invalid rows up front
    # with a comparison (CUDA-safe, no indexing) before decoding.
    codebook_size = visual_tok.encoder.bottleneck.regularizer.codebook_size
    valid_rows = ((image_ids >= 0) & (image_ids < codebook_size)).all(dim=1)  # (B,)
    if not bool(valid_rows.all()):
        n_bad = int((~valid_rows).sum())
        print(f"[{device}] WARNING: {n_bad}/{image_ids.shape[0]} samples had "
              f"out-of-range image tokens (codebook_size={codebook_size}); dropping them.",
              flush=True)
        keep = valid_rows.nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            return [], [], []
        image_ids = image_ids[keep]
        rows      = [rows[i] for i in keep.tolist()]

    decoded = visual_tok.decode_from_encoder_indices(image_ids, {"cfg_scale": args.cfg_scale})

    # Copy the whole batch to host once, here, instead of once per image inside the
    # encode threads: a single sync, and the GPU buffer is freed immediately rather
    # than being held alive until the last image is compressed.
    decoded = decoded.to("cpu", non_blocking=False).numpy()
    return rows, list(decoded), image_ids.cpu().tolist()


# ── output ─────────────────────────────────────────────────────────────────────

class ShardWriter:
    """Parquet writer driven from a single background thread.

    Compressing a chunk of images is expensive enough to stall the GPU if it
    happens inline, and a ParquetWriter is not thread safe, so every write is
    funnelled through one worker in submission order.
    """

    def __init__(self, path: Path, schema: pa.Schema, row_group_size: int):
        self._path = path
        self._schema = schema
        self._row_group_size = row_group_size
        self._writer: pq.ParquetWriter | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: list[Future] = []
        self._rows = 0

    def submit(self, table: pa.Table) -> None:
        # Surface a failed write at the next submit rather than at close, so a
        # full disk stops the job instead of silently dropping shards.
        self._reap()
        self._pending.append(self._executor.submit(self._write, table))

    def _write(self, table: pa.Table) -> None:
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._path, self._schema)
        self._writer.write_table(table, row_group_size=self._row_group_size)
        self._rows += len(table)

    def _reap(self) -> None:
        still_pending = []
        for future in self._pending:
            if future.done():
                future.result()
            else:
                still_pending.append(future)
        self._pending = still_pending

    def close(self) -> int:
        """Flush every queued write and return the number of rows written."""
        try:
            for future in self._pending:
                future.result()
            self._pending = []
            if self._writer is None:
                # Nothing was generated for this shard: still emit a valid,
                # empty file so the shard counts as done on a rerun.
                pq.write_table(pa.Table.from_batches([], schema=self._schema), self._path)
            else:
                self._writer.close()
                self._writer = None
        finally:
            self._executor.shutdown(wait=True)
        return self._rows

    def abort(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001 - best effort on the error path
                pass
            self._writer = None


def shard_path_for(output_dir: Path, group: int, rank: int, world_size: int) -> Path:
    if world_size == 1:
        return output_dir / f"shard_{group:05d}.parquet"
    return output_dir / f"shard_{group:05d}.rank{rank:02d}.parquet"


def render_shard(model, visual_tok, tokenizer, image_start_id: int,
                 rows: list[dict], destination: Path, schema: pa.Schema,
                 extra_columns: list[str], args, device, progress: tqdm) -> tuple[int, int]:
    """Render one shard's rows and write it. Returns (rows written, rows failed).

    `progress` spans the rank's whole run, not this shard, so its ETA covers
    every shard still to be rendered.
    """
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    writer = ShardWriter(temporary, schema, args.row_group_size)
    buffered: list[pa.Table] = []
    buffered_rows = 0
    failed_rows = 0
    # Encoding of a batch runs on CPU while the GPU starts the next batch, so a
    # batch's rows are only resolved after the following generate call returns.
    pending: tuple[list[dict], list[Future], list[list[int]]] | None = None

    def flush(final: bool = False) -> None:
        nonlocal buffered, buffered_rows
        if not buffered or (buffered_rows < args.write_rows and not final):
            return
        writer.submit(pa.concat_tables(buffered))
        buffered = []
        buffered_rows = 0

    def resolve() -> None:
        nonlocal pending, buffered, buffered_rows
        if pending is None:
            return
        batch_rows, futures, batch_tokens = pending
        pending = None
        buffered.append(build_table(batch_rows, [f.result() for f in futures],
                                    batch_tokens, schema, extra_columns))
        buffered_rows += len(batch_rows)

    dataset = RowDataset(rows, tokenizer, args.scale, args.system_prompt)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=partial(collate_rows, pad_id=tokenizer.pad_token_id),
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=False,
    )

    try:
        with ThreadPoolExecutor(max_workers=args.encode_workers) as encoder:
            progress.set_postfix_str(destination.name, refresh=False)
            for idxs, prefix_ids, prefix_mask in loader:
                batch = [rows[i] for i in idxs]
                attempted = len(batch)
                batch, images, tokens = generate_batch(
                    model, visual_tok, batch, prefix_ids, prefix_mask,
                    image_start_id, args, device,
                )
                progress.update(attempted)
                failed_rows += attempted - len(batch)
                if not batch:
                    continue

                resolve()
                flush()

                pending = (
                    batch,
                    [
                        encoder.submit(
                            encode_image, img_arr, args.image_format,
                            args.jpeg_quality, args.png_compress_level,
                        )
                        for img_arr in images
                    ],
                    tokens,
                )

            resolve()
        flush(final=True)

        rows_written = writer.close()
        os.replace(temporary, destination)
    except BaseException:
        writer.abort()
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise

    return rows_written, failed_rows


# ── planning / reporting (shared with the vLLM mirror) ─────────────────────────

def subsample_prompts(prompts: list[dict], args, rank: int) -> list[dict]:
    if args.samples is None or args.samples >= len(prompts):
        return prompts
    # Seeded, so every rank draws the same subset and the union across ranks
    # is exactly `args.samples` prompts.
    prompts = random.Random(args.seed).sample(prompts, args.samples)
    if rank == 0:
        print(f"Randomly sampled {len(prompts)} prompts (seed={args.seed}).", flush=True)
    return prompts


def get_image_start_id(tokenizer) -> int:
    # One-time check so we don't repeat it per batch.
    image_start_id = tokenizer.convert_tokens_to_ids("<I0>")
    if image_start_id is None or image_start_id == tokenizer.unk_token_id:
        raise RuntimeError("<I0> not in tokenizer — checkpoint must include image vocab.")
    return image_start_id


def plan_shards(prompts: list[dict], extra_columns: list[str], args, rank: int,
                world_size: int, output_dir: Path, device):
    """(all groups, [(group index, rows)] still to render, rows still to render)."""
    # Strided partition of the (prompt x sample) grid, then fixed-size groups:
    # a shard's name depends only on (rank, group), so reruns skip finished ones.
    all_rows = build_rows(prompts, args.images_per_prompt)
    rows = all_rows[rank::world_size]
    groups = [
        rows[start : start + args.chunk_rows]
        for start in range(0, len(rows), args.chunk_rows)
    ]
    # Shards that already exist are resolved BEFORE any rendering starts, so the
    # progress bar's total (and therefore its ETA) covers only the work left.
    todo = [
        (group, group_rows)
        for group, group_rows in enumerate(groups)
        if args.overwrite
        or not shard_path_for(output_dir, group, rank, world_size).exists()
    ]
    todo_rows = sum(len(group_rows) for _, group_rows in todo)

    print(
        f"Rank {rank}/{world_size} on {device}: {len(prompts)} prompts x "
        f"{args.images_per_prompt} = {len(all_rows)} images, {len(rows)} on this "
        f"rank in {len(groups)} shards ({len(groups) - len(todo)} already done, "
        f"{todo_rows} images to render), batch_size={args.batch_size}, "
        f"gen_seq_len={args.gen_seq_len}, scale={args.scale}, "
        f"cfg_scale={args.cfg_scale}, extra columns={extra_columns}",
        flush=True,
    )
    return groups, todo, todo_rows


def report_rank(rank: int, total: int, total_failed: int, elapsed: float,
                output_dir: Path) -> None:
    attempted = total + total_failed
    failure_rate = 100.0 * total_failed / attempted if attempted else 0.0
    print(
        f"Rank {rank} done in {elapsed / 3600:.2f} h. Saved {total} rows to "
        f"{output_dir} ({total / elapsed if elapsed else 0:.2f} img/s). "
        f"Dropped {total_failed}/{attempted} rows on generation failure "
        f"({failure_rate:.2f}%).",
        flush=True,
    )


def report_run(world_size: int, run_total: int, run_failed: int, output_dir: Path) -> None:
    run_attempted = run_total + run_failed
    rate = 100.0 * run_failed / run_attempted if run_attempted else 0.0
    print(
        f"All {world_size} ranks done. Saved {run_total} rows to "
        f"{output_dir}. Dropped {run_failed}/{run_attempted} rows on "
        f"generation failure ({rate:.2f}%).",
        flush=True,
    )


# ── entry point ────────────────────────────────────────────────────────────────

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def generate(args: argparse.Namespace) -> None:
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(args.device if args.device else f"cuda:{local_rank}")

    prompts_file = Path(os.path.expanduser(args.prompts_file))
    output_dir = Path(os.path.expanduser(args.output_dir))

    # Reading the prompt file is pure I/O, so run it on a background thread while
    # the (slow, GPU-bound) model load proceeds.
    input_pool = ThreadPoolExecutor(max_workers=1)
    prompts_future = input_pool.submit(read_prompts, prompts_file, args.prompt_key)

    tokenizer, model = load_model(args, device)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    visual_tok = load_visual_tokenizer(args, dtype, device)

    prompts, extra_columns = prompts_future.result()
    input_pool.shutdown()
    prompts = subsample_prompts(prompts, args, rank)
    schema = build_schema(extra_columns)

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    image_start_id = get_image_start_id(tokenizer)
    groups, todo, todo_rows = plan_shards(prompts, extra_columns, args, rank,
                                          world_size, output_dir, device)

    total = 0
    total_failed = 0
    started = time.monotonic()
    # One bar for the rank's whole run. smoothing=0 makes the rate (and the ETA)
    # a plain average over everything rendered so far. Only rank 0 draws, but
    # every rank prints its per-shard lines through tqdm.write so a drawn bar is
    # never broken up.
    with tqdm(total=todo_rows, unit="img", desc=f"rank {rank}", smoothing=0,
              disable=rank != 0, dynamic_ncols=True) as progress:
        for group, group_rows in todo:
            destination = shard_path_for(output_dir, group, rank, world_size)
            written, failed = render_shard(
                model, visual_tok, tokenizer, image_start_id, group_rows,
                destination, schema, extra_columns, args, device, progress,
            )
            total += written
            total_failed += failed
            elapsed = time.monotonic() - started
            rate = total / elapsed if elapsed > 0 else 0.0
            progress.write(
                f"[rank {rank}: shard {group + 1}/{len(groups)}] {destination.name}: "
                f"{written} rows written, {failed} dropped on generation failure "
                f"({rate:.2f} img/s, {total} total)"
            )

    report_rank(rank, total, total_failed, time.monotonic() - started, output_dir)
    if world_size > 1:
        counters = torch.tensor([total, total_failed], dtype=torch.int64, device=device)
        dist.all_reduce(counters, op=dist.ReduceOp.SUM)
        dist.barrier()
        if rank == 0:
            report_run(world_size, *counters.tolist(), output_dir)
        dist.destroy_process_group()


if __name__ == "__main__":
    generate(parse_args())
