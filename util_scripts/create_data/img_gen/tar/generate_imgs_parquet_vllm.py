"""vLLM mirror of `generate_imgs_parquet.py`: same inputs, same output dataset.

The image-token LM runs under vLLM instead of HF `generate`; everything else
(prompt template, TA-Tok/LlamaGen decoding, parquet schema, shard naming,
resume-by-skipping-existing-shards, --samples, per-row seeding) is imported from
`generate_imgs_parquet.py`, so the two scripts write interchangeable shards. With
the same --chunk_rows and the same number of ranks (= GPU groups here, torchrun
processes there) a shard left unfinished by one can be finished by the other.

Differences from the HF script:
  * Multi-GPU is NOT torchrun. One process per GPU group is spawned from this
    script (--num_gpus / --tensor_parallel_size), each with its own vLLM engine
    and its own visual decoder on the group's first GPU.
  * --batch_size is the number of rows handed to one `llm.generate` call (vLLM's
    scheduler batches inside it, up to --max_num_seqs); the visual decoder runs
    in chunks of --decode_batch_size.
  * Sampling seeds are per row (from the row's prompt + sample index) rather
    than per batch, so an image does not depend on which batch it landed in.
  * --repetition_penalty defaults to the checkpoint's generation_config, which is
    what HF `generate` applies implicitly.
  * --restrict_to_image_tokens (off by default) masks sampling to the image
    codebook, so no row is ever dropped for out-of-range tokens.

The pipeline is three-stage and runs across shard boundaries: while vLLM
generates batch N, a thread decodes batch N-1 into pixels on the GPU, and a
thread pool JPEG/PNG-encodes batch N-2.

Usage:
    python util_scripts/create_data/img_gen/tar/generate_imgs_parquet_vllm.py \\
        --model <path-or-hf-id> \\
        --prompts_file <path-to-prompts> \\
        --output_dir <out> \\
        --ar_path ... --encoder_path ... --decoder_path ... \\
        --num_gpus 4
"""

import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate_imgs_parquet as base  # noqa: E402  (also puts REPO_ROOT on sys.path)
import pyarrow as pa  # noqa: E402
from tqdm import tqdm  # noqa: E402


def parse_args():
    p = base.build_parser()
    p.description = ("vLLM version: render a list of prompts with Tar and write the "
                     "images out as a tar-format parquet dataset.")
    p.set_defaults(batch_size=256)
    p.add_argument("--decode_batch_size", type=int, default=32,
                   help="Images per visual-decoder call.")
    p.add_argument("--num_gpus", type=int, default=0,
                   help="GPUs to use (default: all visible).")
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="GPUs per vLLM replica; replicas = num_gpus / this.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.6,
                   help="vLLM's share of GPU memory; the rest is left for the "
                        "visual decoder that runs on the same GPU.")
    p.add_argument("--max_num_seqs", type=int, default=256)
    p.add_argument("--max_model_len", type=int, default=4096,
                   help="Prompt + --gen_seq_len budget; longer prompts are dropped.")
    p.add_argument("--enforce_eager", action="store_true",
                   help="Disable CUDA graph capture in vLLM.")
    p.add_argument("--repetition_penalty", type=float, default=None,
                   help="Default: the checkpoint's generation_config value (as HF "
                        "generate uses), else 1.0.")
    p.add_argument("--restrict_to_image_tokens", action="store_true",
                   help="Only sample image-codebook tokens (no out-of-range drops).")
    args = p.parse_args()
    base.check_args(p, args)
    if args.decode_batch_size < 1 or args.tensor_parallel_size < 1:
        p.error("--decode_batch_size and --tensor_parallel_size must be at least 1")
    return args


# ── per-shard state ────────────────────────────────────────────────────────────

@dataclass
class Shard:
    group: int
    destination: Path
    writer: base.ShardWriter
    temporary: Path
    buffered: list = field(default_factory=list)
    buffered_rows: int = 0
    failed_rows: int = 0

    def add(self, table: pa.Table | None, write_rows: int, final: bool) -> None:
        if table is not None and len(table):
            self.buffered.append(table)
            self.buffered_rows += len(table)
        if self.buffered and (self.buffered_rows >= write_rows or final):
            self.writer.submit(pa.concat_tables(self.buffered))
            self.buffered = []
            self.buffered_rows = 0

    def finish(self) -> int:
        rows_written = self.writer.close()
        os.replace(self.temporary, self.destination)
        return rows_written

    def abort(self) -> None:
        self.writer.abort()
        try:
            self.temporary.unlink()
        except FileNotFoundError:
            pass


# ── worker ─────────────────────────────────────────────────────────────────────

def run_worker(rank: int, world_size: int, gpu_ids: list[int], args, results) -> None:
    """Process entry point. vLLM's engine subprocess is non-daemonic and its
    shutdown hook is atexit-based, which never runs in a multiprocessing child:
    left alone, the child would hang joining the engine at exit (and, on error,
    before multiprocessing even prints the traceback). So report the error
    here, stop the engine explicitly and hard-exit."""
    holder: dict = {}
    exit_code = 0
    try:
        render(rank, world_size, gpu_ids, args, results, holder)
    except BaseException:  # noqa: BLE001 - report everything, then exit non-zero
        traceback.print_exc()
        exit_code = 1
    finally:
        llm = holder.get("llm")
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception:  # noqa: BLE001 - best effort on the way out
                traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        results.close()
        results.join_thread()
        os._exit(exit_code)


def render(rank: int, world_size: int, gpu_ids: list[int], args, results,
           holder: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    # Heavy imports after CUDA_VISIBLE_DEVICES is set.
    import torch
    from transformers import AutoTokenizer, GenerationConfig
    from vllm import LLM, ModelRegistry, SamplingParams
    from vllm.inputs import TokensPrompt

    # Registered lazily by "module:Class" so vLLM's subprocesses (including the
    # plain `python -m` one that inspects the architecture) can import it, which
    # needs this file's directory on PYTHONPATH, not just sys.path.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (script_dir, os.environ.get("PYTHONPATH")) if p)
    ModelRegistry.register_model("TarQwen2ForCausalLM",
                                 "tar_vllm_model:TarQwen2ForCausalLM")

    device = torch.device("cuda:0")
    prompts_file = Path(os.path.expanduser(args.prompts_file))
    output_dir = Path(os.path.expanduser(args.output_dir))

    input_pool = ThreadPoolExecutor(max_workers=1)
    prompts_future = input_pool.submit(base.read_prompts, prompts_file, args.prompt_key)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_start_id = base.get_image_start_id(tokenizer)

    repetition_penalty = args.repetition_penalty
    if repetition_penalty is None:
        try:
            repetition_penalty = GenerationConfig.from_pretrained(args.model).repetition_penalty
        except OSError:
            repetition_penalty = None
        repetition_penalty = repetition_penalty or 1.0

    print(f"[rank {rank} gpus {gpu_ids}] Loading LM from {args.model} into vLLM "
          f"(tp={len(gpu_ids)}, repetition_penalty={repetition_penalty}) …", flush=True)
    # vLLM must be created before anything else allocates on the GPU: it checks
    # that its --gpu_memory_utilization share is free at startup.
    llm = LLM(
        model=args.model,
        # Tar checkpoints are saved as LlavaQwenForCausalLM; generation only
        # needs the Qwen2 LM (the bundled vision tower is unused, as with the
        # HF script's Qwen2ForCausalLM.from_pretrained). See tar_vllm_model.py.
        hf_overrides={"architectures": ["TarQwen2ForCausalLM"]},
        dtype="bfloat16" if args.bf16 else "float32",
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=True,  # shared system-prompt prefix
        disable_log_stats=True,
        seed=args.seed,
    )
    holder["llm"] = llm
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    # The TA-Tok / AR checkpoints pickle an EasyDict config, which torch>=2.6's
    # default weights_only load rejects (allowlisting does not help: its
    # unpickler cannot SETITEMS on an EasyDict). Restore the full pickle load
    # the `tar` env's torch 2.1 does, for these trusted files only.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        visual_tok = base.load_visual_tokenizer(args, dtype, device)
    finally:
        del os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]
    codebook_size = visual_tok.encoder.bottleneck.regularizer.codebook_size
    allowed_token_ids = (list(range(image_start_id, image_start_id + codebook_size))
                         if args.restrict_to_image_tokens else None)

    prompts, extra_columns = prompts_future.result()
    input_pool.shutdown()
    prompts = base.subsample_prompts(prompts, args, rank)
    schema = base.build_schema(extra_columns)
    groups, todo, todo_rows = base.plan_shards(prompts, extra_columns, args, rank,
                                               world_size, output_dir, device)
    D = args.gen_seq_len

    def sampling_params(row: dict) -> SamplingParams:
        return SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=repetition_penalty,
            seed=base.batch_seed(args.seed, [row]),
            min_tokens=D,
            max_tokens=D,
            ignore_eos=True,
            detokenize=False,
            allowed_token_ids=allowed_token_ids,
        )

    def generate(batch: list[dict]) -> tuple[list[dict], list[list[int]]]:
        """vLLM stage: (rows that produced exactly D tokens, their raw token ids)."""
        dataset = base.RowDataset(batch, tokenizer, args.scale, args.system_prompt)
        requests, kept = [], []
        for i, row in enumerate(batch):
            ids = dataset[i][1].tolist()
            if len(ids) + D > args.max_model_len:
                print(f"[rank {rank}] WARNING: prompt {row['prompt_index']} is "
                      f"{len(ids)} tokens, over --max_model_len - gen_seq_len; dropping.",
                      flush=True)
                continue
            requests.append(TokensPrompt(prompt_token_ids=ids))
            kept.append(row)
        if not requests:
            return [], []
        outputs = llm.generate(requests, [sampling_params(r) for r in kept], use_tqdm=False)
        rows, tokens = [], []
        for row, output in zip(kept, outputs):
            token_ids = list(output.outputs[0].token_ids)
            if len(token_ids) == D:
                rows.append(row)
                tokens.append(token_ids)
        return rows, tokens

    def decode(batch: list[dict], tokens: list[list[int]]):
        """Decoder-thread stage: pixels on GPU, then queue CPU image encoding.
        Returns (kept rows, encode futures, image token ids, rows dropped)."""
        rows, arrays, image_tokens = [], [], []
        with torch.inference_mode():
            for start in range(0, len(batch), args.decode_batch_size):
                chunk = batch[start : start + args.decode_batch_size]
                # Seeds the decoder's AR sampling, as the HF script's per-batch
                # torch.manual_seed does.
                torch.manual_seed(base.batch_seed(args.seed, chunk))
                image_ids = torch.tensor(tokens[start : start + args.decode_batch_size],
                                         dtype=torch.long, device=device) - image_start_id
                r, a, t = base.decode_batch(visual_tok, chunk, image_ids, args, device)
                rows += r
                arrays += a
                image_tokens += t
        futures = [
            encoder.submit(base.encode_image, arr, args.image_format,
                           args.jpeg_quality, args.png_compress_level)
            for arr in arrays
        ]
        return rows, futures, image_tokens, len(batch) - len(rows)

    # Every (shard, batch) in order, so the pipeline keeps flowing across shard
    # boundaries; a shard is finalized when its last batch clears the last stage.
    work = []
    for group, group_rows in todo:
        starts = range(0, len(group_rows), args.batch_size)
        for start in starts:
            work.append((group, group_rows[start : start + args.batch_size],
                         start == starts[-1]))

    shards: dict[int, Shard] = {}
    total = 0
    total_failed = 0
    started = time.monotonic()

    def open_shard(group: int) -> Shard:
        destination = base.shard_path_for(output_dir, group, rank, world_size)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        return Shard(group, destination, base.ShardWriter(temporary, schema,
                                                          args.row_group_size), temporary)

    def finish_stage(item) -> None:
        """Last stage: wait for encoding, buffer/write, finalize a done shard."""
        nonlocal total, total_failed
        group, is_last, (rows, futures, image_tokens, dropped) = item
        shard = shards[group]
        shard.failed_rows += dropped
        table = (base.build_table(rows, [f.result() for f in futures], image_tokens,
                                  schema, extra_columns) if rows else None)
        shard.add(table, args.write_rows, final=is_last)
        if not is_last:
            return
        written = shard.finish()
        del shards[group]
        total += written
        total_failed += shard.failed_rows
        elapsed = time.monotonic() - started
        progress.write(
            f"[rank {rank}: shard {group + 1}/{len(groups)}] {shard.destination.name}: "
            f"{written} rows written, {shard.failed_rows} dropped on generation failure "
            f"({total / elapsed if elapsed > 0 else 0.0:.2f} img/s, {total} total)"
        )

    decoder = ThreadPoolExecutor(max_workers=1)
    encoder = ThreadPoolExecutor(max_workers=args.encode_workers)
    # (group, is_last, Future) waiting on the decoder; (group, is_last, result)
    # whose image encodes are in flight.
    decoding: tuple[int, bool, Future] | None = None
    encoding: tuple | None = None
    try:
        with tqdm(total=todo_rows, unit="img", desc=f"rank {rank}", smoothing=0,
                  disable=rank != 0, dynamic_ncols=True) as progress:
            for group, batch, is_last in work:
                if group not in shards:
                    shards[group] = open_shard(group)
                    progress.set_postfix_str(shards[group].destination.name, refresh=False)
                rows, tokens = generate(batch)
                shards[group].failed_rows += len(batch) - len(rows)
                progress.update(len(batch))

                decoded = None
                if decoding is not None:
                    g, last, future = decoding
                    decoded = (g, last, future.result())
                decoding = (group, is_last, decoder.submit(decode, rows, tokens))
                if encoding is not None:
                    finish_stage(encoding)
                encoding = decoded

            if decoding is not None:
                g, last, future = decoding
                decoded = (g, last, future.result())
                if encoding is not None:
                    finish_stage(encoding)
                encoding = decoded
            if encoding is not None:
                finish_stage(encoding)
    except BaseException:
        for shard in shards.values():
            shard.abort()
        raise
    finally:
        decoder.shutdown(wait=False, cancel_futures=True)
        encoder.shutdown(wait=False, cancel_futures=True)

    base.report_rank(rank, total, total_failed, time.monotonic() - started, output_dir)
    results.put((rank, total, total_failed))


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    import torch
    total_gpus = args.num_gpus or torch.cuda.device_count()
    if total_gpus < 1:
        raise RuntimeError("No GPUs visible")
    if total_gpus % args.tensor_parallel_size != 0:
        raise ValueError(f"--num_gpus ({total_gpus}) must be divisible by "
                         f"--tensor_parallel_size ({args.tensor_parallel_size})")
    # Children overwrite CUDA_VISIBLE_DEVICES, so map indices through the
    # parent's own list (e.g. SLURM's "2,3") rather than assuming 0..N-1.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = ([d.strip() for d in visible.split(",") if d.strip()] if visible
                else [str(i) for i in range(total_gpus)])
    if total_gpus > len(physical):
        raise ValueError(f"--num_gpus ({total_gpus}) exceeds visible GPUs ({len(physical)})")
    gpu_groups = [physical[start : start + args.tensor_parallel_size]
                  for start in range(0, total_gpus, args.tensor_parallel_size)]
    world_size = len(gpu_groups)

    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # A fresh (spawned, non-daemon) process per replica: vLLM starts its own
    # subprocesses, and each replica needs CUDA_VISIBLE_DEVICES set before CUDA
    # is touched.
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    procs = [ctx.Process(target=run_worker, args=(rank, world_size, gpus, args, results))
             for rank, gpus in enumerate(gpu_groups)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    counts = []
    while not results.empty():
        counts.append(results.get())
    failed_ranks = [rank for rank, proc in enumerate(procs) if proc.exitcode != 0]
    base.report_run(world_size, sum(c[1] for c in counts), sum(c[2] for c in counts),
                    output_dir)
    if failed_ranks:
        print(f"Ranks {failed_ranks} exited with an error; their finished shards "
              "are kept, relaunch to resume the rest.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
