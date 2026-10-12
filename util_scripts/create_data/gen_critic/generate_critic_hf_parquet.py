'''
Relevant input columns:
    - conversations:
        human: <image_description>
        gpt: <image>
    - conversations_short
        human: <image_description_short>
        gpt: <image>
    - images

Infer the VLM model to generate criticism of the image w.r.t the image description. Basically, if the image is not faithful to the image description, write what is wrong and what needs to be edited.

The critique follows the decomposed iterative-refinement data (gpt-edit-simpler_hqedt2_slfref_tar/train, written by
util_scripts/create_data/reduce_gap_w_ood/decompose_reflection_correction_vllm.sh), so that a model trained on both
learns one critique format and executes the Correction steps it writes here the way it learned to in that data:
    - "Self-reflect:" marker, then numbered items, one discrepancy each, in the form "<what the image shows> instead of
      <what the caption asks>", "<object> is missing" or "<object> should not be present".
    - "Correction:" marker, then numbered steps, one editing goal each (one action, add / remove / replace / move /
      change, on one target), imperative, no preservation steps ("keep X unchanged").
The VLM answers in the decompose script's JSON ({"self_reflect": [...], "correction": [...]}), with its rules and
few-shot outputs, and the reply goes through the same step validators (one action per step, no keep steps, no
edit-pair narration). A rejected reply is retried with the reply and the reason added to the conversation; rows that
never validate are dropped.

Output columns:
    - conversations:
        human: <image_descrition>
        gpt: <image>\nSelf-reflect:\n1. {discrepancy}\n2. ...\nCorrection:\n1. {edit step}\n2. ...
             or, when the image is faithful (both lists empty):
             <image>\nSelf-reflect: no issues, it matches the prompt.\nCorrection: looks good.
    - conversations_short:
        human: <image_descrition>
        gpt: same as conversations
    - images

    Keep rest of the input columns in the output

There is no trailing <image>: these rows have no corrected image. The no-issue text is the loader's NO_CORRECTION_SUFFIX
(llava/data/dataset.py) with the single newline that every other critique row uses.
'''

import argparse
import bisect
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

# The decompose script holds the step format: its few-shot outputs are the examples below and its validators check
# every reply, so a change there carries over here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reduce_gap_w_ood"))
from decompose_reflection_correction import (  # noqa: E402
    FEW_SHOT,
    KEEP_STEP,
    _clean_steps,
    has_second_action,
    numbered,
)

DEFAULT_MODEL = "/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/Qwen3.6-27B"
DEFAULT_INPUT_DIR = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/T2I_datasets/geneval2_50K/tar_format"
DEFAULT_OUTPUT_DIR = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/critique/geneval2_50K"

# LazyParquetDataset loads a WHOLE row group into RAM per DataLoader worker, so
# keep the output row groups small (see generate_critic_vllm_paraquet.py).
ROW_GROUP_SIZE = 128

SYSTEM_PROMPT = """You are the critic of a text-to-image model. You are given the image the model generated and the caption it was generated from. Find where the image is not faithful to the caption and write two numbered lists: the discrepancies ("self_reflect") and the edit steps that fix them ("correction"). The model then edits its image by following the correction steps one by one, so every step must be ONE editing goal, one action applied to one object, that it can carry out on its own. A goal is not a sentence: do not split one goal into several steps, and do not put two goals in one step.

What to report:
- Only clear, objective discrepancies with what the caption asks for: missing or extra objects, counts, colors, materials, sizes, shapes, positions and spatial relations, poses and actions, rendered text (spelling and letter case), style, lighting, time of day, viewpoint. Count the objects one by one before you report a count.
- Precision matters more than recall: report a discrepancy only when you are sure of it after looking carefully. Never critique aesthetics or image quality.
- Details the caption does not mention are fine; the model may add them to fit the scene. Do not ask to remove them unless the caption excludes them ("no iron", "without people").

The actions (always name the target object explicitly with its distinguishing attributes, in the caption's words):
- add: "Add <object with all its attributes> <location relative to a named object or the frame>."
- remove: "Remove <object>."
- replace (swap): "Replace <object A> with <object B with all its attributes>."
- move: "Move <object> <relation> <reference object>." (left of, right of, above, below, behind, in front of, on, under, next to, into, out of, center, foreground, background).
- change: modifies a target in place. Attributes ("Change the color of <object> from <old> to <new> and its material to <new>"), count ("Change the number of <objects> from <current> to <target> by adding/removing <k> <objects>"), size ("Change the size of <object> so that it is <larger/smaller> than <reference object>"), pose/action/expression ("Change <subject> so that <subject> is <new pose, action, gaze or expression>"), text ("Change the text on <object> from "<old>" to "<new>""), and image-level properties, where the target is the image, the background, the sky or the scene (style, color palette, lighting, time of day, weather, season, viewpoint).

Rules:
1. Split and merge by goal. Start a new step only when the target object changes or the action changes. Everything to do to the same target with the same action is ONE step, however many attributes it touches: "make the car red and shiny" is one change step; "add a red kite with a long tail" is one add step. Different targets or different actions are separate steps, and each step has exactly one action verb: "replace X, remove Y, add W" is three steps; "remove the lamp and the rug" is two remove steps because they are two objects.
2. Write every correction step as one imperative sentence (under 45 words) that starts with Add, Remove, Replace, Move or Change. Never write a step that keeps, maintains or preserves something.
3. Write every self_reflect item as one discrepancy, grouped like the correction steps (one item per target and kind of problem, however many attributes are wrong), in the form "<what the image shows> instead of <what the caption asks>", "<object with its attributes> is missing", or "<object> should not be present". A self_reflect item describes the image against the caption; it is never an instruction.
4. Every correction step fixes a listed discrepancy, every discrepancy is fixed by a step, and both lists follow the same order. Never write a step for something the image already gets right.
5. Resolve pronouns and vague references ("it", "them", "the second one") to concrete nouns.
6. If the image is faithful to the caption, return empty lists: {"self_reflect": [], "correction": []}.

Respond with one JSON object and nothing else:
{"self_reflect": ["...", "..."], "correction": ["...", "..."]}"""


def _few_shot_examples() -> str:
    """The decompose script's few-shot outputs that have a caption, as caption -> JSON reply examples."""
    blocks = []
    for example_in, example_out in FEW_SHOT:
        caption = example_out.get("caption") or example_in.get("caption")
        if not caption or not example_out.get("self_reflect"):
            continue
        reply = {"self_reflect": example_out["self_reflect"], "correction": example_out["correction"]}
        blocks.append(f"Caption: {caption}\n{json.dumps(reply, ensure_ascii=False)}")
    return "\n\n".join(blocks)


SYSTEM_PROMPT += (
    "\n\nExamples of replies for other captions (their images are not shown here):\n\n" + _few_shot_examples()
)

# Same text as the loader's NO_CORRECTION_SUFFIX (llava/data/dataset.py) apart from its double newline; the loader
# tells a faithful image by "looks good" / "no issues" in the Correction.
NO_ISSUE_CRITIQUE = "<image>\nSelf-reflect: no issues, it matches the prompt.\nCorrection: looks good."
NO_ISSUE_RE = re.compile(r"(?i)\b(?:looks\s+good|no\s+issues)\b")
# First words of 98% of the correction steps in the decomposed hqedt2 train shards.
ACTION_VERBS = ("Add", "Remove", "Replace", "Move", "Change")

# Sampling parameters recommended by the Qwen3.6-27B model card (Best Practices).
# `presence_penalty` has no equivalent in `generate()`, so it is applied through
# the custom logits processor below.
MODE_SAMPLING = {
    "instruct": {
        "temperature": 0.7,
        "top_p": 0.80,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "max_new_tokens": 768,
    },
    "thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 4096,
    },
}


class PresencePenaltyLogitsProcessor:
    """OpenAI/vLLM-style presence penalty, which `generate()` does not implement.

    A flat `penalty` is subtracted from the logit of every token already emitted
    in the *generated* continuation (the prompt is excluded, matching vLLM).
    """

    def __init__(self, penalty: float, prompt_len: int):
        self.penalty = penalty
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        generated = input_ids[:, self.prompt_len:]
        if generated.shape[1] == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, generated, True)
        return scores - self.penalty * seen


DTYPES = {
    "auto": "auto",
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model(model_path: str, dtype: str, gpus_per_replica: int):
    """Load Qwen3.6 (`Qwen3_5ForConditionalGeneration`) with its processor.

    The caller has already pinned this process to its own GPU group through
    `CUDA_VISIBLE_DEVICES`, so a one-GPU replica is placed wholesale on cuda:0
    and only a multi-GPU replica pays for `device_map="auto"` sharding (which
    is pipeline-, not data-parallel: layers split across GPUs, one active at a
    time). Qwen3.6-27B in bf16 needs ~54 GB, so it fits one 80 GB card.
    """
    # The auto class resolves the architecture from config.json, so the same
    # code path also serves the smaller Qwen3.5 checkpoints.
    from transformers import AutoModelForImageTextToText as model_class
    from transformers import AutoProcessor

    device_map = "auto" if gpus_per_replica > 1 else {"": 0}
    model = model_class.from_pretrained(
        model_path, dtype=DTYPES[dtype], device_map=device_map
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)
    # Batched generation needs left padding so every sequence ends at the same
    # position as the generation prompt.
    processor.tokenizer.padding_side = "left"
    return model, processor


def collect_parquet_files(input_dir: str) -> list[Path]:
    input_path = Path(input_dir).expanduser()
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")
    files = sorted(input_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_path}")
    return files


def sample_rows(files: list[Path], num_samples: int, seed: int) -> dict[Path, list[int]]:
    """Pick `num_samples` rows uniformly at random from the whole file list.

    Row counts come from the parquet footers, so this never touches pixel data.
    Files that no sampled row falls into are dropped from the returned mapping.
    """
    counts = [pq.ParquetFile(path).metadata.num_rows for path in files]
    total = sum(counts)
    if num_samples > total:
        raise ValueError(f"Requested {num_samples} samples but only {total} rows are available")

    global_indices = random.Random(seed).sample(range(total), num_samples)

    offsets = []
    running = 0
    for count in counts:
        offsets.append(running)
        running += count

    selected: dict[Path, list[int]] = {}
    for global_index in global_indices:
        file_index = bisect.bisect_right(offsets, global_index) - 1
        selected.setdefault(files[file_index], []).append(global_index - offsets[file_index])
    return {path: sorted(rows) for path, rows in selected.items()}


def read_progress(path: Path) -> dict:
    """Per-shard resume state; a half-written file just restarts the shard."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def write_progress(path: Path, progress: dict) -> None:
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(progress))
    tmp_path.replace(path)


def extract_prompt(conversations) -> str:
    """Return the human turn (the image description) of a conversation list."""
    for turn in conversations:
        if turn.get("from") == "human":
            return turn["value"]
    raise ValueError("No human turn found in conversations")


def decode_image(image_field) -> Image.Image:
    """Decode a HuggingFace Image column value into a PIL RGB image."""
    if isinstance(image_field, dict):
        if image_field.get("bytes") is not None:
            image = Image.open(BytesIO(image_field["bytes"]))
            image.load()  # force decode; the BytesIO does not outlive this call
            return image.convert("RGB")
        if image_field.get("path"):
            with Image.open(image_field["path"]) as image:
                return image.convert("RGB")
    if isinstance(image_field, (bytes, bytearray)):
        image = Image.open(BytesIO(bytes(image_field)))
        image.load()
        return image.convert("RGB")
    raise TypeError(f"Unsupported image column value: {type(image_field)!r}")


def resize_to_max_pixels(image: Image.Image, max_pixels: int | None) -> Image.Image:
    """Cap the pixel count, the single biggest lever on VL throughput."""
    if max_pixels is None:
        return image
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (width * height))
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BICUBIC)


def _text(text: str) -> list[dict]:
    # List content for every turn: the vLLM wrapper walks each message's parts to collect the images.
    return [{"type": "text", "text": text}]


def build_messages(image: Image.Image, description: str) -> list[dict]:
    return [
        {"role": "system", "content": _text(SYSTEM_PROMPT)},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Caption: {description}"},
            ],
        },
    ]


def retry_messages(messages: list[dict], answer: str, reason: str) -> list[dict]:
    """The conversation plus the rejected answer and why it was rejected, as in the decompose script."""
    return messages + [
        {"role": "assistant", "content": _text(answer)},
        {"role": "user", "content": _text(f"That answer was rejected: {reason} Answer again with the full JSON object.")},
    ]


def strip_thinking(text: str) -> str:
    """Drop the `<think>...</think>` block Qwen3.6 emits in thinking mode."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def parse_reply(text: str, max_steps: int) -> tuple[dict | None, str | None]:
    """({"self_reflect": [...], "correction": [...]}, None), or (None, why the reply was rejected).

    Both lists empty means the image is faithful. The reason is sent back to the model on the retry.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None, "The reply has no JSON object."
    try:
        reply = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None, "The reply is not valid JSON."
    if not isinstance(reply, dict):
        return None, "The reply is not a JSON object."
    correction, reason = _clean_steps(reply.get("correction"), "correction")
    if reason:
        return None, reason
    self_reflect, reason = _clean_steps(reply.get("self_reflect"), "self_reflect")
    if reason:
        return None, reason
    if not correction and not self_reflect:
        return {"self_reflect": [], "correction": []}, None
    if not correction or not self_reflect:
        return None, ('"self_reflect" and "correction" must both list the problems, or both be empty when the image '
                      'is faithful to the caption.')
    if len(correction) > max_steps or len(self_reflect) > max_steps:
        return None, (f"There are more than {max_steps} items; merge the ones that act on the same target with the "
                      f"same action.")
    for name, steps in (("self_reflect", self_reflect), ("correction", correction)):
        for step in steps:
            if NO_ISSUE_RE.search(step):
                return None, (f'The {name} item "{step}" says there is no problem. List only problems; when the image '
                              f'is faithful, return empty lists.')
    for number, step in enumerate(self_reflect, 1):
        if step.split(maxsplit=1)[0] in ACTION_VERBS:
            return None, (f'Self_reflect item {number} ("{step}") is an instruction. Describe what the image shows '
                          f'against the caption, and put the edit in "correction".')
    for number, step in enumerate(correction, 1):
        if step.split(maxsplit=1)[0] not in ACTION_VERBS:
            return None, (f'Correction step {number} ("{step}") must start with one of '
                          f'{", ".join(ACTION_VERBS)}.')
        if has_second_action(step):
            return None, (f'Correction step {number} ("{step}") joins two different actions. Split it into one step '
                          f"per action.")
        if KEEP_STEP.search(step):
            return None, (f'Correction step {number} ("{step}") says what should stay the same. Remove that step or '
                          f"clause.")
    return {"self_reflect": self_reflect, "correction": correction}, None


def format_critique(result: dict) -> str:
    if not result["correction"]:
        return NO_ISSUE_CRITIQUE
    return f"<image>\nSelf-reflect:\n{numbered(result['self_reflect'])}\nCorrection:\n{numbered(result['correction'])}"


def critique_batch(batch_messages: list[list[dict]], generate_fn, args, stats: Counter) -> list[dict | None]:
    """Generate, validate and retry rejected replies; one result (None when it never validated) per conversation."""
    results: list[dict | None] = [None] * len(batch_messages)
    pending = list(enumerate(batch_messages))
    for attempt in range(args.max_retries + 1):
        responses = generate_fn([messages for _, messages in pending])
        retry = []
        for (index, messages), response in zip(pending, responses):
            if args.mode == "thinking" and "</think>" not in response:
                # Ran out of tokens while thinking: a fresh sample, not a reply to correct.
                result, reason, messages_next = None, "thinking truncated", messages
            else:
                answer = strip_thinking(response)
                result, reason = parse_reply(answer, args.max_steps)
                messages_next = retry_messages(messages, answer, reason) if result is None else None
            if result is not None:
                results[index] = result
                stats["no_issue" if not result["correction"] else "critique"] += 1
                stats["retried_ok"] += attempt > 0
            elif attempt < args.max_retries:
                retry.append((index, messages_next))
                stats["retries"] += 1
            else:
                # Bucket by the reason's first words; the full text quotes the reply.
                stats["failed: " + " ".join(reason.split()[:4])] += 1
        pending = retry
        if not pending:
            break
    return results


@torch.inference_mode()
def generate_batch(model, processor, batch_messages: list[list[dict]], mode: str,
                   sampling: dict) -> list[str]:
    """Run one batched chat-template generation and return the new text only."""
    from transformers import LogitsProcessorList

    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        # Padding is a processor argument; passing it in **kwargs makes
        # `apply_chat_template` mistake it for a template variable.
        processor_kwargs={"padding": True},
        # Qwen3.6 thinks by default; the template opens and immediately closes
        # `<think>` for us when thinking is disabled.
        enable_thinking=(mode == "thinking"),
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]
    logits_processor = LogitsProcessorList()
    if sampling["presence_penalty"]:
        logits_processor.append(
            PresencePenaltyLogitsProcessor(sampling["presence_penalty"], prompt_len)
        )

    generated_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        top_k=sampling["top_k"],
        min_p=sampling["min_p"],
        repetition_penalty=sampling["repetition_penalty"],
        max_new_tokens=sampling["max_new_tokens"],
        logits_processor=logits_processor,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    return processor.batch_decode(
        generated_ids[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def build_hf_generate_fn(args: argparse.Namespace, sampling: dict, gpu_ids: list[int]):
    """Backend hook: load the HF model and return `messages -> texts`.

    `run_worker` calls one of these builders after pinning the process to its
    GPU group, so the vLLM script can swap in its own without duplicating any
    of the sharding/resume/parquet machinery below.
    """
    model, processor = load_model(args.model, args.dtype, args.gpus_per_replica)

    def generate_fn(batch_messages: list[list[dict]]) -> list[str]:
        return generate_batch(model, processor, batch_messages, args.mode, sampling)

    return generate_fn


def critique_rows(table: pa.Table, generate_fn, args, tag: str,
                  start_row: int, on_chunk) -> tuple[int, int]:
    """Critique rows [start_row, num_rows) and return (num_kept, num_failed).

    Every input column is preserved; only the `gpt` turn of `conversations` and
    `conversations_short` gains the self-reflection and correction, and rows
    whose reply never validated (after --max_retries retries) are dropped.

    Generation is slow enough that a whole shard rarely fits in one SLURM
    allocation, so finished rows are handed to `on_chunk(chunk, consumed)`
    every `--flush_rows` input rows instead of only at the end of the shard.
    `consumed` is the number of input rows processed so far, which is what the
    caller records to resume from. `chunk` is None when the flush interval
    produced no parsable row.
    """
    conversations = table.column("conversations").to_pylist()
    short_conversations = table.column("conversations_short").to_pylist()
    images = table.column("image").to_pylist()

    conversations_type = table.schema.field("conversations").type
    short_type = table.schema.field("conversations_short").type

    keep_indices: list[int] = []
    new_conversations: list[list[dict]] = []
    new_short_conversations: list[list[dict]] = []
    failed = 0
    kept_total = 0
    since_flush = 0
    stats: Counter = Counter()

    def flush(consumed: int) -> None:
        nonlocal keep_indices, new_conversations, new_short_conversations, kept_total
        chunk = None
        if keep_indices:
            chunk = table.take(pa.array(keep_indices, type=pa.int64()))
            chunk = chunk.set_column(
                chunk.schema.get_field_index("conversations"),
                table.schema.field("conversations"),
                pa.array(new_conversations, type=conversations_type),
            )
            chunk = chunk.set_column(
                chunk.schema.get_field_index("conversations_short"),
                table.schema.field("conversations_short"),
                pa.array(new_short_conversations, type=short_type),
            )
            kept_total += chunk.num_rows
            keep_indices = []
            new_conversations = []
            new_short_conversations = []
        on_chunk(chunk, consumed)

    for start in range(start_row, table.num_rows, args.batch_size):
        stop = min(start + args.batch_size, table.num_rows)
        batch_messages = []
        for row in range(start, stop):
            image = resize_to_max_pixels(decode_image(images[row]), args.max_pixels)
            batch_messages.append(build_messages(image, extract_prompt(conversations[row])))

        results = critique_batch(batch_messages, generate_fn, args, stats)

        for row, result in zip(range(start, stop), results):
            if result is None:
                failed += 1
                continue
            critique = format_critique(result)
            keep_indices.append(row)
            new_conversations.append(
                [turn if turn["from"] != "gpt" else {"from": "gpt", "value": critique}
                 for turn in conversations[row]]
            )
            new_short_conversations.append(
                [turn if turn["from"] != "gpt" else {"from": "gpt", "value": critique}
                 for turn in short_conversations[row]]
            )

        since_flush += stop - start
        print(f"{tag}    rows {start}-{stop - 1}: {kept_total + len(keep_indices)} kept, "
              f"{failed} failed; {dict(stats)}", flush=True)

        if since_flush >= args.flush_rows or stop == table.num_rows:
            flush(stop)
            since_flush = 0

    return kept_total, failed


def run_worker(rank: int, world_size: int, gpu_ids: list[int], args: argparse.Namespace,
               sampling: dict, files: list[Path],
               selected_rows: dict[Path, list[int]] | None,
               build_generate_fn=build_hf_generate_fn) -> None:
    """One model replica, pinned to its own GPU group, over its slice of files.

    Replicas are data-parallel: each owns a disjoint set of input shards and
    writes its own output shards, so nothing has to be merged afterwards.
    """
    if gpu_ids:
        # Must happen before the first CUDA call in this process. Under the
        # "spawn" start method the child re-imports this module with CUDA
        # uninitialized, so the assignment takes effect.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpu_ids)

    tag = f"[rank {rank}] " if world_size > 1 else ""
    torch.manual_seed(args.seed + rank)

    files = files[rank::world_size]
    output_dir = Path(args.output_dir).expanduser()
    print(f"{tag}{len(files)} shard(s) on GPU(s) {gpu_ids}; loading {args.model}", flush=True)
    generate_fn = build_generate_fn(args, sampling, gpu_ids)

    total_kept = total_failed = 0
    for index, parquet_path in enumerate(files, start=1):
        stem = parquet_path.stem
        progress_path = output_dir / f"{stem}.rank{rank:02d}.progress.json"
        progress = read_progress(progress_path) if not args.overwrite else {}
        if progress.get("complete"):
            print(f"{tag}[{index}/{len(files)}] {parquet_path.name}: already complete, skipping",
                  flush=True)
            continue
        # Older runs wrote one file per shard under the input name.
        if (output_dir / parquet_path.name).exists() and not args.overwrite:
            print(f"{tag}[{index}/{len(files)}] {parquet_path.name}: output exists, skipping",
                  flush=True)
            continue

        table = pq.read_table(parquet_path)
        if selected_rows is not None:
            table = table.take(pa.array(selected_rows[parquet_path], type=pa.int64()))
        missing = {"conversations", "conversations_short", "image"} - set(table.column_names)
        if missing:
            raise ValueError(f"Missing column(s) {sorted(missing)} in {parquet_path}")

        start_row = int(progress.get("consumed", 0))
        state = {"parts": int(progress.get("parts", 0))}
        print(f"{tag}[{index}/{len(files)}] {parquet_path.name}: {table.num_rows} rows"
              + (f" (resuming at row {start_row})" if start_row else ""), flush=True)

        def on_chunk(chunk, consumed, _stem=stem, _state=state):
            """Persist one flush interval: a self-contained part file, then progress."""
            if chunk is not None:
                part_path = output_dir / f"{_stem}.rank{rank:02d}.part{_state['parts']:05d}.parquet"
                # Write to a temp name first so a kill mid-write never leaves a
                # truncated part that a later run would read as data.
                tmp_path = part_path.with_suffix(".parquet.tmp")
                pq.write_table(chunk, tmp_path, row_group_size=ROW_GROUP_SIZE)
                tmp_path.replace(part_path)
                _state["parts"] += 1
                print(f"{tag}  wrote {part_path.name}: {chunk.num_rows} rows", flush=True)
            # Written after the part, so a crash in between only re-does work.
            write_progress(progress_path,
                           {"consumed": consumed, "parts": _state["parts"],
                            "total_rows": table.num_rows, "complete": consumed >= table.num_rows})

        kept, failed = critique_rows(table, generate_fn, args, tag, start_row, on_chunk)
        total_failed += failed
        total_kept += kept
        if kept == 0:
            print(f"{tag}  no parsable critique in {parquet_path.name}; nothing written",
                  flush=True)

    print(f"{tag}Done. {total_kept} rows written, {total_failed} rows dropped (never validated).",
          flush=True)


def resolve_replicas(args: argparse.Namespace) -> tuple[int, list[list[int]]]:
    """Return (world_size, per-replica GPU id groups)."""
    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No CUDA device found; Qwen3.6-27B cannot be run on CPU here.")
    world_size = num_gpus // args.gpus_per_replica
    if world_size < 1:
        raise ValueError(
            f"--gpus_per_replica {args.gpus_per_replica} exceeds the {num_gpus} available GPU(s)"
        )
    gpu_groups = [
        list(range(rank * args.gpus_per_replica, (rank + 1) * args.gpus_per_replica))
        for rank in range(world_size)
    ]
    return world_size, gpu_groups


def main(args: argparse.Namespace, build_generate_fn=build_hf_generate_fn) -> None:
    sampling = dict(MODE_SAMPLING[args.mode])
    for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                "repetition_penalty", "max_new_tokens"):
        override = getattr(args, key)
        if override is not None:
            sampling[key] = override
    print(f"Mode: {args.mode}; dtype: {args.dtype}; sampling: {sampling}", flush=True)

    # Seeds the row sampling below and the token sampling in `generate()`, so a
    # debugging run with the same --seed is reproducible end to end.
    random.seed(args.seed)

    files = collect_parquet_files(args.input_dir)
    files = files[args.shard_id::args.num_shards]
    if args.limit_files is not None:
        files = files[: args.limit_files]
    print(f"{len(files)} parquet shard(s) assigned to shard {args.shard_id}/{args.num_shards}",
          flush=True)

    selected_rows: dict[Path, list[int]] | None = None
    if args.samples is not None:
        selected_rows = sample_rows(files, args.samples, args.seed)
        files = [path for path in files if path in selected_rows]
        print(f"Sampling {args.samples} random row(s) from {len(files)} shard(s) "
              f"(seed={args.seed})", flush=True)

    Path(args.output_dir).expanduser().mkdir(parents=True, exist_ok=True)

    world_size, gpu_groups = resolve_replicas(args)
    world_size = min(world_size, len(files)) or 1
    print(f"{world_size} data-parallel replica(s) x {args.gpus_per_replica} GPU(s)", flush=True)

    if world_size == 1:
        # Stay in-process: easier to debug, and CUDA_VISIBLE_DEVICES from the
        # job environment is left untouched.
        run_worker(0, 1, [], args, sampling, files, selected_rows, build_generate_fn)
        return

    context = mp.get_context("spawn")
    workers = [
        context.Process(
            target=run_worker,
            args=(rank, world_size, gpu_groups[rank], args, sampling, files, selected_rows,
                  build_generate_fn),
        )
        for rank in range(world_size)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    failed_ranks = [rank for rank, worker in enumerate(workers) if worker.exitcode != 0]
    if failed_ranks:
        raise RuntimeError(f"Replica(s) {failed_ranks} exited with a non-zero status")
    print("All replicas finished.", flush=True)


def parse_args(argv: list[str] | None = None, add_arguments=None,
               description: str = "Generate self-reflection/correction critique data with "
                                  "Qwen3.6 (HF transformers).") -> argparse.Namespace:
    """Shared CLI. `add_arguments(parser)` lets a backend add/redefault flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", type=str.lower, choices=sorted(MODE_SAMPLING),
                        default="instruct",
                        help="Qwen3.6 response mode (case-insensitive): 'Thinking' keeps the "
                             "model's default reasoning pass, 'Instruct' disables it via the "
                             "chat template. Each mode also selects the model card's "
                             "recommended sampling parameters.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Rows generated at once per replica.")
    parser.add_argument("--flush_rows", type=int, default=64,
                        help="Write finished rows to a part file every this many input "
                             "rows, so a job killed at its wall clock keeps its work and "
                             "the next run resumes from there. Rounded up to a batch.")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16",
                        help="Weight/compute precision; 'auto' takes the checkpoint's dtype.")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="GPUs to use (default: every visible CUDA device).")
    parser.add_argument("--gpus_per_replica", type=int, default=1,
                        help="GPUs each model replica is sharded over. 1 gives one "
                             "data-parallel replica per GPU (fastest when the model fits on "
                             "a single card); raise it only when it does not fit.")
    parser.add_argument("--max_pixels", type=int, default=262144,
                        help="Downscale images above this pixel count (0 disables).")
    parser.add_argument("--limit_files", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None,
                        help="Debugging: critique only this many rows, drawn uniformly at "
                             "random from the assigned shards, instead of every row.")
    parser.add_argument("--seed", type=int, default=887,
                        help="Seed for the --samples row draw and for token sampling.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the shard list across this many parallel jobs.")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate shards whose output file already exists.")
    # Sampling overrides; unset values fall back to the model card's per-mode defaults.
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--presence_penalty", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=2,
                        help="Retries of a reply that fails validation; the retry shows the model its "
                             "rejected reply and the reason. Rows still invalid after them are dropped.")
    parser.add_argument("--max_steps", type=int, default=12,
                        help="Reject replies with more self_reflect items or correction steps than this "
                             "(the decomposed hqedt2 data has at most 12).")

    if add_arguments is not None:
        add_arguments(parser)

    args = parser.parse_args(argv)
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard_id must satisfy 0 <= shard_id < num_shards")
    if args.flush_rows < 1:
        parser.error("--flush_rows must be >= 1")
    if args.gpus_per_replica < 1:
        parser.error("--gpus_per_replica must be >= 1")
    if args.num_gpus is not None and args.num_gpus < 1:
        parser.error("--num_gpus must be >= 1")
    if args.max_retries < 0:
        parser.error("--max_retries must be >= 0")
    if args.samples is not None and args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.max_pixels is not None and args.max_pixels <= 0:
        args.max_pixels = None
    return args


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
