'''
Script to edit an iterative data (caption, image1, self-reflection and correction instructions, image2) such that the edited caption is no longer faithful to the image2.
Training on the original data makes the image generation model to bypass the interim instructions and use the caption for generating image2.

Input:
    conversations:
        human: {caption}
        gpt: <img1> {self-reflection and correction instruction} <img2>

errored_caption <- introduce_errors(caption)

Output:
    conversations:
        human: {errored_caption}
        gpt: <img1> {new self-reflection and correction instruction} <img2>

    old_conversations:
        human: {caption}
        gpt: <img1> {self-reflection and correction instruction} <img2>


Two LLM stages per row, both batched:

    legit_errors = sample_errors(caption, correction)          # what MAY be broken
    n_errors     = random.randint(1, N)
    errors       = random.sample(legit_errors, n_errors)
    errored_caption, new_self_reflect, new_correction = inject_errors(...)

# caption: A small dog with tiger-like painting peeks out from a pastel green handbag with a bow on the front.
# img1: an image with everything described but a pink handbag
# Correction: Change the color of the handbag from pastel pink to pastel green
# img2: image faithful to the caption
# Sample a type or two for the error to introduce in the caption. The error can't contradict the correction instruction like error: pastel green to orange

Why the errors must avoid whatever the correction talks about: img1 and img2 differ
ONLY by the correction, so any detail the correction does not mention is identical in
both images. Injecting the error into such a detail is what makes the caption
unfaithful to img2 while leaving the correction well defined.

img2 is fixed data, so the gpt turn has to be rewritten to still imply it:

    self_reflect   critiques img1 against the ERRORED caption, so it lists the genuine
                   defect of img1 *and* the injected mismatches.
    correction     resolves every listed point to what img2 actually shows: it fixes the
                   genuine defect and explicitly declines to apply the injected ones
                   ("keep the bow on the front"). The correction, not the caption, is the
                   authority on img2 - which is exactly the behaviour being trained.

Error taxonomy pulled from the benchmarks this data is evaluated on:
Geneval2: position, attributes: color, texture, shape, etc., object, count
TIIF: attribute: color, texture, shape, relations: 2D, 3D, action, reasoning: numeracy, differentiation, comparison, negation
Genai-bench: attribute, scene, spatial, action, part, count, differentiation, comparison, negation, universal
'''

from __future__ import annotations

import argparse
import logging
import os
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


logger = logging.getLogger(__name__)


input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/gpt-edit-simpler_hqedt_slfref_tar/train"
output_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/caption_error/gpt-edit-simpler_hqedt_slfref_tar/train"
# Read the model card to understand how to use the model - input preparation and sampling parameters
# Model card https://huggingface.co/Qwen/Qwen3.6-35B-A3B
llm_path = "/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/Qwen3.6-35B-A3B"
N = 2

# LazyParquetDataset loads a WHOLE row group into RAM per DataLoader worker, and these
# rows carry two images each, so keep row groups small. Same value as the script that
# produced the input data (add_slfref_in_edit_llm.py).
ROW_GROUP_SIZE = 64
RANDOM_SEED = 29991

TURN_STRUCT = pa.struct([pa.field("from", pa.string()), pa.field("value", pa.string())])
CONVERSATIONS_TYPE = pa.list_(TURN_STRUCT)

# The gpt turn laid out by add_slfref_in_edit_llm.py:
#   <image>\nSelf-reflect: {...}\nCorrection: {...}\n<image>
GPT_TURN_RE = re.compile(
    r"^\s*<image>\s*\n"
    r"Self-reflect:\s*(?P<self_reflect>.*?)\s*\n"
    r"Correction:\s*(?P<correction>.*?)\s*\n"
    r"<image>\s*$",
    flags=re.DOTALL,
)


# ---------------------------------------------------------------- stage 1: sample_errors

ERROR_TYPES = """\
- color: the colour of an object or surface.
- texture: the surface finish or material appearance (glossy, furry, rusty, woven, ...).
- shape: the geometric form of an object (round -> square, tall -> squat, ...).
- size: the absolute or relative size of an object.
- object: swap a named object for a different one, or name an object that is not there.
- part: a component or body part of an object (a handle, a tail, a collar, ...).
- count: how many of something there are (numeracy).
- position: 2D placement in the frame (left/right, top/bottom, centre, foreground/background).
- spatial relation: 3D relation between objects (on top of, behind, inside, between, under).
- action: what a person, animal, or object is doing.
- scene: the setting or surroundings the subject is in.
- text: the wording of legible text that the caption quotes.
- comparison: a stated comparison between two things (taller than, darker than, ...).
- differentiation: a property that tells two otherwise similar things apart.
- negation: assert the absence of something that is present, or drop a stated absence."""

SAMPLE_ERRORS_SYSTEM_PROMPT = f"""You propose factual errors that could be injected into an image caption.

You are given:
<caption> - a faithful description of a target image.
<correction> - the instruction that turns a flawed draft of that image into the target image.

The draft and the target differ ONLY by what <correction> describes. Every other detail in <caption> is already identical in both images.

Propose candidate errors that would make <caption> factually wrong about the target image.

Rules:
1. Never touch a detail that <correction> mentions or relies on. That detail is the one thing the correction is about, and the correction must stay valid and unambiguous after the error is injected. If <correction> is about the handbag's colour, no error about the handbag's colour.
2. Only propose errors that are visually checkable. Never anything about mood, style, artistic quality, camera settings, or the medium.
3. Each error either rewrites a detail <caption> already states, or states a detail that is plainly absent from the target image.
4. Keep each error small and local: a phrase, not a rewritten sentence. The caption must stay fluent and plausible - it has to read like a caption someone would really write, not like a corrupted one.
5. Propose diverse types. Do not give two errors of the same type unless the caption offers nothing else.

Error types:
{ERROR_TYPES}

Return between 3 and 6 candidates as exactly this XML and no other text:
<errors>
<error><type>TYPE</type><original>exact phrase from the caption, or "absent" when the error states something new</original><errored>the replacement phrase</errored></error>
...
</errors>"""

SAMPLE_ERRORS_USER1 = """<caption>
A small dog with tiger-like painting peeks out from a pastel green handbag with a bow on the front. The handbag rests on a wooden bench in a sunlit park.
</caption>

<correction>
change the color of the handbag from pastel pink to pastel green
</correction>"""

SAMPLE_ERRORS_ASSISTANT1 = """<errors>
<error><type>position</type><original>a bow on the front</original><errored>a bow on the back</errored></error>
<error><type>object</type><original>a wooden bench</original><errored>a stone ledge</errored></error>
<error><type>count</type><original>A small dog</original><errored>Two small dogs</errored></error>
<error><type>scene</type><original>a sunlit park</original><errored>a dim parking garage</errored></error>
<error><type>texture</type><original>absent</original><errored>the handbag is made of woven straw</errored></error>
</errors>"""

SAMPLE_ERRORS_USER2 = """<caption>
A golden wheat field at dusk with a dark sky filled with dark stormy clouds, lightning, and a darker overall atmosphere.
</caption>

<correction>
add dark stormy clouds, lightning, and change the sky color to a darker shade
</correction>"""

SAMPLE_ERRORS_ASSISTANT2 = """<errors>
<error><type>color</type><original>A golden wheat field</original><errored>A pale green wheat field</errored></error>
<error><type>object</type><original>absent</original><errored>a red barn stands at the edge of the field</errored></error>
<error><type>spatial relation</type><original>absent</original><errored>a dirt road cuts through the middle of the field</errored></error>
<error><type>count</type><original>absent</original><errored>three tall trees line the horizon</errored></error>
</errors>"""


def _sample_errors_messages(caption: str, correction: str) -> list[dict]:
    return [
        {"role": "system", "content": SAMPLE_ERRORS_SYSTEM_PROMPT},
        {"role": "user", "content": SAMPLE_ERRORS_USER1},
        {"role": "assistant", "content": SAMPLE_ERRORS_ASSISTANT1},
        {"role": "user", "content": SAMPLE_ERRORS_USER2},
        {"role": "assistant", "content": SAMPLE_ERRORS_ASSISTANT2},
        {
            "role": "user",
            "content": (
                "<caption>\n"
                f"{caption.strip()}\n"
                "</caption>\n\n"
                "<correction>\n"
                f"{correction.strip()}\n"
                "</correction>"
            ),
        },
    ]


ERROR_RE = re.compile(
    r"<error>\s*"
    r"<type>\s*(?P<type>.*?)\s*</type>\s*"
    r"<original>\s*(?P<original>.*?)\s*</original>\s*"
    r"<errored>\s*(?P<errored>.*?)\s*</errored>\s*"
    r"</error>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _strip_response(text: str) -> str:
    """Drop a reasoning block and any code fence the model wrapped the XML in."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # An unterminated <think> means the token budget ran out mid-reasoning; there is
    # no answer to salvage in that case.
    if "<think>" in text:
        raise ValueError("Response is an unterminated <think> block")
    return text.removeprefix("```xml").removeprefix("```").removesuffix("```").strip()


def _parse_errors(text: str) -> list[dict]:
    text = _strip_response(text)
    errors = [
        {
            "type": match.group("type"),
            "original": match.group("original"),
            "errored": match.group("errored"),
        }
        for match in ERROR_RE.finditer(text)
    ]
    errors = [error for error in errors if error["type"] and error["errored"]]
    if not errors:
        raise ValueError(f"No <error> elements in the response:\n{text}")
    return errors


# ---------------------------------------------------------------- stage 2: inject_errors

INJECT_ERRORS_SYSTEM_PROMPT = """You rewrite one training sample for an iterative image-generation model.

A sample is a caption plus a draft image (image 1), a critique of that draft, a correction instruction, and the corrected image (image 2). Image 2 is fixed and cannot change.

You are given <caption>, <self_reflect> and <correction> describing the original sample, and a list of <errors> to inject into the caption.

Produce three things.

1. <caption>: the original caption with every listed error applied, and nothing else changed. Keep the wording, order, and tone of the original. The result must read like a natural caption, not like a caption with something obviously wrong bolted on.

2. <self_reflect>: a critique of image 1 measured against the NEW, errored caption. It must state
   (a) the genuine defects of image 1, i.e. what the original <self_reflect> already said, and
   (b) one discrepancy per injected error, because image 1 shows the original detail, not the errored one.
   Write it in the present tense as a plain description of image 1. Never mention an edit, a change, an instruction, a caption, an error, image 1, image 2, or the fact that anything was rewritten. Never say a detail is correct or unchanged.

3. <correction>: the instruction that turns image 1 into image 2. It must resolve every point raised in <self_reflect>:
   (a) fix the genuine defects exactly as the original <correction> did, keeping its wording where possible, and
   (b) for each injected error, explicitly say to KEEP the detail as image 1 already shows it ("keep the bow on the front", "leave the field golden"). Image 2 still shows the original detail, so the injected error must not be applied.
   One instruction, written as a single flowing sentence or a short comma-separated list. Never mention the caption or the fact that anything was injected.

Return exactly this XML and no other text:
<caption>...</caption>
<self_reflect>...</self_reflect>
<correction>...</correction>"""

INJECT_ERRORS_USER1 = """<caption>
A small dog with tiger-like painting peeks out from a pastel green handbag with a bow on the front. The handbag rests on a wooden bench in a sunlit park.
</caption>

<self_reflect>
The handbag is pastel pink instead of pastel green.
</self_reflect>

<correction>
change the color of the handbag from pastel pink to pastel green
</correction>

<errors>
<error><type>position</type><original>a bow on the front</original><errored>a bow on the back</errored></error>
<error><type>object</type><original>a wooden bench</original><errored>a stone ledge</errored></error>
</errors>"""

INJECT_ERRORS_ASSISTANT1 = """<caption>
A small dog with tiger-like painting peeks out from a pastel green handbag with a bow on the back. The handbag rests on a stone ledge in a sunlit park.
</caption>
<self_reflect>
The handbag is pastel pink instead of pastel green. The bow sits on the front of the handbag rather than on the back, and the handbag rests on a wooden bench rather than a stone ledge.
</self_reflect>
<correction>
change the color of the handbag from pastel pink to pastel green, keep the bow on the front, and leave the handbag resting on the wooden bench
</correction>"""

INJECT_ERRORS_USER2 = """<caption>
A golden wheat field at dusk with a dark sky filled with dark stormy clouds, lightning, and a darker overall atmosphere.
</caption>

<self_reflect>
The sky is clear with orange clouds instead of dark stormy clouds and lightning, and the sky color is bright rather than a darker shade.
</self_reflect>

<correction>
add dark stormy clouds, lightning, and change the sky color to a darker shade
</correction>

<errors>
<error><type>count</type><original>absent</original><errored>three tall trees line the horizon</errored></error>
</errors>"""

INJECT_ERRORS_ASSISTANT2 = """<caption>
A golden wheat field at dusk with three tall trees lining the horizon, under a dark sky filled with dark stormy clouds, lightning, and a darker overall atmosphere.
</caption>
<self_reflect>
The sky is clear with orange clouds instead of dark stormy clouds and lightning, the sky color is bright rather than a darker shade, and the horizon is bare with no trees along it.
</self_reflect>
<correction>
add dark stormy clouds, lightning, and change the sky color to a darker shade, and leave the horizon bare without any trees
</correction>"""


def _inject_errors_messages(
    caption: str, self_reflect: str, correction: str, errors: list[dict]
) -> list[dict]:
    errors_xml = "\n".join(
        f"<error><type>{error['type']}</type>"
        f"<original>{error['original']}</original>"
        f"<errored>{error['errored']}</errored></error>"
        for error in errors
    )
    return [
        {"role": "system", "content": INJECT_ERRORS_SYSTEM_PROMPT},
        {"role": "user", "content": INJECT_ERRORS_USER1},
        {"role": "assistant", "content": INJECT_ERRORS_ASSISTANT1},
        {"role": "user", "content": INJECT_ERRORS_USER2},
        {"role": "assistant", "content": INJECT_ERRORS_ASSISTANT2},
        {
            "role": "user",
            "content": (
                "<caption>\n"
                f"{caption.strip()}\n"
                "</caption>\n\n"
                "<self_reflect>\n"
                f"{self_reflect.strip()}\n"
                "</self_reflect>\n\n"
                "<correction>\n"
                f"{correction.strip()}\n"
                "</correction>\n\n"
                "<errors>\n"
                f"{errors_xml}\n"
                "</errors>"
            ),
        },
    ]


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE
    )
    if match is None:
        raise ValueError(f"Response has no <{tag}> element:\n{text}")
    value = " ".join(match.group(1).split())
    if not value:
        raise ValueError(f"Response has an empty <{tag}> element:\n{text}")
    return value


def _parse_injection(text: str) -> tuple[str, str, str]:
    text = _strip_response(text)
    return (
        _extract_tag(text, "caption"),
        _extract_tag(text, "self_reflect"),
        _extract_tag(text, "correction"),
    )


# ---------------------------------------------------------------- generation

class PresencePenalty:
    """The card's `presence_penalty` has no `generate()` equivalent, so apply it as a
    logits processor: once a token has been emitted, subtract a flat penalty from it."""

    def __init__(self, penalty: float, prompt_length: int):
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        import torch

        generated = input_ids[:, self.prompt_length:]
        if generated.numel():
            seen = torch.zeros_like(scores, dtype=torch.bool)
            seen.scatter_(1, generated, True)
            scores = scores - seen.to(scores.dtype) * self.penalty
        return scores


def _resolve_no_think_kwargs(tokenizer) -> dict:
    """Return the `apply_chat_template` kwargs that actually switch thinking off.

    Qwen3.5/3.6 think by default: the template appends a bare `<think>\\n` unless
    thinking is disabled, in which case it emits a pre-closed `<think>\\n\\n</think>`.
    A reasoning block would eat the whole `--max-tokens` budget here, and Transformers
    routes extra kwargs differently across releases (the model card's nested
    `chat_template_kwargs={...}` is silently dropped by some 5.x releases), so probe
    the rendered template instead of assuming.
    """
    probe = [{"role": "user", "content": "probe"}]
    for candidate in ({"enable_thinking": False}, {"chat_template_kwargs": {"enable_thinking": False}}):
        try:
            prompt = tokenizer.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=True, **candidate
            )
        except (TypeError, ValueError):
            continue
        # Templates with no <think> at all (e.g. Qwen3-VL-Instruct) land here too and
        # simply ignore the unknown variable.
        if "<think>" not in prompt or "</think>" in prompt.split("<think>")[-1]:
            return candidate
    raise RuntimeError(
        "Could not disable thinking for this checkpoint: its chat template still opens "
        "a <think> block, so generation would spend its budget on reasoning."
    )


def generate(model, tokenizer, batch_messages, args, no_think_kwargs, max_tokens):
    """Run one batched chat completion; return the newly generated text per row."""
    import torch

    prompts = tokenizer.apply_chat_template(
        batch_messages, tokenize=False, add_generation_prompt=True, **no_think_kwargs
    )
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    input_device = model.get_input_embeddings().weight.device
    inputs = {name: value.to(input_device) for name, value in inputs.items()}

    logits_processors = []
    if args.presence_penalty:
        logits_processors.append(PresencePenalty(args.presence_penalty, input_length))

    sampling = {}
    if args.temperature > 0:
        sampling = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
        }
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=args.temperature > 0,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=max_tokens,
            logits_processor=logits_processors,
            **sampling,
        )
    return tokenizer.batch_decode(
        output_ids[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _generate_parsed(model, tokenizer, messages, args, no_think_kwargs, max_tokens, parse):
    """Generate for every row, parse each response, and return None where parsing failed."""
    if not messages:
        return []
    parsed = []
    for response in generate(model, tokenizer, messages, args, no_think_kwargs, max_tokens):
        try:
            parsed.append(parse(response))
        except Exception:
            logger.exception("Skipping a response that could not be parsed")
            parsed.append(None)
    return parsed


# ---------------------------------------------------------------- per-batch pipeline

def _split_gpt_turn(conversation) -> tuple[str, str, str]:
    """Pull (caption, self_reflect, correction) out of one `conversations` value."""
    if not conversation or len(conversation) < 2:
        raise ValueError("Conversation does not have a human and a gpt turn")
    human = next((turn for turn in conversation if turn.get("from") == "human"), None)
    gpt = next((turn for turn in conversation if turn.get("from") == "gpt"), None)
    if human is None or gpt is None:
        raise ValueError("Conversation is missing its human or gpt turn")
    match = GPT_TURN_RE.match(gpt.get("value") or "")
    if match is None:
        raise ValueError(f"gpt turn is not <image>/Self-reflect/Correction/<image>:\n{gpt.get('value')!r}")
    caption = (human.get("value") or "").strip()
    if not caption:
        raise ValueError("Empty caption in the human turn")
    return caption, match.group("self_reflect").strip(), match.group("correction").strip()


def _build_conversation(caption: str, self_reflect: str, correction: str):
    return [
        {"from": "human", "value": caption},
        {
            "from": "gpt",
            "value": f"<image>\nSelf-reflect: {self_reflect}\nCorrection: {correction}\n<image>",
        },
    ]


def process_batch(batch, model, tokenizer, args, no_think_kwargs, rng):
    """Error the caption of every row in `batch`.

    Returns (kept_row_indices, new_conversations). Rows whose gpt turn will not parse,
    or whose generation fails at either stage, are dropped rather than written through
    unerrored - an unerrored row would reintroduce exactly the shortcut this data is
    meant to remove.
    """
    conversations = batch.column(batch.schema.get_field_index("conversations")).to_pylist()

    parsed_rows = []
    for index, conversation in enumerate(conversations):
        try:
            parsed_rows.append((index, *_split_gpt_turn(conversation)))
        except Exception as error:
            logger.warning("Skipping row %d: %s", index, error)
    if not parsed_rows:
        return [], []

    # Stage 1: what may legitimately be broken in each caption.
    candidates = _generate_parsed(
        model,
        tokenizer,
        [_sample_errors_messages(caption, correction) for _, caption, _, correction in parsed_rows],
        args,
        no_think_kwargs,
        args.max_tokens_errors,
        _parse_errors,
    )

    # Stage 2: inject a random 1..N of them and rewrite the gpt turn to match.
    stage2_rows = []
    stage2_messages = []
    for (index, caption, self_reflect, correction), legit_errors in zip(parsed_rows, candidates):
        if not legit_errors:
            continue
        n_errors = rng.randint(1, min(args.max_errors, len(legit_errors)))
        errors = rng.sample(legit_errors, n_errors)
        stage2_rows.append(index)
        stage2_messages.append(
            _inject_errors_messages(caption, self_reflect, correction, errors)
        )

    injected = _generate_parsed(
        model,
        tokenizer,
        stage2_messages,
        args,
        no_think_kwargs,
        args.max_tokens_inject,
        _parse_injection,
    )

    kept_indices = []
    new_conversations = []
    for index, result in zip(stage2_rows, injected):
        if result is None:
            continue
        kept_indices.append(index)
        new_conversations.append(_build_conversation(*result))
    return kept_indices, new_conversations


def _output_schema(input_schema: pa.Schema) -> pa.Schema:
    """Input schema with `old_conversations` inserted right after `conversations`."""
    fields = []
    for field in input_schema:
        fields.append(field)
        if field.name == "conversations":
            fields.append(pa.field("old_conversations", CONVERSATIONS_TYPE))
    return pa.schema(fields)


def _build_table(batch, kept_indices, new_conversations, schema: pa.Schema) -> pa.Table:
    """Keep every input column for the surviving rows, swap in the errored caption, and
    carry the untouched original along as `old_conversations`."""
    kept = pa.Table.from_batches([batch]).take(pa.array(kept_indices, type=pa.int64()))
    columns = []
    for field in schema:
        if field.name == "old_conversations":
            columns.append(kept.column("conversations").cast(CONVERSATIONS_TYPE))
        elif field.name == "conversations":
            columns.append(pa.chunked_array([pa.array(new_conversations, type=CONVERSATIONS_TYPE)]))
        else:
            columns.append(kept.column(field.name).cast(field.type))
    return pa.Table.from_arrays([column.combine_chunks() for column in columns], schema=schema)


# ---------------------------------------------------------------- driver

def _output_is_complete(path: Path, schema: pa.Schema) -> bool:
    """A finished shard; row counts are not comparable because failed rows are dropped."""
    try:
        return pq.ParquetFile(path).schema_arrow.equals(schema, check_metadata=False)
    except (OSError, pa.ArrowException):
        return False


def convert_file(input_path: Path, output_path: Path, model, tokenizer, args) -> tuple[int, int, bool]:
    """Returns (rows read, rows written, whether the file was skipped as already done)."""
    parquet_file = pq.ParquetFile(input_path)
    input_schema = parquet_file.schema_arrow
    if "conversations" not in input_schema.names:
        raise ValueError(f"{input_path}: no `conversations` column")
    if "old_conversations" in input_schema.names:
        raise ValueError(f"{input_path}: already has an `old_conversations` column")
    if "conversations_short" in input_schema.names:
        logger.warning(
            "%s: `conversations_short` is copied through unerrored; its caption still "
            "matches image 2 and would keep the shortcut open",
            input_path.name,
        )
    schema = _output_schema(input_schema)
    if not args.overwrite and _output_is_complete(output_path, schema):
        return 0, 0, True

    rng = random.Random(f"{args.seed}:{input_path.name}")
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    writer = None
    rows_in = 0
    rows_out = 0
    try:
        for batch in parquet_file.iter_batches(batch_size=args.batch_size, use_threads=True):
            rows_in += batch.num_rows
            kept_indices, new_conversations = process_batch(
                batch, model, tokenizer, args, args.no_think_kwargs, rng
            )
            if not kept_indices:
                continue
            table = _build_table(batch, kept_indices, new_conversations, schema)
            if writer is None:
                writer = pq.ParquetWriter(temporary, schema)
            writer.write_table(table, row_group_size=args.row_group_size)
            rows_out += len(kept_indices)
            logger.info("  %s: %d/%d rows errored", input_path.name, rows_out, rows_in)
        if writer is not None:
            writer.close()
            writer = None
        else:
            pq.write_table(pa.Table.from_batches([], schema=schema), temporary)
        os.replace(temporary, output_path)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    return rows_in, rows_out, False


def load_model(args):
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # Batched generation with a decoder-only model needs left padding, otherwise the
    # shorter prompts in a batch continue from pad tokens.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Qwen3.6 (`model_type: qwen3_5_moe`) only exists in Transformers 5.x. 4.57.0 and
    # older raise here, before any weight is touched, so say what is actually wrong.
    try:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    except ValueError as error:
        raise RuntimeError(
            f"Transformers {transformers.__version__} does not recognise the checkpoint "
            f"at {args.model_path}. Qwen3.6 needs a Transformers release that registers "
            f"`qwen3_5_moe`.\n{error}"
        ) from error

    # Qwen3.5/3.6 ship as `*ForConditionalGeneration` (a vision encoder bolted onto the
    # LM) and are not registered under AutoModelForCausalLM; plain Qwen3.5-*B dense
    # text checkpoints are. Try the multimodal auto classes first, newest name first.
    auto_classes = [
        getattr(transformers, name, None)
        for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM")
    ]
    kwargs = {"dtype": "auto", "device_map": args.device, "trust_remote_code": True}
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
        if args.attn_implementation == "flash_attention_2":
            kwargs["dtype"] = torch.bfloat16

    errors = []
    for auto_class in auto_classes:
        if auto_class is None:
            continue
        try:
            model = auto_class.from_pretrained(args.model_path, **kwargs)
            break
        except (ValueError, KeyError) as error:
            errors.append(f"{auto_class.__name__}: {error}")
    else:
        raise RuntimeError(
            f"Transformers {transformers.__version__} cannot load a "
            f"`{config.model_type}` checkpoint from {args.model_path}.\n"
            + "\n".join(errors)
        )
    model.eval()
    return model, tokenizer


def main(args) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    source_dir = Path(args.input_dir).expanduser()
    destination_dir = Path(args.output_dir).expanduser()
    input_files = sorted(source_dir.glob("*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No .parquet files found in {source_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Data-parallel over independent jobs: each rank owns a disjoint file slice.
    my_files = [
        path for index, path in enumerate(input_files)
        if index % args.world_size == args.rank
    ]
    logger.info(
        "%d parquet files in %s; rank %d/%d handles %d",
        len(input_files), source_dir, args.rank, args.world_size, len(my_files),
    )

    model, tokenizer = load_model(args)
    args.no_think_kwargs = _resolve_no_think_kwargs(tokenizer)
    logger.info("thinking disabled via %s", args.no_think_kwargs)

    total_in = 0
    total_out = 0
    for path in my_files:
        rows_in, rows_out, skipped = convert_file(
            path, destination_dir / path.name, model, tokenizer, args
        )
        if skipped:
            logger.info("%s: already done, skipping", path.name)
            continue
        total_in += rows_in
        total_out += rows_out
        logger.info("%s: wrote %d of %d rows", path.name, rows_out, rows_in)

    logger.info(
        "Done. Wrote %d of %d rows to %s (%d dropped)",
        total_out, total_in, destination_dir, total_in - total_out,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Inject factual errors into the caption of iterative image-generation data "
            "so the caption alone no longer determines the final image, and rewrite the "
            "self-reflection/correction turn to still imply it."
        )
    )
    parser.add_argument("--input-dir", default=input_dir)
    parser.add_argument("--output-dir", default=output_dir)
    parser.add_argument("--model-path", default=llm_path)
    parser.add_argument("--device", default="auto", help="`device_map` for from_pretrained.")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="e.g. flash_attention_2 (forces bf16).",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Rows per LLM batch.")
    parser.add_argument("--row-group-size", type=int, default=ROW_GROUP_SIZE)
    parser.add_argument(
        "--max-errors",
        type=int,
        default=N,
        help="Upper bound on how many errors get injected per caption; 1..this many are sampled.",
    )
    parser.add_argument("--max-tokens-errors", type=int, default=512)
    parser.add_argument("--max-tokens-inject", type=int, default=640)
    # Qwen3.6's card recommends these for instruct (non-thinking) mode, which is how
    # this script runs.
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="0 switches to greedy decoding (top_p/top_k/min_p are then ignored).",
    )
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    parser.add_argument(
        "--overwrite", action="store_true", help="Redo input files that already have an output shard."
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
