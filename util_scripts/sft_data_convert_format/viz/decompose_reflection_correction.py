'''
Visualize the output of util_scripts/create_data/reduce_gap_w_ood/decompose_reflection_correction.py.

One PNG per row, in the style of enrich_caption.py: the draft and refined images, a word diff of the original caption
("v1-conversations") against the rewritten one when the caption was flagged as an edit delta, then the original
Self-reflect and Correction (edit-pair framing such as "the second image has been" highlighted) next to the numbered
steps they were decomposed into. Rows the LLM never validated show the rejected reply; rows dropped as junk
(placeholders, no-edit replies, diptych critiques) are drawn from the source shards with the reason.

Rows are drawn grouped by outcome (caption rewritten, caption rewrite rejected, decomposed, failed, dropped), up to
--num_samples. summary.txt lists the outcome counts and one line per drawn row.

Usage:
    python decompose_reflection_correction.py <decomposed dir> <output dir> [--source_dir <input dir>] [--cache_dir <dir>]
--source_dir is the directory the decompose run read; it locates the cache (<source_dir>/.decompose_cache) and the
dropped rows. For an in-place run, the decomposed dir is the source dir and dropped rows are gone, so they are not drawn.
'''

import argparse
import collections
import glob
import io
import json
import os
import random
import re
import sys

import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from enrich_caption import (BOLD, DELETED, FONT, IMAGE_HEIGHT, INSERTED, INSERTED_BG, MARGIN, MIN_WIDTH, MUTED,
                            RichText, _fit, diff_spans)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "create_data", "reduce_gap_w_ood"))
import decompose_reflection_correction as decompose  # noqa: E402

NUM_SAMPLES = 50
SEED = 8811
V1_COLUMN = "v1-conversations"
ID_COLUMNS = ("id", "key")  # UnicEdit shards carry "key"
TASK_COLUMNS = ("edit_task", "edit_subtask")
FLAGGED = "#c77c02"  # edit-delta wording in a caption, edit-pair framing in a critique
OUTCOMES = {  # tag -> (header text, color); also the drawing order
    "caption": ("decomposed, caption rewritten", INSERTED),
    "caption_kept": ("decomposed, caption rewrite rejected (original kept)", FLAGGED),
    "caption_same": ("decomposed, caption returned unchanged by the LLM", FLAGGED),
    "decomposed": ("decomposed", INSERTED),
    "failed": ("failed validation (original text kept)", DELETED),
    "dropped": ("dropped as junk", DELETED),
    "missing": ("no cache record (original text kept)", MUTED),
}


# ----------------------------------------------------------------------------------------------------- loading

def load_cache(cache_dir):
    """(caption, self_reflect, correction) -> newest record; several prompt versions may sit in one cache dir."""
    records = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.jsonl")), key=os.path.getmtime):
        with open(path) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records[(record.get("caption"), record.get("self_reflect"), record.get("correction"))] = record
    return records


def outcome(conversations, records):
    """(tag, edit-turn parts, record, junk reason) of a source conversation."""
    found = decompose.find_edit_turn(conversations)
    if found is None:
        return "missing", None, None, None
    parts = found[1]
    reason = decompose.junk_reason(parts)
    if reason is not None:
        return "dropped", parts, None, reason
    record = records.get((decompose.caption_of(conversations), parts["self_reflect"], parts["correction"]))
    if record is None:
        return "missing", parts, None, None
    if record["status"] != "ok":
        return "failed", parts, record, None
    if record.get("rewrite_caption"):
        caption = record["result"].get("caption")
        tag = "caption_kept" if caption is None else "caption_same" if caption == record["caption"] else "caption"
        return tag, parts, record, None
    return "decomposed", parts, record, None


def id_column(parquet_file):
    return next(name for name in ID_COLUMNS if name in parquet_file.schema_arrow.names)


def row_id(row):
    return next((row[name] for name in ID_COLUMNS if name in row), None)


def scan(paths, column, records, exclude_ids=()):
    """[(tag, path, row_group, index)] reading only the conversation and id columns."""
    rows = []
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(group, columns=[column, id_column(parquet_file)])
            for index, (conversations, row_id) in enumerate(zip(table.column(0).to_pylist(), table.column(1).to_pylist())):
                if row_id in exclude_ids:
                    continue
                rows.append((outcome(conversations, records)[0], path, group, index))
    return rows


def iter_rows(selected):
    """Yield (order, row) reading each needed row group once."""
    by_group = collections.defaultdict(list)
    for order, (_, path, group, index) in enumerate(selected):
        by_group[(path, group)].append((order, index))
    for (path, group), wanted in sorted(by_group.items()):
        rows = pq.ParquetFile(path).read_row_group(group).to_pylist()
        for order, index in wanted:
            yield order, rows[index]
        del rows


# ----------------------------------------------------------------------------------------------------- drawing

def marked(text, pattern, style, base=None):
    """Spans of text with every match of pattern drawn in style."""
    base = base or {}
    spans, last = [], 0
    for match in pattern.finditer(text):
        if match.start() > last:
            spans.append((text[last:match.start()], base))
        # The word wrapper drops a span's leading space, so the space after a match goes at the end of the match.
        end = match.end() + len(text[match.end():]) - len(text[match.end():].lstrip())
        spans.append((text[match.start():end], style))
        last = end
    if last < len(text):
        spans.append((text[last:], base))
    return spans


def steps_block(text, title, steps):
    text.line([(f"{title}:", {"bold": True})], gap=0)
    for number, step in enumerate(steps, 1):
        text.line([(f"   {number}. ", {"bold": True, "fill": INSERTED}), (step, {"fill": INSERTED})], gap=0)
    text.space(8)


def render_row(row, tag, parts, record, reason):
    images = [Image.open(io.BytesIO(im["bytes"])).convert("RGB") for im in (row.get("image") or []) if im and im.get("bytes")]
    panels = [(_fit(image, IMAGE_HEIGHT)[0], "refined (image-2)" if index == len(images) - 1 else "draft (image-1)")
              for index, image in enumerate(images)]
    width = max(MIN_WIDTH, sum(p.width for p, _ in panels) + MARGIN * (len(panels) + 1))
    text = RichText(width - 2 * MARGIN)

    source = row.get(V1_COLUMN) or row["conversations"]
    header, color = OUTCOMES[tag]
    if reason:
        header += f": {reason}"
    header += "    "
    attempts = f"attempts: {record['attempts']}" if record else ""
    task = " / ".join(str(row[name]) for name in TASK_COLUMNS if row.get(name))
    text.line([(f"{row_id(row)}    ", {"bold": True}), (header, {"fill": color, "bold": True}), (attempts, {"fill": MUTED})])
    if task:
        text.line([("edit task: ", {"fill": MUTED}), (task, {"fill": MUTED, "bold": True})], gap=0)
    text.space(4)

    old_caption = decompose.caption_of(source) or ""
    new_caption = old_caption if tag == "dropped" else decompose.caption_of(row["conversations"]) or old_caption
    if new_caption != old_caption:
        old_spans, new_spans = diff_spans(old_caption, new_caption)
        text.line([("Original caption: ", {"bold": True})] + old_spans)
        text.line([("Rewritten caption: ", {"bold": True})] + new_spans, gap=10)
    elif old_caption:  # UnicEdit rows have no caption: nothing to show
        flagged = bool(decompose.CAPTION_DELTA.search(old_caption))
        title = "Caption (flagged as an edit delta, kept): " if flagged else "Caption (unchanged): "
        text.line([(title, {"bold": True})] + marked(old_caption, decompose.CAPTION_DELTA, {"fill": FLAGGED, "bold": True}), gap=10)
        if tag == "caption_kept" and record.get("raw"):
            if record.get("reason"):
                text.line([("Why: ", {"bold": True, "fill": DELETED}), (record["reason"], {"fill": DELETED})], gap=0)
            text.line([("Rejected caption rewrite (last attempt): ", {"bold": True, "fill": DELETED}),
                       (" ".join(record["raw"].split())[:600], {"fill": MUTED})], gap=10)

    leak_style = {"fill": FLAGGED, "bold": True}
    if parts is not None and parts["self_reflect"] is not None:
        text.line([("Original Self-reflect: ", {"bold": True})] +
                  marked(parts["self_reflect"], decompose.STEP_LEAK, leak_style, {"fill": MUTED}))
    if parts is not None:
        text.line([("Original Correction: ", {"bold": True})] +
                  marked(parts["correction"], decompose.STEP_LEAK, leak_style, {"fill": MUTED}), gap=10)

    result = (record or {}).get("result")
    if result:
        if result["self_reflect"]:
            steps_block(text, "Decomposed Self-reflect", result["self_reflect"])
        steps_block(text, "Decomposed Correction", result["correction"])
    elif tag == "failed" and record.get("raw"):
        if record.get("reason"):
            text.line([("Why: ", {"bold": True, "fill": DELETED}), (record["reason"], {"fill": DELETED})], gap=0)
        text.line([("Rejected reply (last attempt): ", {"bold": True, "fill": DELETED}),
                   (" ".join(record["raw"].split())[:800], {"fill": MUTED})])
    elif tag == "dropped":
        text.line([(f"Not sent to the LLM and left out of the output shard ({reason} pattern matched).", {"fill": DELETED})])

    text_height = text.render(None, 0, 0)
    height = MARGIN + IMAGE_HEIGHT + 24 + MARGIN + text_height + MARGIN
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x = MARGIN
    for panel, label in panels:
        canvas.paste(panel, (x, MARGIN))
        draw.text((x, MARGIN + IMAGE_HEIGHT + 4), label, font=FONT, fill=MUTED)
        x += panel.width + MARGIN
    text.render(draw, MARGIN, MARGIN + IMAGE_HEIGHT + 24 + MARGIN)
    return canvas


# -------------------------------------------------------------------------------------------------------- main

def main(input_dir, output_dir, source_dir, cache_dir, num_samples, seed):
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    source_dir = source_dir or input_dir
    cache_dir = cache_dir or os.path.join(source_dir, ".decompose_cache")
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"No decompose cache at {cache_dir}; pass --source_dir or --cache_dir")
    records = load_cache(cache_dir)

    rows = scan(files, V1_COLUMN, records)
    if os.path.abspath(source_dir) != os.path.abspath(input_dir):
        kept_ids = {i for path in files for i in pq.read_table(path, columns=[id_column(pq.ParquetFile(path))]).column(0).to_pylist()}
        sources = sorted(glob.glob(os.path.join(source_dir, "*.parquet")))
        rows += scan(sources, decompose.source_column(pq.ParquetFile(sources[0])), records, kept_ids)
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows.sort(key=lambda r: list(OUTCOMES).index(r[0]))
    selected = rows[:num_samples]
    counts = collections.Counter(r[0] for r in rows)
    print("=======================")
    print(f"Rows: {len(rows)}, by outcome: {dict(counts)}, drawing: {len(selected)}")
    print("=======================")
    os.makedirs(output_dir, exist_ok=True)

    lines = {}
    for order, row in iter_rows(selected):
        source = row.get(V1_COLUMN) or row["conversations"]
        tag, parts, record, reason = outcome(source, records)
        name = f"sample_{order:02d}_{tag}.png"
        render_row(row, tag, parts, record, reason).save(os.path.join(output_dir, name))
        result = (record or {}).get("result") or {}
        detail = reason or (record or {}).get("reason") if tag in ("dropped", "failed", "caption_kept") else None
        detail = detail or (f"self-reflect {len(result.get('self_reflect') or [])} / correction "
                            f"{len(result.get('correction') or [])} steps" if result else "-")
        lines[order] = f"{name}  {row_id(row)}  [{tag}]  {detail}"
        if order == 0:
            print(f"Saved {os.path.join(output_dir, name)}")

    with open(os.path.join(output_dir, "summary.txt"), "w") as handle:
        handle.write(f"source: {source_dir}\ndecomposed: {input_dir}\ncache: {cache_dir}\n")
        handle.write(f"rows: {len(rows)}, drawn: {len(selected)}\noutcomes: {dict(counts)}\n\n")
        handle.writelines(lines[k] + "\n" for k in sorted(lines))
    print(f"Done. Outputs in {output_dir}")
    print("=========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory of decomposed parquet shards (with a v1-conversations column).")
    parser.add_argument("output")
    parser.add_argument("--source_dir", default=None, help="Directory the decompose run read (default: input_dir).")
    parser.add_argument("--cache_dir", default=None, help="Default: <source_dir>/.decompose_cache")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args.input_dir, args.output, args.source_dir, args.cache_dir, args.num_samples, args.seed)
