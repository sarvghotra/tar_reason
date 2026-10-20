'''
Visualize the output of util_scripts/create_data/reduce_gap_w_ood/enrich_captions.py.

One PNG per row: the draft and final images (detector boxes drawn on the final), a word diff of the original caption
("v1-conversations") against the enriched one ("conversations", insertions in green, deletions struck through in
red), the same for the short caption when the data has one, the critique, what the detectors measured, and every
fact derived for the row with what happened to it (used in the caption / selected but skipped by the rewrite / not
selected). Facts and boxes come from the enrich cache; without it only the caption diff is drawn.

Enriched rows are drawn first, then unchanged ones, up to --num_samples. summary.txt lists the status and fact-type
counts of the drawn rows and a one-line diff per row.

Usage:
    python enrich_caption.py <enriched dir> <output dir> [--cache_dir <dir>] [--num_samples 50]
The cache defaults to <enriched dir>/.enrich_cache, where an in-place enrich run leaves it; pass --cache_dir when the
run used --output_dir or --cache_dir.
'''

import argparse
import collections
import difflib
import glob
import io
import json
import os
import random
import re

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

NUM_SAMPLES = 50
SEED = 8811
IMAGE_HEIGHT = 480
MARGIN = 16
MIN_WIDTH = 1400
FONT_SIZE = 17
V1_PREFIX = "v1-"
SHORT_COLUMNS = ("conversations_short", "short_conversations")

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
BOX_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#9a6324"]
TEXT = "#222222"
MUTED = "#777777"
INSERTED_BG = "#c8f7c5"
INSERTED = "#0b6e0b"
DELETED = "#c0392b"
FACT_STYLE = {  # marker, color
    "used": ("[used]", "#0b6e0b"),
    "skipped": ("[skipped by rewrite]", "#c77c02"),
    "vetoed": ("[vetoed by VLM]", "#8e44ad"),
    "not selected": ("[not selected]", "#999999"),
}


def _font(bold=False, size=FONT_SIZE):
    path = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size=size)


FONT, BOLD = _font(), _font(bold=True)


# ----------------------------------------------------------------------------------------------------- loading

def _caption(convs):
    for turn in convs or []:
        if turn.get("from") == "human":
            return " ".join((turn.get("value") or "").split())
    return None


def _critique(convs):
    for turn in convs or []:
        if turn.get("from") == "gpt":
            value = (turn.get("value") or "").replace("<image>", "").strip()
            return " ".join(value.split())
    return None


def load_cache(cache_dir):
    """id -> the records seen for it (several prompt versions may sit in one cache dir)."""
    records = collections.defaultdict(list)
    paths = sorted(glob.glob(os.path.join(cache_dir, "*.jsonl")), key=os.path.getmtime)
    for path in paths:
        with open(path) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records[record["id"]].append(record)
    return records


def match_record(records, caption_v1, caption_new):
    """The record that produced this row: same source caption and, when enriched, the caption now in the data."""
    for record in reversed(records):  # newest first
        if record.get("caption") != caption_v1:
            continue
        result = record.get("result") or {}
        if caption_new == caption_v1 or result.get("caption") == caption_new:
            return record
    return None


def select_rows(files, n, seed):
    """[(path, row_group, row_index, enriched)] with enriched rows first; reads only the conversation columns."""
    rng = random.Random(seed)
    enriched, unchanged = [], []
    for path in files:
        parquet_file = pq.ParquetFile(path)
        names = parquet_file.schema_arrow.names
        if V1_PREFIX + "conversations" not in names:
            raise ValueError(f"{path} has no {V1_PREFIX}conversations column; is it enrich_captions.py output?")
        for group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(group, columns=["conversations", V1_PREFIX + "conversations"])
            for index, (new, old) in enumerate(zip(table.column(0).to_pylist(), table.column(1).to_pylist())):
                (enriched if _caption(new) != _caption(old) else unchanged).append((path, group, index))
    rng.shuffle(enriched)
    rng.shuffle(unchanged)
    picked = [(p, g, i, True) for p, g, i in enriched] + [(p, g, i, False) for p, g, i in unchanged]
    return picked[:n], len(enriched), len(enriched) + len(unchanged)


def iter_rows(selected):
    """Yield (order, row) reading each needed row group once, so image memory stays at one row group."""
    by_group = collections.defaultdict(list)
    for order, (path, group, index, _) in enumerate(selected):
        by_group[(path, group)].append((order, index))
    for (path, group), wanted in sorted(by_group.items()):
        rows = pq.ParquetFile(path).read_row_group(group).to_pylist()
        for order, index in wanted:
            yield order, rows[index]
        del rows


# ----------------------------------------------------------------------------------------------------- drawing

class RichText:
    """Word-wrapped text with per-word color, highlight and strike-through. Lay out once to measure, once to draw."""

    def __init__(self, width):
        self.width = width
        self.items = []  # (kind, payload)

    def line(self, spans, gap=4):
        """spans: [(text, style dict)]; one paragraph, wrapped."""
        self.items.append(("line", (spans, gap)))

    def space(self, pixels):
        self.items.append(("space", pixels))

    def render(self, draw, x0, y0):
        y = y0
        line_height = FONT_SIZE + 6
        for kind, payload in self.items:
            if kind == "space":
                y += payload
                continue
            spans, gap = payload
            x = x0
            for text, style in spans:
                font = BOLD if style.get("bold") else FONT
                for word in re.findall(r"\S+\s*", text):
                    length = font.getlength(word)
                    if x + font.getlength(word.rstrip()) > x0 + self.width and x > x0:
                        x, y = x0, y + line_height
                    if draw is not None:
                        word_width = font.getlength(word.rstrip())
                        if style.get("bg"):
                            draw.rectangle([x - 1, y - 1, x + word_width + 1, y + FONT_SIZE + 3], fill=style["bg"])
                        draw.text((x, y), word.rstrip(), font=font, fill=style.get("fill", TEXT))
                        if style.get("strike"):
                            mid = y + FONT_SIZE // 2 + 2
                            draw.line([x, mid, x + word_width, mid], fill=style.get("fill", TEXT), width=2)
                    x += length
            y += line_height + gap
        return y


def diff_spans(old, new):
    """(spans for the original, spans for the enriched caption) from a word diff."""
    a, b = old.split(), new.split()
    key = lambda words: [w.strip(",.;:!?\"'").lower() for w in words]  # "hair" -> "hair," is not an edit
    old_spans, new_spans = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, key(a), key(b), autojunk=False).get_opcodes():
        if op == "equal":
            old_spans.append((" ".join(a[i1:i2]) + " ", {}))
            new_spans.append((" ".join(b[j1:j2]) + " ", {}))
            continue
        if i2 > i1:
            old_spans.append((" ".join(a[i1:i2]) + " ", {"fill": DELETED, "strike": True}))
            new_spans.append((" ".join(a[i1:i2]) + " ", {"fill": DELETED, "strike": True}))
        if j2 > j1:
            new_spans.append((" ".join(b[j1:j2]) + " ", {"fill": INSERTED, "bg": INSERTED_BG, "bold": True}))
    return old_spans, new_spans


def _fit(image, height):
    scale = height / image.height
    return image.resize((max(1, round(image.width * scale)), height), Image.BICUBIC), scale


def draw_boxes(image, scale, measured):
    draw = ImageDraw.Draw(image)
    legend = []
    for k, (name, entry) in enumerate(sorted((measured or {}).get("objects", {}).items())):
        color = BOX_COLORS[k % len(BOX_COLORS)]
        for box in entry.get("boxes", []):
            draw.rectangle([v * scale for v in box], outline=color, width=3)
        if entry.get("boxes"):
            legend.append((name, color))
    return legend


def _proposals_by_region(record):
    """Qwen's raw region answers (stored under each region's reference) as short strings, in region order."""
    raw = record.get("region_raw") or {}
    return [_proposal_text(raw.get(region["ref"]) or {}) for region in record.get("regions") or []]


def _proposal_text(answer):
    bits = [f"{k} {answer[k]}" for k in ("color", "material") if answer.get(k)]
    bits += [p.get("name", "?") for p in answer.get("parts") or [] if isinstance(p, dict)]
    bits += ["wearing " + str(w.get("item")) for w in answer.get("wearing") or [] if isinstance(w, dict)]
    hair = answer.get("hair") or {}
    if isinstance(hair, dict) and hair.get("color"):
        bits.append(f"{hair.get('length') or ''} {hair['color']} hair".strip())
    return ", ".join(bits)


def fact_status(record, fact):
    used = set((record.get("result") or {}).get("used") or [])
    if fact.get("id") in used:
        return "used"
    if fact.get("key") in (record.get("vetoed") or []):
        return "vetoed"
    return "skipped" if fact.get("key") in (record.get("selected") or []) else "not selected"


def render_row(row, record, short_column):
    images = [Image.open(io.BytesIO(im["bytes"])).convert("RGB") for im in (row.get("image") or []) if im and im.get("bytes")]
    panels, legend = [], []
    for index, image in enumerate(images):
        fitted, scale = _fit(image, IMAGE_HEIGHT)
        if index == len(images) - 1 and record:
            legend = draw_boxes(fitted, scale, record.get("measured"))
        label = "final (the caption describes it)" if index == len(images) - 1 else "draft"
        panels.append((fitted, label))
    width = max(MIN_WIDTH, sum(p.width for p, _ in panels) + MARGIN * (len(panels) + 1))
    text_width = width - 2 * MARGIN

    old_caption = _caption(row[V1_PREFIX + "conversations"]) or ""
    new_caption = _caption(row["conversations"]) or ""
    status = record["status"] if record else "no cache record"
    used = (record.get("result") or {}).get("used") or [] if record else []
    text = RichText(text_width)
    # Leading spaces of a span are dropped by the word wrapper, so separators go at the end of the previous span.
    text.line([(f"{row.get('id')}    ", {"bold": True}),
               (f"status: {status}   facts used: {len(used)}   "
                f"{'ENRICHED' if new_caption != old_caption else 'unchanged'}", {"fill": MUTED})])
    if legend:
        text.line([("boxes: ", {"fill": MUTED})] + [(f"■ {name}  ", {"fill": color, "bold": True}) for name, color in legend])
    old_spans, new_spans = diff_spans(old_caption, new_caption)
    text.space(4)
    text.line([("Original caption: ", {"bold": True})] + old_spans)
    text.line([("Enriched caption: ", {"bold": True})] + new_spans, gap=10)
    if short_column and row.get(short_column) is not None:
        # A synthesized short column (enrich --synth_short) has no v1- copy: show it on its own.
        old_short, new_short = _caption(row.get(V1_PREFIX + short_column)) or "", _caption(row[short_column]) or ""
        if V1_PREFIX + short_column in row:
            old_spans, new_spans = diff_spans(old_short, new_short)
            text.line([("Original short: ", {"bold": True})] + old_spans)
            text.line([("Enriched short: ", {"bold": True})] + new_spans, gap=10)
        else:
            synthesized = new_short != old_caption and new_short != new_caption
            text.line([("Synthesized short: " if synthesized else "Short (copy of caption): ", {"bold": True}),
                       (new_short if synthesized else "-", {"fill": INSERTED if synthesized else MUTED})], gap=10)
    old_critique, critique = _critique(row.get(V1_PREFIX + "conversations")), _critique(row["conversations"])
    if critique and old_critique and critique != old_critique:
        old_spans, new_spans = diff_spans(old_critique, critique)
        text.line([("Original gpt turn: ", {"bold": True})] + old_spans)
        text.line([("Enriched gpt turn: ", {"bold": True})] + new_spans, gap=10)
    elif critique:
        text.line([("gpt turn (unchanged): ", {"bold": True}), (critique, {"fill": MUTED})], gap=10)

    if record:
        measured = record.get("measured") or {}
        objects = measured.get("objects") or {}
        if objects:
            parts = []
            for name, entry in objects.items():
                if "count" in entry:  # v2 records: detector vote
                    reliable = entry["count"] is not None
                    votes = f"GDINO {entry['n']}"
                    if "verify" in entry:  # before 2026-10-09: OWLv2 and Florence-2 voted too
                        votes += f", OWLv2 {entry['verify']}, Florence {entry.get('od') or '-'}"
                    if entry.get("sam3") is not None:  # v6 records: SAM 3 is the primary counter
                        votes = f"SAM3 {entry['sam3']}" + (f"+{entry['sam3_gray']}?" if entry.get("sam3_gray") else "") + ", " + votes
                    if entry["gray"]:
                        votes += f", {entry['gray']} borderline"
                    source = " [from Florence <OD>]" if entry.get("source") == "od" else ""
                    label = f"{name} = {entry['count']}" if reliable else f"{name} = ?"
                    if reliable and not entry.get("boxes_match"):
                        label += " (count only)"
                    parts.append((f"{label} ({votes}){source};  ", {"fill": TEXT if reliable else MUTED}))
                else:
                    reliable = entry["n"] > 0 and not entry["gray"] and entry["verify"] == entry["n"]
                    note = "" if reliable else f" (unreliable: {entry['gray']} borderline, OWLv2 {entry['verify']})"
                    parts.append((f"{name} {entry['n']}{note};  ", {"fill": TEXT if reliable else MUTED}))
            text.line([("Measured: ", {"bold": True})] + parts)
        regions = record.get("regions") or []
        if regions:
            text.line([("Regions (verified; Qwen's unverified proposals in grey): ", {"bold": True})], gap=0)
        for region, proposal in zip(regions, _proposals_by_region(record)):
            found = []
            if region.get("facing"):
                found.append(f"facing {region['facing']} (Orient-Anything)")
            for key in ("color", "material"):
                if region.get(key):
                    found.append(f"{key} {region[key]}")
            for part in region.get("parts") or []:
                found.append(f"{part['count']} {(part.get('color') or '') + ' '}{part['name']}" +
                             (f" on its {part['side']}" if part.get("side") else ""))
            for item in region.get("wearing") or []:
                found.append(f"wearing {(item.get('color') or '') + ' '}{item['item']}")
            if region.get("hair"):
                found.append(f"{region['hair']} hair")
            text.line([(f"  {region['ref']}: ", {"bold": True}),
                       ("; ".join(found) + "  " if found else "nothing verified  ", {"fill": INSERTED if found else MUTED}),
                       (f"| proposed: {proposal}" if proposal else "", {"fill": MUTED})], gap=0)
        if regions:
            text.space(4)
        extras = []
        absent = [name for name, is_absent in (measured.get("absent") or {}).items() if is_absent]
        if absent:
            extras.append((f"verified absent: {', '.join(absent)};  ", {"fill": MUTED}))
        if measured.get("texts"):
            extras.append((f"OCR: {' | '.join(measured['texts'])}", {"fill": MUTED}))
        if extras:
            text.line(extras)
        facts = record.get("facts") or []
        text.space(4)
        text.line([("Facts: ", {"bold": True}), ("none derived" if not facts else "", {"fill": MUTED})])
        order = {"used": 0, "skipped": 1, "vetoed": 2, "not selected": 3}
        for fact in sorted(facts, key=lambda f: order[fact_status(record, f)]):
            marker, color = FACT_STYLE[fact_status(record, fact)]
            text.line([(f"{marker} ", {"fill": color, "bold": True}), (f"{fact['type']}: ", {"fill": MUTED}),
                       (fact["text"], {"fill": color})], gap=0)
        if record.get("status") == "failed" and record.get("raw"):
            text.space(6)
            text.line([("Rejected rewrite (last attempt): ", {"bold": True, "fill": DELETED}),
                       (" ".join(record["raw"].split())[:600], {"fill": MUTED})])

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

def main(input_dir, output_dir, cache_dir, num_samples, seed):
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    cache_dir = cache_dir or os.path.join(input_dir, ".enrich_cache")
    cache = load_cache(cache_dir) if os.path.isdir(cache_dir) else {}
    if not cache:
        print(f"No enrich cache at {cache_dir}: drawing caption diffs only (no facts or boxes).")
    names = pq.ParquetFile(files[0]).schema_arrow.names
    short_column = next((c for c in SHORT_COLUMNS if c in names), None)

    selected, n_enriched, n_total = select_rows(files, num_samples, seed)
    print("=======================")
    print(f"Rows: {n_total}, enriched: {n_enriched}, drawing: {len(selected)}")
    print("=======================")
    os.makedirs(output_dir, exist_ok=True)

    status_counts, used_types, lines = collections.Counter(), collections.Counter(), {}
    for order, row in iter_rows(selected):
        old, new = _caption(row[V1_PREFIX + "conversations"]), _caption(row["conversations"])
        record = match_record(cache.get(row.get("id"), []), old, new)
        status = record["status"] if record else "no record"
        status_counts[status] += 1
        if record:
            by_id = {f.get("id"): f for f in record.get("facts") or [] if f.get("id")}
            for fact_id in (record.get("result") or {}).get("used") or []:
                if fact_id in by_id:
                    used_types[by_id[fact_id]["type"].split("_")[0]] += 1
        tag = "enriched" if old != new else "unchanged"
        name = f"sample_{order:02d}_{tag}.png"
        render_row(row, record, short_column).save(os.path.join(output_dir, name))
        added = [" ".join(new.split()[j1:j2]) for op, _, _, j1, j2
                 in difflib.SequenceMatcher(None, (old or "").split(), (new or "").split(), autojunk=False).get_opcodes()
                 if op in ("insert", "replace")]
        lines[order] = f"{name}  {row.get('id')}  [{status}]  + {' / '.join(added) if added else '-'}"
        if order == 0:
            print(f"Saved {os.path.join(output_dir, name)}")

    with open(os.path.join(output_dir, "summary.txt"), "w") as handle:
        handle.write(f"source: {input_dir}\ncache: {cache_dir}\n")
        handle.write(f"rows: {n_total}, enriched: {n_enriched}, drawn: {len(selected)}\n")
        handle.write(f"status of drawn rows: {dict(status_counts)}\n")
        handle.write(f"facts used in drawn rows, by type: {dict(used_types)}\n\n")
        handle.writelines(lines[k] + "\n" for k in sorted(lines))
    print(f"Done. Outputs in {output_dir}")
    print("=========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory of enriched parquet shards.")
    parser.add_argument("output")
    parser.add_argument("--cache_dir", default=None, help="Default: <input_dir>/.enrich_cache")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args.input_dir, args.output, args.cache_dir, args.num_samples, args.seed)
