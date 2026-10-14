'''
Sheets of the replies the critic rejected (generate_critic_hf_parquet.py --rejections_log): per row the image, the
caption and detector counts, then every rejected attempt's Self-reflect, Correction and the reason, and whether the row
was finally kept or dropped. One PNG per row in <output_dir>.

    python util_scripts/create_data/gen_critic/viz_rejections.py <rejections.jsonl> <input_dir> <output_dir>
'''

import argparse
import glob
import io
import json
import os
import textwrap
from collections import OrderedDict

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
IMAGE_SIZE = 512
TEXT_WIDTH = 900
WRAP = 105  # characters per line at 15 px
MARGIN = 16


def load_rejections(path):
    """caption -> {"counts": str, "attempts": [(attempt, reply dict | raw str, reason)]}, in log order."""
    rows = OrderedDict()
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            caption, _, counts = record["caption"].partition("\n")
            caption = caption.removeprefix("Caption: ")
            reply = record["reply"]
            start, end = reply.find("{"), reply.rfind("}")
            try:
                reply = json.loads(reply[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
            entry = rows.setdefault(caption, {"counts": counts, "attempts": []})
            entry["attempts"].append((record["attempt"], reply, record["reason"]))
    return rows


def find_images(input_dir, captions):
    """caption -> PIL image, from the input shards' human turns."""
    wanted, found = set(captions), {}
    for path in sorted(glob.glob(os.path.join(input_dir, "*.parquet"))):
        table = pq.read_table(path, columns=["conversations", "image"])
        for conversations, image in zip(table.column("conversations").to_pylist(), table.column("image").to_pylist()):
            caption = next((turn["value"] for turn in conversations if turn["from"] == "human"), None)
            if caption in wanted and caption not in found:
                found[caption] = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if len(found) == len(wanted):
            break
    return found


def text_blocks(caption, entry, max_retries):
    """[(text, font, color)] lines for the right-hand column."""
    final = any(attempt >= max_retries for attempt, _, _ in entry["attempts"])
    blocks = [("DROPPED" if final else "kept after retry", "bold", (180, 0, 0) if final else (0, 120, 0)),
              ("Caption: " + caption, "bold", (0, 0, 0))]
    if entry["counts"]:
        blocks.append((entry["counts"], "regular", (0, 70, 160)))
    for attempt, reply, reason in entry["attempts"]:
        blocks.append((f"--- attempt {attempt}", "bold", (90, 90, 90)))
        if isinstance(reply, dict):
            for key, title in (("self_reflect", "Self-reflect"), ("correction", "Correction")):
                items = reply.get(key)
                if isinstance(items, list) and items:
                    blocks.append((title + ":\n" + "\n".join(f"{n}. {item}" for n, item in enumerate(items, 1)),
                                   "regular", (0, 0, 0)))
                elif key in reply:
                    blocks.append((f"{title}: {json.dumps(items)}", "regular", (0, 0, 0)))
            if not reply:
                blocks.append(("Reply: {}", "regular", (0, 0, 0)))
        else:
            blocks.append(("Reply: " + str(reply)[:600], "regular", (0, 0, 0)))
        blocks.append(("REJECTED: " + reason, "regular", (180, 0, 0)))
    return blocks


def render(image, blocks):
    fonts = {"regular": ImageFont.truetype(FONT, 15), "bold": ImageFont.truetype(BOLD, 15)}
    lines = []
    for text, font, color in blocks:
        for paragraph in text.split("\n"):
            for line in textwrap.wrap(paragraph, WRAP) or [""]:
                lines.append((line, fonts[font], color))
        lines.append(("", fonts["regular"], color))
    height = max(IMAGE_SIZE + 2 * MARGIN, len(lines) * 19 + 2 * MARGIN)
    sheet = Image.new("RGB", (IMAGE_SIZE + TEXT_WIDTH + 3 * MARGIN, height), "white")
    if image is not None:
        sheet.paste(image.resize((IMAGE_SIZE, IMAGE_SIZE)), (MARGIN, MARGIN))
    draw = ImageDraw.Draw(sheet)
    y = MARGIN
    for line, font, color in lines:
        draw.text((IMAGE_SIZE + 2 * MARGIN, y), line, font=font, fill=color)
        y += 19
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rejections")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--max_retries", type=int, default=2, help="The run's --max_retries: a row rejected at this "
                                                                    "attempt was dropped.")
    args = parser.parse_args()
    rows = load_rejections(args.rejections)
    images = find_images(args.input_dir, list(rows))
    os.makedirs(args.output_dir, exist_ok=True)
    for index, (caption, entry) in enumerate(rows.items(), 1):
        final = any(attempt >= args.max_retries for attempt, _, _ in entry["attempts"])
        name = f"{index:03d}_{'dropped' if final else 'kept'}.png"
        render(images.get(caption), text_blocks(caption, entry, args.max_retries)).save(os.path.join(args.output_dir, name))
    print(f"{len(rows)} sheets in {args.output_dir} ({len(images)} images found)")


if __name__ == "__main__":
    main()
