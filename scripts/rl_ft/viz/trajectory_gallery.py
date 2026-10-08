"""Build an HTML gallery of the validation trajectories an RL run logged to wandb.

At every validation, train_grpo.py decodes the first ``--log_images`` leaves on
rank 0 and logs them as ``val/images``: one draft|refined strip per prompt,
captioned ``prompt: ... | AM0=... | r1: <critique> -> AM=...``. This script reads
those records straight from the run's local wandb folder (no network, no GPU),
adds each prompt's checklist from the validation JSONL, and writes a
self-contained folder:

    <out>/index.html      open in a browser, or publish as an artifact
    <out>/img/*.jpg       one 2:1 strip per (step, prompt)

Usage, from the repo root:

    python scripts/rl_ft/viz/trajectory_gallery.py wandb/run-<date>-<RUN_NAME>
    python scripts/rl_ft/viz/trajectory_gallery.py <run_dir> --out /some/dir --val_jsonl path/to/val.jsonl

By default the gallery goes to ``<output_dir from the run config>/gallery``,
next to the run's checkpoints. Needs PyYAML, Pillow and wandb (all in the
training env). Only rounds 0 and 1 are shown (the caption format of a
1-refinement chain).
"""

import argparse
import json
import os
import re
import sys

import yaml
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
CAPTION_RE = re.compile(r"^prompt: (.*?) \| AM0=([0-9.]+) \| r1: (.*) -> AM=([0-9.]+)$", re.S)
LOOKS_GOOD_RE = re.compile(r"looks good", re.IGNORECASE)
VAL_KEYS = ("val/am_0", "val/am_1", "val/am_final", "val/improved_1", "val/degraded_1", "val/looks_good_1")


def read_history(run_dir):
    """Yield (step, {key: value}) for every history record in the run's .wandb file."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    files = [f for f in os.listdir(run_dir) if f.endswith(".wandb")]
    if not files:
        sys.exit(f"No .wandb file in {run_dir}")
    ds = datastore.DataStore()
    ds.open_for_scan(os.path.join(run_dir, files[0]))
    while True:
        data = ds.scan_data()
        if data is None:
            return
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        items = {(i.key or "/".join(i.nested_key)): json.loads(i.value_json) for i in rec.history.item}
        yield items.get("_step"), items


def run_config(run_dir):
    path = os.path.join(run_dir, "files", "config.yaml")
    if not os.path.isfile(path):
        return {}
    cfg = yaml.safe_load(open(path)) or {}
    return {k: v.get("value") for k, v in cfg.items() if isinstance(v, dict) and "value" in v}


def val_rows(val_path):
    """Prompt -> row, from a val .jsonl or a data yaml listing .jsonl files."""
    if val_path.endswith((".yaml", ".yml")):
        cfg = yaml.safe_load(open(val_path))
        paths = []
        for ds in cfg.get("datasets", []):
            jp = ds.get("json_path")
            paths.extend(jp if isinstance(jp, list) else [jp] if jp else [])
    else:
        paths = [val_path]
    rows = {}
    for path in paths:
        for line in open(os.path.expanduser(path)):
            if line.strip():
                d = json.loads(line)
                rows[d["prompt"].strip()] = d
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="the run's wandb folder, e.g. wandb/run-20261007_014828-darshan_rl_2gpu_mila")
    ap.add_argument("--out", default=None, help="output folder (default: <run output_dir>/gallery)")
    ap.add_argument("--val_jsonl", default=None, help="validation .jsonl or data yaml (default: the run's eval_data_path)")
    ap.add_argument("--width", type=int, default=896, help="width of each saved strip in pixels")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    cfg = run_config(run_dir)
    run_name = cfg.get("run_name") or os.path.basename(run_dir).split("-", 2)[-1]

    def from_repo(p):
        return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)

    out = args.out or (from_repo(os.path.join(cfg["output_dir"], "gallery")) if cfg.get("output_dir")
                       else os.path.join(run_dir, "gallery"))
    os.makedirs(os.path.join(out, "img"), exist_ok=True)

    val_path = args.val_jsonl or (from_repo(cfg["eval_data_path"]) if cfg.get("eval_data_path") else None)
    rows = val_rows(val_path) if val_path and os.path.exists(val_path) else {}
    if not rows:
        print("warning: no validation file found, checklists will be empty (pass --val_jsonl)")
    n_val = len(rows)
    if cfg.get("eval_max_prompts"):
        n_val = min(n_val, int(cfg["eval_max_prompts"])) if n_val else int(cfg["eval_max_prompts"])

    steps, glob, prompts, order = [], {}, {}, []
    height = args.width // 2
    for step, items in read_history(run_dir):
        if "val/images/filenames" not in items:
            continue
        steps.append(step)
        glob[str(step)] = {k: items[k] for k in VAL_KEYS if k in items}
        for fn, cap in zip(items["val/images/filenames"], items["val/images/captions"]):
            m = CAPTION_RE.match(cap)
            if not m:
                print(f"warning: step {step}: caption not in the 1-refinement format, skipped: {cap[:80]!r}")
                continue
            prompt, am0, refl, am1 = m.group(1).strip(), float(m.group(2)), m.group(3).strip(), float(m.group(4))
            if prompt not in prompts:
                prompts[prompt] = {}
                order.append(prompt)
            rel = f"img/s{step:04d}_p{order.index(prompt)}.jpg"
            Image.open(os.path.join(run_dir, "files", fn)).convert("RGB") \
                 .resize((args.width, height), Image.LANCZOS).save(os.path.join(out, rel), quality=82, optimize=True)
            prompts[prompt][str(step)] = {"img": rel, "am0": am0, "am1": am1, "refl": refl,
                                          "lg": bool(LOOKS_GOOD_RE.search(refl))}
    if not steps:
        sys.exit(f"No val/images records in {run_dir} (was the run launched with --log_images > 0?)")

    # Keep prompts that appear at every logged step, so the strips line up.
    keep = [p for p in order if len(prompts[p]) == len(steps)]
    if len(keep) < len(order):
        print(f"note: {len(order) - len(keep)} prompt(s) missing at some steps were left out")
    data = {
        "run": run_name, "n_val": n_val, "steps": steps, "global": glob,
        "prompts": [{"prompt": p,
                     "checklist": [[q, a, s] for (q, a), s in zip(rows[p]["vqa_list"], rows[p].get("skills") or ["?"] * len(rows[p]["vqa_list"]))]
                     if p in rows else [],
                     "steps": prompts[p]} for p in keep],
    }
    tpl = open(os.path.join(HERE, "gallery_template.html")).read()
    page = tpl.replace("{{TITLE}}", f"{run_name} trajectories").replace("{{DATA}}", json.dumps(data).replace("</", "<\\/"))
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write(page)
    print(f"{len(keep)} prompts x {len(steps)} evaluations -> {os.path.join(out, 'index.html')}")


if __name__ == "__main__":
    main()
