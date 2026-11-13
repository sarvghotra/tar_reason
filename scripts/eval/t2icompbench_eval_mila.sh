#!/bin/bash
# Mila (singhsd) wrapper around the official T2I-CompBench++ evaluators (Karine-Huang/T2I-CompBench, checkout in
# $T2I below, scripts run unchanged): score one category of already-generated images. No generation.
#   color / shape / texture  BLIPvqa_eval/BLIP_vqa.py           -> annotation_blip/vqa_result.json
#   spatial (2D)             UniDet_eval/2D_spatial_eval.py      -> labels/annotation_obj_detection_2d/vqa_result.json
#   3d_spatial               UniDet_eval/3D_spatial_eval.py      -> labels/depth/ + labels/annotation_obj_detection_3d/
#   numeracy                 UniDet_eval/numeracy_eval.py        -> annotation_num/vqa_result.json
#   non_spatial              CLIPScore_eval/CLIP_similarity.py   -> annotation_clip/vqa_result.json
#   complex                  BLIP_vqa + 2D_spatial --complex True + CLIP_similarity --complex True, then
#                            3_in_1_eval/3_in_1.py               -> annotation_3_in_1/vqa_result.json
# Score = mean of the per-image "answer" in that vqa_result.json (what each official script prints/saves).
#
# IMG_DIR must hold samples/<prompt>_<6-digit index>.png and nothing else in samples/ (question_id = the index;
# the prompt is the text before the "_", so prompts must not contain "_" or "/"; name files with the prompt line
# exactly as in examples/dataset/<cat>_val.txt, trailing "." included). Official protocol: the 300 val prompts x
# 10 images, index = prompt_idx*10 + image_idx. Outputs land in IMG_DIR (the scripts' own subfolders, cleared
# first) plus IMG_DIR/score_<category>.json = {category, score, n_images, evaluator}.
# complex: 3_in_1.py hard-codes 10 images per prompt and reads its prompt list from --data_path/complex_val.txt;
# we pass a data dir built from IMG_DIR's prompts (IMG_DIR/3_in_1_data/, official spatial/action lists copied).
# Other layouts (not 10 consecutive images per prompt) fall back to the same per-image formula keyed on each
# file's prompt (evaluator field says which ran).
#
#   CATEGORY=color IMG_DIR=/path/to/run sbatch scripts/eval/t2icompbench_eval_mila.sh
#   sbatch scripts/eval/t2icompbench_eval_mila.sh complex /path/to/run          # positional form
#
# Env ~/envs/t2icompbench (uv, py3.9, torch 2.0.1+cu118, detectron2 @5aeb252); weights in
# $T2I/UniDet_eval/experts/expert_weights; BLIP ckpt in $TORCH_HOME, CLIP ViT-B/32 in ~/.cache/clip.
#
#SBATCH --time=2:00:00
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --job-name=t2icb_eval
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.out

set -euo pipefail
CATEGORY=${1:-${CATEGORY:-}}
IMG_DIR=${2:-${IMG_DIR:-}}
T2I=/network/scratch/s/singhsd/benchmarks/T2I-CompBench
PY=/home/mila/s/singhsd/envs/t2icompbench/bin/python
export HF_HOME=/network/scratch/s/singhsd/hf_home TORCH_HOME=/network/scratch/s/singhsd/hf_home/torch
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1   # the checkout tracks its .pyc files
unset PYTHONPATH

case "$CATEGORY" in color|shape|texture|spatial|3d_spatial|non_spatial|numeracy|complex) ;;
  *) echo "CATEGORY must be one of color shape texture spatial 3d_spatial non_spatial numeracy complex (got '$CATEGORY')"; exit 1;;
esac
[ -n "$IMG_DIR" ] && [ -d "$IMG_DIR/samples" ] || { echo "IMG_DIR/samples not found (IMG_DIR='$IMG_DIR')"; exit 1; }
IMG_DIR=$(realpath "$IMG_DIR")   # absolute, no trailing "/" (3D_spatial_eval.py builds depth paths from it)
echo "CATEGORY=${CATEGORY} IMG_DIR=${IMG_DIR} host=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# File-name check: every official loader parses <prompt>_<index>.png (UniDet uses split("_")[1]).
N_IMAGES=$($PY - "$IMG_DIR/samples" <<'EOF'
import os, re, sys
names = os.listdir(sys.argv[1])
bad = [n for n in names if not re.fullmatch(r"[^_/]+_\d{6}\.png", n)]
if bad:
    sys.exit(f"{len(bad)} file(s) in samples/ do not match <prompt>_<6 digits>.png, e.g. {bad[:3]}")
idx = sorted(int(n[-10:-4]) for n in names)
if len(set(idx)) != len(idx):
    sys.exit("duplicate indices in samples/")
if idx != list(range(len(idx))):
    print("WARNING: indices are not 0..N-1; BLIP/CLIP/3_in_1 number question_ids by sorted position", file=sys.stderr)
print(len(names))
EOF
)
echo "n_images=${N_IMAGES}"

run_blip()   { rm -rf "$IMG_DIR"/annotation[0-9]*_blip "$IMG_DIR/annotation_blip"
               (cd "$T2I/BLIPvqa_eval" && $PY BLIP_vqa.py --out_dir="$IMG_DIR"); }
run_2d()     { rm -rf "$IMG_DIR/labels/annotation_obj_detection_2d"
               (cd "$T2I/UniDet_eval" && $PY 2D_spatial_eval.py --outpath "$IMG_DIR" "$@"); }
run_clip()   { rm -rf "$IMG_DIR/annotation_clip"
               (cd "$T2I" && $PY CLIPScore_eval/CLIP_similarity.py --outpath "$IMG_DIR" "$@"); }
run_3d()     { rm -rf "$IMG_DIR/labels/depth" "$IMG_DIR/labels/annotation_obj_detection_3d"   # depth is skipped if present
               (cd "$T2I/UniDet_eval" && $PY 3D_spatial_eval.py --outpath "$IMG_DIR"); }
run_num()    { rm -rf "$IMG_DIR/annotation_num"
               (cd "$T2I/UniDet_eval" && $PY numeracy_eval.py --outpath "$IMG_DIR"); }
timed()      { local t0=$(date +%s); "$@"; echo "[time] $1: $(( $(date +%s) - t0 ))s for ${N_IMAGES} images"; }

T_START=$(date +%s)
case "$CATEGORY" in
  color|shape|texture) timed run_blip; RESULT=annotation_blip/vqa_result.json; EVALUATOR="BLIPvqa_eval/BLIP_vqa.py" ;;
  spatial)     timed run_2d;   RESULT=labels/annotation_obj_detection_2d/vqa_result.json; EVALUATOR="UniDet_eval/2D_spatial_eval.py" ;;
  3d_spatial)  timed run_3d;   RESULT=labels/annotation_obj_detection_3d/vqa_result.json; EVALUATOR="UniDet_eval/3D_spatial_eval.py" ;;
  numeracy)    timed run_num;  RESULT=annotation_num/vqa_result.json; EVALUATOR="UniDet_eval/numeracy_eval.py" ;;
  non_spatial) timed run_clip; RESULT=annotation_clip/vqa_result.json; EVALUATOR="CLIPScore_eval/CLIP_similarity.py" ;;
  complex)
    # argparse type=bool: any non-empty value means True, so only ever pass "True" (never "False")
    timed run_blip
    timed run_2d --complex True
    timed run_clip --complex True
    rm -rf "$IMG_DIR/annotation_3_in_1" "$IMG_DIR/3_in_1_data"
    MODE=$($PY - "$IMG_DIR" "$T2I/examples/dataset" <<'EOF'
import os, sys, json, shutil
import numpy as np
img_dir, ds = sys.argv[1], sys.argv[2]
names = sorted(os.listdir(f"{img_dir}/samples"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
prompts = [n.split("_")[0] for n in names]
groups = [prompts[i:i + 10] for i in range(0, len(prompts), 10)]
if len(prompts) % 10 == 0 and all(len(set(g)) == 1 for g in groups):
    # official layout: 10 consecutive images per prompt -> run 3_in_1.py unchanged on our prompt list
    d = f"{img_dir}/3_in_1_data"; os.makedirs(d)
    with open(f"{d}/complex_val.txt", "w") as f:
        f.write("".join(g[0] + "\n" for g in groups))
    for fn in ("complex_val_spatial.txt", "complex_val_action.txt"):
        shutil.copy(f"{ds}/{fn}", d)
    print("official")
    sys.exit()
# fallback: same formula as 3_in_1.py, per image, keyed on the image's own prompt
L = lambda fn: [i.strip("\n").split(".")[0].lower() for i in open(f"{ds}/{fn}")]
spatial, action = L("complex_val_spatial.txt"), L("complex_val_action.txt")
rd = lambda p: np.array([float(i["answer"]) for i in json.load(open(f"{img_dir}/{p}"))])
attr, spa, act = rd("annotation_blip/vqa_result.json"), rd("labels/annotation_obj_detection_2d/vqa_result.json"), rd("annotation_clip/vqa_result.json")
out = []
for k, p in enumerate(prompts):
    key = p.split(".")[0].lower()
    if key in spatial:  s = (spa[k] + attr[k]) * 0.5
    elif key in action: s = (act[k] + attr[k]) * 0.5
    else:               s = (attr[k] + spa[k] + act[k]) / 3
    out.append({"question_id": k, "answer": float(s)})
os.makedirs(f"{img_dir}/annotation_3_in_1")
json.dump(out, open(f"{img_dir}/annotation_3_in_1/vqa_result.json", "w"))
print("fallback")
EOF
)
    if [ "$MODE" = "official" ]; then
      (cd "$T2I" && $PY 3_in_1_eval/3_in_1.py --outpath "$IMG_DIR" --data_path "$IMG_DIR/3_in_1_data")
      EVALUATOR="BLIP_vqa + 2D_spatial_eval --complex True + CLIP_similarity --complex True -> 3_in_1_eval/3_in_1.py"
    else
      echo "complex: images are not 10 consecutive per prompt; 3_in_1 formula applied per image by the wrapper"
      EVALUATOR="BLIP_vqa + 2D_spatial_eval --complex True + CLIP_similarity --complex True -> 3_in_1 formula (wrapper, per-image prompt)"
    fi
    RESULT=annotation_3_in_1/vqa_result.json ;;
esac
echo "[time] total: $(( $(date +%s) - T_START ))s for ${N_IMAGES} images"

$PY - "$IMG_DIR" "$CATEGORY" "$RESULT" "$EVALUATOR" <<'EOF'
import json, sys
img_dir, cat, result, evaluator = sys.argv[1:]
r = json.load(open(f"{img_dir}/{result}"))
score = sum(float(i["answer"]) for i in r) / len(r)
out = {"category": cat, "score": score, "n_images": len(r), "evaluator": evaluator}
json.dump(out, open(f"{img_dir}/score_{cat}.json", "w"), indent=1)
print(f"T2I-CompBench {cat}: {score:.4f}  (n_images={len(r)}, {img_dir}/score_{cat}.json)")
EOF
echo "================ DONE ============="
