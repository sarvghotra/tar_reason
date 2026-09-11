#!/bin/bash
source "${TAR_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/cluster_env.sh"
tar_init_run "one_shot"
MODEL_PATH="${MODEL_PATH:-$SFT_MODEL}"
PROMPTS_FILE="${PROMPTS_FILE:-$DATA_ROOT/T2I_datasets/geneval2_50K/evaluation_metadata_shuf_val256.jsonl}"
tar_require_paths "$MODEL_PATH/config.json" "$PROMPTS_FILE" "$AR_MODEL" "$VISION_MODEL" "$DECODER"
tar_launch python tts/eval/1shot_gen.py \
    --model "$MODEL_PATH" \
    --out_dir "$EVAL_DIR" \
    --prompts_file "$PROMPTS_FILE" \
    --generate_images \
    --ar_path "$AR_MODEL" \
    --decoder_path "$DECODER" \
    --encoder_path "$VISION_MODEL" \
    "$@"
