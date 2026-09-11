#!/bin/bash
source "${TAR_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/cluster_env.sh"
tar_init_run "generation_benchmarks"
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL}"
N_GPUS=${N_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
DPG_PROMPTS="${DPG_PROMPTS:-$DATA_ROOT/dpg_bench/prompts}"
GENEVAL_PROMPTS="${GENEVAL_PROMPTS:-$DATA_ROOT/geneval/prompts/evaluation_metadata.jsonl}"
tar_require_paths "$MODEL_PATH/config.json" "$AR_MODEL" "$VISION_MODEL" "$DECODER" "$DPG_PROMPTS" "$GENEVAL_PROMPTS"
# Produces benchmark images. External benchmark scoring is a separate step.
tar_launch torchrun --standalone --nproc_per_node="$N_GPUS" \
    eval/eval_dpg_bench.py \
    --model "$MODEL_PATH" --prompts "$DPG_PROMPTS" \
    --ar_path "$AR_MODEL" --encoder_path "$VISION_MODEL" --decoder_path "$DECODER" \
    --save_dir "$EVAL_DIR/dpgbench" "$@"
tar_launch torchrun --standalone --nproc_per_node="$N_GPUS" \
    eval/eval_geneval.py \
    --model "$MODEL_PATH" --prompts "$GENEVAL_PROMPTS" \
    --ar_path "$AR_MODEL" --encoder_path "$VISION_MODEL" --decoder_path "$DECODER" \
    --save_dir "$EVAL_DIR/geneval" "$@"
