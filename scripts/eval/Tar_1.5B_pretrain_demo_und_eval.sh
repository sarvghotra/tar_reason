#!/bin/bash
source "${TAR_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/cluster_env.sh"
tar_init_run "understanding"
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL}"
N_GPUS=${N_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
export POOL_SCALE="${POOL_SCALE:-1}"
tar_require_paths "$MODEL_PATH/config.json" "$VISION_MODEL"
if [[ "${TAR_DRY_RUN:-0}" != 1 ]]; then
    python -c 'import lmms_eval' || { echo 'lmms_eval is not installed in tar; see scripts/CLUSTER.md.' >&2; exit 1; }
fi
tar_launch torchrun --standalone --nproc_per_node="$N_GPUS" \
    -m lmms_eval \
    --model llava_onevision \
    --model_args "pretrained=${MODEL_PATH},conv_template=qwen_1_5,model_name=llava_qwen" \
    --tasks "${TASKS:-mme}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path "$EVAL_DIR" \
    "$@"
