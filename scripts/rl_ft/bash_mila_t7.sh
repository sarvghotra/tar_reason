#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --open-mode=append
#SBATCH -J darshan_t7_mila
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.err

# Single-node Mila port of the collaborator's rl_ft_oracle_t7 (Fir, 2-4 nodes x 4 H100).
# GRPO refiner training under frozen hybrid critiques with GenEval2's own pixel judge
# (Qwen3-VL-8B on decoded PNGs). Same trainer flags as t7; differences, all forced
# by the hardware or the data we have:
#   * 1 node x 4 A100 80GB: 3 training ranks + 1 judge GPU (t7's JUDGE_COLOCATE=0
#     layout on one node), so 3 x 6 = 18 prompts/step (t7: 36 on 2 nodes, 72 on 4).
#   * short-unkillable caps a job at 3 h: chain jobs (see below); training resumes
#     from the latest checkpoint-N. SAVE_STEPS=5 (t7: 50) so a time-limit kill loses
#     at most ~5 steps.
#   * Validation is t7's 800 prompts (~1.5 h per validation on 3 ranks). A job
#     killed mid-validation has the next job redo it (eval_done_<step> markers).
#     logging_steps 1 (t7: 8), which does not affect training.
#
# Submit a chain of N jobs (each starts when the previous ends, for any reason):
#   cd ~/CODE/LatentDCR/tar_reason
#   jid=$(sbatch --parsable scripts/rl_ft/bash_mila_t7.sh)
#   for i in $(seq 2 N); do jid=$(sbatch --parsable --dependency=afterany:$jid scripts/rl_ft/bash_mila_t7.sh); done
# Once MAX_STEPS is reached the remaining jobs start, see nothing to do, and exit.
# Cancel the whole chain: scancel --name=darshan_t7_mila --user=$USER
#
# Smoke test (one short job): MAX_STEPS=2 EVAL_STEPS=2 SAVE_STEPS=2 EVAL_MAX_PROMPTS=12 \
#   RUN_NAME=t7_smoke_mila sbatch --time=01:00:00 scripts/rl_ft/bash_mila_t7.sh

REPO=/home/mila/s/singhsd/CODE/LatentDCR/tar_reason
MODELS=/network/scratch/s/singhsd/models/tar
# The collaborator's leak-free copy (Fir /scratch/jeet/tmp/data/T2I_datasets/geneval2_50K, 2026-10-08):
# train = first 49,000 rows of the old 49,488-row file, no overlap with val800/val256.
DATA=/network/scratch/s/singhsd/data/geneval2_50K_v2
TRAIN_JSONL=${TRAIN_JSONL:-${DATA}/evaluation_metadata_shuf_train.jsonl}
VAL_JSONL=${VAL_JSONL:-${DATA}/evaluation_metadata_shuf_val800.jsonl}   # t7's val set
JUDGE_MODEL=/network/scratch/s/singhsd/models/vlm/Qwen3-VL-8B-Instruct
JUDGE_PY=/home/mila/s/singhsd/envs/qwen_judge/bin/python

RUN_NAME=${RUN_NAME:-darshan_t7_mila}
LOCAL_DIR="output_dir/${RUN_NAME}"
DATA_PATH="output_dir/${RUN_NAME}/data.yaml"
VAL_DATA_PATH="output_dir/${RUN_NAME}/val.yaml"

# ===================== Config params (t7 values) ========================
N_GPUS=4
JUDGE_GPU=3                       # last GPU judges and nothing else
TRAIN_GPUS=0,1,2
N_TRAIN_GPUS=3

PREV_STAGE_CHECKPOINT=${MODELS}/sft/slf_ref_edit_t21_ckpt17000

REWARD_QUESTION_BATCH=32
REWARD_IMAGES_PER_REQ=8
REWARD_DECODE_BATCH=24
REWARD_LOAD_TIMEOUT=1800

PROMPTS_PER_GPU=${PROMPTS_PER_GPU:-6}
BRANCH=${BRANCH:-2,4}
EVAL_BRANCH=1,1
GEN_BATCH_SIZE=16
TRAIN_MICRO_BATCH=2

LR=1e-4
LORA_R=64
LORA_ALPHA=128
KL_COEF=0.05
CLIP_EPS=0.2
ALPHA=1.0
NUM_PPO_EPOCHS=2
REFLECT_TOKEN_WEIGHT=1.0
GROUP=parent
ADV_NORM=std

CRITIQUE_SOURCE=hybrid
HYBRID_ORACLE_FRAC=0.5
ORACLE_FAIL_THRESHOLD=0.5

BREAK_WEIGHT=1.0
DRAFT_REWARD=children
STOP_PENALTY=0.5
STOP_THRESHOLD=0.8
MIN_REFINES=0
REFINE_RESAMPLE_TRIES=3

REFLECT_TOKENS=128
REFLECT_TEMPERATURE=1.0
ANSWER_SUFFIX=geneval2

MAX_STEPS=${MAX_STEPS:-300}
EVAL_STEPS=${EVAL_STEPS:-50}
SAVE_STEPS=${SAVE_STEPS:-5}
EVAL_MAX_PROMPTS=${EVAL_MAX_PROMPTS:-}     # empty = full val set
WARMUP_RATIO=0.05
LOG_IMAGES=8
SEED=421
# Array jobs (sbatch --array=...): offset the base seed by the array task id.
# Chained jobs of one run have no array id, so they all keep the same value and
# resume the same prompt stream.
DATASET_SEED=$(( ${DATASET_SEED:-19} + ${SLURM_ARRAY_TASK_ID:-0} ))
echo "DATASET_SEED: ${DATASET_SEED}"
# ===================== Config params END =====================

cd ${REPO}
export PYTHONPATH=$PYTHONPATH:${REPO}/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_NAME=$RUN_NAME
export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}
export WANDB_PROJECT="tar_reasoning"
export WANDB_ENTITY="darshan-singh"
export WANDB_MODE="${WANDB_MODE:-online}"
export SIGLIP2_PATH=${MODELS}/siglip2-so400m-patch14-384   # TA-Tok encoder config

[ -f "${VAL_JSONL}" ] || { echo "Validation file not found: ${VAL_JSONL}" >&2; exit 1; }
mkdir -p "$LOCAL_DIR"
if [ ! -f "$DATA_PATH" ]; then
    printf 'datasets:\n    - json_path:\n      - %s\n      ratio: 1\n' "${TRAIN_JSONL}" > "$DATA_PATH"
fi
if [ ! -f "$VAL_DATA_PATH" ]; then
    printf 'datasets:\n    - json_path:\n      - %s\n      ratio: 1\n' "${VAL_JSONL}" > "$VAL_DATA_PATH"
fi

# Stop early if a previous job in the chain already finished the run.
LAST=$(ls -d ${LOCAL_DIR}/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
if [ -n "$LAST" ] && [ "$LAST" -ge "$MAX_STEPS" ]; then
    echo "checkpoint-${LAST} >= MAX_STEPS=${MAX_STEPS}: run already complete, nothing to do."
    exit 0
fi

TMP_DIR=${SLURM_TMPDIR:-/tmp}
[ -f ${TMP_DIR}/ta_tok.pth ] || cp ${MODELS}/ta_tok.pth ${TMP_DIR}/
AR_MODEL=${MODELS}/ar_dtok_lp_512px.pth
DECODER=${MODELS}/vq_ds16_t2i.pt
ENCODER=${TMP_DIR}/ta_tok.pth

echo "RUN_NAME=${RUN_NAME} job=${SLURM_JOB_ID} host=$(hostname) resume_from=${LAST:-none}"
echo "train GPUs ${TRAIN_GPUS}, judge GPU ${JUDGE_GPU}; prompts/gpu ${PROMPTS_PER_GPU}, branch ${BRANCH}, prompts/step $((N_TRAIN_GPUS * PROMPTS_PER_GPU))"

# ---------------- Pixel judge (its own env, started before the CUDA module) ----------------
# The judge env's torch 2.6 ships its own CUDA 12.4 libraries; starting it before
# `module load cuda` keeps the module's older libnvJitLink off its library path.
REWARD_PORT=$((8000 + ${SLURM_JOB_ID:-0} % 1000))
REWARD_URL="http://127.0.0.1:${REWARD_PORT}"
REWARD_LOG="${LOCAL_DIR}/reward_server_${SLURM_JOB_ID:-local}.log"
READY_FILE="${LOCAL_DIR}/.reward_ready_${SLURM_JOB_ID:-local}"
rm -f "${READY_FILE}"
CUDA_VISIBLE_DEVICES=${JUDGE_GPU} ${JUDGE_PY} llava/train/rl/pixel_reward_server.py \
    --model "${JUDGE_MODEL}" --host 127.0.0.1 --port ${REWARD_PORT} \
    --batch_size ${REWARD_QUESTION_BATCH} --answer_suffix ${ANSWER_SUFFIX} \
    --answer_id_mode geneval2 --attn_implementation sdpa \
    --ready_file "${READY_FILE}" > "${REWARD_LOG}" 2>&1 &
REWARD_PID=$!
trap "kill ${REWARD_PID} 2>/dev/null" EXIT INT TERM
waited=0
until [ -f "${READY_FILE}" ]; do
    if ! kill -0 ${REWARD_PID} 2>/dev/null; then
        echo "judge died while loading; tail of ${REWARD_LOG}:" >&2; tail -40 "${REWARD_LOG}" >&2; exit 1
    fi
    if [ ${waited} -ge ${REWARD_LOAD_TIMEOUT} ]; then
        echo "judge not ready after ${REWARD_LOAD_TIMEOUT}s; tail of ${REWARD_LOG}:" >&2; tail -40 "${REWARD_LOG}" >&2; exit 1
    fi
    sleep 10; waited=$((waited + 10))
done
echo "judge ready after ${waited}s at ${REWARD_URL}"

# ---------------- Training ----------------
source ~/envs/tar/bin/activate
module load cuda/12.2.2

EVAL_ARGS=()
[ -n "${EVAL_MAX_PROMPTS}" ] && EVAL_ARGS+=(--eval_max_prompts "${EVAL_MAX_PROMPTS}")

CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} \
torchrun --standalone --nproc_per_node=${N_TRAIN_GPUS} \
    llava/train/rl/train_grpo.py \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --attn_implementation flash_attention_2 \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --data_path ${DATA_PATH} \
    --eval_data_path ${VAL_DATA_PATH} \
    --prompts_per_gpu ${PROMPTS_PER_GPU} \
    --branch ${BRANCH} \
    --eval_branch ${EVAL_BRANCH} \
    --scale 0 \
    --gen_seq_len 729 \
    --img_temperature 1.0 \
    --img_top_k 1200 \
    --img_top_p 0.95 \
    --reflect_tokens ${REFLECT_TOKENS} \
    --reflect_temperature ${REFLECT_TEMPERATURE} \
    --reflect_top_k 0 \
    --reflect_top_p 0.95 \
    --gen_batch_size ${GEN_BATCH_SIZE} \
    --alpha ${ALPHA} \
    --answer_suffix ${ANSWER_SUFFIX} \
    --reward_batch_size 16 \
    --reward_kind pixel \
    --reward_server_url ${REWARD_URL} \
    --reward_images_per_request ${REWARD_IMAGES_PER_REQ} \
    --reward_decode_batch ${REWARD_DECODE_BATCH} \
    --draft_reward ${DRAFT_REWARD} \
    --stop_penalty ${STOP_PENALTY} \
    --stop_threshold ${STOP_THRESHOLD} \
    --min_refines ${MIN_REFINES} \
    --refine_resample_tries ${REFINE_RESAMPLE_TRIES} \
    --critique_source ${CRITIQUE_SOURCE} \
    --hybrid_oracle_frac ${HYBRID_ORACLE_FRAC} \
    --oracle_fail_threshold ${ORACLE_FAIL_THRESHOLD} \
    --break_weight ${BREAK_WEIGHT} \
    --eval_oracle \
    --group ${GROUP} \
    --adv_norm ${ADV_NORM} \
    --clip_eps ${CLIP_EPS} \
    --kl_coef ${KL_COEF} \
    --reflect_token_weight ${REFLECT_TOKEN_WEIGHT} \
    --num_ppo_epochs ${NUM_PPO_EPOCHS} \
    --train_micro_batch ${TRAIN_MICRO_BATCH} \
    --learning_rate ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type constant \
    --max_grad_norm 1.0 \
    --max_steps ${MAX_STEPS} \
    --output_dir ${LOCAL_DIR} \
    --eval_steps ${EVAL_STEPS} \
    "${EVAL_ARGS[@]}" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 5 \
    --logging_steps 1 \
    --report_to wandb \
    --run_name ${RUN_NAME} \
    --seed ${SEED} \
    --dataset_seed ${DATASET_SEED} \
    --log_images ${LOG_IMAGES} \
    --ar_path ${AR_MODEL} \
    --encoder_path ${ENCODER} \
    --decoder_path ${DECODER} \
    --cfg_scale 4.0
