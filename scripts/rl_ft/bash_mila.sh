#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --open-mode=append
#SBATCH -J darshan_rl_2gpu_mila
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.err

# Mila version of bash_darshan.sh (GRPO on generate -> self-reflect -> refine with
# the latent GenEval2-style VQA reward, llava/train/rl/train_grpo.py).
#
# Batch (2x A100 80GB on `long`; preemptible, requeued jobs resume from the latest
# checkpoint-N, and --open-mode=append keeps the log across requeues):
#   RUN_NAME=<name> sbatch scripts/rl_ft/bash_mila.sh
# For 4 GPUs (8 prompts/step, the collaborator's batch): sbatch --gres=gpu:a100l:4 --mem=256G --cpus-per-task=24 ...
# N_GPUS defaults to the GPUs SLURM allocated.
# 1-GPU smoke test inside an existing allocation (run from the repo root):
#   N_GPUS=1 MAX_STEPS=3 EVAL_STEPS=3 SAVE_STEPS=3 EVAL_MAX_PROMPTS=16 bash scripts/rl_ft/bash_mila.sh
# `main` caps a user at 2 GPUs / 8 CPUs / 48G RAM, so multi-GPU runs go on `long`.
# output_dir is a symlink to $SCRATCH/tar_reason/output_dir; the -o/-e directory
# above must exist before sbatch.

source ~/envs/tar/bin/activate
module load cuda/12.2.2

REPO=/home/mila/s/singhsd/CODE/LatentDCR/tar_reason
MODELS=/network/scratch/s/singhsd/models/tar
DATA=/network/scratch/s/singhsd/data/geneval2_50K

RUN_NAME=${RUN_NAME:-darshan_rl_test_mila}
LOCAL_DIR="output_dir/${RUN_NAME}"
DATA_PATH="output_dir/${RUN_NAME}/data.yaml"
VAL_DATA_PATH="output_dir/${RUN_NAME}/val.yaml"

# ===================== Config params ========================
N_GPUS=${N_GPUS:-${SLURM_GPUS_ON_NODE:-4}}

PREV_STAGE_CHECKPOINT=${MODELS}/sft/slf_ref_edit_t21_ckpt17000

# Checkpoint scoring the VQA reward. Leave empty to score with the policy's own
# frozen base weights (LoRA disabled), which costs no extra GPU memory. A separate
# reward model adds one bf16 copy of the weights per GPU (~15G for 7B), so watch
# the memory headroom before raising GEN_BATCH_SIZE / PROMPTS_PER_GPU.
REWARD_MODEL_PATH=${MODELS}/Tar-7B

# Rollout tree: BRANCH="G0,G1[,G2...]" = drafts per prompt, children per node
# per refinement round. Number of refinement rounds = number of entries - 1.
PROMPTS_PER_GPU=2
BRANCH=${BRANCH:-4,2}
GEN_BATCH_SIZE=16
TRAIN_MICRO_BATCH=2

LR=1e-5
LORA_R=64
LORA_ALPHA=128
KL_COEF=0.01
CLIP_EPS=0.2
ALPHA=1.0                 # reward = ALPHA*AM + (1-ALPHA)*GM
NUM_PPO_EPOCHS=1
REFLECT_TOKEN_WEIGHT=1.0
GROUP=parent              # parent | prompt
ADV_NORM=std           # std | mean

REFLECT_TOKENS=128
REFLECT_TEMPERATURE=1.0
ANSWER_SUFFIX=llava   # llava | geneval2 | none

MAX_STEPS=${MAX_STEPS:-500}
EVAL_STEPS=${EVAL_STEPS:-25}
SAVE_STEPS=${SAVE_STEPS:-25}
EVAL_MAX_PROMPTS=${EVAL_MAX_PROMPTS:-}     # empty = full val set
WARMUP_RATIO=0.03
LOG_IMAGES=8
SEED=421
# ===================== Config params END =====================

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "REWARD_MODEL_PATH: ${REWARD_MODEL_PATH:-<policy base weights>}"
echo "RUN_NAME: ${RUN_NAME}  GPUs: ${N_GPUS}  prompts/gpu: ${PROMPTS_PER_GPU}  branch: ${BRANCH}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_NAME=$RUN_NAME
export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}
export WANDB_PROJECT="tar_reasoning"
export WANDB_ENTITY="darshan-singh"
# Mila compute nodes have internet.
export WANDB_MODE="${WANDB_MODE:-online}"

export PYTHONPATH=$PYTHONPATH:${REPO}/
cd ${REPO}

# Visual de-tokenizer weights, only used to log decoded samples to wandb.
TMP_DIR=${SLURM_TMPDIR:-/tmp}
if [ ! -f ${TMP_DIR}/ta_tok.pth ]; then
    cp ${MODELS}/ta_tok.pth ${TMP_DIR}/
fi
AR_MODEL=${MODELS}/ar_dtok_lp_512px.pth
DECODER=${MODELS}/vq_ds16_t2i.pt
ENCODER=${TMP_DIR}/ta_tok.pth
export SIGLIP2_PATH=${MODELS}/siglip2-so400m-patch14-384   # TA-Tok encoder config

mkdir -p "$LOCAL_DIR"

# Data configs live in the run dir (output_dir is gitignored); write them on first use.
if [ ! -f "$DATA_PATH" ]; then
    printf 'datasets:\n    - json_path:\n      - %s\n      ratio: 1\n' "${DATA}/evaluation_metadata_shuf_train.jsonl" > "$DATA_PATH"
fi
if [ ! -f "$VAL_DATA_PATH" ]; then
    printf 'datasets:\n    - json_path:\n      - %s\n      ratio: 1\n' "${DATA}/evaluation_metadata_shuf_val256.jsonl" > "$VAL_DATA_PATH"
fi

REWARD_ARGS=()
if [ -n "${REWARD_MODEL_PATH}" ]; then
    REWARD_ARGS+=(--reward_model_name_or_path "${REWARD_MODEL_PATH}")
fi
EVAL_ARGS=()
if [ -n "${EVAL_MAX_PROMPTS}" ]; then
    EVAL_ARGS+=(--eval_max_prompts "${EVAL_MAX_PROMPTS}")
fi

torchrun --standalone --nproc_per_node=${N_GPUS} \
    llava/train/rl/train_grpo.py \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --attn_implementation flash_attention_2 \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --data_path ${DATA_PATH} \
    --eval_data_path ${VAL_DATA_PATH} \
    --prompts_per_gpu ${PROMPTS_PER_GPU} \
    --branch ${BRANCH} \
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
    "${REWARD_ARGS[@]}" \
    --group ${GROUP} \
    --adv_norm ${ADV_NORM} \
    --clip_eps ${CLIP_EPS} \
    --kl_coef ${KL_COEF} \
    --reflect_token_weight ${REFLECT_TOKEN_WEIGHT} \
    --num_ppo_epochs ${NUM_PPO_EPOCHS} \
    --train_micro_batch ${TRAIN_MICRO_BATCH} \
    --learning_rate ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_grad_norm 1.0 \
    --max_steps ${MAX_STEPS} \
    --output_dir ${LOCAL_DIR} \
    --eval_steps ${EVAL_STEPS} \
    "${EVAL_ARGS[@]}" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 3 \
    --logging_steps 1 \
    --report_to wandb \
    --run_name ${RUN_NAME} \
    --seed ${SEED} \
    --log_images ${LOG_IMAGES} \
    --ar_path ${AR_MODEL} \
    --encoder_path ${ENCODER} \
    --decoder_path ${DECODER} \
    --cfg_scale 4.0
