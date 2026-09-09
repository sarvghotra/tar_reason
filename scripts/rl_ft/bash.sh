#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0G
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH -J dbg_rl_ft
#SBATCH -o /home/jeet/scratch/git/tar_reason/output_dir/dbg_rl_ft/%j_%t.out
#SBATCH -e /home/jeet/scratch/git/tar_reason/output_dir/dbg_rl_ft/%j_%t.err

# GRPO fine-tuning on generate -> self-reflect -> refine with the latent
# GenEval2-style VQA reward (llava/train/rl/train_grpo.py).
#
# The partition kills jobs after 3 h. Training auto-resumes from the latest
# output_dir/${RUN_NAME}/checkpoint-N, so just resubmit this script (or chain
# it with `sbatch --dependency=afterany:<jobid>`) until MAX_STEPS is reached.
# Keep WANDB_RUN_ID fixed across resubmissions so the wandb run continues.

source /scratch/jeet/installs/miniforge3/bin/activate tar
module load cuda/12.2

RUN_NAME="dbg_rl_ft"
LOCAL_DIR="output_dir/${RUN_NAME}"
DATA_PATH="output_dir/${RUN_NAME}/data.yaml"
VAL_DATA_PATH="output_dir/${RUN_NAME}/val.yaml"

# ===================== Config params ========================
N_GPUS=4

PREV_STAGE_CHECKPOINT=/home/jeet/scratch/git/tar_reason/output_dir/slf_ref_edit_t6_repro_w_corr_t2/weights_only/checkpoint-20000

# Checkpoint scoring the VQA reward. Leave empty to score with the policy's own
# frozen base weights (LoRA disabled), which costs no extra GPU memory. A separate
# reward model adds one bf16 copy of the weights per GPU (~15G for 7B), so watch
# the memory headroom before raising GEN_BATCH_SIZE / PROMPTS_PER_GPU.
REWARD_MODEL_PATH=/scratch/jeet/models/pre_train/tar/Tar-7B

# Rollout tree: BRANCH="G0,G1[,G2...]" = drafts per prompt, children per node
# per refinement round. Number of refinement rounds = number of entries - 1.
PROMPTS_PER_GPU=2
BRANCH="4,2"
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

MAX_STEPS=500
EVAL_STEPS=25
SAVE_STEPS=25
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
export WANDB_ENTITY="diffusion_aaron"
export WANDB_MODE="online"

export PYTHONPATH=$PYTHONPATH:/home/jeet/scratch/git/tar_reason/
cd /home/jeet/scratch/git/tar_reason

# Visual de-tokenizer weights, only used to log decoded samples to wandb.
if [ ! -f /tmp/ta_tok.pth ]; then
    cp /scratch/jeet/models/pre_train/tar/ta_tok.pth /tmp/
fi
AR_MODEL=/scratch/jeet/models/pre_train/tar/ar_dtok_lp_512px.pth
DECODER=/scratch/jeet/models/pre_train/tar/vq_ds16_t2i.pt
ENCODER=/tmp/ta_tok.pth

mkdir -p "$LOCAL_DIR"

REWARD_ARGS=()
if [ -n "${REWARD_MODEL_PATH}" ]; then
    REWARD_ARGS+=(--reward_model_name_or_path "${REWARD_MODEL_PATH}")
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
