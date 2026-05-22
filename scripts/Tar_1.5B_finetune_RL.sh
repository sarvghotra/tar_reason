#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=24
#SBATCH --job-name=tar_grpo
#SBATCH --partition short-unkillable
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/scratch/git/tar_reason/output_dir/debug_rl/%j_%t.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/scratch/git/tar_reason/output_dir/debug_rl/%j_%t.err

source ~/.bashrc
mamba activate tar
module load cuda/11.8

RUN_NAME="debug_rl"

export PYTHONPATH=$PYTHONPATH:/home/mila/s/sarvjeet-singh.ghotra/scratch/git/tar_reason/
PREV_STAGE_CHECKPOINT=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/Tar-1.5B

N_GPUS=4

# Per-GPU prompt batch size; each prompt is expanded to GROUP_SIZE rollouts.
# The effective per-rank micro-batch fed to the policy = PROMPT_BS * GROUP_SIZE.
PROMPT_BS=${PROMPT_BS:-2}
GROUP_SIZE=${GROUP_SIZE:-4}
# Number of PPO inner-epoch optimizer steps per rollout (real GRPO updates,
# not gradient accumulation — the script forces DeepSpeed grad-accum=1 since
# inner-epoch backwards must each actually step).
NUM_PPO_EPOCHS=${NUM_PPO_EPOCHS:-4}

MAX_STEPS=${MAX_STEPS:-1000}
LR=${LR:-1e-6}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
KL_COEF=${KL_COEF:-0.04}
CLIP_EPS=${CLIP_EPS:-0.2}

# T2I generation hyperparameters (match t2i_inference.py defaults).
# scale: int = 0  # choose from [0, 1, 2]
# seq_len: int = 729  # choose from [729, 169, 81]
GEN_SCALE=${GEN_SCALE:-2}
GEN_SEQ_LEN=${GEN_SEQ_LEN:-81}

GEN_TEMPERATURE=${GEN_TEMPERATURE:-1.0}
GEN_TOP_P=${GEN_TOP_P:-0.95}
GEN_TOP_K=${GEN_TOP_K:-1200}

# Self-critique rollout: free-text tokens sampled for the critique segment
# between the draft and final image draws.

# FIXME: Critical!!! Increase the CRITIQUE_SEQ_LEN length
CRITIQUE_SEQ_LEN=${CRITIQUE_SEQ_LEN:-64}

VQA_MAX_QUESTIONS=${VQA_MAX_QUESTIONS:-6}

DATA_PATH=${DATA_PATH:-scripts/data_demo.yaml}

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "RUN_NAME: ${RUN_NAME}"
echo "PROMPT_BS=${PROMPT_BS} GROUP_SIZE=${GROUP_SIZE} NUM_PPO_EPOCHS=${NUM_PPO_EPOCHS}"

export WANDB_NAME=$RUN_NAME
export WANDB_PROJECT="tar_reasoning"
export WANDB_ENTITY="diffusion_aaron"
export WANDB_MODE="offline"

LOCAL_DIR="output_dir/${RUN_NAME}"
mkdir -p "$LOCAL_DIR"

torchrun \
    --nproc_per_node=$N_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29505 \
    llava/train/rl/train_grpo.py \
    --deepspeed_config scripts/zero1.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --reward_model_path $PREV_STAGE_CHECKPOINT \
    --num_image_tokens 65536 \
    --bf16 \
    --attn_implementation sdpa \
    --data_path ${DATA_PATH} \
    --dataloader_num_workers 2 \
    --per_device_prompt_batch_size ${PROMPT_BS} \
    --group_size ${GROUP_SIZE} \
    --num_ppo_epochs ${NUM_PPO_EPOCHS} \
    --learning_rate ${LR} \
    --weight_decay 0.0 \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_steps ${MAX_STEPS} \
    --gradient_clipping 1.0 \
    --gen_scale ${GEN_SCALE} \
    --gen_seq_len ${GEN_SEQ_LEN} \
    --gen_temperature ${GEN_TEMPERATURE} \
    --gen_top_p ${GEN_TOP_P} \
    --gen_top_k ${GEN_TOP_K} \
    --clip_eps ${CLIP_EPS} \
    --kl_coef ${KL_COEF} \
    --reward_type vqa \
    --vqa_max_questions $VQA_MAX_QUESTIONS \
    --reward_question "Describe the image shortly." \
    --reward_max_prompt_tokens 128 \
    --system_prompt "You are an image-generation assistant capable of iterative self-correction. Generate an image for the user's prompt, then critique your output identifying any deviations from the prompt (missing objects, wrong counts, wrong attributes, wrong relations). If issues are found, regenerate an improved image and repeat. Stop when the image faithfully matches the prompt." \
    --enable_self_critique \
    --critique_seq_len ${CRITIQUE_SEQ_LEN} \
    --output_dir ${LOCAL_DIR} \
    --logging_steps 1 \
    --save_steps 200 \
    --save_total_limit 2 \
    --report_to wandb \
    --run_name $RUN_NAME
