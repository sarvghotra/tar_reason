#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --open-mode=append
#SBATCH -J darshan_mr_rorqual
#SBATCH -o /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.err

# Rorqual copy of bash_fir_mr.sh (same paths and account; Rorqual is laid out like Fir). Staging stays on until Rorqual's Lustre proves safe.
# Task F of the 2026-10-12 plan: multi-round (N=4) refiner training with t8's recipe. Image tokens trained
# (drafts and refinements), hybrid frozen critiques (50% oracle / 50% policy, as t7/t8), t8's stabilisers
# (1 PPO epoch, adv std floor 0.04, break weight 2). New: chain-mode rollouts (--chain_siblings): every draft
# grows one chain of single refinements up to a random round k* in 1..4 and gets 4 siblings at k* (same
# history and critique), so every depth gets clean groups at ~t8's cost (~13 images/prompt vs 10).
# Context: CONTEXT_MODE (default mdp; full = whole chain). Start: the merged t8 (base + KL reference = t8, fresh LoRA),
# LR 5e-5 (continuing from an RL'd model with deeper chains). Evals: depth-4 chains every 100 steps.
# Open with jeet: DRAFT_REWARD children (t8) vs own for deep chains.
#
#   smoke:   MAX_STEPS=2 EVAL_STEPS=2 SAVE_STEPS=2 EVAL_MAX_PROMPTS=24 RUN_NAME=mr_smoke_rorqual \
#              sbatch --nodes=1 --time=01:00:00 scripts/rl_ft/bash_rorqual_mr.sh
#   resume:  resubmit the same command (or chain with --dependency=afterany:<jobid>).
#   wandb:   offline; sync each job's folder oldest first, later ones with wandb sync --append.
#   cancel:  scancel --name=darshan_mr_rorqual --user=$USER

REPO=${REPO:-/home/dars11/scratch/git/tar_reason}   # override to run from another checkout (e.g. the tar_reason_v2 worktree)
MODELS=/home/dars11/scratch/models/tar
DATA=/home/dars11/scratch/data/geneval2_50K_v2
TRAIN_JSONL=${TRAIN_JSONL:-${DATA}/evaluation_metadata_shuf_train.jsonl}
VAL_JSONL=${VAL_JSONL:-${DATA}/evaluation_metadata_shuf_val800.jsonl}
JUDGE_MODEL=/home/dars11/scratch/models/vlm/Qwen3-VL-8B-Instruct
JUDGE_PY=/home/dars11/envs/qwen_judge/bin/python
# Node-local staging (default on): Fir's Lustre /home and /scratch have killed jobs with SIGBUS (memory-mapped
# pages that became unreadable: t7 step 41, t11 judge step 13, t11 trainer step 123). With STAGE_LOCAL=1 every
# node unpacks both Python envs (+ uv's base Python) from one archive and copies the policy, judge and
# (de-)tokenizer weights to $SLURM_TMPDIR, then runs everything from there. Rebuild the archive after any env change:
#   cd /home/dars11 && tar -cf - --transform 's,^\.local/share/uv/python/,uvpy/,' envs/tar envs/qwen_judge \
#     .local/share/uv/python/cpython-3.10.22-linux-x86_64-gnu | zstd -T16 -3 -o ~/scratch/stage/envs_tar_qwenjudge.tar.zst
STAGE_LOCAL=${STAGE_LOCAL:-1}
ENV_ARCHIVE=/home/dars11/scratch/stage/envs_tar_qwenjudge.tar.zst
UV_PY_DIR=cpython-3.10.22-linux-x86_64-gnu

RUN_NAME=${RUN_NAME:-darshan_mr_rorqual}
LOCAL_DIR="output_dir/${RUN_NAME}"
DATA_PATH="output_dir/${RUN_NAME}/data.yaml"
VAL_DATA_PATH="output_dir/${RUN_NAME}/val.yaml"

# ===================== Config params (task F: t8 recipe + chain mode) ========================
N_GPUS=4
N_NODES=${SLURM_NNODES:-1}
JUDGE_GPU=3
N_TRAIN_GPUS=3
TRAIN_GPUS=0,1,2
WORLD_SIZE_GPUS=$((N_TRAIN_GPUS * N_NODES))

PREV_STAGE_CHECKPOINT=${PREV_STAGE_CHECKPOINT:-${MODELS}/sft/rl_ft_oracle_t8_merged500}   # base + KL reference; fresh LoRA

REWARD_QUESTION_BATCH=32
REWARD_IMAGES_PER_REQ=8
REWARD_DECODE_BATCH=${REWARD_DECODE_BATCH:-32}   # jeet's speed settings (2026-10-19, with his faster AR decoding)
REWARD_LOAD_TIMEOUT=1800

PROMPTS_PER_GPU=${PROMPTS_PER_GPU:-3}
BRANCH=${BRANCH:-2,1,1,1,1}      # 2 drafts per prompt, up to 4 refinement rounds (chain mode uses only these two facts)
CHAIN_SIBLINGS=${CHAIN_SIBLINGS:-4}   # siblings at each draft's random branch round
CONTEXT_MODE=${CONTEXT_MODE:-mdp}     # mdp: each round sees prompt + latest image only (tasks B/C: full history degrades t8 after round 1)
IMAGES_PER_CRITIQUE=1
TRAIN_DRAFTS=1
TRAIN_IMAGES=1
IMAGE_ADV_SCALE=1.0            # image tokens trained (t8 recipe)
EVAL_BRANCH=${EVAL_BRANCH:-1,1,1,1,1}   # greedy depth-4 chains
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-48}            # jeet's speed settings (was 16)
TRAIN_MICRO_BATCH=${TRAIN_MICRO_BATCH:-4}   # jeet's speed settings (was 2; mdp rows ~1.5k tokens, full-history rows up to ~3.8k: use 2 there)

LR=${LR:-5e-5}
LORA_R=64
LORA_ALPHA=128
KL_COEF=0.05
CLIP_EPS=0.2
ALPHA=1.0
NUM_PPO_EPOCHS=1
REFLECT_TOKEN_WEIGHT=1.0
GROUP=parent
ADV_NORM=std
ADV_STD_FLOOR=0.04

CRITIQUE_SOURCE=hybrid
HYBRID_ORACLE_FRAC=0.5
ORACLE_FAIL_THRESHOLD=0.5

BREAK_WEIGHT=2.0
DRAFT_REWARD=${DRAFT_REWARD:-children_r1}  # every draft judged after one refinement (chain depths differ)
STOP_PENALTY=0.5
STOP_THRESHOLD=0.8
MIN_REFINES=0
REFINE_RESAMPLE_TRIES=3

REFLECT_TOKENS=128
REFLECT_TEMPERATURE=1.0
ANSWER_SUFFIX=geneval2

MAX_STEPS=${MAX_STEPS:-300}
EVAL_STEPS=${EVAL_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-10}
EVAL_MAX_PROMPTS=${EVAL_MAX_PROMPTS:-}     # empty = full val set
WARMUP_RATIO=0.05
LOG_IMAGES=8
SEED=421
# Prompt order: a different seed per training run (jeet: never reuse a previous run's order; runs with the same seed,
# data and GPU layout see identical prompt sequences), the same across resumes of one run (resume skips the batches
# already consumed). Default derived from RUN_NAME; override with DATASET_SEED=<n>.
DATASET_SEED=$(( ${DATASET_SEED:-$(( $(printf "%s" "$RUN_NAME" | cksum | cut -d" " -f1) % 100000 ))} + ${SLURM_ARRAY_TASK_ID:-0} ))
# ===================== Config params END =====================

cd ${REPO}
mkdir -p output_dir/slurm_logs
[ -f "${VAL_JSONL}" ] || { echo "Validation file not found: ${VAL_JSONL}" >&2; exit 1; }
[ -x "${JUDGE_PY}" ] || { echo "Judge env not found: ${JUDGE_PY}" >&2; exit 1; }
mkdir -p "$LOCAL_DIR"
[ -f "$DATA_PATH" ] || printf 'datasets:\n    - json_path:\n      - %s\n      ratio: 1\n' "${TRAIN_JSONL}" > "$DATA_PATH"
[ -f "$VAL_DATA_PATH" ] || printf 'datasets:\n    - json_path:\n      - %s\n      ratio: 1\n' "${VAL_JSONL}" > "$VAL_DATA_PATH"

LAST=$(ls -d ${LOCAL_DIR}/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
if [ -n "$LAST" ] && [ "$LAST" -ge "$MAX_STEPS" ]; then
    echo "checkpoint-${LAST} >= MAX_STEPS=${MAX_STEPS}: run already complete, nothing to do."
    exit 0
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29507
REWARD_PORT=$((8000 + ${SLURM_JOB_ID:-0} % 1000))
REWARD_URL="http://127.0.0.1:${REWARD_PORT}"

echo "RUN_NAME=${RUN_NAME} job=${SLURM_JOB_ID} nodes=${N_NODES} resume_from=${LAST:-none} DATASET_SEED=${DATASET_SEED} EXTRA_ARGS=${EXTRA_ARGS}"
echo "multi-round: critique_source=${CRITIQUE_SOURCE} branch=${BRANCH} chain_siblings=${CHAIN_SIBLINGS} context=${CONTEXT_MODE} draft_reward=${DRAFT_REWARD} lr=${LR} eval_branch=${EVAL_BRANCH} base=${PREV_STAGE_CHECKPOINT}"
echo "train ranks ${WORLD_SIZE_GPUS} (${N_TRAIN_GPUS}/node), judge GPU ${JUDGE_GPU} per node; prompts/step $((WORLD_SIZE_GPUS * PROMPTS_PER_GPU)); master ${MASTER_ADDR}:${MASTER_PORT}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_NAME=$RUN_NAME WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}
export WANDB_PROJECT="tar_reasoning" WANDB_ENTITY="darshan-singh"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SIGLIP2_PATH=${MODELS}/siglip2-so400m-patch14-384
export STAGE_LOCAL ENV_ARCHIVE UV_PY_DIR RUN_NAME REPO MODELS LOCAL_DIR DATA_PATH VAL_DATA_PATH PREV_STAGE_CHECKPOINT JUDGE_MODEL JUDGE_PY
export N_NODES N_TRAIN_GPUS TRAIN_GPUS JUDGE_GPU MASTER_ADDR MASTER_PORT REWARD_PORT REWARD_URL
export REWARD_QUESTION_BATCH REWARD_IMAGES_PER_REQ REWARD_DECODE_BATCH REWARD_LOAD_TIMEOUT
export PROMPTS_PER_GPU BRANCH CHAIN_SIBLINGS CONTEXT_MODE IMAGES_PER_CRITIQUE TRAIN_DRAFTS TRAIN_IMAGES IMAGE_ADV_SCALE EVAL_BRANCH GEN_BATCH_SIZE TRAIN_MICRO_BATCH
export LR LORA_R LORA_ALPHA KL_COEF CLIP_EPS ALPHA NUM_PPO_EPOCHS REFLECT_TOKEN_WEIGHT GROUP ADV_NORM ADV_STD_FLOOR
export CRITIQUE_SOURCE HYBRID_ORACLE_FRAC ORACLE_FAIL_THRESHOLD BREAK_WEIGHT DRAFT_REWARD
export STOP_PENALTY STOP_THRESHOLD MIN_REFINES REFINE_RESAMPLE_TRIES REFLECT_TOKENS REFLECT_TEMPERATURE
# Extra train_grpo.py flags appended at the end of the torchrun call.
EXTRA_ARGS=${EXTRA_ARGS:-}
export ANSWER_SUFFIX MAX_STEPS EVAL_STEPS SAVE_STEPS EVAL_MAX_PROMPTS WARMUP_RATIO LOG_IMAGES SEED DATASET_SEED EXTRA_ARGS
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

# One task per node: each copies the tokenizer locally, starts its own judge on GPU 3
# (127.0.0.1 is node-local, so every node needs one), then launches its 3 training ranks.
srun --ntasks-per-node=1 --mem=0 --export=ALL bash -c '
set -e
cd ${REPO}
export PYTHONPATH=$PYTHONPATH:${REPO}/
TMP_DIR=${SLURM_TMPDIR:-/tmp}
TRAIN_PY=$HOME/envs/tar/bin/python
POLICY=${PREV_STAGE_CHECKPOINT}; AR_PATH=${MODELS}/ar_dtok_lp_512px.pth; DEC_PATH=${MODELS}/vq_ds16_t2i.pt
if [ "${STAGE_LOCAL}" = "1" ]; then
    S=${TMP_DIR}/stage; mkdir -p $S/models
    t0=$(date +%s)
    zstd -dc ${ENV_ARCHIVE} | tar -xf - -C $S &
    cp -r ${PREV_STAGE_CHECKPOINT} $S/models/policy &
    cp -r ${JUDGE_MODEL} $S/models/judge &
    cp ${MODELS}/ar_dtok_lp_512px.pth ${MODELS}/vq_ds16_t2i.pt ${MODELS}/ta_tok.pth $S/models/ &
    wait
    for e in tar qwen_judge; do
        ln -sfn $S/uvpy/${UV_PY_DIR}/bin/python3.10 $S/envs/$e/bin/python
        sed -i "s,^home = .*,home = $S/uvpy/${UV_PY_DIR}/bin," $S/envs/$e/pyvenv.cfg
    done
    TRAIN_PY=$S/envs/tar/bin/python; JUDGE_PY=$S/envs/qwen_judge/bin/python; JUDGE_MODEL=$S/models/judge
    POLICY=$S/models/policy; AR_PATH=$S/models/ar_dtok_lp_512px.pth; DEC_PATH=$S/models/vq_ds16_t2i.pt
    cp $S/models/ta_tok.pth ${TMP_DIR}/ta_tok.pth
    ${TRAIN_PY} -c "import sys, torch; assert sys.prefix.startswith(\"$S\"), sys.prefix" \
        || { echo "[node ${SLURM_NODEID}] staged training env broken" >&2; exit 1; }
    echo "[node ${SLURM_NODEID}] $(hostname) staged envs + weights to $S in $(( $(date +%s) - t0 ))s"
else
    [ -f ${TMP_DIR}/ta_tok.pth ] || cp ${MODELS}/ta_tok.pth ${TMP_DIR}/
fi

# Judge first, before any CUDA module is loaded, so its own CUDA 12.4 libraries win.
REWARD_LOG="${LOCAL_DIR}/reward_server_${SLURM_JOB_ID}_n${SLURM_NODEID}.log"
READY_FILE="${LOCAL_DIR}/.reward_ready_${SLURM_JOB_ID}_n${SLURM_NODEID}"
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
        echo "[node ${SLURM_NODEID}] judge died while loading:" >&2; tail -40 "${REWARD_LOG}" >&2; exit 1
    fi
    if [ ${waited} -ge ${REWARD_LOAD_TIMEOUT} ]; then
        echo "[node ${SLURM_NODEID}] judge not ready after ${REWARD_LOAD_TIMEOUT}s:" >&2; tail -40 "${REWARD_LOG}" >&2; exit 1
    fi
    sleep 10; waited=$((waited + 10))
done
echo "[node ${SLURM_NODEID}] $(hostname) judge ready after ${waited}s"

export VIRTUAL_ENV=$(dirname $(dirname ${TRAIN_PY})); export PATH=${VIRTUAL_ENV}/bin:$PATH
module load cuda/12.2

EVAL_ARGS=""
[ -n "${EVAL_MAX_PROMPTS}" ] && EVAL_ARGS="--eval_max_prompts ${EVAL_MAX_PROMPTS}"

# No --standalone: that would confine the job to one node.
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} \
${TRAIN_PY} -m torch.distributed.run --nproc_per_node=${N_TRAIN_GPUS} --nnodes=${N_NODES} --node_rank=${SLURM_NODEID} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    llava/train/rl/train_grpo.py \
    --model_name_or_path ${POLICY} \
    --attn_implementation flash_attention_2 \
    --lora_r ${LORA_R} --lora_alpha ${LORA_ALPHA} \
    --data_path ${DATA_PATH} --eval_data_path ${VAL_DATA_PATH} \
    --prompts_per_gpu ${PROMPTS_PER_GPU} --branch ${BRANCH} --eval_branch ${EVAL_BRANCH} \
    --images_per_critique ${IMAGES_PER_CRITIQUE} --image_adv_scale ${IMAGE_ADV_SCALE} \
    --chain_siblings ${CHAIN_SIBLINGS} --context_mode ${CONTEXT_MODE} \
    $([ "${TRAIN_DRAFTS}" = "0" ] && echo "--no-train_drafts") $([ "${TRAIN_IMAGES}" = "0" ] && echo "--no-train_images") \
    --scale 0 --gen_seq_len 729 \
    --img_temperature 1.0 --img_top_k 1200 --img_top_p 0.95 \
    --reflect_tokens ${REFLECT_TOKENS} --reflect_temperature ${REFLECT_TEMPERATURE} \
    --reflect_top_k 0 --reflect_top_p 0.95 \
    --gen_batch_size ${GEN_BATCH_SIZE} \
    --alpha ${ALPHA} --answer_suffix ${ANSWER_SUFFIX} --reward_batch_size 16 \
    --reward_kind pixel --reward_server_url ${REWARD_URL} \
    --reward_images_per_request ${REWARD_IMAGES_PER_REQ} --reward_decode_batch ${REWARD_DECODE_BATCH} \
    --draft_reward ${DRAFT_REWARD} --stop_penalty ${STOP_PENALTY} --stop_threshold ${STOP_THRESHOLD} \
    --min_refines ${MIN_REFINES} --refine_resample_tries ${REFINE_RESAMPLE_TRIES} \
    --critique_source ${CRITIQUE_SOURCE} --hybrid_oracle_frac ${HYBRID_ORACLE_FRAC} \
    --oracle_fail_threshold ${ORACLE_FAIL_THRESHOLD} --break_weight ${BREAK_WEIGHT} \
    --eval_oracle \
    --group ${GROUP} --adv_norm ${ADV_NORM} --adv_std_floor ${ADV_STD_FLOOR} --clip_eps ${CLIP_EPS} --kl_coef ${KL_COEF} \
    --reflect_token_weight ${REFLECT_TOKEN_WEIGHT} --num_ppo_epochs ${NUM_PPO_EPOCHS} \
    --train_micro_batch ${TRAIN_MICRO_BATCH} \
    --learning_rate ${LR} --warmup_ratio ${WARMUP_RATIO} --lr_scheduler_type constant \
    --max_grad_norm 1.0 --max_steps ${MAX_STEPS} \
    --output_dir ${LOCAL_DIR} --eval_steps ${EVAL_STEPS} ${EVAL_ARGS} \
    --val_dump_dir ${LOCAL_DIR}/val_dump \
    --save_steps ${SAVE_STEPS} --save_total_limit 5 --logging_steps 1 \
    --report_to wandb --run_name ${RUN_NAME} \
    --seed ${SEED} --dataset_seed ${DATASET_SEED} \
    --log_images ${LOG_IMAGES} \
    --ar_path ${AR_PATH} --encoder_path ${TMP_DIR}/ta_tok.pth \
    --decoder_path ${DEC_PATH} --cfg_scale 4.0 ${EXTRA_ARGS}
'
