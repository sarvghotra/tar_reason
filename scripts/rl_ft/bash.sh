#!/bin/bash
# GRPO fine-tuning on generate -> self-reflect -> refine with the pixel-space
# GenEval2-style VQA reward (llava/train/rl/train_grpo.py).
#
source "${TAR_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/cluster_env.sh"
if [[ -n "${BRANCH+x}" || -n "${GROUP+x}" ]]; then
    echo 'BRANCH/GROUP are obsolete. Use NUM_ROLLOUTS and MAX_REFINEMENTS for final-image GRPO.' >&2
    exit 1
fi
tar_init_run "rl_pixel_qwen"
DATA_PATH="${DATA_PATH:-scripts/rl_ft/data.yaml}"
VAL_DATA_PATH="${VAL_DATA_PATH:-scripts/rl_ft/val.yaml}"

# ===================== Config params ========================
N_GPUS=${N_GPUS:-${SLURM_GPUS_ON_NODE:-4}}

PREV_STAGE_CHECKPOINT="${PREV_STAGE_CHECKPOINT:-$SFT_MODEL}"

# Official GenEval2 pixel judge. Qwen3-VL needs a newer Transformers stack.
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$RESULTS_ROOT/pretrained/Qwen3-VL-8B-Instruct}"
REWARD_PYTHON="${REWARD_PYTHON:-$REPO_ROOT/qwen_reward/bin/python}"

# Independent complete trajectories; each group belongs to one prompt.
PROMPTS_PER_GPU="${PROMPTS_PER_GPU:-4}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
MAX_REFINEMENTS="${MAX_REFINEMENTS:-3}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
RESET_CONTEXT_PER_TURN="${RESET_CONTEXT_PER_TURN:-0}"
case "$RESET_CONTEXT_PER_TURN" in
    1|true) CONTEXT_ARGS=(--reset_context_per_turn) ;;
    0|false) CONTEXT_ARGS=(--no-reset_context_per_turn) ;;
    *) echo 'RESET_CONTEXT_PER_TURN must be 0, 1, false, or true.' >&2; exit 1 ;;
esac
# Opt-in tree rollouts: comma list "R[xM]" per refinement round (after image k),
# e.g. TREE_BRANCH=2 (main-like: 4 drafts x 2 continuations) or 2,2x2. Empty:
# independent episodes. Leaves/prompt <= NUM_ROLLOUTS x prod(R*M).
TREE_BRANCH="${TREE_BRANCH:-}"
MAX_TREE_LEAVES_PER_PROMPT="${MAX_TREE_LEAVES_PER_PROMPT:-64}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
TRAIN_MICRO_BATCH="${TRAIN_MICRO_BATCH:-2}"

LR="${LR:-1e-5}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
KL_COEF="${KL_COEF:-0.01}"
CLIP_EPS="${CLIP_EPS:-0.2}"
NUM_PPO_EPOCHS="${NUM_PPO_EPOCHS:-1}"
REFLECT_TOKEN_WEIGHT="${REFLECT_TOKEN_WEIGHT:-1.0}"
TEXT_POLICY_GRADIENT="${TEXT_POLICY_GRADIENT:-all}"  # all | transitions (prose: KL only)
# final_episode | janus_one_iteration (one refinement; draft / verdict / final rewards;
# requires MAX_REFINEMENTS=1; ADV_NORM=std matches Janus-Pro-R1) | teacher_tree
# (judge-model reflections drive a tree of policy images, e.g. TREE_BRANCH=1x2;
# teacher tokens distilled, shadow reflections rewarded for verdict correctness).
OBJECTIVE="${OBJECTIVE:-final_episode}"
VERDICT_RULE="${VERDICT_RULE:-correction}"   # correction | anywhere ("looks good" location)
ADV_STD_FLOOR="${ADV_STD_FLOOR:-0}"
VERDICT_TARGET="${VERDICT_TARGET:-all_pass}"        # all_pass | soft_am (Janus formula)
VERDICT_PASS_THRESHOLD="${VERDICT_PASS_THRESHOLD:-0.5}"
SHADOW_REFLECTIONS="${SHADOW_REFLECTIONS:-1}"   # teacher_tree: policy reflections per image (not continued)
SHADOW_COEF="${SHADOW_COEF:-1.0}"               # teacher_tree: mean shadow-reflection loss weight
TEACHER_CE_COEF="${TEACHER_CE_COEF:-0.1}"       # teacher_tree: mean teacher-reflection NLL weight
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-96}"  # teacher_tree: teacher reply cap (judge tokens)
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-16}"  # teacher_tree: images per teacher generate call
RENDER_BATCH_SIZE="${RENDER_BATCH_SIZE:-1}"           # images per renderer call
REWARD_QUESTION_BATCH_SIZE="${REWARD_QUESTION_BATCH_SIZE:-1}"  # judge rows per forward
case "$TEXT_POLICY_GRADIENT" in
    all|transitions) ;;
    *) echo 'TEXT_POLICY_GRADIENT must be all or transitions.' >&2; exit 1 ;;
esac
ADV_NORM="${ADV_NORM:-mean}"           # std | mean
LENGTH_NORMALIZATION="${LENGTH_NORMALIZATION:-episode}"  # episode | constant (MAX_SEQ_LEN)
KL_GRADIENT_CORRECTION="${KL_GRADIENT_CORRECTION:-0}"
case "$KL_GRADIENT_CORRECTION" in
    1|true) KL_GRADIENT_ARGS=(--kl_gradient_correction) ;;
    0|false) KL_GRADIENT_ARGS=(--no-kl_gradient_correction) ;;
    *) echo 'KL_GRADIENT_CORRECTION must be 0, 1, false, or true.' >&2; exit 1 ;;
esac

REFLECT_TOKENS="${REFLECT_TOKENS:-128}"
REFLECT_TEMPERATURE="${REFLECT_TEMPERATURE:-1.0}"  # RL requires 1.0 to match PPO log-probs.

MAX_STEPS="${MAX_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-25}"
SAVE_STEPS="${SAVE_STEPS:-25}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LOG_IMAGES="${LOG_IMAGES:-8}"
SEED="${SEED:-421}"
# ===================== Config params END =====================

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "REWARD_MODEL_PATH: ${REWARD_MODEL_PATH}"
echo "RUN_NAME: ${RUN_NAME}  GPUs: ${N_GPUS}  prompts/gpu: ${PROMPTS_PER_GPU}  rollouts/prompt: ${NUM_ROLLOUTS}  max refinements: ${MAX_REFINEMENTS}"
echo "tree branch: '${TREE_BRANCH}'  text policy gradient: ${TEXT_POLICY_GRADIENT}  objective: ${OBJECTIVE} (${VERDICT_RULE})"

export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_NAME}"
ENCODER="$VISION_MODEL"
tar_require_paths "$PREV_STAGE_CHECKPOINT/config.json" "$DATA_PATH" "$VAL_DATA_PATH"
tar_require_paths "$REWARD_MODEL_PATH/config.json" "$REWARD_PYTHON" "$AR_MODEL" "$ENCODER" "$DECODER"

tar_launch torchrun --standalone --nproc_per_node=${N_GPUS} \
    llava/train/rl/train_grpo.py \
    --model_name_or_path "$PREV_STAGE_CHECKPOINT" \
    --attn_implementation flash_attention_2 \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --data_path "${DATA_PATH}" \
    --eval_data_path "${VAL_DATA_PATH}" \
    --prompts_per_gpu "${PROMPTS_PER_GPU}" \
    --num_rollouts "${NUM_ROLLOUTS}" \
    --max_refinements "${MAX_REFINEMENTS}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    "${CONTEXT_ARGS[@]}" \
    --tree_branch "${TREE_BRANCH}" \
    --max_tree_leaves_per_prompt "${MAX_TREE_LEAVES_PER_PROMPT}" \
    --scale 0 \
    --gen_seq_len 729 \
    --img_temperature 1.0 \
    --img_top_k 0 \
    --img_top_p 1.0 \
    --reflect_tokens "${REFLECT_TOKENS}" \
    --reflect_temperature "${REFLECT_TEMPERATURE}" \
    --reflect_top_k 0 \
    --reflect_top_p 1.0 \
    --gen_batch_size "${GEN_BATCH_SIZE}" \
    --reward_model_name_or_path "$REWARD_MODEL_PATH" \
    --reward_python "$REWARD_PYTHON" \
    --adv_norm "${ADV_NORM}" \
    --clip_eps "${CLIP_EPS}" \
    --kl_coef "${KL_COEF}" \
    --reflect_token_weight "${REFLECT_TOKEN_WEIGHT}" \
    --text_policy_gradient "${TEXT_POLICY_GRADIENT}" \
    --objective "${OBJECTIVE}" \
    --verdict_rule "${VERDICT_RULE}" \
    --adv_std_floor "${ADV_STD_FLOOR}" \
    --verdict_target "${VERDICT_TARGET}" \
    --verdict_pass_threshold "${VERDICT_PASS_THRESHOLD}" \
    --shadow_reflections "${SHADOW_REFLECTIONS}" \
    --shadow_coef "${SHADOW_COEF}" \
    --teacher_ce_coef "${TEACHER_CE_COEF}" \
    --teacher_max_tokens "${TEACHER_MAX_TOKENS}" \
    --teacher_batch_size "${TEACHER_BATCH_SIZE}" \
    --render_batch_size "${RENDER_BATCH_SIZE}" \
    --reward_question_batch_size "${REWARD_QUESTION_BATCH_SIZE}" \
    --length_normalization "${LENGTH_NORMALIZATION}" \
    "${KL_GRADIENT_ARGS[@]}" \
    --num_ppo_epochs "${NUM_PPO_EPOCHS}" \
    --train_micro_batch "${TRAIN_MICRO_BATCH}" \
    --learning_rate "${LR}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --max_grad_norm 1.0 \
    --max_steps "${MAX_STEPS}" \
    --output_dir "${LOCAL_DIR}" \
    --eval_output_dir "$EVAL_DIR" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 3 \
    --logging_steps 1 \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --seed "${SEED}" \
    --log_images "${LOG_IMAGES}" \
    --ar_path "${AR_MODEL}" \
    --encoder_path "${ENCODER}" \
    --decoder_path "${DECODER}" \
    --cfg_scale 4.0 \
    "$@"
