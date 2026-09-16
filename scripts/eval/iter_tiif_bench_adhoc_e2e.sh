#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=iter_tiif_bench_adhoc
#SBATCH --partition short-unkillable
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/iter_tiif_bench_adhoc/%j_%t.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/iter_tiif_bench_adhoc/%j_%t.err

# End-to-end TIIF-Bench for the forced-schedule iterative model:
#   1. generate the draft (iter-1) and self-corrected (iter-2) image for every
#      TIIF-Bench prompt, in both the short and the long register;
#   2. judge them with a local Qwen2.5-VL (eval/tiif_bench_vlm_judge.py, which
#      reuses TIIF-Bench's own prompts, parsing and result schema);
#   3. summarise with TIIF-Bench's own summary scripts. Both iterations appear
#      as separate rows of the same table, so iter-1 vs iter-2 is a direct read.

source ~/.bashrc
eval "$(mamba shell hook --shell bash)"

set -euo pipefail

if [ ! -f /tmp/ta_tok.pth ]; then
    cp /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ta_tok.pth /tmp/
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    N_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
    N_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi
echo "Using ${N_GPUS} GPUs"

REPEAT=1

TIIF_DIR=/home/mila/s/sarvjeet-singh.ghotra/scratch/git/TIIF-Bench
PROMPTS_DIR=${TIIF_DIR}/data/testmini_prompts
EVAL_PROMPTS_DIR=${TIIF_DIR}/data/testmini_eval_prompts
EVAL_SET_NAME=tiif-bench-testmini_eval

# MODEL_NAME=7B
# MODEL_PATH=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/Tar-7B

MODEL_NAME=slf_ref_edit_t20_16K_greedy_slf_ref_draft_S0
MODEL_PATH=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t20/checkpoint-16000

AR_RES=512
AR_MODEL=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ar_dtok_lp_${AR_RES}px.pth

# Qwen2.5-VL judge (local HF snapshot; nothing is downloaded at run time).
JUDGE_PATH=$(ls -d /network/scratch/s/sarvjeet-singh.ghotra/hf_home/models/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*/ | head -1)
VLLM_PYTHON=/network/scratch/s/sarvjeet-singh.ghotra/installs/miniforge3/envs/vllm/bin/python

SEEDS=(13 17 91)
OUTPUT_DIR=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/${EVAL_SET_NAME}/

# RL="_RL_rl_ft_oracle_t2_ckpt_500"
# LORA_PATH=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/rl_ft_oracle_t2/checkpoint-500
RL=""
LORA_PATH=${5:-}

LORA_ARG=""
if [ -n "$LORA_PATH" ]; then
    LORA_ARG="--lora_path $LORA_PATH"
fi

CURR_DIR=$(pwd)
for SEED in "${SEEDS[@]}"; do

cd "$CURR_DIR"
GEN_MODEL=iter_adhoc_${MODEL_NAME}_${AR_RES}px_repeat${REPEAT}${RL}/seed_${SEED}
SAVE_DIR=${OUTPUT_DIR}/${GEN_MODEL}
IMAGE_DIR=${SAVE_DIR}/images
RESULT_DIR=${SAVE_DIR}/eval_results

############################ 1. Generate images ############################
mamba activate tar
module load cuda/12.1.1

export LD_LIBRARY_PATH="/network/scratch/s/sarvjeet-singh.ghotra/installs/miniforge3/envs/tar/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

# torchrun --standalone --nproc_per_node=$N_GPUS \
#     eval/iterative_tiif_bench_adhoc.py \
#     --model ${MODEL_PATH} \
#     --gen_model ${GEN_MODEL} \
#     --output_dir ${OUTPUT_DIR} \
#     --prompts_dir ${PROMPTS_DIR} \
#     --eval_model_name ${MODEL_NAME} \
#     --repeat $REPEAT \
#     --seed $SEED \
#     --temperature 1.0 \
#     --top_k 1200 \
#     --top_p 0.95 \
#     --reflect_temperature 1.0 \
#     --reflect_top_k 1200 \
#     --reflect_top_p 0.95 \
#     --batch_size 32 \
#     --decode_batch_size 32 \
#     --reflect_tokens 128 \
#     --gen_seq_len 729 \
#     --cfg_scale 4.0 \
#     --ar_path $AR_MODEL \
#     --encoder_path /tmp/ta_tok.pth \
#     --decoder_path /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/vq_ds16_t2i.pt \
#     --verbose \
#     --draft_img_scale 0 \
#     --draft_gen_seq_len 729 \
#     $LORA_ARG


############################ 2. Qwen2.5-VL judge ############################
mamba activate vllm
module load cuda/12.6.0

# eval_with_vlm.py runs from the TIIF-Bench checkout; keep tar_reason (and the
# tar env's pinned transformers) off its import path.
unset PYTHONPATH

# TIIF-Bench's summary script writes an .xlsx.
$VLLM_PYTHON -c "import openpyxl" 2>/dev/null || $VLLM_PYTHON -m pip install --quiet openpyxl

mkdir -p ${RESULT_DIR}

# eval/tiif_bench_vlm_judge.py stands in for TIIF-Bench's own eval_with_vlm.py:
# same prompts, same yes/no parsing, same per-prompt result files, but talking
# to a local checkpoint instead of an OpenAI endpoint. vLLM's server cannot host
# Qwen2.5-VL in this env (FusedInputNorm calls F.batch_norm with eps=0.0, which
# this torch rejects), and the summary scripts below are TIIF-Bench's own, so
# the reported metrics are still the benchmark's.
#
# The vllm env lives on $SCRATCH, and a read of it can fail outright under
# load -- an ImportError on a stdlib .so ("cannot open shared object file" for
# termios.cpython-310...so) that is there and imports fine a minute later.
# Judging is resumable (collect_tasks skips any prompt whose result json
# exists), so just re-run: a second attempt picks up where the first died.
JUDGE_ATTEMPTS=3
run_judge() {
    local ITER=$1 GPU=$2 ITER_LOG=$3
    local TRY
    for TRY in $(seq 1 ${JUDGE_ATTEMPTS}); do
        if [ ${TRY} -gt 1 ]; then
            echo "=== retry ${TRY}/${JUDGE_ATTEMPTS} ===" >> ${ITER_LOG}
        fi
        CUDA_VISIBLE_DEVICES=${GPU} $VLLM_PYTHON ${CURR_DIR}/eval/tiif_bench_vlm_judge.py \
            --jsonl_dir ${EVAL_PROMPTS_DIR} \
            --image_dir ${IMAGE_DIR} \
            --eval_model ${MODEL_NAME}_iter${ITER} \
            --output_dir ${RESULT_DIR} \
            --model ${JUDGE_PATH} \
            --batch_size 8 \
            --seed ${SEED} \
            >> ${ITER_LOG} 2>&1 && return 0
        sleep 30
    done
    return 1
}

# One iteration per GPU, both judged concurrently.
ITERS=(1 2)
PIDS=()
for I in "${!ITERS[@]}"; do
    ITER=${ITERS[$I]}
    ITER_LOG=${SAVE_DIR}/judge_log_iter${ITER}.txt
    : > ${ITER_LOG}
    run_judge ${ITER} ${I} ${ITER_LOG} &
    PIDS+=($!)
done

STATUS=0
for PID in "${PIDS[@]}"; do
    wait ${PID} || STATUS=$?
done

if [ $STATUS -ne 0 ]; then
    echo "ERROR: at least one judging run failed (exit $STATUS)"
    for ITER in "${ITERS[@]}"; do
        echo "--- iter ${ITER} tail ---"
        tail -30 ${SAVE_DIR}/judge_log_iter${ITER}.txt
    done
    exit $STATUS
fi

# Report coverage: a missing per-prompt json silently drops that prompt from the
# accuracy denominator, so make any gap visible next to the scores.
EXPECTED=$(grep -c . ${EVAL_PROMPTS_DIR}/*.jsonl | awk -F: '{n+=$2} END {print 2*n}')
for ITER in "${ITERS[@]}"; do
    JUDGED=$(find ${RESULT_DIR}/${MODEL_NAME}_iter${ITER} -name '*.json' 2>/dev/null | wc -l)
    echo "iter-${ITER}: judged ${JUDGED}/${EXPECTED} (prompt, register) pairs"
done

############################ 3. Summarise ############################
RESULT_FILE=${SAVE_DIR}/tiif_bench_log.txt

(cd ${TIIF_DIR} && $VLLM_PYTHON eval/summary_results.py --input_dir ${RESULT_DIR}) \
    | tee ${RESULT_FILE}

# summary_dimension_results.py exits 120 even on success: its Tee helper
# flushes an already-closed file at interpreter shutdown, long after the report
# is written. Tolerate that, but only when the report actually landed.
DIM_TXT=${RESULT_DIR}/result_summary_dimension.txt
rm -f ${DIM_TXT}
(cd ${TIIF_DIR} && $VLLM_PYTHON eval/summary_dimension_results.py \
    --input_excel ${RESULT_DIR}/result_summary.xlsx \
    --output_txt ${DIM_TXT}) 2>&1 \
    | tee -a ${RESULT_FILE} || true
if [ ! -s ${DIM_TXT} ]; then
    echo "ERROR: summary_dimension_results.py produced no report"
    exit 1
fi

echo " "
echo "OUTPUT DIR: ${SAVE_DIR}"
echo "  per-dimension accuracy : ${RESULT_DIR}/result_summary.xlsx"
echo "  grouped TIIF scores    : ${RESULT_DIR}/result_summary_dimension.txt"
echo "  rows '${MODEL_NAME}_iter1' / '${MODEL_NAME}_iter2' are the draft and the self-corrected image"

done
