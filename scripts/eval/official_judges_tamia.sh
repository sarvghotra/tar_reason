#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --account=aip-agrawal
#SBATCH -J official_judges
#SBATCH -o /scratch/d/dars11/git/tar_reason_pm/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /scratch/d/dars11/git/tar_reason_pm/output_dir/slurm_logs/%x_%j.err
# Re-score already generated benchmark images with the benchmarks' *official* judges (user, 2026-10-30), so our
# numbers are comparable with published ones. jeet's pipeline (bench_jeet_*.sh) judges TIIF with a local
# Qwen2.5-VL-7B and GenAI-Bench with Qwen3.5-27B VQAScore; the official setups are
#   TIIF-Bench : Qwen2.5-VL-72B-Instruct served by vLLM, judged by TIIF-Bench's own eval/eval_with_vlm.py
#                (OpenAI-compatible API, temperature 1.0, 3 prompt variants) + its summary scripts;
#   GenAI-Bench: VQAScore with CLIP-FlanT5-XXL (eval/genai_bench_clipt5.py = his genai_bench/evaluate.py on the
#                old t2v_metrics CLIP-FlanT5 code).
# No generation happens here: RUN_DIR must hold the seed_<S>/ folders written by bench_jeet_tamia.sh (or copied
# from Mila). Results go next to the existing ones, never overwriting them:
#   tiif : seed_<S>/eval_results_qwen72b/{<MODEL_NAME>_iter<k>/..., result_summary.xlsx, result_summary_dimension.txt},
#          seed_<S>/tiif_bench_log_qwen72b.txt;  aggregate: eval/stats/tiff_bench.py --results_subdir eval_results_qwen72b
#   genai: seed_<S>/<k>/clip-flant5-xxl_1600_prompts.pt + eval_log_clip-flant5-xxl.txt, seed_<S>/clip-flant5-xxl_log.txt;
#          aggregate: eval/stats/genai_bench.py --score_model clip-flant5-xxl
# Usage:
#   BENCH=tiif  MODEL_NAME=darshan_agrpo_tar_380 N_ITERS=2 \
#     RUN_DIR=$SCRATCH/tar_reason/results_jeet/tiif-bench-testmini_eval/iter_adhoc_MDP_n2_darshan_agrpo_tar_380_512px_repeat1 \
#     sbatch scripts/eval/official_judges_tamia.sh
#   BENCH=genai MODEL_NAME=darshan_agrpo_tar_380 N_ITERS=2 RUN_DIR=.../genai-bench-1600/iter_adhoc_MDP_n2_..._repeat1 sbatch ...
#   SEEDS="13 17 91" (default: every seed_* in RUN_DIR), N_PAR (TIIF: concurrent judge clients per iteration, default 8).
# Both are resumable: TIIF skips prompts whose result json exists, GenAI skips iterations whose .pt exists.
set -u
REPO=/scratch/d/dars11/git/tar_reason_pm
BENCH_DIR=/scratch/d/dars11/benchmarks
export HF_HOME=/scratch/d/dars11/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BENCH=${BENCH:?set BENCH=tiif|genai}
MODEL_NAME=${MODEL_NAME:?set MODEL_NAME (the label used at generation time, e.g. darshan_agrpo_tar_380)}
RUN_DIR=${RUN_DIR:?set RUN_DIR (holds seed_*/)}
N_ITERS=${N_ITERS:-5}
N_GPUS=4
if [ -n "${SEEDS:-}" ]; then read -r -a SEEDS <<< "$SEEDS"; else
  SEEDS=($(ls -d ${RUN_DIR}/seed_* 2>/dev/null | sed 's/.*seed_//' | sort -n)); fi
[ ${#SEEDS[@]} -gt 0 ] || { echo "no seed_* under ${RUN_DIR}"; exit 1; }
OUTPUT_DIR=$(dirname "$RUN_DIR")
TMP=${SLURM_TMPDIR:-/tmp/${USER}_official_judges}; mkdir -p "$TMP"
echo "BENCH=${BENCH} MODEL_NAME=${MODEL_NAME} N_ITERS=${N_ITERS} SEEDS=${SEEDS[*]} RUN_DIR=${RUN_DIR} job=${SLURM_JOB_ID:-local} host=$(hostname)"
TAR_PY=/home/d/dars11/envs/tar/bin/python
ITERS=($(seq 1 ${N_ITERS}))

# ================================================================ TIIF-Bench: Qwen2.5-VL-72B via vLLM
tiif_official() {
  local TIIF_DIR=${BENCH_DIR}/TIIF-Bench EVAL_PROMPTS_DIR=${BENCH_DIR}/TIIF-Bench/data/testmini_eval_prompts
  local VLLM=/home/d/dars11/envs/vllm_judge/bin/vllm CLIENT_PY=/home/d/dars11/envs/vllm_judge/bin/python
  local SUM_PY=/home/d/dars11/envs/qwen_judge/bin/python      # pandas + openpyxl for the summary scripts
  local JUDGE=$(ls -d ${HF_HOME}/hub/models--Qwen--Qwen2.5-VL-72B-Instruct/snapshots/*/ | head -1)
  [ -n "$JUDGE" ] || { echo "Qwen2.5-VL-72B-Instruct not in ${HF_HOME}/hub"; exit 1; }
  local SERVED=qwen2.5-vl-72b-instruct PORT=$((8100 + ${SLURM_JOB_ID:-0} % 800)) N_PAR=${N_PAR:-8}
  local EXPECTED=$(cat ${EVAL_PROMPTS_DIR}/*.jsonl | grep -c .); EXPECTED=$((2 * EXPECTED))   # short + long per prompt
  # Server: TP over the 4 GPUs (146 GB of bf16 weights), one image per request.
  local SERVER_LOG=${RUN_DIR}/vllm_qwen72b_${SLURM_JOB_ID:-local}.log
  echo "$(date +%H:%M:%S) starting vLLM ${SERVED} on :${PORT} (TP=${N_GPUS}); log ${SERVER_LOG}"
  env -u PYTHONPATH ${VLLM} serve "${JUDGE}" --served-model-name ${SERVED} --tensor-parallel-size ${N_GPUS} \
      --dtype bfloat16 --max-model-len 8192 --limit-mm-per-prompt '{"image":1}' --gpu-memory-utilization 0.92 \
      --host 127.0.0.1 --port ${PORT} > "${SERVER_LOG}" 2>&1 &
  local SERVER_PID=$!
  trap "kill ${SERVER_PID} 2>/dev/null" EXIT INT TERM
  local waited=0
  until curl -sf http://127.0.0.1:${PORT}/v1/models > /dev/null 2>&1; do
    kill -0 ${SERVER_PID} 2>/dev/null || { echo "vLLM died:"; tail -40 "${SERVER_LOG}"; exit 1; }
    [ ${waited} -ge 2400 ] && { echo "vLLM not ready after ${waited}s:"; tail -40 "${SERVER_LOG}"; exit 1; }
    sleep 15; waited=$((waited + 15))
  done
  echo "$(date +%H:%M:%S) vLLM ready after ${waited}s"
  # The official client is sequential; run N_PAR copies per iteration on disjoint groups of the 39 prompt files
  # (symlink dirs), all against the one server. It skips prompts whose result json exists -> resumable.
  local GROUPS_DIR=${TMP}/tiif_jsonl_groups; rm -rf "$GROUPS_DIR"; mkdir -p "$GROUPS_DIR"
  local i=0 f
  for f in ${EVAL_PROMPTS_DIR}/*.jsonl; do mkdir -p ${GROUPS_DIR}/g$((i % N_PAR)); ln -s "$f" ${GROUPS_DIR}/g$((i % N_PAR))/; i=$((i + 1)); done
  local SEED SAVE_DIR IMAGE_DIR RESULT_DIR LOG ITER g PIDS PID STATUS JUDGED
  for SEED in "${SEEDS[@]}"; do
    SAVE_DIR=${RUN_DIR}/seed_${SEED}; IMAGE_DIR=${SAVE_DIR}/images; RESULT_DIR=${SAVE_DIR}/eval_results_qwen72b
    LOG=${SAVE_DIR}/tiif_bench_log_qwen72b.txt
    [ -d "$IMAGE_DIR" ] || { echo "seed ${SEED}: no images at ${IMAGE_DIR}"; exit 1; }
    if [ -s "${RESULT_DIR}/result_summary_dimension.txt" ] && [ -s "$LOG" ]; then echo "seed ${SEED}: done, skipping"; continue; fi
    mkdir -p "$RESULT_DIR"
    echo "$(date +%H:%M:%S) seed ${SEED}: judging ${N_ITERS} iterations x ${N_PAR} clients"
    PIDS=(); STATUS=0
    for ITER in "${ITERS[@]}"; do
      for g in $(seq 0 $((N_PAR - 1))); do
        (cd ${TIIF_DIR} && env -u PYTHONPATH ${CLIENT_PY} eval/eval_with_vlm.py \
            --jsonl_dir ${GROUPS_DIR}/g${g} --image_dir ${IMAGE_DIR} --eval_model ${MODEL_NAME}_iter${ITER} \
            --output_dir ${RESULT_DIR} --api_key EMPTY --base_url http://127.0.0.1:${PORT}/v1 --model ${SERVED} \
            > ${RESULT_DIR}/client_iter${ITER}_g${g}.log 2>&1) &
        PIDS+=($!)
      done
    done
    for PID in "${PIDS[@]}"; do wait ${PID} || STATUS=$?; done
    for ITER in "${ITERS[@]}"; do
      JUDGED=$(find ${RESULT_DIR}/${MODEL_NAME}_iter${ITER} -name '*.json' 2>/dev/null | wc -l)
      echo "iter-${ITER}: judged ${JUDGED}/${EXPECTED} (prompt, register) pairs"
      [ "$JUDGED" -ge "$EXPECTED" ] || STATUS=1
    done
    [ $STATUS -eq 0 ] || { echo "ERROR: judgments incomplete for seed ${SEED}; resubmit (judged prompts are kept)"; exit 1; }
    rm -f ${RESULT_DIR}/client_iter*_g*.log
    # TIIF-Bench's own summaries (summary_dimension_results.py exits non-zero even on success: check the report).
    (cd ${TIIF_DIR} && env -u PYTHONPATH ${SUM_PY} eval/summary_results.py --input_dir ${RESULT_DIR}) | tee ${LOG}
    rm -f ${RESULT_DIR}/result_summary_dimension.txt
    (cd ${TIIF_DIR} && env -u PYTHONPATH ${SUM_PY} eval/summary_dimension_results.py \
        --input_excel ${RESULT_DIR}/result_summary.xlsx --output_txt ${RESULT_DIR}/result_summary_dimension.txt) 2>&1 | tee -a ${LOG} || true
    [ -s ${RESULT_DIR}/result_summary_dimension.txt ] || { echo "ERROR: no dimension report for seed ${SEED}"; exit 1; }
    echo "seed ${SEED}: done -> ${RESULT_DIR}"
  done
  kill ${SERVER_PID} 2>/dev/null; wait ${SERVER_PID} 2>/dev/null; trap - EXIT INT TERM
  echo "$(date +%H:%M:%S) all seeds done. Aggregate + plot:"
  echo "  cd ${REPO} && ${TAR_PY} eval/stats/tiff_bench.py --results_subdir eval_results_qwen72b --run \"${MODEL_NAME}_qwen72b=${RUN_DIR}\" --out_dir ${OUTPUT_DIR}/figures_qwen72b"
  (cd ${REPO} && ${TAR_PY} eval/stats/tiff_bench.py --results_subdir eval_results_qwen72b --run "${MODEL_NAME}_qwen72b=${RUN_DIR}" --out_dir ${OUTPUT_DIR}/figures_qwen72b) || true
}

# ================================================================ GenAI-Bench: CLIP-FlanT5-XXL VQAScore
genai_official() {
  local SCORE_MODEL=clip-flant5-xxl NUM_PROMPTS=1600 GENAI_ROOT=${BENCH_DIR}/GenAI-Bench
  local CLIPT5_PY=/home/d/dars11/envs/clipt5_judge/bin/python
  export T2V_CLIPT5_DIR=/scratch/d/dars11/git/t2v_metrics_clipt5 T2V_FORK_DIR=/scratch/d/dars11/git/t2v_metrics_jeet
  local PT=${SCORE_MODEL}_${NUM_PROMPTS}_prompts.pt
  local SEED SEED_DIR GEN_MODEL RESULT_FILE ITER I PIDS PID STATUS ITER_LOG
  for SEED in "${SEEDS[@]}"; do
    SEED_DIR=${RUN_DIR}/seed_${SEED}; GEN_MODEL=${RUN_DIR#${OUTPUT_DIR}/}/seed_${SEED}
    RESULT_FILE=${SEED_DIR}/${SCORE_MODEL}_log.txt
    if [ "$(grep -c '^Tag Group: overall' "$RESULT_FILE" 2>/dev/null)" -ge "$N_ITERS" ]; then echo "seed ${SEED}: done, skipping"; continue; fi
    echo "$(date +%H:%M:%S) seed ${SEED}: VQAScore ${SCORE_MODEL}, ${NUM_PROMPTS} prompts, one iteration per GPU"
    PIDS=(); STATUS=0
    for I in "${!ITERS[@]}"; do
      ITER=${ITERS[$I]}; ITER_LOG=${SEED_DIR}/${ITER}/eval_log_${SCORE_MODEL}.txt
      [ -d "${SEED_DIR}/${ITER}" ] || { echo "missing ${SEED_DIR}/${ITER}"; exit 1; }
      if [ -s "${SEED_DIR}/${ITER}/${PT}" ] && grep -q '^Tag Group: overall' "$ITER_LOG" 2>/dev/null; then continue; fi
      # cwd = a neutral dir: the stub t2v_metrics package must win over the fork's and the old checkout's.
      ( L=${TMP}/genai_${SEED}_${ITER}; mkdir -p "$L"; cd "$L" && CUDA_VISIBLE_DEVICES=$((I % N_GPUS)) env -u PYTHONPATH \
          ${CLIPT5_PY} ${REPO}/eval/genai_bench_clipt5.py --model ${SCORE_MODEL} --root_dir ${GENAI_ROOT} \
          --output_dir ${OUTPUT_DIR} --gen_model ${GEN_MODEL}/${ITER} --num_prompts ${NUM_PROMPTS} \
          --cache_dir ${HF_HOME}/hub --result_dir "$L" --batch_size 16 > "$L/eval_log.txt" 2>&1; rc=$?
        cp -f "$L/eval_log.txt" "$ITER_LOG"
        [ $rc -eq 0 ] && cp -f "$L/${GEN_MODEL}/${ITER}/${PT}" "${SEED_DIR}/${ITER}/${PT}"; exit $rc ) &
      PIDS+=($!)
      if [ ${#PIDS[@]} -eq $N_GPUS ]; then for PID in "${PIDS[@]}"; do wait $PID || STATUS=$?; done; PIDS=(); fi
    done
    for PID in "${PIDS[@]}"; do wait $PID || STATUS=$?; done
    : > "$RESULT_FILE"
    for ITER in "${ITERS[@]}"; do { echo " "; echo "ITER-${ITER}:"; cat "${SEED_DIR}/${ITER}/eval_log_${SCORE_MODEL}.txt"; } | tee -a "$RESULT_FILE"; done
    [ $STATUS -eq 0 ] || { echo "ERROR: at least one scoring run failed (exit $STATUS)"; exit $STATUS; }
  done
  echo "$(date +%H:%M:%S) all seeds done. Aggregate + plot:"
  echo "  cd ${REPO} && ${TAR_PY} eval/stats/genai_bench.py --score_model ${SCORE_MODEL} --run \"${MODEL_NAME}_clipt5=${RUN_DIR}\" --out_dir ${OUTPUT_DIR}/figures_clipt5"
  (cd ${REPO} && ${TAR_PY} eval/stats/genai_bench.py --score_model ${SCORE_MODEL} --run "${MODEL_NAME}_clipt5=${RUN_DIR}" --out_dir ${OUTPUT_DIR}/figures_clipt5) || true
}

case $BENCH in
  tiif)  tiif_official ;;
  genai) genai_official ;;
  *) echo "BENCH must be tiif or genai"; exit 1 ;;
esac
