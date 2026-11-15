#!/bin/bash
# Final model t4 (or another checkpoint via MODEL_PATH / LORA_PATH / MODEL_NAME / RL_FT) (jeet's rl_ft_s2ce_pmix_oracle_t4: critique SFT adapter sft/final on darshan_mr_E450_orc_merged500)
# on extra prompt sets (GenEval++, T2I-CompBench++), generated exactly as jeet's
# iter_*_adhoc_e2e_MDP_vllm.sh runs of t4 (`iter_adhoc_MDP_n4_E-450_ORC_512px_repeat1_Cric-SFT-t4`): MDP, N_ITERS=4,
# vLLM backend, same sampling (T 1.0, top-k 1200, top-p 0.95 for images and critiques, CFG 4.0, 128 critique tokens).
# Code and env are jeet's: a frozen copy of his checkout ($SNAP, 137f9bd + his uncommitted changes, 2026-11-13) run
# with his vllm env. Images only; scoring is separate.
#
# One job generates the sets in SETS for one SEED, one after another; a set with a .gen_done marker is skipped,
# so a re-submitted job resumes at set granularity (the generator itself restarts a set from scratch).
#
#   sbatch --export=ALL,SEED=13,SETS="geneval_pp color shape" scripts/eval/t4_gen_mila.sh
#   LIMIT=8 SETS=geneval_pp SEED=13 ...   # smoke: first 8 prompts, output under results_t4_smoke/
#
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=bj_t4_gen
#SBATCH --exclude=cn-g005,cn-g011
#SBATCH --partition=short-unkillable
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.out

set -e
SEED=${SEED:?SEED}
SETS=(${SETS:?SETS})
N_GPUS=${N_GPUS:-4}
N_ITERS=${N_ITERS:-4}
SNAP=/network/scratch/s/singhsd/jeet_snapshot/tar_reason_20261113
JEET=/network/scratch/s/sarvjeet-singh.ghotra
# default model = t4; run A: MODEL_NAME=t4_e2eA500 RL_FT= MODEL_PATH=<t4-merged base> LORA_PATH=<run A adapter>
MODEL_PATH=${MODEL_PATH:-${JEET}/models/tar/sft/darshan_mr_E450_orc_merged500}
LORA_PATH=${LORA_PATH-${JEET}/git/tar_reason/output_dir/rorqual/rl_ft_s2ce_pmix_oracle_t4/sft/final}
MODEL_NAME=${MODEL_NAME:-E-450_ORC}
RL_FT=${RL_FT-_Cric-SFT-t4}
REPEAT=${REPEAT:-1}            # samples per prompt (GenEval: 4)
MODELS=/network/scratch/s/singhsd/models/tar          # ta_tok / ar_dtok_lp_512px / vq_ds16_t2i (same files as jeet's)
B=/network/scratch/s/singhsd/benchmarks
RUN=iter_adhoc_MDP_n${N_ITERS}_${MODEL_NAME}_512px_repeat${REPEAT}${RL_FT}
OUT_ROOT=/network/scratch/s/singhsd/tar_reason/results_t4${LIMIT:+_smoke}

prompts_of() {   # set name -> prompt file (one prompt per line)
  case $1 in
    geneval_pp) echo $B/Echo-4o/test_scripts/Geneval++.txt ;;
    geneval) echo ${JEET}/git/geneval/prompts/evaluation_metadata.jsonl ;;   # GenEval (553), metadata per line
    wise) echo $B/WISE/wise_verified_prompts.jsonl ;;   # WISE_Verified (1,000), line i = prompt_id i+1
    color|shape|texture|spatial|3d_spatial|non_spatial|numeracy|complex) echo $B/T2I-CompBench/examples/dataset/$1_val.txt ;;
    *) echo "unknown set $1" >&2; return 1 ;;
  esac
}
bench_of() { case $1 in geneval_pp) echo GenEval++ ;; geneval) echo GenEval ;; wise) echo WISE_Verified ;; *) echo T2I-CompBench++/$1 ;; esac; }

# his conda.sh points at /home/mila/s/sarvjeet-singh.ghotra (not readable to us): activate by hand
export CONDA_PREFIX=${JEET}/installs/miniforge3/envs/vllm
export PATH=${CONDA_PREFIX}/bin:${PATH} LD_LIBRARY_PATH=${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
module load cuda/12.6.0
VLLM_PYTHON=$(which python)
export PYTHONPATH=${SNAP} HF_HUB_OFFLINE=1 HF_HOME=/network/scratch/s/singhsd/hf_home
export TRITON_CACHE_DIR=/tmp/triton_cache_gen_${SLURM_JOB_ID:-$$} VLLM_CACHE_ROOT=/network/scratch/s/singhsd/.cache/vllm
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
# jeet's vllm_gen_activate: compile Triton against a node-local copy of the Python headers
PY_INC=$(${VLLM_PYTHON} -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
LOCAL_PY_INC=/tmp/py_include_gen_${SLURM_JOB_ID:-$$}
PY_INC_COPY=/network/scratch/s/singhsd/jeet_snapshot/py_include_vllm   # copy of ${PY_INC}: reading his env failed on some nodes
for TRY in 1 2 3 4 5; do   # $SCRATCH reads fail transiently on some nodes (as in jeet's launcher)
  rm -rf ${LOCAL_PY_INC}
  cp -r ${PY_INC_COPY} ${LOCAL_PY_INC} 2>/dev/null \
    && [ "$(find ${LOCAL_PY_INC} -type f | wc -l)" -eq "$(find ${PY_INC_COPY} -type f | wc -l)" ] && break
  echo "copy of Python headers failed (try $TRY)"; sleep 20
done
[ -f ${LOCAL_PY_INC}/Python.h ] || { echo "ERROR: could not copy ${PY_INC}"; exit 1; }
export CC=/tmp/triton_cc_gen_${SLURM_JOB_ID:-$$}.sh
cat > ${CC} <<EOS
#!/bin/bash
ARGS=()
for A in "\$@"; do [ "\$A" = "-I${PY_INC}" ] && A="-I${LOCAL_PY_INC}"; ARGS+=("\$A"); done
exec /usr/bin/gcc "\${ARGS[@]}"
EOS
chmod +x ${CC}
[ -f /tmp/ta_tok.pth ] || cp ${MODELS}/ta_tok.pth /tmp/

GENEVAL2_ARG=""; [ "$REPEAT" = 1 ] && GENEVAL2_ARG="--geneval2"   # {prompt: path} json, needs --repeat 1
LORA_ARG=""; [ -n "$LORA_PATH" ] && LORA_ARG="--lora_path $LORA_PATH"
cd ${SNAP}
for SET in "${SETS[@]}"; do
  PROMPTS=$(prompts_of $SET)
  RESULT_DIR=${OUT_ROOT}/$(bench_of $SET)/${RUN}/seed_${SEED}
  if [ -f ${RESULT_DIR}/.gen_done ]; then echo "$SET seed $SEED: done, skipping"; continue; fi
  if [ -n "${LIMIT:-}" ]; then
    mkdir -p ${OUT_ROOT}; head -n ${LIMIT} ${PROMPTS} > ${OUT_ROOT}/${SET}_${LIMIT}.txt; PROMPTS=${OUT_ROOT}/${SET}_${LIMIT}.txt
  fi
  echo "$(date +%H:%M:%S) $SET seed $SEED: $(grep -c . ${PROMPTS}) prompts -> ${RESULT_DIR}"
  T0=$(date +%s)
  mkdir -p ${RESULT_DIR}/gen_logs
  # A rank that dies at start-up (transient $SCRATCH read errors in the env, seen 11-14) leaves the others blocked
  # at a --sync_dir barrier: reap ranks as they finish, kill the rest on the first failure, retry the set.
  OK=0
  for TRY in 1 2 3; do
    SYNC_DIR=${RESULT_DIR}/.sync_${SLURM_JOB_ID:-$$}_${TRY}; rm -rf ${SYNC_DIR}
    PIDS=()
    for (( R=0; R<N_GPUS; R++ )); do
      RANK=${R} WORLD_SIZE=${N_GPUS} CUDA_VISIBLE_DEVICES=${R} setsid ${VLLM_PYTHON} eval/iterative_generation_adhoc_MDP.py \
        --lm_backend vllm --sync_dir ${SYNC_DIR} \
        --n_iters $N_ITERS --model $MODEL_PATH \
        --out_dir $RESULT_DIR --prompts_file $PROMPTS --repeat $REPEAT $GENEVAL2_ARG --seed $SEED \
        --temperature 1.0 --top_k 1200 --top_p 0.95 \
        --reflect_temperature 1.0 --reflect_top_k 1200 --reflect_top_p 0.95 \
        --batch_size ${GEN_BATCH:-100} --decode_batch_size ${DECODE_BATCH:-32} \
        --vllm_gpu_memory_utilization ${VLLM_MEM:-0.45} --reflect_tokens ${REFLECT_TOKENS:-128} \
        --gen_seq_len 729 --cfg_scale 4.0 \
        --ar_path ${MODELS}/ar_dtok_lp_512px.pth --decoder_path ${MODELS}/vq_ds16_t2i.pt --encoder_path /tmp/ta_tok.pth \
        --draft_img_scale 0 --draft_gen_seq_len 729 --verbose --prompt_format sft $LORA_ARG \
        > ${RESULT_DIR}/gen_logs/rank${R}_try${TRY}.log 2>&1 &
      PIDS+=($!)
    done
    LEFT=("${PIDS[@]}"); FAILED=""
    while [ ${#LEFT[@]} -gt 0 ]; do
      DONE=""; wait -n -p DONE "${LEFT[@]}" && RC=0 || RC=$?
      [ -n "$DONE" ] || break
      KEEP=(); for P in "${LEFT[@]}"; do [ "$P" != "$DONE" ] && KEEP+=("$P"); done; LEFT=("${KEEP[@]}")
      [ $RC -eq 0 ] || { FAILED="pid $DONE rc $RC"; break; }
    done
    if [ -z "$FAILED" ]; then OK=1; break; fi
    echo "$(date +%H:%M:%S) $SET seed $SEED try $TRY failed ($FAILED); killing the other ranks"
    for P in "${LEFT[@]}"; do kill -TERM -- -$P 2>/dev/null || true; done
    sleep 15; for P in "${LEFT[@]}"; do kill -KILL -- -$P 2>/dev/null || true; done; wait 2>/dev/null
    for R in "${!PIDS[@]}"; do grep -v "import_utils.py:408" ${RESULT_DIR}/gen_logs/rank${R}_try${TRY}.log | tail -3; done
    sleep 30
  done
  [ $OK = 1 ] || { echo "ERROR: $SET seed $SEED generation failed 3 times"; exit 1; }
  rm -rf ${SYNC_DIR}
  touch ${RESULT_DIR}/.gen_done
  echo "$(date +%H:%M:%S) $SET seed $SEED: done in $(( $(date +%s) - T0 )) s"
done
echo "================ DONE ============="
