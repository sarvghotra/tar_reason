#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=aip-agrawal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --open-mode=append
#SBATCH -J bench_jeet
#SBATCH -o /scratch/d/dars11/git/tar_reason_pm/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /scratch/d/dars11/git/tar_reason_pm/output_dir/slurm_logs/%x_%j.err

# TamIA port of bench_jeet_mila.sh (2026-10-28): only the paths, the Slurm header (whole H100 node, 3 h partition) and the
# CUDA module differ. Compute nodes have no internet: judge weights are pre-downloaded into $HF_HOME/hub on a login node.
# Submit with a script file (not --wrap: TamIA's --wrap ignores these #SBATCH lines). Envs: ~/envs/{tar,qwen_judge,genai_judge}.
# Official evals with jeet's pipeline, on Mila, from our repo. One script for his six launchers
# (scripts/eval/iter_{geneval2_adhoc_e2e,genai_bench_adhoc,tiif_bench_adhoc_e2e}{,_MDP}.sh in his
# checkout): same generators, generator flags, sampling, batch sizes, scorers, judge models, prompt
# counts, seeds, per-iteration GPU layout and output layout, so his eval/stats scripts read the
# results unchanged. Only the paths and envs are ours (uv envs instead of his mamba envs).
#
# Usage (one benchmark per job; MODEL_NAME is the label that goes into the folder name):
#   MODEL_NAME=slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0 BENCH=geneval2 sbatch scripts/eval/bench_jeet_mila.sh
#   MODEL_NAME=slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0 RL_FT=_rl_ft_promptsmix_oracle_t1_500 \
#     LORA_PATH=$SCRATCH/tar_reason/kept/<run>/checkpoint-500 BENCH=genai sbatch scripts/eval/bench_jeet_mila.sh
#   base Tar (plain prompt, draft only, every later iteration repeats the draft):
#   MODEL_NAME=7B MODEL_PATH=$SCRATCH/models/tar/Tar-7B SINGLE_PASS=1 BENCH=tiif sbatch scripts/eval/bench_jeet_mila.sh
#   Janus baselines (eval/baselines/janus_pro_r1_gen.py, env ~/envs/janus, N_ITERS forced to 2:
#   iteration 1 = first image, iteration 2 = the R1 loop's final image; single = iteration 2 repeats 1):
#   MODEL_NAME=janus_pro_r1 GEN_FAMILY=janus JANUS_MODE=r1_loop BENCH=geneval2 SEEDS=713 sbatch scripts/eval/bench_jeet_mila.sh
#   MODEL_NAME=janus_pro_7b GEN_FAMILY=janus JANUS_MODE=single  BENCH=geneval2 SEEDS=713 sbatch scripts/eval/bench_jeet_mila.sh
#
# Env vars:
#   MODEL_NAME   required. Folder label (jeet's convention: <ckpt>_greedy_slf_ref_draft_S0; his TIIF
#                runs used the typo "sslf_..."). For TIIF it is also --eval_model_name (rows <MODEL_NAME>_iter<k>).
#   BENCH        required: geneval2 | genai | tiif
#   MODEL_PATH   merged model dir (default: SFT t21 checkpoint-17000)
#   LORA_PATH    optional LoRA adapter on top of MODEL_PATH
#   RL_FT        optional folder suffix, as in his scripts (e.g. _rl_ft_oracle_t8_500); default empty
#   MDP          1 (default) = *_MDP generator: each correction sees only the prompt + the previous image;
#                0 = full history (draft, critique, image, critique, ...)
#   N_ITERS      images per sample INCLUDING the draft (default 5 = draft + 4 corrections, as in training). With
#                N_ITERS=2 MDP and full history are identical.
#   SEEDS        space-separated (defaults = his: geneval2 "2213 713 13", genai "13 17 91", tiif "17 91 13")
#   NUM_PROMPTS  GenAI-Bench prompts: 1600 (default, the full official set) or 800 (jeet's subset, same ids)
#   SINGLE_PASS  1 = base-model baseline (--single_pass)
#   OUT_ROOT     default $SCRATCH/tar_reason/results_jeet
#   GEN_FAMILY   tar (default) | janus. janus: GEN_SCRIPT=eval/baselines/janus_pro_r1_gen.py run with ~/envs/janus,
#                no CUDA module, N_ITERS=2; MODEL_PATH / LORA_PATH / MDP / SINGLE_PASS are not used.
#   JANUS_MODE   r1_loop (default: generate, self-check, regenerate if "No") | single (plain text-to-image)
#   JANUS_MODEL  HF id or dir (default midbee/Janus-Pro-R1-7B for r1_loop, deepseek-ai/Janus-Pro-7B for single)
#   JANUS_MAX_IMAGES  images in the R1 loop incl. the first (default 2, as Janus-Pro-R1's inference.py)
#   JANUS_ARGS   extra generator flags (e.g. "--janus_batch_size 8")
#   GEN_FAMILY=janus folders: GEN_PREFIX = janus_<JANUS_MODE>_n2, resolution tag 384px (Janus' native size).
#   Diffusion baselines (eval/baselines/diffusers_gen.py, env ~/envs/diffusers; single pass, every iteration the same image):
#   MODEL_NAME=flux1_dev   GEN_FAMILY=diffusers DIFF_MODEL=flux1-dev  BENCH=geneval2 SEEDS=713 sbatch scripts/eval/bench_jeet_mila.sh
#   MODEL_NAME=sd35_large  GEN_FAMILY=diffusers DIFF_MODEL=sd35-large ...     MODEL_NAME=qwen_image GEN_FAMILY=diffusers DIFF_MODEL=qwen-image ...
#   DIFF_RES (1024; qwen-image 1328), DIFF_ARGS (e.g. "--steps 28 --diff_batch_size 2 --cpu_offload"), DIFF_MODEL_PATH, DIFF_ENV.
#   FLUX.1-dev and SD3.5-Large are gated on HF: accept their licences and export HF_TOKEN before sbatch.
#   Folders: GEN_PREFIX = diffusers_n<N_ITERS>, resolution tag <DIFF_RES>px.
#   Unified-model baselines (eval/baselines/unified_gen.py + backends/<family>.py; one uv env per family, UNI_ENV):
#   MODEL_NAME=showo_512   GEN_FAMILY=showo   ...   MODEL_NAME=showo2_7b   GEN_FAMILY=showo2  ...   MODEL_NAME=bagel_7b GEN_FAMILY=bagel ...
#   MODEL_NAME=blip3o_8b   GEN_FAMILY=blip3o  ...   MODEL_NAME=emu3_gen    GEN_FAMILY=emu3    ...   MODEL_NAME=unicot_7b GEN_FAMILY=unicot ...
#   UNI_ARGS = extra backend flags (e.g. "--gen_batch 4 --steps 50", repo paths "--bagel_repo ..."), UNI_MODEL_PATH, UNI_RES (folder tag only).
#   RESUME=1 (unified families only): survive preemption on `long`. Generation keeps a copy of its local outputs in
#   ${OUT_ROOT}_partial/<same path> (rsync every RESUME_EVERY s, default 600), a requeued job starts from that copy and
#   unified_gen.py --resume skips the prompts already done.
#   unicot / alphagrpo are self-correction loops (N_ITERS=2: first image -> final; UNI_N_ITERS=5 keeps every round like our models).
#   MODEL_NAME=alphagrpo_srr GEN_FAMILY=alphagrpo UNI_ARGS="--mode reflect" ...   MODEL_NAME=alphagrpo_rt2i GEN_FAMILY=alphagrpo UNI_ARGS="--mode think" ...
#
# Output layout (identical to his, OUTPUT_DIR = OUT_ROOT/<eval set name>):
#   geneval2: $OUT_ROOT/GenEval2/<GEN_MODEL>/seed_<S>/{1..N}/{geneval2_results.json,results_geneval2.jsonl,eval_log.txt}, log.txt
#   genai   : $OUT_ROOT/genai-bench-800/<GEN_MODEL>/seed_<S>/{1..N}/{<idx>.jpeg,qwen3.5-27b_800_prompts.pt,eval_log.txt}, qwen3.5-27b_log.txt
#   tiif    : $OUT_ROOT/tiif-bench-testmini_eval/<GEN_MODEL>/seed_<S>/{images,eval_results/result_summary_dimension.txt}, tiif_bench_log.txt
#   GEN_MODEL = iter_adhoc_MDP_n<N>_<MODEL_NAME>_512px_repeat1<RL_FT>   (MDP=1, all benches)
#               iter_adhoc_n<N>_...     (MDP=0, genai / tiif)
#               iter_adhoc_niter<N>_... (MDP=0, geneval2; no niter tag when N=2)
#
# Resumable: a seed whose final score files exist is skipped; within a seed, generation leaves a
# .gen_done marker (an unfinished generation restarts from scratch, as his generators wipe their
# output dirs), finished per-iteration scores are kept, and the TIIF judge skips judged prompts. A job
# cut off by the 3 h limit just needs the same sbatch again (or --dependency=afterany:<jobid>).
#
# Aggregate 3 seeds + plot (CPU, login node is fine; the dir is the one holding seed_*/; repeat --run
# to put several models on the same axes):
#   cd /home/mila/s/singhsd/CODE/LatentDCR/tar_reason
#   ~/envs/tar/bin/python eval/stats/geneval2.py    --run "t8=$OUT_ROOT/GenEval2/<GEN_MODEL>"                 --out_dir $OUT_ROOT/GenEval2/figures
#   ~/envs/tar/bin/python eval/stats/genai_bench.py --run "t8=$OUT_ROOT/genai-bench-800/<GEN_MODEL>"          --out_dir $OUT_ROOT/genai-bench-800/figures
#   ~/envs/tar/bin/python eval/stats/tiff_bench.py  --run "t8=$OUT_ROOT/tiif-bench-testmini_eval/<GEN_MODEL>" --out_dir $OUT_ROOT/tiif-bench-testmini_eval/figures
# They read seed_*/log.txt (GenEval2), seed_*/qwen3.5-27b_log.txt (GenAI) and
# seed_*/eval_results/result_summary_dimension.txt + result_summary.xlsx (TIIF). This script prints
# the exact command for its run at the end.

set -uo pipefail
REPO=/scratch/d/dars11/git/tar_reason_pm
MODELS=/scratch/d/dars11/models/tar
BENCH_DIR=/scratch/d/dars11/benchmarks
T2V_DIR=/scratch/d/dars11/git/t2v_metrics_jeet    # copy of jeet's t2v_metrics fork (code only)
MODEL_NAME=${MODEL_NAME:?set MODEL_NAME (label for the results folder)}
BENCH=${BENCH:?set BENCH to geneval2, genai or tiif}
MODEL_PATH=${MODEL_PATH:-${MODELS}/sft/slf_ref_edit_t21_ckpt17000}
LORA_PATH=${LORA_PATH:-}
RL_FT=${RL_FT:-}
MDP=${MDP:-1}
N_ITERS=${N_ITERS:-5}   # official: draft + 4 corrections (user, 2026-10-21; jeet used 4)
NUM_PROMPTS=${NUM_PROMPTS:-1600}   # full official GenAI-Bench (jeet used 800)
SINGLE_PASS=${SINGLE_PASS:-0}
OUT_ROOT=${OUT_ROOT:-/scratch/d/dars11/tar_reason/results_jeet}
REPEAT=1
AR_RES=512
GEN_FAMILY=${GEN_FAMILY:-tar}
case $GEN_FAMILY in
  tar) ;;
  janus)
    JANUS_MODE=${JANUS_MODE:-r1_loop}
    case $JANUS_MODE in
      r1_loop) JANUS_MODEL=${JANUS_MODEL:-midbee/Janus-Pro-R1-7B} ;;
      single)  JANUS_MODEL=${JANUS_MODEL:-deepseek-ai/Janus-Pro-7B} ;;
      *) echo "JANUS_MODE must be r1_loop or single (got '$JANUS_MODE')"; exit 1 ;;
    esac
    JANUS_MAX_IMAGES=${JANUS_MAX_IMAGES:-2}
    N_ITERS=2 ;;
  diffusers)   # single-pass diffusion baselines (eval/baselines/diffusers_gen.py): FLUX.1-dev, SD3.5-Large, Qwen-Image
    DIFF_MODEL=${DIFF_MODEL:?set DIFF_MODEL=flux1-dev | sd35-large | qwen-image}
    case $DIFF_MODEL in flux1-dev|sd35-large) DIFF_RES=${DIFF_RES:-1024} ;; qwen-image) DIFF_RES=${DIFF_RES:-1328} ;;
      *) echo "DIFF_MODEL must be flux1-dev, sd35-large or qwen-image (got '$DIFF_MODEL')"; exit 1 ;; esac
    N_ITERS=2 ;;   # every iteration = the same single-pass image (forced: N_ITERS already defaults to 5 above)
  showo|showo2|bagel|blip3o|emu3)   # unified-model baselines, single pass (eval/baselines/unified_gen.py + backends/)
    N_ITERS=2 ;;
  unicot|alphagrpo)                 # self-reflection loops: iteration 1 = first image, N = final (like Janus R1)
    N_ITERS=${UNI_N_ITERS:-2} ;;
  *) echo "GEN_FAMILY must be tar, janus, diffusers, showo, showo2, bagel, blip3o, emu3, unicot or alphagrpo (got '$GEN_FAMILY')"; exit 1 ;;
esac
case $BENCH in
  geneval2) DEFAULT_SEEDS="2213 713 13" ;;
  genai)    DEFAULT_SEEDS="13 17 91" ;;
  tiif)     DEFAULT_SEEDS="17 91 13" ;;
  *) echo "BENCH must be geneval2, genai or tiif (got '$BENCH')"; exit 1 ;;
esac
read -r -a SEEDS <<< "${SEEDS:-$DEFAULT_SEEDS}"
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  N_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  N_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
  N_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi

TAR_ENV=/home/d/dars11/envs/tar
TAR_PY=${TAR_ENV}/bin/python
JUDGE_PY=/home/d/dars11/envs/qwen_judge/bin/python    # GenEval2 scorer (Qwen3-VL-8B), TIIF judge (Qwen2.5-VL-7B) + summaries
GENAI_PY=/home/d/dars11/envs/genai_judge/bin/python   # t2v_metrics (Qwen3.5-27B); its bin/ has ffmpeg
export HF_HOME=/scratch/d/dars11/hf_home            # all judge weights are in $HF_HOME/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1    # TamIA compute nodes have no internet: fail fast on a missing weight
export SIGLIP2_PATH=${MODELS}/siglip2-so400m-patch14-384
TMP_DIR=${SLURM_TMPDIR:-/tmp}
[ "$GEN_FAMILY" != tar ] || [ -f ${TMP_DIR}/ta_tok.pth ] || cp ${MODELS}/ta_tok.pth ${TMP_DIR}/   # Tar only
AR_MODEL=${MODELS}/ar_dtok_lp_${AR_RES}px.pth
cd "$REPO"

# ---------------------------------------------------------------- per-bench settings (his scripts)
MDP_SUFFIX=""; [ "$MDP" = 1 ] && MDP_SUFFIX="_MDP"
VERBOSE_ARG=()
case $BENCH in
  geneval2)
    EVAL_SET_NAME=GenEval2
    EVAL_SET=${BENCH_DIR}/GenEval2/geneval2_data.jsonl
    GEN_SCRIPT=eval/iterative_generation_adhoc${MDP_SUFFIX}.py
    if [ "$MDP" = 1 ]; then
      GEN_PREFIX=iter_adhoc_MDP_n${N_ITERS}
    else
      N_ITERS_TAG=""; [ "$N_ITERS" -ne 2 ] && N_ITERS_TAG="_niter${N_ITERS}"
      GEN_PREFIX=iter_adhoc${N_ITERS_TAG}
    fi
    BATCH_SIZE=32; VERBOSE_ARG=(--verbose) ;;
  genai)
    EVAL_SET_DIR=GenAI-Image-${NUM_PROMPTS}
    EVAL_SET_NAME=genai-bench-${NUM_PROMPTS}
    GENAI_ROOT=${BENCH_DIR}/GenAI-Bench      # holds GenAI-Image-800 and GenAI-Image-1600 (same json as his t2v_metrics/datasets)
    SCORE_MODEL=qwen3.5-27b
    GEN_SCRIPT=eval/iterative_genai_bench_adhoc${MDP_SUFFIX}.py
    GEN_PREFIX=iter_adhoc${MDP_SUFFIX}_n${N_ITERS}
    BATCH_SIZE=32; [ "$MDP" = 1 ] && BATCH_SIZE=64 ;;
  tiif)
    TIIF_DIR=${BENCH_DIR}/TIIF-Bench
    PROMPTS_DIR=${TIIF_DIR}/data/testmini_prompts
    EVAL_PROMPTS_DIR=${TIIF_DIR}/data/testmini_eval_prompts
    EVAL_SET_NAME=tiif-bench-testmini_eval
    JUDGE_PATH=$(ls -d ${HF_HOME}/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*/ | head -1)
    GEN_SCRIPT=eval/iterative_tiif_bench_adhoc${MDP_SUFFIX}.py
    GEN_PREFIX=iter_adhoc${MDP_SUFFIX}_n${N_ITERS}
    BATCH_SIZE=64; [ "$MDP" = 0 ] && VERBOSE_ARG=(--verbose) ;;
esac
RES_TAG=${AR_RES}px
if [ "$GEN_FAMILY" = janus ]; then
  GEN_SCRIPT=eval/baselines/janus_pro_r1_gen.py
  GEN_PREFIX=janus_${JANUS_MODE}_n${N_ITERS}
  RES_TAG=384px
fi
if [ "$GEN_FAMILY" = diffusers ]; then
  GEN_SCRIPT=eval/baselines/diffusers_gen.py
  GEN_PREFIX=diffusers_n${N_ITERS}
  RES_TAG=${DIFF_RES}px
fi
case $GEN_FAMILY in showo|showo2|bagel|blip3o|emu3|unicot|alphagrpo)
  GEN_SCRIPT=eval/baselines/unified_gen.py
  GEN_PREFIX=${GEN_FAMILY}_n${N_ITERS}
  case $GEN_FAMILY in showo) UNI_RES=${UNI_RES:-512} ;; showo2) UNI_RES=${UNI_RES:-432} ;; emu3) UNI_RES=${UNI_RES:-720} ;; *) UNI_RES=${UNI_RES:-1024} ;; esac
  RES_TAG=${UNI_RES}px ;;
esac
OUTPUT_DIR=${OUT_ROOT}/${EVAL_SET_NAME}
RUN_DIR=${OUTPUT_DIR}/${GEN_PREFIX}_${MODEL_NAME}_${RES_TAG}_repeat${REPEAT}${RL_FT}

LORA_ARG=(); [ -n "$LORA_PATH" ] && LORA_ARG=(--lora_path "$LORA_PATH")
SP_ARG=(); [ "$SINGLE_PASS" = 1 ] && SP_ARG=(--single_pass)
# Shared by all his generator calls.
GEN_COMMON=(--model "$MODEL_PATH" --n_iters "$N_ITERS" --repeat $REPEAT
            --temperature 1.0 --top_k 1200 --top_p 0.95
            --reflect_temperature 1.0 --reflect_top_k 1200 --reflect_top_p 0.95
            --batch_size $BATCH_SIZE --decode_batch_size 64 --reflect_tokens 128
            --gen_seq_len 729 --cfg_scale 4.0
            --ar_path "$AR_MODEL" --encoder_path ${TMP_DIR}/ta_tok.pth --decoder_path ${MODELS}/vq_ds16_t2i.pt
            --draft_img_scale 0 --draft_gen_seq_len 729
            "${VERBOSE_ARG[@]}" "${SP_ARG[@]}" "${LORA_ARG[@]}")
# Generation env (Tar: the tar env + CUDA module). The Janus generator ignores the Tar-only flags above.
GEN_ENV=$TAR_ENV
GEN_CUDA_MODULE=cuda/12.2
if [ "$GEN_FAMILY" = janus ]; then
  GEN_ENV=/home/d/dars11/envs/janus   # torch 2.5.1+cu121 wheel bundles its CUDA libs: no module
  GEN_CUDA_MODULE=""
  read -r -a JANUS_EXTRA <<< "${JANUS_ARGS:-}"
  GEN_COMMON+=(--model_path "$JANUS_MODEL" --mode "$JANUS_MODE" --max_images "$JANUS_MAX_IMAGES" "${JANUS_EXTRA[@]}")
  echo "GEN_FAMILY=janus JANUS_MODE=${JANUS_MODE} JANUS_MODEL=${JANUS_MODEL} JANUS_MAX_IMAGES=${JANUS_MAX_IMAGES} (N_ITERS forced to 2; MODEL_PATH unused)"
fi
if [ "$GEN_FAMILY" = diffusers ]; then
  GEN_ENV=${DIFF_ENV:-/home/d/dars11/envs/diffusers}   # torch 2.6+cu124 wheel bundles its CUDA libs: no module
  GEN_CUDA_MODULE=""
  read -r -a DIFF_EXTRA <<< "${DIFF_ARGS:-}"
  GEN_COMMON+=(--diff_model "$DIFF_MODEL" --height "$DIFF_RES" --width "$DIFF_RES" "${DIFF_EXTRA[@]}")
  [ -n "${DIFF_MODEL_PATH:-}" ] && GEN_COMMON+=(--diff_model_path "$DIFF_MODEL_PATH")
  echo "GEN_FAMILY=diffusers DIFF_MODEL=${DIFF_MODEL} DIFF_RES=${DIFF_RES} DIFF_ARGS=${DIFF_ARGS:-} (single pass, N_ITERS=${N_ITERS} copies; MODEL_PATH unused)"
fi
case $GEN_FAMILY in showo|showo2|bagel|blip3o|emu3|unicot|alphagrpo)
  GEN_ENV=${UNI_ENV:-/home/d/dars11/envs/${GEN_FAMILY}}   # one uv env per family (bagel + unicot can share)
  GEN_CUDA_MODULE=""
  read -r -a UNI_EXTRA <<< "${UNI_ARGS:-}"
  GEN_COMMON+=(--family "$GEN_FAMILY" "${UNI_EXTRA[@]}")
  [ -n "${UNI_MODEL_PATH:-}" ] && GEN_COMMON+=(--model_path "$UNI_MODEL_PATH")
  [ "${RESUME:-0}" = 1 ] && GEN_COMMON+=(--resume)
  echo "GEN_FAMILY=${GEN_FAMILY} UNI_ENV=${GEN_ENV} UNI_RES=${UNI_RES} UNI_ARGS=${UNI_ARGS:-} N_ITERS=${N_ITERS} (MODEL_PATH unused)" ;;
esac

echo "BENCH=${BENCH} MDP=${MDP} N_ITERS=${N_ITERS} SEEDS=${SEEDS[*]} N_GPUS=${N_GPUS} MODEL_NAME=${MODEL_NAME} MODEL_PATH=${MODEL_PATH}" \
     "LORA_PATH=${LORA_PATH:-none} RL_FT=${RL_FT:-none} SINGLE_PASS=${SINGLE_PASS} job=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "RUN_DIR=${RUN_DIR}"

# Generation runs in a subshell with the tar env + CUDA module, so the judges below start from a
# clean environment (the module's CUDA libraries break the judge envs' torch).
run_gen() {
  ( set +u; source ${GEN_ENV}/bin/activate && { [ -z "$GEN_CUDA_MODULE" ] || module load $GEN_CUDA_MODULE; } \
    && export PYTHONPATH=${REPO} \
    && torchrun --standalone --nproc_per_node=$N_GPUS "$GEN_SCRIPT" "${GEN_COMMON[@]}" "$@" )
}
# gen_retry '<success check>' <run_gen args...>: generate on the node's local disk, then copy to scratch.
# Mila's scratch (BeeGFS) returned intermittent "Remote I/O error" (Errno 121) on mkdir from many compute nodes
# (2026-10-21), killing generations that create thousands of folders there. So every output path under $OUT_ROOT
# is redirected to $LOCAL_ROOT for the generation, the result is copied with rsync (retried: it is idempotent),
# absolute image paths in geneval2_results.json are rewritten to the scratch copy, and the check then runs on
# scratch. Up to GEN_TRIES generation attempts, 2 min apart. The check may use $rc (torchrun's exit code).
GEN_TRIES=${GEN_TRIES:-3}
LOCAL_ROOT=${SLURM_TMPDIR:-/tmp/${USER}_bench_jeet}/results_jeet
copy_to_scratch() {
  local t
  for t in 1 2 3 4 5 6; do
    rsync -a "$LOCAL_ROOT/" "$OUT_ROOT/" && return 0
    echo "$(date +%H:%M:%S) rsync to scratch attempt $t failed; retrying in 60 s"; sleep 60
  done
  return 1
}
copy_retry() {   # copy_retry <src> <dst>: cp to scratch, retried (same BeeGFS errors)
  local t; for t in 1 2 3 4 5 6; do cp -f "$1" "$2" && return 0; sleep 30; done; return 1
}
# RESUME=1: resume_begin <scratch dir> seeds the local copy of <scratch dir> from ${OUT_ROOT}_partial and syncs it
# back every RESUME_EVERY s in the background; resume_end stops that and syncs once more. No-ops otherwise.
PARTIAL_ROOT=${OUT_ROOT}_partial
# rsync creates only the last path component: without the parents, every copy_to_scratch fails (TamIA 653114, 2026-10-30).
mkdir -p "$OUT_ROOT" "$PARTIAL_ROOT"
RESUME_PID=""
resume_begin() {
  [ "${RESUME:-0}" = 1 ] || return 0
  local L=${1/#$OUT_ROOT/$LOCAL_ROOT} P=${1/#$OUT_ROOT/$PARTIAL_ROOT}
  mkdir -p "$L" "$P"
  rsync -a "$P/" "$L/" && echo "$(date +%H:%M:%S) resume: seeded $L from $P ($(find "$L/records" -name '*.json' 2>/dev/null | wc -l) prompts done)"
  ( while sleep ${RESUME_EVERY:-600}; do rsync -a "$L/" "$P/" || true; done ) &
  RESUME_PID=$!; RESUME_L=$L; RESUME_P=$P
}
resume_end() {
  [ -n "$RESUME_PID" ] || return 0
  kill $RESUME_PID 2>/dev/null; wait $RESUME_PID 2>/dev/null; RESUME_PID=""
  rsync -a "$RESUME_L/" "$RESUME_P/" || true
}
gen_retry() {
  local check=$1 t rc a f; shift
  local args=(); for a in "$@"; do args+=("${a//$OUT_ROOT/$LOCAL_ROOT}"); done
  for t in $(seq 1 $GEN_TRIES); do
    rm -rf "$LOCAL_ROOT"; mkdir -p "$LOCAL_ROOT"
    [ -n "${RESUME_DIR:-}" ] && resume_begin "$RESUME_DIR"
    run_gen "${args[@]}"; rc=$?
    resume_end
    if copy_to_scratch; then
      while IFS= read -r f; do
        sed -i "s|$LOCAL_ROOT|$OUT_ROOT|g" "$OUT_ROOT/${f#$LOCAL_ROOT/}"
      done < <(find "$LOCAL_ROOT" -name geneval2_results.json)
      eval "$check" && { rm -rf "$LOCAL_ROOT"; return 0; }
    fi
    echo "$(date +%H:%M:%S) generation attempt $t/${GEN_TRIES} failed (exit $rc)"; [ "$t" -lt "$GEN_TRIES" ] && sleep 120
  done
  rm -rf "$LOCAL_ROOT"
  return 1
}
n_images() {     # n_images <dir> <glob>: number of matching files directly in <dir>
  find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l
}
iters_have() {   # iters_have <dir> <relative file>: true if <dir>/<k>/<file> exists for k = 1..N_ITERS
  local k; for k in $(seq 1 $N_ITERS); do [ -s "$1/$k/$2" ] || return 1; done
}

# ================================================================ GenEval2
# jeet's evaluation_optz.py (copied from Mila; `--method soft_tifa` reports AM and GM). The official evaluation.py is not a
# drop-in (methods soft_tifa_am / soft_tifa_gm, one per run), so require the copy.
GENEVAL2_SCORER=evaluation_optz.py
# The scorer hardcodes the hub id of its Qwen3-VL-8B judge; TamIA compute nodes have no internet, so the TamIA copy of
# evaluation_optz.py reads MODEL_ID from GENEVAL2_JUDGE_PATH (patched 2026-10-30; original kept as evaluation_optz.py.orig).
export GENEVAL2_JUDGE_PATH=${GENEVAL2_JUDGE_PATH:-/scratch/d/dars11/models/vlm/Qwen3-VL-8B-Instruct}   # sibling of $MODELS, not under it
if [ "$BENCH" = geneval2 ] && [ ! -f ${BENCH_DIR}/GenEval2/${GENEVAL2_SCORER} ]; then
  echo "missing ${BENCH_DIR}/GenEval2/${GENEVAL2_SCORER}: copy it from Mila (\$SCRATCH/benchmarks/GenEval2/)"; exit 1
fi
geneval2_seed() {
  local SEED=$1 RESULT_DIR=${RUN_DIR}/seed_$1
  local LOG_FILE=${RESULT_DIR}/log.txt
  if iters_have "$RESULT_DIR" results_geneval2.jsonl && grep -q "GenEval2 soft-TIFA scores" "$LOG_FILE" 2>/dev/null; then
    echo "seed ${SEED}: done, skipping"; return 0
  fi
  # Images are scored from wherever they are complete: the scratch copy (.gen_done) or, when this job
  # generated them, the local copy. Some scratch folders refuse new files even on retry (BeeGFS, 2026-10-22),
  # so copying the images to scratch is best effort; only the score files and logs must reach scratch.
  local SCORE_DIR=$RESULT_DIR
  if [ ! -f "$RESULT_DIR/.gen_done" ]; then
    # Success = all iteration outputs on disk, not torchrun's exit code: a rank can segfault at
    # teardown after writing everything (seen 2026-10-20).
    echo "$(date +%H:%M:%S) seed ${SEED}: generation (local disk)"
    local LRES=${RESULT_DIR/#$OUT_ROOT/$LOCAL_ROOT} T OK=0
    for T in $(seq 1 $GEN_TRIES); do
      rm -rf "$LRES"; mkdir -p "$LRES"
      resume_begin "$RESULT_DIR"
      run_gen --out_dir "$LRES" --prompts_file "$EVAL_SET" --geneval2 --seed "$SEED"
      resume_end
      iters_have "$LRES" geneval2_results.json && { OK=1; break; }
      echo "$(date +%H:%M:%S) generation attempt $T/${GEN_TRIES} failed"; sleep 60
    done
    [ $OK = 1 ] || { echo "GenEval2 generation failed (seed ${SEED})"; exit 1; }
    SCORE_DIR=$LRES
    mkdir -p "$RESULT_DIR"
    for T in 1 2 3; do
      if rsync -a "$LRES/" "$RESULT_DIR/"; then
        local f; for f in "$RESULT_DIR"/*/geneval2_results.json; do sed -i "s|$LRES|$RESULT_DIR|g" "$f"; done
        touch "$RESULT_DIR/.gen_done"; break
      fi
      echo "$(date +%H:%M:%S) rsync of images attempt $T failed (scoring uses the local copy)"; sleep 30
    done
    local k; for k in $(seq 1 $N_ITERS); do mkdir -p "$RESULT_DIR/$k" 2>/dev/null; done
  fi
  echo "$(date +%H:%M:%S) seed ${SEED}: scoring ITER-1..ITER-${N_ITERS} (evaluation_optz.py soft_tifa, one iteration per GPU)"
  local FAILED="" START ITER STATUS
  for (( START=1; START<=N_ITERS; START+=N_GPUS )); do
    declare -A PIDS=()
    for (( ITER=START; ITER<START+N_GPUS && ITER<=N_ITERS; ITER++ )); do
      if [ -s "${RESULT_DIR}/${ITER}/results_geneval2.jsonl" ] && grep -q "^GM Score" "${RESULT_DIR}/${ITER}/eval_log.txt" 2>/dev/null; then
        continue
      fi
      # Score into local files, then copy them to scratch (scratch file creation fails intermittently).
      ( L=${LOCAL_ROOT}/geneval2_score_${SEED}_${ITER}; mkdir -p "$L"
        (cd ${BENCH_DIR}/GenEval2 && CUDA_VISIBLE_DEVICES=$(( ITER - START )) env -u PYTHONPATH $JUDGE_PY ${GENEVAL2_SCORER} \
          --benchmark_data "$EVAL_SET" \
          --image_filepath_data "${SCORE_DIR}/${ITER}/geneval2_results.json" \
          --method soft_tifa \
          --output_file "$L/results_geneval2.jsonl") > "$L/eval_log.txt" 2>&1; rc=$?
        copy_retry "$L/eval_log.txt" "${RESULT_DIR}/${ITER}/eval_log.txt" || rc=1
        [ $rc -eq 0 ] && { copy_retry "$L/results_geneval2.jsonl" "${RESULT_DIR}/${ITER}/results_geneval2.jsonl" || rc=1; }
        exit $rc ) &
      PIDS[$ITER]=$!
    done
    for ITER in "${!PIDS[@]}"; do
      STATUS=0; wait ${PIDS[$ITER]} || STATUS=$?
      [ $STATUS -ne 0 ] && FAILED="$FAILED iter-${ITER}:${STATUS}"
    done
    unset PIDS
  done
  : > "$LOG_FILE"
  for ITER in $(seq 1 $N_ITERS); do
    { echo " "; echo "ITER-${ITER}:"; cat "${RESULT_DIR}/${ITER}/eval_log.txt"; } >> "$LOG_FILE"
  done
  if [ -n "$FAILED" ]; then echo "Evaluation failed:$FAILED"; exit 1; fi
  $TAR_PY ${REPO}/eval/analyze/viz_geneval2_iter_results.py \
      --results-dir "$RESULT_DIR" --delta-threshold 0.1 --no_images >> "$LOG_FILE"
  echo "================ DONE (seed ${SEED}) ============="
  cat "$LOG_FILE"
}

# ================================================================ GenAI-Bench
genai_images_complete() {   # every iteration folder holds one <prompt_idx>.jpeg per prompt
  local k; for k in $(seq 1 $N_ITERS); do
    [ "$(n_images "$1/$k" '*.jpeg')" -ge "$NUM_PROMPTS" ] || { echo "iteration $k: $(n_images "$1/$k" '*.jpeg')/${NUM_PROMPTS} images"; return 1; }
  done
}
genai_seed() {
  local SEED=$1 GEN_MODEL=${RUN_DIR#${OUTPUT_DIR}/}/seed_$1
  local SEED_DIR=${OUTPUT_DIR}/${GEN_MODEL}
  local PT=${SCORE_MODEL}_${NUM_PROMPTS}_prompts.pt
  local RESULT_FILE=${SEED_DIR}/${SCORE_MODEL}_log.txt
  if iters_have "$SEED_DIR" "$PT" && [ "$(grep -c '^Tag Group: overall' "$RESULT_FILE" 2>/dev/null)" -ge "$N_ITERS" ]; then
    echo "seed ${SEED}: done, skipping"; return 0
  fi
  if [ ! -f "$SEED_DIR/.gen_done" ]; then
    echo "$(date +%H:%M:%S) seed ${SEED}: generation"
    RESUME_DIR=$SEED_DIR gen_retry 'genai_images_complete "$SEED_DIR"' \
        --gen_model "$GEN_MODEL" --output_dir "$OUTPUT_DIR" --root_dir "$GENAI_ROOT" \
        --eval_set_dir "$EVAL_SET_DIR" --seed "$SEED" \
      && touch "$SEED_DIR/.gen_done" \
      || { echo "GenAI-Bench generation failed (seed ${SEED})"; exit 1; }
  fi
  echo "$(date +%H:%M:%S) seed ${SEED}: VQAScore ${SCORE_MODEL}, ${NUM_PROMPTS} prompts, one iteration per GPU (${N_GPUS} at a time)"
  # His fork's genai_bench/evaluate.py (accepts --num_prompts 800), run from the fork's folder so its
  # t2v_metrics / dataset.py shadow the env's editable upstream install; no tar_reason on PYTHONPATH
  # (t2v_metrics has its own `llava`). Qwen3.5-27B loads from $HF_HOME/hub.
  local ITERS=($(seq 1 $N_ITERS)) STATUS=0 PIDS=() I ITER ITER_LOG PID
  for I in "${!ITERS[@]}"; do
    ITER=${ITERS[$I]}
    ITER_LOG=${SEED_DIR}/${ITER}/eval_log.txt
    if [ -s "${SEED_DIR}/${ITER}/${PT}" ] && grep -q '^Tag Group: overall' "$ITER_LOG" 2>/dev/null; then
      continue
    fi
    # Scores (.pt) and log go to local disk, then are copied to scratch (scratch file creation fails intermittently).
    ( L=${LOCAL_ROOT}/genai_score_${SEED}_${ITER}; mkdir -p "$L"
      (cd ${T2V_DIR} && CUDA_VISIBLE_DEVICES=$((I % N_GPUS)) env -u PYTHONPATH PATH=$(dirname $GENAI_PY):$PATH \
        $GENAI_PY -m genai_bench.evaluate \
        --model ${SCORE_MODEL} \
        --root_dir ${GENAI_ROOT} \
        --output_dir ${OUTPUT_DIR} \
        --gen_model ${GEN_MODEL}/${ITER} \
        --num_prompts ${NUM_PROMPTS} \
        --cache_dir ${HF_HOME}/hub \
        --result_dir "$L") > "$L/eval_log.txt" 2>&1; rc=$?
      copy_retry "$L/eval_log.txt" "$ITER_LOG" || rc=1
      [ $rc -eq 0 ] && { copy_retry "$L/${GEN_MODEL}/${ITER}/${PT}" "${SEED_DIR}/${ITER}/${PT}" || rc=1; }
      exit $rc ) &
    PIDS+=($!)
    if [ ${#PIDS[@]} -eq $N_GPUS ]; then
      for PID in "${PIDS[@]}"; do wait $PID || STATUS=$?; done
      PIDS=()
    fi
  done
  for PID in "${PIDS[@]}"; do wait $PID || STATUS=$?; done
  : > "$RESULT_FILE"
  for ITER in "${ITERS[@]}"; do
    { echo " "; echo "ITER-${ITER}:"; cat "${SEED_DIR}/${ITER}/eval_log.txt"; } | tee -a "$RESULT_FILE"
  done
  if [ $STATUS -ne 0 ]; then
    echo "ERROR: at least one genai_bench.evaluate run failed (exit $STATUS)"; exit $STATUS
  fi
  # His step 3 (viz_genai_bench_iter_results.py ... --no_images) is left out: that script has no
  # --no_images flag, so in his pipeline it exits with a usage error (hidden by `| tee`) and adds
  # nothing to the log. eval/stats/genai_bench.py only reads the ITER blocks above.
  echo "OUTPUT DIR: ${SEED_DIR}"
}

# ================================================================ TIIF-Bench testmini
tiif_seed() {
  local SEED=$1 GEN_MODEL=${RUN_DIR#${OUTPUT_DIR}/}/seed_$1
  local SAVE_DIR=${OUTPUT_DIR}/${GEN_MODEL}
  local IMAGE_DIR=${SAVE_DIR}/images RESULT_DIR=${SAVE_DIR}/eval_results
  local DIM_TXT=${RESULT_DIR}/result_summary_dimension.txt RESULT_FILE=${SAVE_DIR}/tiif_bench_log.txt
  local EXPECTED ITER JUDGED
  EXPECTED=$(grep -c . ${EVAL_PROMPTS_DIR}/*.jsonl | awk -F: '{n+=$2} END {print 2*n}')
  tiif_complete() {   # tiif_complete [dir]: every iteration has all judgments (default: scratch results)
    local d=${1:-$RESULT_DIR}
    for ITER in $(seq 1 $N_ITERS); do
      JUDGED=$(find ${d}/${MODEL_NAME}_iter${ITER} -name '*.json' 2>/dev/null | wc -l)
      [ "$JUDGED" -ge "$EXPECTED" ] || return 1
    done
  }
  if [ -s "$DIM_TXT" ] && [ -s "$RESULT_FILE" ] && { [ -f "$SAVE_DIR/.judge_done" ] || tiif_complete; }; then
    echo "seed ${SEED}: done, skipping"; return 0
  fi
  if [ ! -f "$SAVE_DIR/.gen_done" ]; then
    echo "$(date +%H:%M:%S) seed ${SEED}: generation (short + long)"
    RESUME_DIR=$SAVE_DIR gen_retry '[ "$rc" -eq 0 ] && [ -n "$(ls -A "$IMAGE_DIR" 2>/dev/null)" ]' \
        --gen_model "$GEN_MODEL" --output_dir "$OUTPUT_DIR" --prompts_dir "$PROMPTS_DIR" \
        --eval_model_name "$MODEL_NAME" --seed "$SEED" \
      && touch "$SAVE_DIR/.gen_done" \
      || { echo "TIIF generation failed (seed ${SEED})"; exit 1; }
  fi
  mkdir -p "$RESULT_DIR"
  # The judge writes one folder per (dimension, register, iteration): judge into a local copy (seeded with any
  # results already on scratch, so it stays resumable) and rsync back, for the same scratch mkdir errors as above.
  local JUDGE_DIR=${LOCAL_ROOT}/tiif_judge_seed_${SEED}
  rm -rf "$JUDGE_DIR"; mkdir -p "$JUDGE_DIR"; rsync -a "$RESULT_DIR/" "$JUDGE_DIR/"
  # eval/tiif_bench_vlm_judge.py (TIIF-Bench's prompts, parsing and result schema, local Qwen2.5-VL-7B).
  # Resumable: prompts whose result json exists are skipped, so a failed attempt is simply re-run.
  local JUDGE_ATTEMPTS=3
  run_judge() {
    local ITER=$1 GPU=$2 ITER_LOG=$3 TRY
    for TRY in $(seq 1 ${JUDGE_ATTEMPTS}); do
      [ ${TRY} -gt 1 ] && echo "=== retry ${TRY}/${JUDGE_ATTEMPTS} ===" >> ${ITER_LOG}
      CUDA_VISIBLE_DEVICES=${GPU} env -u PYTHONPATH $JUDGE_PY ${REPO}/eval/tiif_bench_vlm_judge.py \
          --jsonl_dir ${EVAL_PROMPTS_DIR} \
          --image_dir ${IMAGE_DIR} \
          --eval_model ${MODEL_NAME}_iter${ITER} \
          --output_dir ${JUDGE_DIR} \
          --model ${JUDGE_PATH} \
          --batch_size 32 \
          --seed ${SEED} \
          >> ${ITER_LOG} 2>&1 && return 0
      sleep 30
    done
    return 1
  }
  echo "$(date +%H:%M:%S) seed ${SEED}: judging (Qwen2.5-VL-7B), one iteration per GPU"
  local ITERS=($(seq 1 ${N_ITERS})) STATUS=0 START I PIDS PID ITER_LOG
  for (( START=0; START<${#ITERS[@]}; START+=N_GPUS )); do
    PIDS=()
    for (( I=START; I<START+N_GPUS && I<${#ITERS[@]}; I++ )); do
      ITER=${ITERS[$I]}
      ITER_LOG=${SAVE_DIR}/judge_log_iter${ITER}.txt
      : > ${ITER_LOG}
      run_judge ${ITER} $(( I - START )) ${ITER_LOG} &
      PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do wait ${PID} || STATUS=$?; done
  done
  # Copy all judgments back (best effort: some scratch folders refuse new files); everything below
  # (completeness check, summaries, viz) runs on the local copy, and only the few summary files are copied.
  local T; for T in 1 2 3; do rsync -a "$JUDGE_DIR/" "$RESULT_DIR/" && break
    echo "$(date +%H:%M:%S) rsync of judge results attempt $T failed (summaries use the local copy)"; sleep 30; done
  if [ $STATUS -ne 0 ]; then
    echo "ERROR: at least one judging run failed (exit $STATUS)"
    for ITER in "${ITERS[@]}"; do echo "--- iter ${ITER} tail ---"; tail -30 ${SAVE_DIR}/judge_log_iter${ITER}.txt; done
    exit $STATUS
  fi
  for ITER in "${ITERS[@]}"; do
    JUDGED=$(find ${JUDGE_DIR}/${MODEL_NAME}_iter${ITER} -name '*.json' 2>/dev/null | wc -l)
    echo "iter-${ITER}: judged ${JUDGED}/${EXPECTED} (prompt, register) pairs"
  done
  tiif_complete "$JUDGE_DIR" || { echo "ERROR: judgments incomplete; rerun this seed (judged prompts are kept)"; exit 1; }
  local LOG_LOCAL=${JUDGE_DIR}.log DIM_LOCAL=${JUDGE_DIR}/result_summary_dimension.txt
  # TIIF-Bench's own summary scripts.
  (cd ${TIIF_DIR} && env -u PYTHONPATH $JUDGE_PY eval/summary_results.py --input_dir ${JUDGE_DIR}) | tee ${LOG_LOCAL}
  # summary_dimension_results.py exits non-zero even on success (its Tee flushes a closed file at
  # shutdown): success = the report exists.
  rm -f ${DIM_LOCAL}
  (cd ${TIIF_DIR} && env -u PYTHONPATH $JUDGE_PY eval/summary_dimension_results.py \
      --input_excel ${JUDGE_DIR}/result_summary.xlsx --output_txt ${DIM_LOCAL}) 2>&1 | tee -a ${LOG_LOCAL} || true
  if [ ! -s ${DIM_LOCAL} ]; then echo "ERROR: summary_dimension_results.py produced no report"; exit 1; fi
  local VIZ=${LOCAL_ROOT}/tiif_viz_seed_${SEED}; rm -rf ${VIZ}; mkdir -p ${VIZ}; ln -s ${JUDGE_DIR} ${VIZ}/eval_results
  $TAR_PY ${REPO}/eval/analyze/viz_tiif_bench_iter_results.py \
      --results-dir ${VIZ} --model-name ${MODEL_NAME} --delta-threshold 0.1 --no_images | tee -a ${LOG_LOCAL}
  copy_retry ${JUDGE_DIR}/result_summary.xlsx ${RESULT_DIR}/result_summary.xlsx \
    && copy_retry ${DIM_LOCAL} ${DIM_TXT} && copy_retry ${LOG_LOCAL} ${RESULT_FILE} \
    || { echo "ERROR: could not copy the summary files to scratch"; exit 1; }
  : > ${LOCAL_ROOT}/.judge_done; copy_retry ${LOCAL_ROOT}/.judge_done ${SAVE_DIR}/.judge_done
  rm -rf "$JUDGE_DIR" ${VIZ} ${LOG_LOCAL}
  echo "OUTPUT DIR: ${SAVE_DIR}"
}

for SEED in "${SEEDS[@]}"; do
  ${BENCH}_seed "$SEED"
done

case $BENCH in
  geneval2) STATS=geneval2.py ;;
  genai)    STATS=genai_bench.py ;;
  tiif)     STATS=tiff_bench.py ;;
esac
echo "$(date +%H:%M:%S) all seeds done. Aggregate + plot:"
echo "  cd ${REPO} && ${TAR_PY} eval/stats/${STATS} --run \"${MODEL_NAME}${RL_FT}=${RUN_DIR}\" --out_dir ${OUTPUT_DIR}/figures"
