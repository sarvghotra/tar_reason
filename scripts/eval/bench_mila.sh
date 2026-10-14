#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --open-mode=append
#SBATCH -J bench_mila
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.err

# Official benchmarks for one model on Mila: GenEval2 (800 official prompts, scored with the
# official facebookresearch/GenEval2 evaluation.py + Qwen3-VL-8B) and TIIF-Bench testmini
# (short + long, judged by eval/tiif_bench_vlm_judge.py with Qwen2.5-VL-7B, summarised with the
# official TIIF scripts). Images come from the repo's adhoc generators: draft -> one greedy
# critique -> one refinement ("looks good" keeps the draft), the same single-round loop the RL
# runs train and validate. Draft and final are scored separately (stage 1 = draft, 2 = final).
#
#   MODEL_NAME=sft_t21 sbatch scripts/eval/bench_mila.sh
#   MODEL_NAME=t7mila_ck200 LORA_PATH=$SCRATCH/tar_reason/kept/darshan_t7_mila/checkpoint-200 sbatch scripts/eval/bench_mila.sh
# Base Tar baseline (single pass: plain prompt, no critique; final = draft):
#   MODEL_NAME=tar7b_base BASE_MODEL=$SCRATCH/models/tar/Tar-7B SINGLE_PASS=1 sbatch scripts/eval/bench_mila.sh
# GenAI-Bench (1,600 prompts, VQAScore with Qwen3.5-27B via t2v_metrics in ~/envs/genai_judge) is opt-in:
#   BENCHES="genai" (or "geneval2 tiif genai").
# Optional: SEED (713), BENCHES ("geneval2 tiif"), BASE_MODEL (the t21 SFT checkpoint), SINGLE_PASS (0),
# SMOKE=1 (24 GenEval2 prompts + the TIIF 2d_spatial_relation dimension only; results go to
# <MODEL_NAME>_smoke and are not comparable).
# Every stage leaves a .done marker; resubmitting the same command skips finished stages, so a
# job cut off by the 3 h limit just needs another sbatch (or --dependency=afterany:<jobid>).
# Results: $SCRATCH/tar_reason/bench/<MODEL_NAME>/seed_<SEED>/{geneval2,tiif}/ and summary.json.

set -uo pipefail
REPO=/home/mila/s/singhsd/CODE/LatentDCR/tar_reason
MODELS=/network/scratch/s/singhsd/models/tar
BENCH=/network/scratch/s/singhsd/benchmarks
BASE_MODEL=${BASE_MODEL:-${MODELS}/sft/slf_ref_edit_t21_ckpt17000}
LORA_PATH=${LORA_PATH:-}
MODEL_NAME=${MODEL_NAME:?set MODEL_NAME (a label for the results folder)}
SEED=${SEED:-713}
BENCHES=${BENCHES:-geneval2 tiif}
SMOKE=${SMOKE:-0}
SINGLE_PASS=${SINGLE_PASS:-0}
[ "$SMOKE" = 1 ] && MODEL_NAME=${MODEL_NAME}_smoke
OUT=/network/scratch/s/singhsd/tar_reason/bench/${MODEL_NAME}/seed_${SEED}
N_GPUS=${N_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
TAR_PY=/home/mila/s/singhsd/envs/tar/bin/python
TAR_TORCHRUN=/home/mila/s/singhsd/envs/tar/bin/torchrun
JUDGE_PY=/home/mila/s/singhsd/envs/qwen_judge/bin/python
GENAI_PY=/home/mila/s/singhsd/envs/genai_judge/bin/python   # t2v_metrics, transformers >= 5; its bin/ has ffmpeg (imported by t2v_metrics)
export HF_HOME=/network/scratch/s/singhsd/hf_home          # Qwen3-VL-8B / Qwen2.5-VL-7B caches
export SIGLIP2_PATH=${MODELS}/siglip2-so400m-patch14-384
export PYTHONPATH=${REPO}
mkdir -p "$OUT"
cd "$REPO"

TMP_DIR=${SLURM_TMPDIR:-/tmp}
[ -f ${TMP_DIR}/ta_tok.pth ] || cp ${MODELS}/ta_tok.pth ${TMP_DIR}/
LORA_ARG=()
[ -n "$LORA_PATH" ] && LORA_ARG=(--lora_path "$LORA_PATH")
# Same image sampling as RL training; the critique is greedy inside the generators.
GEN_ARGS=(--model "$BASE_MODEL" "${LORA_ARG[@]}" --seed "$SEED"
          --temperature 1.0 --top_k 1200 --top_p 0.95 --reflect_tokens 128
          --gen_seq_len 729 --draft_gen_seq_len 729 --draft_img_scale 0 --scale 0 --cfg_scale 4.0
          --batch_size 32 --decode_batch_size 32
          --ar_path ${MODELS}/ar_dtok_lp_512px.pth --encoder_path ${TMP_DIR}/ta_tok.pth
          --decoder_path ${MODELS}/vq_ds16_t2i.pt)
[ "$SINGLE_PASS" = 1 ] && GEN_ARGS+=(--single_pass)
echo "N_GPUS=${N_GPUS} MODEL_NAME=${MODEL_NAME} BASE_MODEL=${BASE_MODEL} LORA_PATH=${LORA_PATH:-none} SEED=${SEED} SINGLE_PASS=${SINGLE_PASS} BENCHES=${BENCHES} job=${SLURM_JOB_ID:-local} host=$(hostname)"
GENEVAL2_DATA=${BENCH}/GenEval2/geneval2_data.jsonl
TIIF_EVAL_DIR=${BENCH}/TIIF-Bench/data/testmini_eval_prompts
DIM_ARG=()
if [ "$SMOKE" = 1 ]; then
  mkdir -p "$OUT/smoke_data/tiif_eval"
  head -24 "$GENEVAL2_DATA" > "$OUT/smoke_data/geneval2_24.jsonl"; GENEVAL2_DATA=$OUT/smoke_data/geneval2_24.jsonl
  cp ${TIIF_EVAL_DIR}/2d_spatial_relation_eval_prompts.jsonl "$OUT/smoke_data/tiif_eval/"; TIIF_EVAL_DIR=$OUT/smoke_data/tiif_eval
  DIM_ARG=(--dimensions 2d_spatial_relation)
fi
GENAI_SET=${BENCH}/GenAI-Bench/GenAI-Image-1600
GENAI_LIMIT=0; [ "$SMOKE" = 1 ] && GENAI_LIMIT=24
done_() { [ -f "$OUT/.done_$1" ]; }
mark() { touch "$OUT/.done_$1"; echo "$(date +%H:%M:%S) stage $1 done"; }

# ---------------------------------------------------------------- GenEval2
if [[ " $BENCHES " == *" geneval2 "* ]]; then
  G=$OUT/geneval2
  if ! done_ geneval2_gen; then
    echo "$(date +%H:%M:%S) GenEval2 generation"
    rm -rf "$G"
    $TAR_TORCHRUN --standalone --nproc_per_node=$N_GPUS eval/iterative_generation_adhoc.py "${GEN_ARGS[@]}" \
        --prompts_file "$GENEVAL2_DATA" --out_dir "$G" --repeat 1 --geneval2 \
      && [ -s "$G/1/geneval2_results.json" ] && [ -s "$G/2/geneval2_results.json" ] && mark geneval2_gen \
      || { echo "GenEval2 generation failed"; exit 1; }
  fi
  if ! done_ geneval2_score; then
    echo "$(date +%H:%M:%S) GenEval2 scoring (official evaluation.py, draft on GPU 0, final on GPU 1)"
    pids=()
    for st in 1 2; do
      (cd ${BENCH}/GenEval2 && CUDA_VISIBLE_DEVICES=$(( (st - 1) % N_GPUS )) $JUDGE_PY evaluation.py \
          --benchmark_data "$GENEVAL2_DATA" --image_filepath_data "$G/$st/geneval2_results.json" \
          --method soft_tifa_am --output_file "$G/$st/score_lists.json" > "$G/$st/eval_log.txt" 2>&1) &
      pids+=($!)
    done
    ok=1; for p in "${pids[@]}"; do wait $p || ok=0; done
    [ $ok = 1 ] && mark geneval2_score || { echo "GenEval2 scoring failed; see $G/*/eval_log.txt"; tail -5 $G/*/eval_log.txt; exit 1; }
  fi
fi

# ---------------------------------------------------------------- TIIF-Bench testmini
if [[ " $BENCHES " == *" tiif "* ]]; then
  T=$OUT/tiif
  if ! done_ tiif_gen; then
    echo "$(date +%H:%M:%S) TIIF generation (testmini, short + long)"
    $TAR_TORCHRUN --standalone --nproc_per_node=$N_GPUS eval/iterative_tiif_bench_adhoc.py "${GEN_ARGS[@]}" \
        --prompts_dir ${BENCH}/TIIF-Bench/data/testmini_prompts --output_dir "$T" --gen_model gen \
        --eval_model_name "$MODEL_NAME" --repeat 1 --skip_existing "${DIM_ARG[@]}" \
      && mark tiif_gen || { echo "TIIF generation failed"; exit 1; }
  fi
  if ! done_ tiif_judge; then
    echo "$(date +%H:%M:%S) TIIF judging (Qwen2.5-VL-7B; iter1 = draft on GPU 0, iter2 = final on GPU 1)"
    JUDGE_PATH=$(ls -d ${HF_HOME}/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*/ | head -1)
    pids=()
    for it in 1 2; do
      CUDA_VISIBLE_DEVICES=$(( (it - 1) % N_GPUS )) $JUDGE_PY eval/tiif_bench_vlm_judge.py \
          --jsonl_dir "$TIIF_EVAL_DIR" --image_dir "$T/gen/images" \
          --eval_model "${MODEL_NAME}_iter${it}" --output_dir "$T/eval_results" --model "$JUDGE_PATH" \
          --batch_size 8 --seed "$SEED" > "$T/judge_log_iter${it}.txt" 2>&1 &
      pids+=($!)
    done
    ok=1; for p in "${pids[@]}"; do wait $p || ok=0; done
    [ $ok = 1 ] && mark tiif_judge || { echo "TIIF judging failed; see $T/judge_log_iter*.txt"; tail -5 $T/judge_log_iter*.txt; exit 1; }
  fi
  if ! done_ tiif_summary; then
    (cd ${BENCH}/TIIF-Bench && $JUDGE_PY eval/summary_results.py --input_dir "$T/eval_results" \
        && $JUDGE_PY eval/summary_dimension_results.py --input_excel "$T/eval_results/result_summary.xlsx" \
             --output_txt "$T/eval_results/result_summary_dimension.txt") > "$T/summary_log.txt" 2>&1
    # summary_dimension_results.py writes its report, then raises at exit (its Tee flushes a closed
    # file), so success = the report exists, not the exit code.
    [ -s "$T/eval_results/result_summary_dimension.txt" ] && mark tiif_summary \
      || { echo "TIIF summary failed; see $T/summary_log.txt"; tail -5 "$T/summary_log.txt"; }
  fi
fi

# ---------------------------------------------------------------- GenAI-Bench (image, 1,600 prompts)
if [[ " $BENCHES " == *" genai "* ]]; then
  A=$OUT/genai
  if ! done_ genai_gen; then
    echo "$(date +%H:%M:%S) GenAI-Bench generation"
    LIMIT_SET=$GENAI_SET
    if [ "$GENAI_LIMIT" != 0 ]; then   # smoke: first N prompts only
      LIMIT_SET=$OUT/smoke_data/GenAI-Image-1600; mkdir -p "$LIMIT_SET"; cp $GENAI_SET/genai_skills.json "$LIMIT_SET/"
      $TAR_PY -c "import json,sys; d=json.load(open(sys.argv[1])); k=sorted(d)[:int(sys.argv[3])]; json.dump({x: d[x] for x in k}, open(sys.argv[2], 'w'))" \
          $GENAI_SET/genai_image.json "$LIMIT_SET/genai_image.json" $GENAI_LIMIT
    fi
    $TAR_TORCHRUN --standalone --nproc_per_node=$N_GPUS eval/iterative_genai_bench_adhoc.py "${GEN_ARGS[@]}" \
        --root_dir "$(dirname "$LIMIT_SET")" --eval_set_dir GenAI-Image-1600 --output_dir "$OUT" --gen_model genai \
        --repeat 1 --skip_existing \
      && mark genai_gen || { echo "GenAI-Bench generation failed"; exit 1; }
  fi
  if ! done_ genai_score; then
    echo "$(date +%H:%M:%S) GenAI-Bench VQAScore (Qwen3.5-27B; draft on GPU 0, final on GPU 1)"
    pids=()
    for st in 1 2; do
      CUDA_VISIBLE_DEVICES=$(( (st - 1) % N_GPUS )) PATH=$(dirname $GENAI_PY):$PATH env -u PYTHONPATH $GENAI_PY eval/genai_bench_score.py \
          --set_dir "$GENAI_SET" --image_dir "$A/$st" --out_json "$A/$st/scores_qwen3.5-27b.json" \
          --limit $GENAI_LIMIT > "$A/$st/score_log.txt" 2>&1 &
      pids+=($!)
    done
    ok=1; for p in "${pids[@]}"; do wait $p || ok=0; done
    [ $ok = 1 ] && mark genai_score || { echo "GenAI-Bench scoring failed; see $A/*/score_log.txt"; tail -5 $A/*/score_log.txt; exit 1; }
  fi
fi

# ---------------------------------------------------------------- summary.json
$JUDGE_PY - "$OUT" "$MODEL_NAME" <<'EOF'
import json, os, sys
from scipy.stats import gmean
out, name = sys.argv[1], sys.argv[2]
res = {"model": name}
for st, label in (("1", "draft"), ("2", "final")):
    f = os.path.join(out, "geneval2", st, "score_lists.json")
    if os.path.exists(f):
        lists = json.load(open(f))
        res[f"geneval2_{label}_am"] = 100 * sum(sum(s) / len(s) for s in lists) / len(lists)
        res[f"geneval2_{label}_gm"] = 100 * sum(gmean(s) for s in lists) / len(lists)
        res[f"geneval2_{label}_all_correct"] = 100 * sum(all(x >= 0.5 for x in s) for s in lists) / len(lists)
    f = os.path.join(out, "genai", st, "scores_qwen3.5-27b.json")
    if os.path.exists(f):
        sk = json.load(open(f))["skills"]
        for g in ("all", "basic", "advanced"):
            if sk.get(g) is not None: res[f"genai_{label}_{g}"] = 100 * sk[g]
dim = os.path.join(out, "tiif", "eval_results", "result_summary_dimension.txt")
if os.path.exists(dim):
    res["tiif_summary_file"] = dim
json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
EOF
echo "$(date +%H:%M:%S) all requested stages done for ${MODEL_NAME}"
