#!/bin/bash
# Nibi submissions for the official-judge re-scoring (user, 2026-10-30), driven from Fir over ssh.
#   bash scripts/eval/nibi_official_jobs.sh e450      # TIIF (Qwen2.5-VL-72B) + GenAI (CLIP-FlanT5-XXL) on E-450's images (rsynced from Mila)
#   bash scripts/eval/nibi_official_jobs.sh tar7b_gen # base Tar-7B: generate TIIF + GenAI images with jeet's pipeline, single pass, no judging
#   bash scripts/eval/nibi_official_jobs.sh tar7b     # ... then re-score them (after tar7b_gen finished)
#   bash scripts/eval/nibi_official_jobs.sh baselines [bench ...]   # Mila's baseline images (BAGEL, AlphaGRPO srr/rt2i, Show-o2,
#       BLIP3-o, ...; user 2026-10-31: "do the rest too") rsynced to $R/<bench>/<run>/seed_*: one judge job per (bench, run) that
#       has no complete result yet; MODEL_NAME / N_ITERS are read off the folder name <prefix>_n<N>_<MODEL_NAME>_<res>px_repeat1.
# Preconditions (Nibi): ~/envs/{vllm_judge,clipt5_judge,tar}, $HF_HOME/hub with the 72B + clip-flant5-xxl + clip-vit-336,
# /scratch/dars11/{benchmarks/{TIIF-Bench,GenAI-Bench},git/{tar_reason,t2v_metrics_jeet,t2v_metrics_clipt5},models/tar}.
set -eu
cd /scratch/dars11/git/tar_reason
export PATH=$PATH:/opt/software/slurm/bin
mkdir -p output_dir/slurm_logs
R=/scratch/dars11/tar_reason/results_jeet
LOGS=(-o /scratch/dars11/git/tar_reason/output_dir/slurm_logs/%x_%j.out -e /scratch/dars11/git/tar_reason/output_dir/slurm_logs/%x_%j.err)
ACC=(--account=rrg-bengioy-ad_gpu)
case ${1:?e450 | tar7b_gen | tar7b | baselines} in
  baselines)
    shift; BENCHES=(${@:-tiif genai})
    for B in "${BENCHES[@]}"; do
      case $B in tiif) S=tiif-bench-testmini_eval ;; genai) S=genai-bench-1600 ;; *) echo "unknown bench $B"; exit 1 ;; esac
      for d in $(ls -d $R/$S/*_n[0-9]*_repeat1 2>/dev/null); do
        run=$(basename $d)
        case $run in iter_adhoc_*) continue ;; esac          # ours (E-450, base Tar, AlphaGRPO-Tar): other cases above
        n=$(echo $run | sed -E 's/^[a-z0-9]+_n([0-9]+)_.*/\1/'); name=$(echo $run | sed -E 's/^[a-z0-9]+_n[0-9]+_(.*)_[0-9]+px_repeat1$/\1/')
        seeds=$(ls -d $d/seed_* 2>/dev/null | wc -l); [ $seeds -gt 0 ] || { echo "skip $run: no seeds"; continue; }
        case $B in
          tiif)  done_n=$(ls $d/seed_*/eval_results_qwen72b/result_summary_dimension.txt 2>/dev/null | wc -l) ;;
          genai) done_n=$(for f in $d/seed_*/clip-flant5-xxl_log.txt; do [ "$(grep -c '^Tag Group: overall' $f 2>/dev/null)" -ge $n ] && echo 1; done | wc -l) ;;
        esac
        [ $done_n -ge $seeds ] && { echo "done  $run ($seeds seeds)"; continue; }
        echo "judge $run: MODEL_NAME=$name N_ITERS=$n seeds=$seeds"
        BENCH=$B MODEL_NAME=$name N_ITERS=$n RUN_DIR=$d \
          sbatch --export=ALL "${ACC[@]}" "${LOGS[@]}" -J official_${B}_${name} scripts/eval/official_judges_tamia.sh
      done
    done ;;
  e450)
    for B in tiif genai; do
      case $B in tiif) S=tiif-bench-testmini_eval ;; genai) S=genai-bench-1600 ;; esac
      BENCH=$B MODEL_NAME=darshan_pm1_E_450 N_ITERS=5 RUN_DIR=$R/$S/iter_adhoc_MDP_n5_darshan_pm1_E_450_512px_repeat1 \
        sbatch --export=ALL "${ACC[@]}" "${LOGS[@]}" -J official_${B}_e450 scripts/eval/official_judges_tamia.sh
    done ;;
  tar7b_gen)
    for B in tiif genai; do
      MODEL_NAME=7B MODEL_PATH=/scratch/dars11/models/tar/Tar-7B SINGLE_PASS=1 GEN_ONLY=1 BENCH=$B \
        sbatch --export=ALL "${ACC[@]}" "${LOGS[@]}" --mem=240G -J gen_${B}_tar7b scripts/eval/bench_official_tamia.sh
    done ;;
  tar7b)
    # bench_official_tamia.sh wrote n5 folders (N_ITERS default 5); with SINGLE_PASS every iteration is the same image
    # (md5-identical, checked 2026-10-31), so only iteration 1 is judged.
    for B in tiif genai; do
      case $B in tiif) S=tiif-bench-testmini_eval ;; genai) S=genai-bench-1600 ;; esac
      BENCH=$B MODEL_NAME=7B N_ITERS=1 RUN_DIR=$R/$S/iter_adhoc_MDP_n5_7B_512px_repeat1 \
        sbatch --export=ALL "${ACC[@]}" "${LOGS[@]}" -J official_${B}_tar7b scripts/eval/official_judges_tamia.sh
    done ;;
esac
squeue -u dars11 -o "%.9i %.22j %.8T %.10M %R"
