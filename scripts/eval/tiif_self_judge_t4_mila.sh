#!/bin/bash
set -e
J=/network/scratch/s/sarvjeet-singh.ghotra
cd /home/mila/s/singhsd/CODE/LatentDCR/tar_reason
export HF_HOME=/network/scratch/s/singhsd/hf_home HF_HUB_OFFLINE=1
~/envs/tar/bin/python eval/tiif_self_judge.py \
  --run_dir $J/git/tar_reason/results/tiif-bench-testmini_eval/iter_adhoc_MDP_n4_E-450_ORC_512px_repeat1_Cric-SFT-t4 \
  --model $J/models/tar/sft/darshan_mr_E450_orc_merged500 \
  --lora_path $J/git/tar_reason/output_dir/rorqual/rl_ft_s2ce_pmix_oracle_t4/sft/final \
  --ta_tok /network/scratch/s/singhsd/models/tar/ta_tok.pth \
  --siglip2 /network/scratch/s/singhsd/models/tar/siglip2-so400m-patch14-384 \
  --out_dir ${OUT:-/network/scratch/s/singhsd/tar_reason/results_t4/tiif_self_judge} "$@"
