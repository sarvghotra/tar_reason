# Fir (Alliance). Local overrides belong in config.local.sh beside this file.
TAR_MODULES=(StdEnv/2023 python/3.10 cuda/12.2)
SLURM_ACCOUNT="${SLURM_ACCOUNT:-rrg-bengioy-ad_gpu}"
GPU_TYPE="${GPU_TYPE:-h100}"

TAR_REWARD_MODULES=(StdEnv/2023 python/3.11)
