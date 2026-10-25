# Fir (Alliance). Local overrides belong in config.local.sh beside this file.
TAR_MODULES=(StdEnv/2023 python/3.10 cuda/12.2)
SLURM_ACCOUNT="${SLURM_ACCOUNT:-rrg-bengioy-ad_gpu}"
GPU_TYPE="${GPU_TYPE:-h100}"
# Legacy cvmfs environment: some GPU nodes need the driver-library shim (cluster_env.sh).
TAR_NVIDIA_DRIVER_SHIM="${TAR_NVIDIA_DRIVER_SHIM:-1}"
# Evaluation placement (scripts/eval/looks_good_suite.py): CPU-only stages and the
# finalize steps (CUDA init, no GPU memory) on the smallest H100 MIG slice.
TAR_CPU_ACCOUNT="${TAR_CPU_ACCOUNT:-def-bengioy_cpu}"
TAR_GPU_TINY="${TAR_GPU_TINY:-nvidia_h100_80gb_hbm3_1g.10gb}"

TAR_REWARD_MODULES=(StdEnv/2023 python/3.11)
