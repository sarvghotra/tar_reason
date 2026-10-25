# Killarney (PAICE): 10 nodes x 8 H100-80GB (training) and 168 nodes x 4
# L40S-48GB (evaluation); compute nodes have internet; jobs cannot be submitted
# from /home. Needs an aip-<PI> allocation. uv mode only.
TAR_MODULES=()
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
GPU_TYPE="${GPU_TYPE:-h100}"
