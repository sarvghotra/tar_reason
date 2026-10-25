# tamIA (PAICE): H100-80GB / H200 nodes, whole 4-GPU nodes only, 24 h maximum,
# no internet on compute nodes. Needs an aip-<PI> allocation (not rrg-/def-).
# uv mode only; set SLURM_ACCOUNT=aip-<PI> in config.local.sh.
TAR_MODULES=()
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
GPU_TYPE="${GPU_TYPE:-h100}"
TAR_GPU_REQUEST_STYLE="${TAR_GPU_REQUEST_STYLE:-whole_node}"
