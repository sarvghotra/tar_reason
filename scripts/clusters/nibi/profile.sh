# Nibi (Alliance): 36 nodes x 8 H100-80GB (AMD MI300A nodes are not usable by
# the CUDA wheels); compute nodes have internet. uv mode only.
# Confirm the account with `sshare -U` and set it in config.local.sh.
TAR_MODULES=()
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
GPU_TYPE="${GPU_TYPE:-h100}"
