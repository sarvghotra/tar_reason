# Sourced by the Mila ports of jeet's *_MDP_vllm.sh launchers (see port.sh). Replaces `mamba shell hook`:
# `mamba activate <env>` puts his env (read-only, /network/scratch/s/sarvjeet-singh.ghotra/installs/miniforge3/envs)
# first on PATH; his conda.sh points at his /home, which we cannot read.
JEET_ENVS=/network/scratch/s/sarvjeet-singh.ghotra/installs/miniforge3/envs
BASE_PATH_JEETPIPE=${BASE_PATH_JEETPIPE:-$PATH}
mamba() {
  if [ "$1" = activate ]; then
    export CONDA_PREFIX=${JEET_ENVS}/$2
    export PATH=${CONDA_PREFIX}/bin:${BASE_PATH_JEETPIPE}
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
  fi
}
export HF_HOME=${HF_HOME:-/network/scratch/s/singhsd/hf_home}
export VLLM_CACHE_ROOT=/network/scratch/s/singhsd/.cache/vllm
