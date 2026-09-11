#!/bin/bash
# Run once on a login node with internet access before GPU jobs.
export HF_HUB_OFFLINE=0
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/cluster_env.sh"
tar_init_run "setup"
# TA-Tok contains the vision weights, but its constructor also needs this config.
python - <<'PY'
from huggingface_hub import hf_hub_download

print(hf_hub_download("google/siglip2-so400m-patch14-384", "config.json"))
PY
