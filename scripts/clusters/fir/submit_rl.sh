#!/bin/bash
# Submit the next RL run with the agreed long allocation and online logging.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export JOB_TIME="${JOB_TIME:-7-00:00:00}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROMPTS_PER_GPU="${PROMPTS_PER_GPU:-4}"
exec bash "$REPO_ROOT/scripts/clusters/fir/submit.sh" scripts/rl_ft/bash.sh "$@"
