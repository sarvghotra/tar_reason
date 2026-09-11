#!/bin/bash
# Compatibility entry point: the previous GRPO arguments are obsolete.
source "${TAR_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/cluster_env.sh"
exec bash "$REPO_ROOT/scripts/rl_ft/bash.sh" "$@"
