#!/bin/bash
set -euo pipefail
export TAR_CLUSTER=fir
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$REPO_ROOT/scripts/submit.sh" "$@"
