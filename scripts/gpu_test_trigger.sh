#!/bin/bash
# Run as a tiny CPU job with --dependency=after:<interactive-job-id>.
set -euo pipefail
cd /home/xiaofeng/tar_reason
module load StdEnv/2023 python/3.10
python3 scripts/monitor_gpu_test.py "${1:?Expected the interactive allocation job ID}"
