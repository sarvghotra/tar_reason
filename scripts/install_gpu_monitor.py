"""Install the once-per-minute allocation monitor, preserving existing cron entries."""
import shlex
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
job_id = "59153811"
directory = root / "results/logs" / f"gpu-test-{job_id}"
directory.mkdir(parents=True, exist_ok=True)
marker = f"# tar-gpu-test-{job_id}"
old = subprocess.run(["crontab", "-l"], text=True, capture_output=True)
if old.returncode and "no crontab for" not in old.stderr:
    raise RuntimeError(old.stderr)
command = f"cd {shlex.quote(str(root))} && python3 scripts/monitor_gpu_test.py {job_id}"
entry = (f"* * * * * /bin/bash -lc {shlex.quote(command)} "
         f">> {shlex.quote(str(directory / 'cron.log'))} 2>&1 {marker}")
lines = [line for line in old.stdout.splitlines() if marker not in line]
lines.append(entry)
subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True, check=True)
print(entry)
