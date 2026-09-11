"""Cron entry point: run the GPU smoke test once when an allocation starts."""
import argparse
import fcntl
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def write_status(directory, state, **fields):
    record = dict(state=state, updated=datetime.now(timezone.utc).isoformat(), **fields)
    temporary = directory / "status.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(directory / "status.json")


def check_once(job_id, root, runner=subprocess.run):
    directory = root / "results/logs" / f"gpu-test-{job_id}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "monitor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        # Never launch twice, including after a monitor crash or failed test.
        if (directory / "attempted").exists() or (directory / "finished").exists():
            return
        try:
            query = runner(["squeue", "--noheader", "--jobs", job_id, "--format=%T"],
                           text=True, capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            write_status(directory, "query_error", error=str(exc))
            return
        if query.returncode:
            write_status(directory, "query_error", error=query.stderr.strip())
            return
        states = query.stdout.split()
        if states == ["RUNNING"]:
            # Exclusive creation also protects across hosts if the shared
            # filesystem's flock is configured as local-only.
            try:
                with (directory / "attempted").open("x") as marker:
                    marker.write(datetime.now(timezone.utc).isoformat())
            except FileExistsError:
                return
            write_status(directory, "testing", job_id=job_id)
            shell = (f"cd {shlex.quote(str(root))} && source scripts/cluster_env.sh && "
                     "exec python -u scripts/test_qwen_reward_gpu.py")
            command = ["srun", "--jobid=" + job_id, "--overlap", "--nodes=1", "--ntasks=1",
                       "--cpus-per-task=8", "--gres=gpu:h100:1", "--time=01:00:00",
                       "bash", "-lc", shell]
            try:
                with (directory / "test.log").open("a") as log:
                    result = runner(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=3900,
                                    env={k: v for k, v in os.environ.items()
                                         if not k.startswith("SLURM_")})
                write_status(directory, "passed" if result.returncode == 0 else "failed",
                             job_id=job_id, exit_code=result.returncode)
            except (OSError, subprocess.TimeoutExpired) as exc:
                write_status(directory, "failed", job_id=job_id, error=str(exc))
            (directory / "finished").touch()
        elif states and all(s in ("PENDING", "CONFIGURING", "SUSPENDED") for s in states):
            write_status(directory, "waiting", job_id=job_id, slurm_states=states)
        else:
            write_status(directory, "allocation_unavailable", job_id=job_id, slurm_states=states)
            (directory / "finished").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    args = parser.parse_args()
    if not args.job_id.isdigit():
        parser.error("job_id must be numeric")
    check_once(args.job_id, Path(__file__).resolve().parents[1])
