"""Guards for one RL run chained as a Slurm array of short jobs (--array=0-K%1).

  check  --output_dir D --target T    exit 0: train; 10: target reached; 11: stalled
  record --output_dir D --job_id J --start_step S --exit_code C
  latest --output_dir D               print the latest published checkpoint step

Stalled means the last STALL_LIMIT elements each ended without a new
checkpoint, so a crashing configuration cannot consume the whole array.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llava.train.rl.training_lifecycle import checkpoint_dirs  # noqa: E402

PROGRESS = 'array_progress.jsonl'
STALL_LIMIT = 2
CONTINUE, COMPLETE, STALLED = 0, 10, 11


def latest_step(output_dir):
    found = checkpoint_dirs(output_dir)
    return found[-1][0] if found else 0


def records(output_dir):
    path = Path(output_dir)/PROGRESS
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def decide(output_dir, target, stall_limit=STALL_LIMIT):
    if latest_step(output_dir) >= target:
        return COMPLETE
    recent = records(output_dir)[-stall_limit:]
    if len(recent) == stall_limit and all(r['end_step'] <= r['start_step'] for r in recent):
        return STALLED
    return CONTINUE


def record(output_dir, job_id, start_step, exit_code):
    entry = dict(job_id=str(job_id), start_step=int(start_step), end_step=latest_step(output_dir),
                 exit_code=int(exit_code), time=time.time())
    with open(Path(output_dir)/PROGRESS, 'a') as stream:
        stream.write(json.dumps(entry) + '\n')
    return entry


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('check', 'record', 'latest'):
        q = sub.add_parser(name)
        q.add_argument('--output_dir', required=True, type=Path)
        if name == 'check':
            q.add_argument('--target', required=True, type=int)
        if name == 'record':
            q.add_argument('--job_id', required=True)
            q.add_argument('--start_step', required=True, type=int)
            q.add_argument('--exit_code', required=True, type=int)
    args = p.parse_args(argv)
    if args.command == 'latest':
        print(latest_step(args.output_dir))
        return 0
    if args.command == 'record':
        print(json.dumps(record(args.output_dir, args.job_id, args.start_step, args.exit_code)))
        return 0
    status = decide(args.output_dir, args.target)
    print({CONTINUE: 'continue', COMPLETE: f'target {args.target} reached',
           STALLED: f'stalled: the last {STALL_LIMIT} elements made no checkpoint progress'}[status])
    return status


if __name__ == '__main__':
    sys.exit(main())
