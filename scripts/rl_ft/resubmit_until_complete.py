"""Submit one successor after a successful, progressing, checkpointed RL segment."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess


def continuation_step(output_dir, job_id, target):
    root = Path(output_dir)
    status = json.loads((root/f'training_status_{job_id}.json').read_text())
    step = status['step']
    if status['target_step'] != target or status['complete'] != (step >= target):
        raise ValueError('Training status target/completion mismatch')
    checkpoint = root/f'checkpoint-{step}'
    if Path(status['checkpoint']).resolve() != checkpoint.resolve():
        raise ValueError('Unexpected checkpoint path')
    for name in ('state.json', 'adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt'):
        path = checkpoint/name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing completed checkpoint file: {path}')
    if json.loads((checkpoint/'state.json').read_text())['step'] != step:
        raise ValueError('Saved step does not match training status')
    if step >= target:
        return None
    if step <= status['start_step']:
        raise ValueError('No progress; refusing an unbounded resubmission loop')
    return step


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_dir', required=True, type=Path)
    p.add_argument('--job_id', required=True)
    p.add_argument('--batch_script', required=True, type=Path)
    p.add_argument('--target', type=int, default=500)
    args = p.parse_args()
    if not args.job_id.isdigit():
        p.error('--job_id must be a numeric Slurm job ID')
    step = continuation_step(args.output_dir, args.job_id, args.target)
    if step is None:
        print(f'Target {args.target} reached; no successor submitted.', flush=True)
        return
    # Exclusive durable submission intent avoids duplicates if this helper is
    # accidentally invoked twice. An ambiguous interrupted sbatch needs review.
    ledger = args.output_dir/f'successor_{args.job_id}.json'
    with (args.output_dir/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ledger.exists():
            prior = json.loads(ledger.read_text())
            if prior.get('job_id'):
                print(f"Successor already submitted: {prior['job_id']}", flush=True)
                return
            raise RuntimeError('An earlier submission is unresolved; check Slurm before retrying')
        ledger.write_text(json.dumps(dict(previous_job=args.job_id, resume_step=step, state='submitting'))+'\n')
        # The parent still owns the training-directory lock while submitting.
        # A dependency prevents an immediately allocated successor from racing it.
        result = subprocess.run(['sbatch', '--parsable', '--dependency=afterok:' + args.job_id,
                                 str(args.batch_script.resolve())],
                                check=True, text=True, capture_output=True)
        job_id = result.stdout.strip().split(';')[0]
        if not job_id.isdigit():
            raise RuntimeError(f'Unexpected sbatch response: {result.stdout!r}')
        ledger.write_text(json.dumps(dict(previous_job=args.job_id, resume_step=step,
                                         job_id=job_id, state='submitted'))+'\n')
        print(f'Submitted continuation job {job_id} from step {step} toward {args.target}.', flush=True)


if __name__ == '__main__':
    main()
