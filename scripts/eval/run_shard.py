"""Run one rank of a frozen eval_geneval2 / generate_instruction_benchmarks job
as an independent single-GPU Slurm task, without a process group.

Both modules shard work deterministically (global batches of gen_batch_size,
each seeded by its start index, assigned by batch_number % WORLD_SIZE) and only
use torch.distributed for: init/destroy, one broadcast of checkpoint metadata
from rank 0, a barrier after the manifest, and a final barrier before rank 0
summarizes. This wrapper replaces those calls so the frozen code runs unchanged:

  SHARD_MODE=work      process this rank's batches, then exit 0 at the final
                       barrier (rank 0 does NOT summarize: other shards may be
                       running or queued).
  SHARD_MODE=finalize  run as rank 0 after every shard finished: all batches are
                       skipped as complete, then the frozen rank-0 summary runs
                       (metrics.json / generation_complete.json, W&B logging).

Non-zero ranks recompute the broadcast metadata with the same deterministic
function rank 0 uses (it only reads checkpoint files).
Usage: python run_shard.py <module> <module args...>
"""
import argparse
import os
import runpy
import sys

ALLOWED = ('llava.train.rl.eval_geneval2', 'llava.train.rl.generate_instruction_benchmarks')


def main():
    module, argv = sys.argv[1], sys.argv[2:]
    if module not in ALLOWED:
        raise SystemExit(f'run_shard.py does not support {module}')
    mode = os.environ['SHARD_MODE']
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if mode not in ('work', 'finalize') or not 0 <= rank < world:
        raise SystemExit(f'Bad shard settings: mode={mode} rank={rank} world={world}')
    if mode == 'finalize' and rank != 0:
        raise SystemExit('finalize must run as rank 0')
    os.environ['LOCAL_RANK'] = '0'  # the task owns exactly one visible GPU
    known = argparse.ArgumentParser(add_help=False)
    known.add_argument('--checkpoint', required=True)
    known.add_argument('--checkpoint_type', required=True)
    checkpoint_args, _ = known.parse_known_args(argv)

    import torch.distributed as dist
    barriers = [0]

    def barrier(*args, **kwargs):
        barriers[0] += 1
        if barriers[0] > 2:
            raise RuntimeError('Unexpected extra barrier: the frozen module changed')
        if barriers[0] == 2 and mode == 'work':
            print(f'SHARD_DONE rank={rank} world={world}', flush=True)
            raise SystemExit(0)  # inside the module's try/finally: workers are closed

    def broadcast_object_list(objects, src=0, *args, **kwargs):
        if src != 0 or len(objects) != 1:
            raise RuntimeError('Unexpected broadcast: the frozen module changed')
        if rank != 0:
            from llava.train.rl.evaluation_policy import checkpoint_metadata
            from llava.train.rl.train_grpo import RL_OBJECTIVE
            objects[0] = checkpoint_metadata(checkpoint_args.checkpoint, checkpoint_args.checkpoint_type,
                                             RL_OBJECTIVE)

    dist.init_process_group = lambda *args, **kwargs: None
    dist.destroy_process_group = lambda *args, **kwargs: None
    dist.barrier = barrier
    dist.broadcast_object_list = broadcast_object_list
    sys.argv = [module] + argv
    runpy.run_module(module, run_name='__main__', alter_sys=True)
    if mode == 'work' and world > 1:
        raise RuntimeError('Shard finished without reaching the final barrier')
    print(f'SHARD_FINALIZED world={world}', flush=True)


if __name__ == '__main__':
    main()
