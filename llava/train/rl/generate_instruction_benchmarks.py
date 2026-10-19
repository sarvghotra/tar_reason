"""Generate final RL-policy images for prepared TIIF and GenAI benchmarks."""
import argparse
import json
import os
from contextlib import nullcontext
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from llava.train.rl.geneval2_eval_io import (
    atomic_json, ensure_manifest, sha256_file)
from llava.train.rl.instruction_eval_io import load_benchmark, read_record


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--checkpoint_type', choices=['rl_adapter', 'sft'], default='rl_adapter')
    p.add_argument('--reset_context_per_turn', action=argparse.BooleanOptionalAction, default=None,
                   help='Full SFT only: opt into reset context. RL adapters use (and must match) '
                        'their checkpoint metadata.')
    p.add_argument('--protocol', choices=['native', 'main_adhoc', 'looks_good'], default='native',
                   help='native: the trained EOS policy. main_adhoc: the main branch\'s forced '
                        'one-refinement protocol (llava/train/rl/main_protocol.py). looks_good: '
                        'up to --max_refinements reflect->refine rounds, stopping on a "looks good" '
                        'verdict (EpisodeRollout.run_looks_good); needs an explicit context flag.')
    p.add_argument('--decoding', choices=['sample', 'greedy'], default='sample',
                   help='looks_good only: greedy = argmax image tokens and reflections.')
    p.add_argument('--verdict_rule', choices=['correction', 'anywhere'], default='correction',
                   help='looks_good only: where "looks good" must appear.')
    p.add_argument('--benchmark', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--ar_path', required=True)
    p.add_argument('--encoder_path', required=True)
    p.add_argument('--decoder_path', required=True)
    p.add_argument('--code_manifest', required=True)
    p.add_argument('--attn_implementation', choices=['flash_attention_2', 'sdpa', 'eager'],
                   default='flash_attention_2')
    p.add_argument('--seed', type=int, default=421)
    p.add_argument('--gen_batch_size', type=int, default=16)
    p.add_argument('--max_refinements', type=int, default=3)
    p.add_argument('--reflect_tokens', type=int, default=128)
    p.add_argument('--max_seq_len', type=int, default=4096)
    p.add_argument('--cfg_scale', type=float, default=4.)
    p.add_argument('--report_to', choices=['wandb', 'none'], default='wandb')
    args = p.parse_args()
    if min(args.gen_batch_size, args.reflect_tokens, args.max_seq_len) <= 0 or args.max_refinements < 0:
        p.error('Invalid generation limits')
    if args.protocol == 'main_adhoc' and (args.max_refinements != 1 or args.reset_context_per_turn is not None):
        p.error('main_adhoc requires --max_refinements 1 and no context-mode flag (one refinement, full history)')
    if args.protocol == 'looks_good' and args.reset_context_per_turn is None:
        p.error('looks_good requires an explicit --reset_context_per_turn or --no-reset_context_per_turn')
    if args.protocol != 'looks_good' and (args.decoding != 'sample' or args.verdict_rule != 'correction'):
        p.error('--decoding and --verdict_rule apply only to --protocol looks_good')
    return args


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from llava.train.rl.evaluation_policy import checkpoint_metadata, load_evaluation_policy, evaluation_resets_context
    from llava.train.rl.rollout import EpisodeRollout, RolloutConfig
    from llava.train.rl.train_grpo import ImageDecoder, RL_OBJECTIVE

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    metadata = [checkpoint_metadata(args.checkpoint, args.checkpoint_type, RL_OBJECTIVE)
                if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(metadata, src=0)
    metadata = metadata[0]
    # The main protocol has one refinement on the full history; context mode does not apply.
    # looks_good is an evaluation protocol in its own right: its context mode is
    # chosen explicitly (recorded below) rather than inherited from training.
    reset_context = (False if args.protocol == 'main_adhoc'
                     else args.reset_context_per_turn if args.protocol == 'looks_good'
                     else evaluation_resets_context(metadata, args.reset_context_per_turn))
    base_path = Path(metadata['base_model'])
    rows = load_benchmark(args.benchmark)
    manifest = dict(protocol='instruction_benchmark_final_pixel_v1', **metadata,
                    benchmark=str(Path(args.benchmark).resolve()), benchmark_sha256=sha256_file(args.benchmark),
                    code_sha256=sha256_file(args.code_manifest), n_prompts=len(rows),
                    seed=args.seed, world_size=world, batch_size=args.gen_batch_size,
                    environment={p: version(p) for p in ('torch', 'transformers', 'peft', 'torchvision')},
                    attn_implementation=args.attn_implementation,
                    max_refinements=args.max_refinements, reflect_tokens=args.reflect_tokens,
                    max_seq_len=args.max_seq_len, first_ar=dict(temperature=1, top_k=0, top_p=1),
                    renderer=dict(cfg=args.cfg_scale, temperature=1, top_k=0, top_p=1,
                                  ar=str(Path(args.ar_path).resolve()), encoder=str(Path(args.encoder_path).resolve()),
                                  decoder=str(Path(args.decoder_path).resolve())))
    directory = Path(args.output_dir)
    if reset_context:
        manifest['reset_context_per_turn'] = True
    if args.protocol == 'main_adhoc':
        from llava.train.rl.main_protocol import SETTINGS
        manifest['generation_protocol'] = SETTINGS
        manifest['first_ar'] = dict(temperature=SETTINGS['image_temperature'], top_k=SETTINGS['image_top_k'],
                                    top_p=SETTINGS['image_top_p'], reflection='greedy')
    if args.protocol == 'looks_good':
        manifest['generation_protocol'] = dict(
            protocol='looks_good_v1', max_refinements=args.max_refinements, verdict_rule=args.verdict_rule,
            decoding=args.decoding, reset_context_per_turn=reset_context,
            stop='"looks good" verdict; otherwise a forced refinement up to max_refinements; EOS does not stop')
        if args.decoding == 'greedy':
            manifest['first_ar'] = dict(images='greedy', reflections='greedy')
    directory.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        ensure_manifest(directory, manifest)
    if world > 1:
        dist.barrier()
    identity = ensure_manifest(directory, manifest)
    wandb_run = None
    if rank == 0 and args.report_to == 'wandb':
        import wandb
        wandb_run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'tar_reasoning'),
                               entity=os.environ.get('WANDB_ENTITY'), job_type='instruction_benchmark_generation',
                               name=os.environ.get('RUN_NAME', directory.name),
                               id=os.environ.get('WANDB_RUN_ID', directory.name), resume='allow',
                               config=manifest)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = load_evaluation_policy(metadata, device, args.attn_implementation)
    assert not any(p.requires_grad for p in model.parameters())
    image_start = tokenizer.convert_tokens_to_ids('<I0>')
    count = sum(t.startswith('<I') and t[2:-1].isdigit() for t in tokenizer.get_vocab())
    cfg = RolloutConfig(num_rollouts=1, max_refinements=args.max_refinements,
                        reset_context_per_turn=reset_context,
                        max_seq_len=args.max_seq_len, reflect_tokens=args.reflect_tokens,
                        gen_batch_size=args.gen_batch_size,
                        img_sample=args.decoding == 'sample', reflect_sample=args.decoding == 'sample')
    rollout = EpisodeRollout(tokenizer, cfg, image_start, count, device)
    if args.protocol == 'main_adhoc':
        from llava.train.rl.main_protocol import MainAdhocRollout
        if hasattr(model, 'merge_and_unload'):
            model = model.merge_and_unload()  # main merges LoRA adapters before generating
        rollout = MainAdhocRollout(tokenizer, device, reflect_tokens=args.reflect_tokens)
    # The existing renderer checkpoints contain EasyDict configuration objects.
    # Allow that specific class with PyTorch >=2.6's weights-only loader.
    from easydict import EasyDict
    loader_context = (torch.serialization.safe_globals([EasyDict])
                      if hasattr(torch.serialization, 'safe_globals') else nullcontext())
    with loader_context:
        decoder = ImageDecoder(SimpleNamespace(ar_path=args.ar_path, encoder_path=args.encoder_path,
                                decoder_path=args.decoder_path, gen_seq_len=729, scale=0,
                                cfg_scale=args.cfg_scale), device)
    decoder.tok.requires_grad_(False)
    try:
        # Global batches are fixed before sharding. Re-run an incomplete batch
        # with the same seed; completed prompt records are never replaced.
        for batch_number, start in enumerate(range(0, len(rows), args.gen_batch_size)):
            if batch_number % world != rank:
                continue
            indices = list(range(start, min(start+args.gen_batch_size, len(rows))))
            existing = {i: read_record(directory, i, rows[i], identity) for i in indices}
            if all(existing.values()):
                continue
            torch.manual_seed(args.seed * 1000003 + start)
            batch = (rollout.run_looks_good(model, [rows[i] for i in indices], num_rollouts=1,
                                            verdict_rule=args.verdict_rule)
                     if args.protocol == 'looks_good'
                     else rollout.run(model, [rows[i] for i in indices], num_rollouts=1))
            if (len(batch.trajectories) != len(indices) or
                    any(t.prompt_idx != i for i, t in enumerate(batch.trajectories))):
                raise RuntimeError('Expected exactly one ordered episode per benchmark prompt')
            for index, trajectory in zip(indices, batch.trajectories):
                if existing[index] is not None:
                    continue
                with torch.random.fork_rng(devices=[device]):
                    torch.manual_seed(args.seed * 1000003 + index * 7919 + 17)
                    image = decoder.decode_codes([trajectory.codes])[0]
                torch.cuda.empty_cache()
                filename = f'prompt-{index:04d}.png'
                image_path = directory/filename
                temporary = directory/(filename+'.tmp')
                image.save(temporary, format='PNG')
                os.replace(temporary, image_path)
                record = dict(index=index, protocol_id=identity, row=rows[index],
                              reflections=trajectory.reflections,
                              stop_reason=trajectory.stop_reason, semantic_codes=trajectory.images,
                              sampled_sequence=trajectory.seq, position_kinds=trajectory.kinds,
                              context_segments=[dict(input_ids=s.seq, position_kinds=s.kinds)
                                                for s in trajectory.segments],
                              image=filename, image_sha256=sha256_file(image_path))
                if args.protocol == 'looks_good':
                    record['verdicts'] = trajectory.verdicts
                atomic_json(directory/f'prompt-{index:04d}.json', record)
                print(json.dumps(dict(rank=rank, prompt_index=index,
                                      stop=trajectory.stop_reason)), flush=True)
        if world > 1:
            dist.barrier()
        if rank == 0:
            records = [read_record(directory, i, row, identity) for i, row in enumerate(rows)]
            if not all(records):
                raise RuntimeError('Generation incomplete')
            atomic_json(directory/'generation_complete.json', dict(n_prompts=len(rows), protocol_id=identity))
            if wandb_run:
                wandb_run.log({'generation/n_prompts': len(rows)})
                table = wandb.Table(columns=['index', 'prompt', 'stop', 'image', 'reflections'])
                for record in records[:8]:
                    table.add_data(record['index'], record['row']['prompt'], record['stop_reason'],
                                   wandb.Image(str(directory/record['image'])),
                                   '\n---\n'.join(record['reflections']))
                wandb_run.log({'generation/examples': table})
            print('GENERATION_COMPLETE ' + str(len(rows)), flush=True)
    finally:
        if wandb_run:
            wandb_run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
