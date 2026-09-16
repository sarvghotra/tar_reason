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
    return args


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen2ForCausalLM
    from llava.train.rl.rollout import EpisodeRollout, RolloutConfig
    from llava.train.rl.train_grpo import ImageDecoder, RL_OBJECTIVE

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    checkpoint = Path(args.checkpoint).resolve()
    state = json.loads((checkpoint/'state.json').read_text())
    if state.get('objective') != RL_OBJECTIVE:
        raise ValueError('Checkpoint is not from the final-pixel RL objective')
    adapter_config = json.loads((checkpoint/'adapter_config.json').read_text())
    # Load the base recorded with this adapter, never the current sft_model link.
    base_path = Path(adapter_config['base_model_name_or_path']).resolve(strict=True)
    rows = load_benchmark(args.benchmark)
    hashes = [sha256_file(checkpoint/'adapter_model.safetensors') if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(hashes, src=0)
    manifest = dict(protocol='instruction_benchmark_final_pixel_v1', checkpoint=str(checkpoint),
                    checkpoint_step=state['step'], checkpoint_state=state, adapter_sha256=hashes[0],
                    base_model=str(base_path), base_config_sha256=sha256_file(base_path/'config.json'),
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
    model = Qwen2ForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16,
                                            attn_implementation=args.attn_implementation)
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False).to(device).eval()
    model.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    image_start = tokenizer.convert_tokens_to_ids('<I0>')
    count = sum(t.startswith('<I') and t[2:-1].isdigit() for t in tokenizer.get_vocab())
    cfg = RolloutConfig(num_rollouts=1, max_refinements=args.max_refinements,
                        max_seq_len=args.max_seq_len, reflect_tokens=args.reflect_tokens,
                        gen_batch_size=args.gen_batch_size)
    rollout = EpisodeRollout(tokenizer, cfg, image_start, count, device)
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
            batch = rollout.run(model, [rows[i] for i in indices], num_rollouts=1)
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
                              image=filename, image_sha256=sha256_file(image_path))
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
