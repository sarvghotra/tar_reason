"""Explicit, fingerprinted full-SFT versus RL-adapter evaluation loading."""
import json
from pathlib import Path

from llava.train.rl.geneval2_eval_io import sha256_file


def checkpoint_resets_context(metadata):
    """Old checkpoints and full SFT models use the historical full context."""
    value = metadata.get('checkpoint_state', {}).get('rl_method', {}).get('reset_context_per_turn', False)
    if not isinstance(value, bool):
        raise ValueError('Invalid checkpoint reset_context_per_turn setting')
    return value


def evaluation_resets_context(metadata, requested=None):
    """Context mode for evaluation generation.

    RL adapters always use their checkpoint's mode; an explicit request must
    agree with it. Full SFT checkpoints have no mode and use the historical
    full context unless reset context is explicitly requested.
    """
    saved = checkpoint_resets_context(metadata)
    if requested is None:
        return saved
    if not isinstance(requested, bool):
        raise ValueError('Invalid requested reset_context_per_turn setting')
    if metadata.get('checkpoint_type', 'rl_adapter') == 'sft':
        return requested
    if requested != saved:
        raise ValueError('Requested reset_context_per_turn differs from the RL checkpoint metadata')
    return saved


def checkpoint_metadata(checkpoint, checkpoint_type, rl_objective):
    checkpoint = Path(checkpoint).resolve(strict=True)
    if checkpoint_type == 'rl_adapter':
        state = json.loads((checkpoint/'state.json').read_text())
        if state.get('objective') != rl_objective:
            raise ValueError('Checkpoint is not from the final-pixel RL objective')
        adapter = json.loads((checkpoint/'adapter_config.json').read_text())
        base = Path(adapter['base_model_name_or_path']).resolve(strict=True)
        return dict(checkpoint=str(checkpoint), checkpoint_step=state['step'], checkpoint_state=state,
                    adapter_sha256=sha256_file(checkpoint/'adapter_model.safetensors'),
                    base_model=str(base), base_config_sha256=sha256_file(base/'config.json'))
    if checkpoint_type != 'sft':
        raise ValueError('Unknown checkpoint type')
    if (checkpoint/'adapter_config.json').exists() or (checkpoint/'adapter_model.safetensors').exists():
        raise ValueError('An adapter cannot be evaluated as a full SFT checkpoint')
    config = json.loads((checkpoint/'config.json').read_text())
    if config.get('model_type') != 'qwen2':
        raise ValueError('Expected the Qwen2 first autoregressive model')
    files = ['config.json']
    index = checkpoint/'model.safetensors.index.json'
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
        if not shards or any(Path(s).name != s for s in shards):
            raise ValueError('Invalid checkpoint shard index')
        files += [index.name, *shards]
    else:
        files += ['model.safetensors']
    files += [name for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                               'added_tokens.json', 'vocab.json', 'merges.txt', 'generation_config.json',
                               'trainer_state.json') if (checkpoint/name).exists()]
    for name in files:
        if not (checkpoint/name).is_file() or not (checkpoint/name).stat().st_size:
            raise ValueError(f'Missing/empty checkpoint file: {name}')
    hashes = {name: sha256_file(checkpoint/name) for name in files}
    trainer_state = checkpoint/'trainer_state.json'
    step = json.loads(trainer_state.read_text()).get('global_step') if trainer_state.exists() else None
    return dict(checkpoint=str(checkpoint), checkpoint_type='sft', checkpoint_step=step,
                checkpoint_state={'source': 'sft', 'step': step}, adapter_sha256=None,
                base_model=str(checkpoint), base_config_sha256=hashes['config.json'],
                sft_files_sha256=hashes)


def load_evaluation_policy(metadata, device, attn_implementation):
    import torch
    from transformers import Qwen2ForCausalLM
    model = Qwen2ForCausalLM.from_pretrained(metadata['base_model'], torch_dtype=torch.bfloat16,
                                            attn_implementation=attn_implementation)
    if metadata.get('checkpoint_type', 'rl_adapter') == 'rl_adapter':
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, metadata['checkpoint'], is_trainable=False)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model
