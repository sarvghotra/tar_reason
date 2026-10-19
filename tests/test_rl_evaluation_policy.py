import json
from pathlib import Path
import tempfile
import unittest

from unittest.mock import patch

from llava.train.rl.evaluation_policy import (checkpoint_metadata, evaluation_resets_context,
                                              load_evaluation_policy)

OBJECTIVE = 'final_pixel_qwen3vl_soft_tifa_am_v1'


class EvaluationPolicyTests(unittest.TestCase):
    def test_sft_fingerprint_covers_weights_and_rejects_incomplete_or_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'config.json').write_text(json.dumps({'model_type': 'qwen2'}))
            (p/'trainer_state.json').write_text(json.dumps({'global_step': 14000}))
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'shard.safetensors'}}))
            with self.assertRaisesRegex(ValueError, 'Missing'):
                checkpoint_metadata(p, 'sft', OBJECTIVE)
            (p/'shard.safetensors').write_bytes(b'first')
            first = checkpoint_metadata(p, 'sft', OBJECTIVE)
            self.assertEqual(first['base_model'], str(p))
            self.assertEqual(first['checkpoint_step'], 14000)
            self.assertIsNone(first['adapter_sha256'])
            (p/'shard.safetensors').write_bytes(b'other')
            self.assertNotEqual(first, checkpoint_metadata(p, 'sft', OBJECTIVE))
            (p/'adapter_config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'adapter'):
                checkpoint_metadata(p, 'sft', OBJECTIVE)

    def test_rl_requires_objective_and_recorded_base_without_sft_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            base = p/'base'; base.mkdir()
            (base/'config.json').write_text('{}')
            (p/'state.json').write_text(json.dumps({'step': 200, 'objective': 'old_semantic_reward'}))
            with self.assertRaisesRegex(ValueError, 'final-pixel'):
                checkpoint_metadata(p, 'rl_adapter', OBJECTIVE)
            (p/'state.json').write_text(json.dumps({'step': 200, 'objective': OBJECTIVE}))
            (p/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path': str(base)}))
            (p/'adapter_model.safetensors').write_bytes(b'adapter')
            m = checkpoint_metadata(p, 'rl_adapter', OBJECTIVE)
            self.assertEqual(m['base_model'], str(base))
            self.assertEqual(m['checkpoint_step'], 200)
            self.assertNotIn('sft_files_sha256', m)
            (p/'state.json').unlink()
            with self.assertRaises(FileNotFoundError):
                checkpoint_metadata(p, 'rl_adapter', OBJECTIVE)

    def test_context_mode_sft_opt_in_and_rl_metadata_guard(self):
        sft = {'checkpoint_type': 'sft', 'checkpoint_state': {'source': 'sft', 'step': 17000}}
        self.assertFalse(evaluation_resets_context(sft))  # historical full context
        self.assertTrue(evaluation_resets_context(sft, True))
        self.assertFalse(evaluation_resets_context(sft, False))
        for saved in (None, False, True):
            state = {'rl_method': {} if saved is None else {'reset_context_per_turn': saved}}
            rl = {'checkpoint_state': state}
            self.assertEqual(evaluation_resets_context(rl), bool(saved))
            self.assertEqual(evaluation_resets_context(rl, bool(saved)), bool(saved))
            with self.assertRaisesRegex(ValueError, 'differs'):
                evaluation_resets_context(rl, not bool(saved))
        with self.assertRaises(ValueError):
            evaluation_resets_context(sft, 'true')
        from llava.train.rl import eval_geneval2, generate_instruction_benchmarks
        required = ['x', '--checkpoint', 'c', '--benchmark', 'b', '--output_dir', 'o', '--ar_path', 'a',
                    '--encoder_path', 'e', '--decoder_path', 'd', '--code_manifest', 'm']
        for module, extra in ((eval_geneval2, ['--reward_model', 'r', '--reward_python', 'p']),
                              (generate_instruction_benchmarks, [])):
            for flags, expected in (([], None), (['--reset_context_per_turn'], True),
                                    (['--no-reset_context_per_turn'], False)):
                with self.subTest(module=module.__name__, flags=flags), \
                        patch('sys.argv', required + extra + flags):
                    self.assertIs(module.parse_args().reset_context_per_turn, expected)

    def test_sft_rejects_shards_outside_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'config.json').write_text(json.dumps({'model_type': 'qwen2'}))
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': '../other'}}))
            with self.assertRaisesRegex(ValueError, 'shard index'):
                checkpoint_metadata(p, 'sft', OBJECTIVE)

    def test_real_sft_and_adapter_loading_are_frozen_and_preserve_parameters(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from peft import LoraConfig, PeftModel, get_peft_model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base = root/'base'; adapter = root/'adapter'
            model = Qwen2ForCausalLM(Qwen2Config(vocab_size=24, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=False))
            model.save_pretrained(base)
            loaded = load_evaluation_policy(checkpoint_metadata(base, 'sft', OBJECTIVE), 'cpu', 'eager')
            self.assertNotIsInstance(loaded, PeftModel)
            self.assertFalse(loaded.training)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            self.assertTrue(torch.equal(loaded.lm_head.weight, model.lm_head.weight.to(torch.bfloat16)))
            lora = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj'], task_type='CAUSAL_LM'))
            lora.peft_config['default'].base_model_name_or_path = str(base)
            with torch.no_grad():
                for n, p in lora.named_parameters():
                    if 'lora_B' in n: p.fill_(0.25)
            lora.save_pretrained(adapter)
            (adapter/'state.json').write_text(json.dumps({'step': 200, 'objective': OBJECTIVE}))
            restored = load_evaluation_policy(checkpoint_metadata(adapter, 'rl_adapter', OBJECTIVE), 'cpu', 'eager')
            self.assertIsInstance(restored, PeftModel)
            self.assertFalse(restored.training)
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))
            self.assertTrue(all(torch.all(p == 0.25) for n, p in restored.named_parameters() if 'lora_B' in n))
