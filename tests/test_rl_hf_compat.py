"""Transformers-version compatibility helpers used by sampling and loading."""
import unittest

import torch
import transformers
from packaging.version import Version
from peft import LoraConfig, get_peft_model
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

from llava.train.rl import hf_compat


def tiny_peft():
    base = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=24, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0, eos_token_id=2, attn_implementation="eager"))
    return get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"],
                                           task_type="CAUSAL_LM"))


class HFCompatTests(unittest.TestCase):
    def test_dtype_keyword_matches_installed_version(self):
        expected = 'dtype' if Version(transformers.__version__) >= Version('4.56.0') else 'torch_dtype'
        self.assertEqual(hf_compat.dtype_kwargs(torch.bfloat16), {expected: torch.bfloat16})

    def test_no_model_defaults_keyword_only_before_5(self):
        self.assertEqual(hf_compat.NO_MODEL_DEFAULTS,
                         {} if hf_compat.TRANSFORMERS_5 else {'use_model_defaults': False})

    def test_guard_blanks_both_levels_and_restores_exactly(self):
        for wrapper_owns_config in (False, True):
            with self.subTest(wrapper_owns_config=wrapper_owns_config):
                model = tiny_peft()
                base = model.get_base_model()
                hostile = GenerationConfig(do_sample=True, top_k=1, temperature=0.3)
                base.generation_config = hostile
                if wrapper_owns_config:
                    model.generation_config = GenerationConfig(top_p=0.1)
                wrapper_before = vars(model).get('generation_config')
                with hf_compat.without_model_generation_defaults(model):
                    if hf_compat.TRANSFORMERS_5:
                        for level in (model, base):
                            self.assertEqual(level.generation_config.top_k, None)
                            self.assertEqual(level.generation_config.temperature, None)
                            self.assertEqual(level.generation_config.top_p, None)
                    else:
                        self.assertIs(base.generation_config, hostile)
                self.assertIs(base.generation_config, hostile)
                self.assertIs(vars(model).get('generation_config'), wrapper_before)
                self.assertEqual('generation_config' in vars(model), wrapper_owns_config)

    def test_guard_restores_after_an_exception(self):
        model = tiny_peft()
        base = model.get_base_model()
        before = base.generation_config
        with self.assertRaises(RuntimeError):
            with hf_compat.without_model_generation_defaults(model):
                raise RuntimeError('boom')
        self.assertIs(base.generation_config, before)
        self.assertNotIn('generation_config', vars(model))

    def test_guard_tolerates_objects_without_a_generation_config(self):
        with hf_compat.without_model_generation_defaults(object()):
            pass


if __name__ == '__main__':
    unittest.main()
