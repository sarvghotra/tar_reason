"""Node-local staging of frozen inputs and canonical paths after staged loads."""
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llava.train.rl import staging


def make_model_dir(root):
    root = Path(root)
    root.mkdir(parents=True)
    (root/'config.json').write_text('{"model_type": "qwen2"}')
    (root/'model-00001-of-00002.safetensors').write_bytes(b'a' * 1000)
    (root/'model-00002-of-00002.safetensors').write_bytes(b'b' * 500)
    (root/'tokenizer.json').write_text('{}')
    (root/'optimizer.pt').write_bytes(b'x' * 5000)      # never read by from_pretrained
    (root/'rng_state_0.pth').write_bytes(b'r' * 10)
    return root


def stage_in_child(args):
    paths, stage_root = args
    return staging.stage_inputs(paths, stage_root, log=lambda *_: None)


class StagingTest(unittest.TestCase):
    def test_without_stage_root_paths_map_to_themselves(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('TAR_STAGE_DIR', None)
            self.assertEqual(staging.stage_inputs(['/a', '/b', None, '/a']), {'/a': '/a', '/b': '/b'})

    def test_copies_model_files_and_single_files_and_skips_training_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = make_model_dir(Path(tmp)/'shared/sft')
            renderer = Path(tmp)/'shared/ar.pth'
            renderer.write_bytes(b'w' * 300)
            mapping = staging.stage_inputs([str(model), str(renderer)], Path(tmp)/'local', log=lambda *_: None)
            staged_model, staged_renderer = Path(mapping[str(model)]), Path(mapping[str(renderer)])
            self.assertNotEqual(staged_model, model)
            self.assertEqual(sorted(p.name for p in staged_model.iterdir()),
                             ['config.json', 'model-00001-of-00002.safetensors',
                              'model-00002-of-00002.safetensors', 'tokenizer.json'])
            self.assertEqual(staged_renderer.read_bytes(), renderer.read_bytes())
            self.assertEqual((staged_model/'model-00002-of-00002.safetensors').read_bytes(), b'b' * 500)

    def test_reuses_a_complete_copy_and_recopies_a_changed_or_damaged_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = make_model_dir(Path(tmp)/'shared/sft')
            local = Path(tmp)/'local'
            first = Path(staging.stage_inputs([str(model)], local, log=lambda *_: None)[str(model)])
            shard = first/'model-00001-of-00002.safetensors'
            mtime = shard.stat().st_mtime_ns
            again = Path(staging.stage_inputs([str(model)], local, log=lambda *_: None)[str(model)])
            self.assertEqual(again, first)
            self.assertEqual(shard.stat().st_mtime_ns, mtime)  # not copied again
            shard.write_bytes(b'')  # e.g. a truncated copy from a killed job
            staging.stage_inputs([str(model)], local, log=lambda *_: None)
            self.assertEqual(shard.read_bytes(), b'a' * 1000)
            (model/'model-00002-of-00002.safetensors').write_bytes(b'c' * 700)  # source changed
            staging.stage_inputs([str(model)], local, log=lambda *_: None)
            self.assertEqual((first/'model-00002-of-00002.safetensors').read_bytes(), b'c' * 700)

    def test_concurrent_processes_agree_on_one_complete_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = make_model_dir(Path(tmp)/'shared/sft')
            local = Path(tmp)/'local'
            with multiprocessing.get_context('spawn').Pool(4) as pool:
                results = pool.map(stage_in_child, [([str(model)], str(local))] * 4)
            self.assertEqual(len({r[str(model)] for r in results}), 1)
            staged = Path(results[0][str(model)])
            self.assertEqual((staged/'model-00001-of-00002.safetensors').read_bytes(), b'a' * 1000)
            self.assertFalse([p for p in staged.parent.iterdir() if p.name.startswith('.incoming-')])

    def test_insufficient_space_falls_back_to_shared_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = make_model_dir(Path(tmp)/'shared/sft')
            usage = type('Usage', (), {'free': 10})()
            with patch.object(staging.shutil, 'disk_usage', return_value=usage):
                mapping = staging.stage_inputs([str(model)], Path(tmp)/'local', log=lambda *_: None)
            self.assertEqual(mapping[str(model)], str(model))

    def test_directory_without_safetensors_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'m').mkdir()
            (Path(tmp)/'m'/'config.json').write_text('{}')
            with self.assertRaises(ValueError):
                staging.stage_inputs([str(Path(tmp)/'m')], Path(tmp)/'local', log=lambda *_: None)


class StagedPolicyLoadTest(unittest.TestCase):
    def test_adapter_records_the_canonical_base_path(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from llava.train.rl import train_grpo
        from test_rl_policy import policy_args
        with tempfile.TemporaryDirectory() as tmp:
            canonical, staged = Path(tmp)/'canonical', Path(tmp)/'staged'
            Qwen2ForCausalLM(Qwen2Config(
                vocab_size=24, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                attention_dropout=0.0, pad_token_id=0, eos_token_id=2)).save_pretrained(staged)
            args = policy_args(str(canonical), gradient_checkpointing=False)
            model = train_grpo.load_policy(args, torch.device('cpu'), load_path=str(staged))
            self.assertEqual(model.peft_config['default'].base_model_name_or_path, str(canonical))
            model.save_pretrained(Path(tmp)/'adapter')
            saved = json.loads((Path(tmp)/'adapter'/'adapter_config.json').read_text())
            self.assertEqual(saved['base_model_name_or_path'], str(canonical))


if __name__ == '__main__':
    unittest.main()
