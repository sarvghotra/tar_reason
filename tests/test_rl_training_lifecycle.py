import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from llava.train.rl.training_lifecycle import checkpoint_directory, stop_requested

spec = importlib.util.spec_from_file_location('resubmit', Path(__file__).resolve().parents[1]/'scripts/rl_ft/resubmit_until_complete.py')
resubmit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resubmit)


def fill_checkpoint(path, step=225):
    path = Path(path)
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt'):
        (path/name).write_bytes(b'nonempty test state')
    (path/'state.json').write_text(json.dumps({'step': step}))


class TrainingLifecycleTest(unittest.TestCase):
    def test_deadline_disabled_and_boundary_and_stopfile(self):
        self.assertFalse(stop_requested(now=100))
        self.assertFalse(stop_requested(101, now=100))
        self.assertTrue(stop_requested(100, now=100))
        with tempfile.TemporaryDirectory() as tmp:
            flag = Path(tmp)/'stop'
            self.assertFalse(stop_requested(stop_file=flag, now=100))
            flag.touch()
            self.assertTrue(stop_requested(stop_file=flag, now=100))

    def test_atomic_save_is_invisible_until_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'checkpoint-225'
            with checkpoint_directory(tmp, 225) as staging:
                self.assertFalse(target.exists())
                fill_checkpoint(staging)
            self.assertTrue(target.exists())
            self.assertEqual(json.loads((target/'state.json').read_text())['step'], 225)
            with self.assertRaises(FileExistsError):
                with checkpoint_directory(tmp, 225):
                    self.fail('Must not replace a valid checkpoint')

    def test_interrupted_or_incomplete_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path(tmp)/'checkpoint-200'
            previous.mkdir()
            fill_checkpoint(previous, 200)
            with self.assertRaises(RuntimeError):
                with checkpoint_directory(tmp, 225) as staging:
                    (Path(staging)/'optimizer.pt').write_bytes(b'partial')
                    raise RuntimeError('simulated write failure')
            with self.assertRaises(ValueError):
                with checkpoint_directory(tmp, 225):
                    pass
            self.assertEqual(list(Path(tmp).iterdir()), [previous])

    def test_resubmit_requires_progress_and_complete_saved_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ck = root/'checkpoint-225'
            ck.mkdir()
            fill_checkpoint(ck)
            status = dict(start_step=200, step=225, target_step=500, complete=False, checkpoint=str(ck))
            path = root/'training_status_123.json'
            path.write_text(json.dumps(status))
            self.assertEqual(resubmit.continuation_step(root, '123', 500), 225)
            path.write_text(json.dumps(dict(status, start_step=225)))
            with self.assertRaisesRegex(ValueError, 'No progress'):
                resubmit.continuation_step(root, '123', 500)
            path.write_text(json.dumps(status))
            (ck/'optimizer.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing'):
                resubmit.continuation_step(root, '123', 500)

    def test_no_successor_at_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ck = root/'checkpoint-500'
            ck.mkdir()
            fill_checkpoint(ck, 500)
            (root/'training_status_123.json').write_text(json.dumps(dict(start_step=400, step=500,
                target_step=500, complete=True, checkpoint=str(ck))))
            self.assertIsNone(resubmit.continuation_step(root, '123', 500))

    def test_successor_waits_for_parent_and_duplicate_submission_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = ['resubmit', '--output_dir', tmp, '--job_id', '123',
                    '--batch_script', str(root/'resume.sbatch')]
            with patch('sys.argv', argv), patch.object(resubmit, 'continuation_step', return_value=225), \
                    patch.object(resubmit.subprocess, 'run', return_value=SimpleNamespace(stdout='456;fir\n')) as submit:
                resubmit.main()
                command = submit.call_args.args[0]
                self.assertIn('--dependency=afterok:123', command)
                self.assertEqual(json.loads((root/'successor_123.json').read_text())['job_id'], '456')
                resubmit.main()
                submit.assert_called_once()

    def test_ambiguous_submission_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'successor_123.json').write_text(json.dumps({'state': 'submitting'}))
            argv = ['resubmit', '--output_dir', tmp, '--job_id', '123',
                    '--batch_script', str(root/'resume.sbatch')]
            with patch('sys.argv', argv), patch.object(resubmit, 'continuation_step', return_value=225), \
                    patch.object(resubmit.subprocess, 'run') as submit:
                with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                    resubmit.main()
                submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
