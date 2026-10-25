import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from llava.train.rl.training_lifecycle import (RESUME_SUBDIR, StepClock, checkpoint_directory, checkpoint_dirs,
                                                deadline_allows_step, prune_resume_checkpoints, stop_requested)

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


class ShortJobLifecycleTest(unittest.TestCase):
    def test_deadline_needs_step_validation_and_save_to_fit(self):
        clock = StepClock(dict(step_seconds=[800, 900], validation_seconds=300, save_seconds=20))
        need = 1.25 * 900 + 1.25 * 20  # the longest recent step and the save, with the margin
        self.assertTrue(deadline_allows_step(1000, 1000 + need, clock))
        self.assertFalse(deadline_allows_step(1000, 1000 + need - 1, clock))
        need_val = need + 1.25 * 300
        self.assertTrue(deadline_allows_step(1000, 1000 + need_val, clock, validation_due=True))
        self.assertFalse(deadline_allows_step(1000, 1000 + need_val - 1, clock, validation_due=True))

    def test_unmeasured_validation_counts_as_a_step_and_unmeasured_save_has_a_default(self):
        clock = StepClock(dict(step_seconds=[100]))
        self.assertFalse(deadline_allows_step(0, 125 + 125 + 120 - 1, clock, validation_due=True))
        self.assertTrue(deadline_allows_step(0, 125 + 125 + 120, clock, validation_due=True))

    def test_no_deadline_or_no_measurement_never_stops(self):
        self.assertTrue(deadline_allows_step(10**9, 0, StepClock(dict(step_seconds=[10**6]))))
        self.assertTrue(deadline_allows_step(10**9, 1, StepClock()))

    def test_clock_keeps_a_window_and_round_trips_through_state(self):
        clock = StepClock(window=3)
        for t in (1, 2, 3, 4):
            clock.add_step(t)
        clock.save, clock.validation = 7.0, 9.0
        again = StepClock(json.loads(json.dumps(clock.to_dict())), window=3)
        self.assertEqual(again.steps, [2.0, 3.0, 4.0])
        self.assertEqual((again.save, again.validation), (7.0, 9.0))

    def test_latest_checkpoint_spans_milestones_and_resume_and_prefers_milestones(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step, sub in ((100, ''), (125, ''), (126, RESUME_SUBDIR), (125, RESUME_SUBDIR)):
                path = root/sub/f'checkpoint-{step}'
                path.mkdir(parents=True)
                fill_checkpoint(path, step)
            (root/RESUME_SUBDIR/'checkpoint-127').mkdir()          # unpublished: no state.json
            found = checkpoint_dirs(root)
            self.assertEqual([s for s, _ in found], [100, 125, 125, 126])
            self.assertEqual(found[-1][1], str(root/RESUME_SUBDIR/'checkpoint-126'))
            self.assertEqual(found[2][1], str(root/'checkpoint-125'))  # milestone after its resume twin

    def test_resume_pruning_never_touches_milestones(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step, sub in ((100, ''), (101, RESUME_SUBDIR), (102, RESUME_SUBDIR)):
                path = root/sub/f'checkpoint-{step}'
                path.mkdir(parents=True)
                fill_checkpoint(path, step)
            prune_resume_checkpoints(root, 102)
            self.assertEqual([s for s, _ in checkpoint_dirs(root)], [100, 102])


if __name__ == '__main__':
    unittest.main()
