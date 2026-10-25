"""Launch tooling: srun rank shim, short-array guard, frozen array jobs, uv-mode paths."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'scripts/rl_ft'))
import short_array  # noqa: E402


def shim_env(**slurm):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'MASTER_', 'CUDA_VISIBLE'))}
    env.update(slurm)
    out = subprocess.run(['bash', str(REPO/'scripts/clusters/srun_ddp.sh'), 'env'], env=env,
                         check=True, capture_output=True, text=True).stdout
    return dict(line.split('=', 1) for line in out.splitlines() if '=' in line)


def publish(output_dir, step, resume=False):
    path = Path(output_dir)/('resume' if resume else '')/f'checkpoint-{step}'
    path.mkdir(parents=True)
    (path/'state.json').write_text(json.dumps({'step': step}))


class SrunShimTest(unittest.TestCase):
    def test_bound_task_uses_its_only_gpu_and_rank_comes_from_slurm(self):
        env = shim_env(SLURM_PROCID='3', SLURM_NTASKS='4', SLURM_LOCALID='1', SLURM_JOB_ID='66990001',
                       CUDA_VISIBLE_DEVICES='2', MASTER_ADDR='fc10001')
        self.assertEqual((env['RANK'], env['WORLD_SIZE'], env['LOCAL_RANK']), ('3', '4', '0'))
        self.assertEqual(env['NCCL_P2P_DISABLE'], '1')  # P2P cannot reach hidden sibling GPUs
        self.assertEqual(env['MASTER_ADDR'], 'fc10001')
        self.assertEqual(env['MASTER_PORT'], str(20000 + 66990001 % 20000))

    def test_unbound_tasks_index_the_node_gpus(self):
        env = shim_env(SLURM_PROCID='1', SLURM_NTASKS='2', SLURM_LOCALID='1', SLURM_JOB_ID='5',
                       CUDA_VISIBLE_DEVICES='0,1', MASTER_ADDR='n1')
        self.assertEqual(env['LOCAL_RANK'], '1')
        self.assertNotIn('NCCL_P2P_DISABLE', env)

    @unittest.skipUnless(shutil.which('scontrol'), 'needs scontrol to expand node lists')
    def test_master_is_the_first_allocated_host(self):
        env = shim_env(SLURM_PROCID='0', SLURM_NTASKS='4', SLURM_LOCALID='0', SLURM_JOB_ID='7',
                       CUDA_VISIBLE_DEVICES='0', SLURM_JOB_NODELIST='fc[10003-10004],fc10010')
        self.assertEqual(env['MASTER_ADDR'], 'fc10003')

    def test_refuses_to_run_outside_srun(self):
        with self.assertRaises(subprocess.CalledProcessError):
            shim_env()


class ShortArrayGuardTest(unittest.TestCase):
    def test_complete_when_any_checkpoint_reaches_the_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(short_array.decide(tmp, 500), short_array.CONTINUE)
            publish(tmp, 500, resume=True)
            self.assertEqual(short_array.decide(tmp, 500), short_array.COMPLETE)

    def test_two_elements_without_progress_stall_the_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            publish(tmp, 100)
            short_array.record(tmp, 1, 90, 0)    # progressed 90 -> 100
            self.assertEqual(short_array.decide(tmp, 500), short_array.CONTINUE)
            short_array.record(tmp, 2, 100, 1)   # crashed, no new checkpoint
            self.assertEqual(short_array.decide(tmp, 500), short_array.CONTINUE)
            short_array.record(tmp, 3, 100, 1)
            self.assertEqual(short_array.decide(tmp, 500), short_array.STALLED)
            publish(tmp, 101, resume=True)       # e.g. after a manual fix and resubmission
            short_array.record(tmp, 4, 100, 0)
            self.assertEqual(short_array.decide(tmp, 500), short_array.CONTINUE)

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            publish(tmp, 7)
            self.assertEqual(short_array.main(['latest', '--output_dir', tmp]), 0)
            self.assertEqual(short_array.main(['check', '--output_dir', tmp, '--target', '7']), 10)
            self.assertEqual(short_array.main(['check', '--output_dir', tmp, '--target', '8']), 0)


class ArraySubmissionTest(unittest.TestCase):
    def dry_run(self, tmp, **env):
        root = Path(tmp)/'tar_root'
        base = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'TAR_'))}
        base.update(TAR_ROOT=str(root), TAR_CLUSTER='fir', TAR_DRY_RUN='1', RUN_NAME='unit_run',
                    VIRTUAL_ENV=sys.prefix, **env)
        result = subprocess.run(['bash', str(REPO/'scripts/rl_ft/submit_short_array.sh')], env=base,
                                capture_output=True, text=True, cwd=REPO)
        return result, root/'results/launches/unit_run'

    def test_freezes_code_settings_and_requests_gpus_per_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, launch = self.dry_run(tmp, OBJECTIVE='janus_multi_iteration', GPUS='8', ARRAY_SIZE='5')
            self.assertEqual(result.returncode, 0, result.stderr)
            config = (launch/'config.env').read_text()
            self.assertIn('export OBJECTIVE=janus_multi_iteration', config)
            self.assertIn('LAUNCHER=srun JOB_DEADLINE=1 RESUME_SAVE_STEPS=1', config)
            self.assertEqual(len(set(l.split('=')[0] for l in config.splitlines() if l.startswith('export ')
                                     and ' ' not in l[7:].split('=')[0])),
                             len([l for l in config.splitlines() if l.startswith('export ')
                                  and ' ' not in l[7:].split('=')[0]]))   # no duplicate settings
            batch = (launch/'array.sbatch').read_text()
            for flag in ('--ntasks=8', '--gpus-per-task=h100:1', '--switches=1@00:30:00',
                         '--time=03:00:00', '--signal=B:USR1@600', '--mem-per-gpu=64G'):
                self.assertIn(flag, batch)
            for absent in ('--nodes', '--gpus-per-node', '--mem=0'):
                self.assertNotIn(absent, batch)
            self.assertTrue((launch/'code/llava/train/rl/train_grpo.py').is_file())
            self.assertFalse((launch/'code/.venv').exists())
            self.assertEqual(os.readlink(launch/'code/data'), str(Path(tmp)/'tar_root/shared/data'))
            self.assertIn(f'array_element.sh" "{launch}/config.env"', batch)
            subprocess.run(['bash', '-n', str(launch/'array.sbatch')], check=True)

    def test_refuses_to_overwrite_a_frozen_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.dry_run(tmp)[0].returncode, 0)
            result, _ = self.dry_run(tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('frozen once', result.stderr)

    def test_empty_switches_omits_the_switch_constraint(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, launch = self.dry_run(tmp, SWITCHES='')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('--switches', (launch/'array.sbatch').read_text())


class UvModePathsTest(unittest.TestCase):
    def test_launcher_uses_the_synced_folder_and_one_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'TAR_'))}
            env.update(TAR_ROOT=tmp, TAR_CLUSTER='fir', TAR_DRY_RUN='1', VIRTUAL_ENV=sys.prefix,
                       PATH=f'{Path(sys.executable).parent}:{env["PATH"]}')
            out = subprocess.run(['bash', str(REPO/'scripts/rl_ft/bash.sh')], env=env, cwd=REPO,
                                 capture_output=True, text=True)
            # The dry run stops at the input check (no models in the temporary root).
            text = out.stdout + out.stderr
            self.assertIn(f'{tmp}/shared/models/sft/slf_ref_edit_t21/checkpoint-17000', text)
            self.assertIn(f'{tmp}/shared/models/judges/Qwen3-VL-8B-Instruct', text)
            self.assertNotIn('qwen_reward', text)


class ArrayElementTest(unittest.TestCase):
    """scripts/rl_ft/array_element.sh with a stand-in for bash.sh."""

    def fixture(self, tmp, train_body):
        code = Path(tmp)/'code'
        for rel in ('scripts/rl_ft/array_element.sh', 'scripts/rl_ft/short_array.py',
                    'llava/__init__.py', 'llava/train/__init__.py', 'llava/train/rl/__init__.py',
                    'llava/train/rl/training_lifecycle.py'):
            (code/rel).parent.mkdir(parents=True, exist_ok=True)
            if (REPO/rel).exists():  # llava/train is a namespace package
                shutil.copy2(REPO/rel, code/rel)
        (code/'scripts/rl_ft/bash.sh').write_text('#!/bin/bash\n' + train_body)
        config = Path(tmp)/'config.env'
        config.write_text(f'export RUN_NAME=unit MAX_STEPS=10 TAR_ROOT={tmp} TAR_REPO_ROOT={code}\n')
        return code, config, Path(tmp)/'results/models/unit'

    def element(self, config):
        env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'TAR_'))}
        return subprocess.run(['bash', str(REPO/'scripts/rl_ft/array_element.sh'), str(config)],
                              env=env, capture_output=True, text=True)

    def test_progress_is_recorded_and_the_trainer_status_is_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = ('out={tmp}/results/models/unit/resume/checkpoint-6; mkdir -p $out; '
                    'echo \'{{"step": 6}}\' > $out/state.json; echo "args: $*"; exit 3\n').format(tmp=tmp)
            _, config, out = self.fixture(tmp, body)
            publish(out, 4)
            result = self.element(config)
            self.assertEqual(result.returncode, 3, result.stderr)
            self.assertIn('--stop_file', result.stdout)
            self.assertIn('resume from step 4', result.stdout)
            record = short_array.records(out)[-1]
            self.assertEqual((record['start_step'], record['end_step'], record['exit_code']), (4, 6, 3))

    def test_complete_run_exits_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, config, out = self.fixture(tmp, 'echo TRAINED; exit 1\n')
            publish(out, 10)
            result = self.element(config)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('TRAINED', result.stdout)

    def test_stalled_chain_refuses_to_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, config, out = self.fixture(tmp, 'echo TRAINED; exit 1\n')
            publish(out, 4)
            self.assertEqual(self.element(config).returncode, 1)   # crash 1
            self.assertEqual(self.element(config).returncode, 1)   # crash 2
            result = self.element(config)
            self.assertEqual(result.returncode, 1)
            self.assertNotIn('TRAINED', result.stdout)
            self.assertIn('stalled', result.stdout)

    def test_a_second_trainer_is_refused_while_the_lock_is_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, config, out = self.fixture(tmp, 'echo TRAINED\n')
            out.mkdir(parents=True)
            import fcntl
            with open(out/'active_job.lock', 'w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                result = self.element(config)
            self.assertEqual(result.returncode, 1)
            self.assertIn('training lock', result.stdout)


if __name__ == '__main__':
    unittest.main()
