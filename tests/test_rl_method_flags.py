"""Adversarial checks for opt-in GRPO normalization and KL gradients."""
import copy
import itertools
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig, get_peft_model

from llava.train.rl import train_grpo as trainer
from llava.train.rl.grpo import GRPOConfig, KIND_IMG, KIND_TXT, grpo_loss, normalize_groups
from llava.train.rl.rollout import Trajectory, RolloutBatch
from test_rl_policy import tiny_base, small_rollout, policy_args, training_batch


def distributed_method_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + directory + '/rendezvous',
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(53)
        model = get_peft_model(tiny_base(), LoraConfig(r=2, lora_alpha=4,
                              target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'lora_B' in name:
                    parameter.normal_(0, .1)
        central = copy.deepcopy(model)
        batches = [training_batch(i) for i in (0, 1)]
        # Deliberately unequal total action counts/weights across ranks.
        batches[1].trajectories[0].seq.extend([7, 7, 7])
        batches[1].trajectories[0].kinds.extend([KIND_TXT]*3)
        args = policy_args(train_micro_batch=1, num_ppo_epochs=2, max_grad_norm=1e6)
        cfg = GRPOConfig(length_normalization='constant', kl_gradient_correction=True)
        rollout = small_rollout()
        rollout.cfg.max_seq_len = 32
        def train(candidate, batch):
            opt = torch.optim.SGD([p for p in candidate.parameters() if p.requires_grad], lr=2.)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
            return trainer.train_on_rollouts(candidate, rollout, batch, args, cfg,
                                             opt, scheduler, torch.device('cpu'))
        distributed_metrics = trainer.reduce_mean(train(model, batches[rank]), torch.device('cpu'))
        if rank == 0:
            combined = RolloutBatch(batches[0].prompts + batches[1].prompts,
                                    batches[0].trajectories + batches[1].trajectories)
            with patch.object(trainer.dist, 'is_initialized', return_value=False):
                central_metrics = train(central, combined)
            for a, b in zip(model.parameters(), central.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
            for key in ('loss', 'pg_loss', 'kl', 'clip_frac', 'approx_kl_old'):
                torch.testing.assert_close(torch.tensor(distributed_metrics[key]),
                                           torch.tensor(central_metrics[key]), rtol=1e-5, atol=1e-7)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class MethodFlagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cli_defaults_opt_ins_and_explicit_disable(self):
        required = ['train_grpo', '--model_name_or_path', 'unused', '--data_path', 'unused',
                    '--output_dir', 'unused']
        for extra, expected in [([], ('episode', False)),
                                (['--length_normalization', 'constant'], ('constant', False)),
                                (['--kl_gradient_correction'], ('episode', True)),
                                (['--length_normalization', 'constant', '--kl_gradient_correction'],
                                 ('constant', True)),
                                (['--kl_gradient_correction', '--no-kl_gradient_correction'],
                                 ('episode', False))]:
            with self.subTest(extra=extra), patch('sys.argv', required + extra):
                args = trainer.parse_args()
                self.assertEqual((args.length_normalization, args.kl_gradient_correction), expected)

    def test_legacy_loss_and_gradient_are_preserved(self):
        lp = torch.tensor([-.4, -1.2, -.8], dtype=torch.float64, requires_grad=True)
        old, ref = lp.detach() + .3, lp.detach() - .2
        adv, w = torch.tensor([.4, -.2, .1]), torch.tensor([.2, .3, .5])
        ratio = (lp-old).exp()
        d = ref-lp
        expected = (w * (-torch.minimum(ratio*adv, ratio.clamp(.8, 1.2)*adv)
                         + .01*(d.exp()-d-1))).sum()/2
        actual, _ = grpo_loss(lp, old, ref, adv, w, torch.tensor([1, 2, 2]), 2, GRPOConfig())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        ga = torch.autograd.grad(actual, lp, retain_graph=True)[0]
        ge = torch.autograd.grad(expected, lp)[0]
        torch.testing.assert_close(ga, ge, rtol=0, atol=0)

    def test_constant_weights_preserve_eos_masks_and_phase_weights(self):
        rollout = small_rollout()
        rollout.cfg.max_seq_len = 32
        rollout.pad_id = 2  # EOS deliberately equals padding.
        trajectories = [Trajectory(0, [1, 8, 2], [0, KIND_IMG, KIND_TXT], adv=.2, stop_reason='eos'),
                        Trajectory(0, [1, 8, 9, 7, 3], [0, 1, 1, 2, 2], adv=-.2,
                                   stop_reason='max_refinements')]
        legacy = rollout.build_training_rows(trajectories, .7)
        explicit = rollout.build_training_rows(trajectories, .7, 'episode')
        constant = rollout.build_training_rows(trajectories, .7, 'constant')
        for key in legacy:
            torch.testing.assert_close(legacy[key], explicit[key], rtol=0, atol=0)
            if key != 'weight':
                torch.testing.assert_close(legacy[key], constant[key], rtol=0, atol=0)
        torch.testing.assert_close(constant['weight'][0], torch.tensor([0., 1/32, .7/32, 0., 0.]))
        torch.testing.assert_close(constant['weight'][1], torch.tensor([0., 1/32, 1/32, .7/32, .7/32]))
        with self.assertRaises(ValueError):
            rollout.build_training_rows(trajectories, 1, 'typo')

    def test_constant_normalization_repairs_exact_reward_direction_counterexample(self):
        rollout = small_rollout()
        rollout.cfg.max_seq_len = 32
        rewards, lengths = [.9, .8, .1], [8, 32, 32]
        theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        lp = (theta*torch.tensor([1., -2., 1.], dtype=torch.float64)).log_softmax(-1)
        p = lp.exp()
        target = torch.autograd.grad((p*torch.tensor(rewards, dtype=torch.float64)).sum(),
                                     theta, retain_graph=True)[0]
        objectives = {mode: theta*0 for mode in ('episode', 'constant')}
        for choices in itertools.product(range(3), repeat=4):
            adv, _ = normalize_groups([rewards[i] for i in choices], [0]*4, GRPOConfig())
            ts = [Trajectory(0, [1]+[8]*lengths[i], [0]+[1]*lengths[i], adv=a)
                  for i, a in zip(choices, adv)]
            token_lp = torch.cat([torch.cat([lp[i:i+1], torch.zeros(lengths[i]-1)]) for i in choices])
            for mode in objectives:
                rows = rollout.build_training_rows(ts, 1, mode)
                sel = rows['train_mask']
                loss, _ = grpo_loss(token_lp, token_lp.detach(), token_lp.detach(),
                                    rows['adv'][sel], rows['weight'][sel], rows['pos_kind'][sel], 4,
                                    GRPOConfig(kl_coef=0, length_normalization=mode))
                objectives[mode] = objectives[mode] + p[list(choices)].prod().detach()*loss
        for mode, expected_sign in [('episode', -1), ('constant', 1)]:
            ascent = -torch.autograd.grad(objectives[mode], theta, retain_graph=True)[0]
            self.assertGreater(float(ascent*target)*expected_sign, 0)
        # Group mean including self scales the unbiased gradient by (G-1)/G.
        ascent = -torch.autograd.grad(objectives['constant'], theta)[0]
        torch.testing.assert_close(ascent, target*.75/32, rtol=1e-6, atol=1e-9)

    def test_corrected_kl_matches_exact_gradient_on_and_off_policy(self):
        p = torch.tensor([.7, .2, .1], dtype=torch.float64)
        q = torch.tensor([.2, .3, .5], dtype=torch.float64)
        for old in (p, torch.tensor([.4, .4, .2], dtype=torch.float64)):
            for correction in (False, True):
                with self.subTest(old=old.tolist(), correction=correction):
                    logits = p.log().requires_grad_()
                    lp = logits.log_softmax(-1)
                    loss, _ = grpo_loss(lp, old.log(), q.log(), torch.zeros(3), old,
                                        torch.full((3,), KIND_TXT), 1,
                                        GRPOConfig(kl_coef=1, kl_gradient_correction=correction))
                    expected = (lp.exp()*(lp-q.log())).sum()
                    actual_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
                    target_grad = torch.autograd.grad(expected, logits)[0]
                    if correction:
                        # Includes ratios outside [0.8, 1.2]: KL must use an unclipped ratio.
                        torch.testing.assert_close(loss, expected)
                        torch.testing.assert_close(actual_grad, target_grad)
                    else:
                        self.assertFalse(torch.allclose(actual_grad, target_grad))

    def test_zero_kl_coefficient_keeps_policy_gradient_identical(self):
        values = []
        for correction in (False, True):
            lp = torch.tensor([-.2, -1.], requires_grad=True)
            loss, _ = grpo_loss(lp, torch.tensor([-1., -.2]), torch.tensor([-.8, -.4]),
                                torch.tensor([.2, -.2]), torch.ones(2), torch.tensor([1, 2]), 2,
                                GRPOConfig(kl_coef=0, kl_gradient_correction=correction))
            values.append((loss.detach(), torch.autograd.grad(loss, lp)[0]))
        for a, b in zip(*values):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_updates_and_objective_metrics_are_microbatch_invariant_all_modes(self):
        torch.manual_seed(52)
        base = get_peft_model(tiny_base(), LoraConfig(r=2, lora_alpha=4,
                              target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        for mode, correction in itertools.product(('episode', 'constant'), (False, True)):
            with self.subTest(mode=mode, correction=correction):
                models, metrics = [], []
                for micro in (1, 2):
                    model = copy.deepcopy(base)
                    rollout = small_rollout()
                    rollout.cfg.max_seq_len = 32
                    args = policy_args(train_micro_batch=micro, num_ppo_epochs=2)
                    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.5)
                    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
                    metrics.append(trainer.train_on_rollouts(model, rollout, training_batch(0), args,
                                   GRPOConfig(length_normalization=mode, kl_gradient_correction=correction),
                                   opt, scheduler, torch.device('cpu')))
                    models.append(model)
                for a, b in zip(models[0].parameters(), models[1].parameters()):
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
                for key in ('loss', 'pg_loss', 'kl', 'clip_frac', 'approx_kl_old'):
                    self.assertAlmostEqual(metrics[0][key], metrics[1][key], places=6)
                for m in metrics:
                    self.assertAlmostEqual(m['loss'], m['pg_loss'] + .01*m['kl'], places=6)

    def test_corrected_distributed_update_matches_combined_batch(self):
        with tempfile.TemporaryDirectory(prefix='tar-method-distributed-') as directory:
            mp.spawn(distributed_method_worker, args=(directory,), nprocs=2, join=True)

    def test_checkpoint_save_and_legacy_resume_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            args = policy_args(output_dir=directory, save_total_limit=3, max_seq_len=32,
                               length_normalization='constant', kl_gradient_correction=True)
            model = get_peft_model(tiny_base(), LoraConfig(r=2, target_modules=['q_proj'],
                                                          task_type='CAUSAL_LM'))
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
            trainer.save_checkpoint(model, opt, scheduler, 1, args, torch.device('cpu'))
            ck = Path(directory)/'checkpoint-1'
            state = trainer.validate_resume_checkpoint(ck, args)
            self.assertEqual(state['rl_method'], trainer.rl_method_settings(args))
            for key, value in [('length_normalization', 'episode'), ('kl_gradient_correction', False),
                               ('max_seq_len', 64)]:
                changed = copy.copy(args)
                setattr(changed, key, value)
                with self.assertRaisesRegex(ValueError, 'RL method settings differ'):
                    trainer.validate_resume_checkpoint(ck, changed)
            # Original pixel checkpoints have no method metadata; legacy flags still work.
            (ck/'state.json').write_text(json.dumps({'step': 1, 'objective': trainer.RL_OBJECTIVE}))
            legacy = policy_args(max_seq_len=64)
            self.assertEqual(trainer.validate_resume_checkpoint(ck, legacy)['step'], 1)
            with self.assertRaisesRegex(ValueError, 'RL method settings differ'):
                trainer.validate_resume_checkpoint(ck, args)


if __name__ == '__main__':
    unittest.main()
