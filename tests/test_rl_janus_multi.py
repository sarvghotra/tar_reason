"""Adversarial checks for --objective janus_multi_iteration (own reflections, several rounds)."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from llava.train.rl import train_grpo as trainer
from llava.train.rl.grpo import GRPOConfig, KIND_IMG, KIND_NONE, KIND_TXT
from test_rl_episodes import ScriptedModel
from test_rl_janus import FakeTok, janus_rollout, PROMPTS
from test_rl_policy import policy_args, tiny_base, get_peft_model, LoraConfig
from test_rl_reset_context import new_model

# e0 looks good + EOS | e1 refine <im_start>, then looks good + <im_start> (mismatch)
# e2 refine ".\n" + EOS (mismatch, EOS dropped), refine, refine at the cap | e3 capped refine, then looks good
CALLS = [('image', [[8] * 3, [9] * 3, [10] * 3, [11] * 3]),
         ('text', [[16, 2], [17, 3], [17, 19, 2], [17, 7, 7, 7]]),
         ('image', [[12] * 3, [13] * 3, [14] * 3]),
         ('text', [[16, 3], [17, 3], [16, 2]]),
         ('image', [[15] * 3]),
         ('text', [[17, 3]])]
AMS = [.9, .3, .6, .2, .5, .7, .4, .8]  # render order: e0 d | e1 d,i1 | e2 d,i1,i2 | e3 d,i1


def fixture(reset=False):
    rollout = janus_rollout(num_rollouts=4)
    rollout.cfg.max_refinements = 2
    rollout.cfg.reset_context_per_turn = reset
    model = ScriptedModel(CALLS)
    batch = rollout.run_janus_multi(model, PROMPTS)
    assert model.calls == [], model.calls
    return rollout, batch


def scored(reset=False, norm='mean'):
    rollout, batch = fixture(reset)
    decoder = SimpleNamespace(decode_codes=lambda codes: [object() for _ in codes])
    judge = SimpleNamespace(score_images=lambda images, vqa: (list(AMS), [a / 2 for a in AMS],
                                                               [[a, a] for a in AMS]))
    trainer.score_janus_multi(batch, judge, decoder, render_batch_size=3)
    counts = trainer.assign_janus_multi_advantages(batch, GRPOConfig(adv_norm=norm))
    return rollout, batch, counts


class JanusMultiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rollout_paths_spans_masks_and_contexts(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                _, batch = fixture(reset)
                e = batch.trajectories
                self.assertEqual([t.stop_reason for t in e], ['looks_good', 'looks_good', 'max_refinements',
                                                              'looks_good'])
                self.assertEqual([len(t.images) for t in e], [1, 2, 3, 2])
                self.assertEqual([[s.kind for s in t.janus_spans] for t in e],
                                 [[1, 2], [1, 2, 1, 2], [1, 2, 1, 2, 1, 2], [1, 2, 1, 2]])
                self.assertEqual([[s.index for s in t.janus_spans] for t in e],
                                 [[0, 0], [0, 0, 1, 1], [0, 0, 1, 1, 2, 2], [0, 0, 1, 1]])
                self.assertEqual(e[1].janus_spans[3].pg, [1.0, 0.0])        # looks good + <im_start>
                self.assertEqual(e[2].janus_spans[1].pg, [1.0, 1.0, 0.0])   # refine + EOS
                self.assertEqual(e[3].janus_spans[1].pg, [1.0] * 4)         # capped: no stop token
                self.assertEqual([[s.mismatch for s in t.janus_spans if s.kind == KIND_TXT] for t in e],
                                 [[None], [None, 'looks_good_im_start'], ['correction_eos', None, None],
                                  ['cap', None]])
                self.assertEqual([t.verdicts for t in e], [[True], [False, True], [False, False, False],
                                                           [False, True]])
                self.assertEqual(e[2].janus_spans[1].actions, [17, 19, 2])  # the EOS stays a trained action
                if reset:
                    self.assertEqual(e[2].janus_spans[3].context, [1, 7, 13, 13, 13, 4, 6])
                    self.assertEqual(e[2].janus_spans[4].context, [1, 7, 13, 13, 13, 4, 6, 17, 3, 5])
                    self.assertEqual(e[2].janus_spans[5].context, [1, 7, 15, 15, 15, 4, 6])
                else:
                    self.assertEqual(e[1].janus_spans[2].context, [1, 7, 9, 9, 9, 4, 6, 17, 3, 5])
                    # EOS dropped; the text already ends in a newline -> no extra newline.
                    self.assertEqual(e[2].janus_spans[2].context, [1, 7, 10, 10, 10, 4, 6, 17, 19, 3, 5])
                    self.assertEqual(e[3].janus_spans[2].context, [1, 7, 11, 11, 11, 4, 6, 17, 7, 7, 7, 6, 3, 5])
                for t in e:
                    for s in t.janus_spans:
                        self.assertEqual(s.context[-2:] if s.kind == KIND_TXT else s.context[-1:],
                                         [4, 6] if s.kind == KIND_TXT else s.context[-1:])
        with self.assertRaises(ValueError):
            r = janus_rollout()
            r.cfg.max_refinements = 0
            r.run_janus_multi(ScriptedModel([]), PROMPTS)

    def test_rewards_and_advantages(self):
        _, batch, counts = scored()
        e = batch.trajectories
        self.assertEqual([t.reward for t in e], [.9, .6, .7, .8])  # final image AM
        verdict = [[s.reward for s in t.janus_spans if s.kind == KIND_TXT] for t in e]
        self.assertEqual(verdict, [[1.0], [1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0]])
        for t, a in zip(e, [.15, -.15, -.05, .05]):
            self.assertAlmostEqual(t.adv, a, places=12)
        for t, rewards in zip(e, verdict):
            for s, r in zip([s for s in t.janus_spans if s.kind == KIND_TXT], rewards):
                self.assertAlmostEqual(s.adv, r - .75, places=12)
        self.assertEqual(counts, {'groups': 1, 'zero_var_final': 0, 'zero_var_verdict': 0})
        _, batch, _ = scored(norm='std')
        std = math.sqrt(.05 / 3)
        self.assertAlmostEqual(batch.trajectories[0].adv, .15 / (std + 1e-4), places=9)
        stats = trainer.finalize_janus_multi_extra(trainer.janus_multi_extra_stats(batch, 'x/'), 'x/')
        self.assertAlmostEqual(stats['x/reward_verdict'], .75)
        self.assertAlmostEqual(stats['x/mismatch_cap_rate'], 1 / 8)
        self.assertAlmostEqual(stats['x/delta_am_refined'], ((.6 - .3) + (.5 - .2) + (.7 - .5) + (.8 - .4)) / 4)
        lg = trainer.finalize_looks_good_stats(trainer.looks_good_stats(batch, 'y/', .5, 3), 'y/', 3)
        self.assertAlmostEqual(lg['y/verdict_accuracy'], 6 / 8)
        self.assertAlmostEqual(lg['y/am_upto_2'], (.9 + .6 + .5 + .8) / 4)

    def test_rows_weights_advantages_and_packing(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                rollout, batch, _ = scored(reset)
                rows = rollout.build_janus_multi_rows(batch.trajectories)
                self.assertAlmostEqual(float(rows['weight'].sum()), 4.0, places=5)  # one per episode
                img = rows['pos_kind'] == KIND_IMG
                self.assertEqual(int(img.sum()), 8 * 3)
                self.assertEqual(int((rows['pos_kind'] == KIND_TXT).sum()), 2 + 4 + 7 + 6)
                self.assertEqual(int(((rows['pg_mask'] == 0) & rows['train_mask']).sum()), 2)
                ids = rows['input_ids']
                first = next(i for i in range(len(ids)) if ids[i, :9].tolist() == [1, 7, 8, 8, 8, 4, 6, 16, 2])
                torch.testing.assert_close(rows['weight'][first, :9],
                                           torch.tensor([0, 0, 1 / 6, 1 / 6, 1 / 6, 0, 0, .25, .25]))
                torch.testing.assert_close(rows['adv'][first, :9],
                                           torch.tensor([0, 0, .15, .15, .15, 0, 0, .25, .25]))
                self.assertEqual(rows['pos_segment'][first, :9].tolist(), [0, 0, 1, 1, 1, 0, 0, 2, 2])
                self.assertEqual(len(ids), 5 if not reset else 9)  # EOS drop and each reset round start new rows
                self.assertTrue(set(rows['pos_segment'][img].tolist()) == {1, 3})

    def test_training_step_is_finite_and_microbatch_invariant(self):
        rollout, batch, _ = scored(reset=True)
        params, metrics = [], []
        for micro in (1, 3):
            model = new_model()
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.2)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
            args = policy_args(train_micro_batch=micro, max_grad_norm=1e6, num_ppo_epochs=2,
                               objective='janus_multi_iteration')
            metrics.append(trainer.train_on_rollouts(model, rollout, batch, args,
                                                     GRPOConfig(kl_gradient_correction=True), opt, scheduler,
                                                     torch.device('cpu')))
            params.append([p.detach().clone() for p in model.parameters() if p.requires_grad])
        for p, q in zip(*params):
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
        for key in ('loss', 'pg_loss', 'kl', 'pg_draft', 'pg_reflection', 'pg_refined'):
            self.assertTrue(math.isfinite(metrics[0][key]), key)
            self.assertAlmostEqual(metrics[0][key], metrics[1][key], places=5)

    def test_cli_and_metadata(self):
        required = ['train_grpo', '--model_name_or_path', 'unused', '--data_path', 'unused', '--output_dir',
                    'unused', '--objective', 'janus_multi_iteration', '--num_rollouts', '8']
        with patch('sys.argv', required + ['--reset_context_per_turn']):
            args = trainer.parse_args()
        method = trainer.rl_method_settings(args)
        self.assertEqual((method['objective'], method['verdict_target'], method['verdict_rule'],
                          method['teacher_ce_coef'], method['reset_context_per_turn']),
                         ('janus_multi_iteration', 'all_pass', 'correction', None, True))
        for extra in (['--tree_branch', '2'], ['--max_refinements', '0'], ['--verdict_target', 'soft_am'],
                      ['--text_policy_gradient', 'transitions'], ['--length_normalization', 'constant']):
            with self.subTest(extra=extra), patch('sys.argv', required + extra), patch('sys.stderr'), \
                    self.assertRaises(SystemExit):
                trainer.parse_args()
        with tempfile.TemporaryDirectory() as directory:
            args = policy_args(output_dir=directory, save_total_limit=3, max_seq_len=64,
                               objective='janus_multi_iteration', verdict_rule='correction',
                               verdict_target='all_pass', verdict_pass_threshold=.5, reset_context_per_turn=True)
            model = get_peft_model(tiny_base(), LoraConfig(r=2, target_modules=['q_proj'], task_type='CAUSAL_LM'))
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
            trainer.save_checkpoint(model, opt, scheduler, 1, args, torch.device('cpu'))
            ck = Path(directory) / 'checkpoint-1'
            self.assertEqual(trainer.validate_resume_checkpoint(ck, args)['step'], 1)
            for key, value in (('objective', 'janus_one_iteration'), ('reset_context_per_turn', False)):
                changed = copy.copy(args)
                setattr(changed, key, value)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'RL method settings differ'):
                    trainer.validate_resume_checkpoint(ck, changed)


if __name__ == '__main__':
    unittest.main()
