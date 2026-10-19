"""Checks for --text_policy_gradient transitions (reflection prose: KL only)."""
import unittest

import torch

from llava.train.rl import train_grpo as trainer
from llava.train.rl.grpo import GRPOConfig, KIND_IMG, KIND_NONE, KIND_TXT, grpo_loss
from test_rl_reset_context import fixture, new_model, policy_args


class TextPolicyGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mask_drops_only_reflection_prose(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                rollout, batch = fixture(reset)
                legacy = rollout.build_training_rows(batch.trajectories, .7)
                rows = rollout.build_training_rows(batch.trajectories, .7, 'episode', 'transitions')
                self.assertNotIn('pg_mask', legacy)
                for key in legacy:
                    torch.testing.assert_close(rows[key], legacy[key], rtol=0, atol=0)
                ids, kinds, mask = rows['input_ids'], rows['pos_kind'], rows['pg_mask']
                prose = (kinds == KIND_TXT) & (ids != rollout.eos_id) & (ids != rollout.im_start_tok)
                # Fixture reflections are [16, <im_start>], [7, <im_start>] and EOS.
                self.assertEqual(sorted(set(ids[prose].tolist())), [7, 16])
                self.assertTrue(torch.all(mask[prose] == 0))
                self.assertTrue(torch.all(mask[~prose] == 1))
                for token in (rollout.eos_id, rollout.im_start_tok):
                    self.assertTrue(torch.all(mask[(kinds == KIND_TXT) & (ids == token)] == 1))
                self.assertTrue(torch.all(mask[kinds == KIND_IMG] == 1))
                with self.assertRaises(ValueError):
                    rollout.build_training_rows(batch.trajectories, .7, 'episode', 'typo')

    def test_loss_keeps_prose_kl_but_not_prose_policy_gradient(self):
        kinds = torch.tensor([KIND_IMG, KIND_TXT, KIND_TXT, KIND_TXT])
        pg_mask = torch.tensor([1., 0., 0., 1.])
        old = torch.tensor([-.5, -1., -.7, -.3], dtype=torch.float64)
        ref = old - .2
        w = torch.tensor([.2, .3, .1, .4], dtype=torch.float64)
        grads = []
        for prose_adv in (.5, -3.):
            adv = torch.tensor([.4, prose_adv, prose_adv, -.2], dtype=torch.float64)
            lp = old.clone().requires_grad_()
            loss, metrics = grpo_loss(lp, old, ref, adv, w, kinds, 2, GRPOConfig(), pg_mask=pg_mask)
            grads.append(torch.autograd.grad(loss, lp)[0])
            self.assertAlmostEqual(float(loss), metrics['pg_loss'] * float(w.sum()) / 2
                                   + .01 * metrics['kl'] * float(w.sum()) / 2, places=12)
        torch.testing.assert_close(grads[0], grads[1], rtol=0, atol=0)  # prose advantage ignored
        self.assertTrue(torch.all(grads[0][1:3] != 0))  # prose KL gradient remains
        lp = old.clone().requires_grad_()
        loss, _ = grpo_loss(lp, old, ref, torch.ones(4, dtype=torch.float64), w, kinds, 2,
                            GRPOConfig(kl_coef=0), pg_mask=pg_mask)
        self.assertTrue(torch.all(torch.autograd.grad(loss, lp)[0][1:3] == 0))
        # An all-ones mask is exactly the legacy loss.
        for mask in (None, torch.ones(4)):
            lp = old.clone().requires_grad_()
            values = grpo_loss(lp, old, ref, torch.ones(4, dtype=torch.float64), w, kinds, 2,
                               GRPOConfig(), pg_mask=mask)
            if mask is None:
                expected = values
            else:
                torch.testing.assert_close(values[0], expected[0], rtol=0, atol=0)
                self.assertEqual(values[1], expected[1])

    def test_training_step_ignores_prose_advantages(self):
        def train(prose_adv, mode='transitions'):
            rollout, batch = fixture(True)
            model = new_model()
            rows = rollout.build_training_rows(batch.trajectories, .7)
            # Give prose tokens a different advantage by editing the episode rows.
            original = rollout.build_training_rows
            def patched(*args, **kwargs):
                out = original(*args, **kwargs)
                ids, kinds = out['input_ids'], out['pos_kind']
                prose = (kinds == KIND_TXT) & (ids != rollout.eos_id) & (ids != rollout.im_start_tok)
                out['adv'][prose] = prose_adv
                return out
            rollout.build_training_rows = patched
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.2)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
            metrics = trainer.train_on_rollouts(
                model, rollout, batch,
                policy_args(train_micro_batch=1, max_grad_norm=1e6, text_policy_gradient=mode),
                GRPOConfig(), opt, scheduler, torch.device('cpu'))
            self.assertAlmostEqual(metrics['loss'], metrics['pg_loss'] + .01 * metrics['kl'], places=6)
            return model
        a, b = train(5.), train(-5.)
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        # Sanity: with all-text policy gradient the prose advantage does matter.
        a, b = train(5., 'all'), train(-5., 'all')
        self.assertFalse(all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters())))


if __name__ == '__main__':
    unittest.main()
