"""Adversarial tests for bounded-context episodes with terminal credit."""
import copy
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llava.train.rl import train_grpo as trainer
from llava.train.rl.evaluation_policy import checkpoint_resets_context
from llava.train.rl.grpo import GRPOConfig, KIND_IMG, KIND_TXT, masked_token_logprobs
from llava.train.rl.rollout import RolloutBatch
from test_rl_episodes import episode_rollout, ScriptedModel, PROMPTS
from test_rl_policy import tiny_base, policy_args, get_peft_model, LoraConfig


def fixture(reset=True, shorter=False, **kwargs):
    rollout = episode_rollout(reset_context_per_turn=reset, **kwargs)
    calls = [("image", [[8]*3, [9]*3]), ("text", [[2], [16, 3]]),
             ("image", [[10]*3]), ("text", [[2]] if shorter else [[7, 3]])]
    if not shorter:
        calls += [("image", [[11]*3]), ("text", [[2]])]
    batch = rollout.run(ScriptedModel(calls), PROMPTS)
    batch.trajectories[0].reward = .1
    batch.trajectories[1].reward = .9
    trainer.assign_advantages(batch, GRPOConfig())
    return rollout, batch


def new_model():
    torch.manual_seed(93)
    model = get_peft_model(tiny_base(), LoraConfig(r=2, lora_alpha=4,
                          target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'lora_B' in name:
                p.normal_(0, .1)
    return model


def update(model, rollout, batch, micro, mode):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
    return trainer.train_on_rollouts(model, rollout, batch,
        policy_args(train_micro_batch=micro, max_grad_norm=1e6),
        GRPOConfig(length_normalization=mode, kl_gradient_correction=True),
        opt, scheduler, torch.device('cpu'))


def distributed_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + directory + '/rendezvous',
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        for mode in ('episode', 'constant'):
            model = new_model()
            central = copy.deepcopy(model)
            rollout, batch = fixture(shorter=bool(rank))
            metrics = update(model, rollout, batch, 1, mode)
            distributed_metrics = dict(metrics)
            for key in ('loss', 'pg_loss', 'kl', 'clip_frac', 'approx_kl_old'):
                value = torch.tensor(metrics[key], dtype=torch.float64)
                dist.all_reduce(value)
                distributed_metrics[key] = float(value)/2
            if rank == 0:
                other = fixture(shorter=True)[1]
                combined = RolloutBatch(PROMPTS, batch.trajectories + other.trajectories)
                with patch.object(trainer.dist, 'is_initialized', return_value=False):
                    expected = update(central, rollout, combined, 2, mode)
                for p, q in zip(model.parameters(), central.parameters()):
                    torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
                for key in ('loss', 'pg_loss', 'kl', 'clip_frac', 'approx_kl_old'):
                    torch.testing.assert_close(torch.tensor(distributed_metrics[key]),
                                               torch.tensor(expected[key]), rtol=1e-5, atol=1e-7)
            dist.barrier()
    finally:
        dist.destroy_process_group()


class ResetContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_context_reset_and_actions_preserved_exactly_once(self):
        rollout, batch = fixture()
        legacy = fixture(False)[1]
        for t, old in zip(batch.trajectories, legacy.trajectories):
            self.assertEqual(t.seq, old.seq)
            self.assertEqual(t.kinds, old.kinds)
            actions = [(v, k) for s in t.segments for v, k in zip(s.seq, s.kinds) if k]
            self.assertEqual(actions, [(v, k) for v, k in zip(t.seq, t.kinds) if k])
        t = batch.trajectories[1]
        self.assertEqual(len(t.segments), 3)
        self.assertEqual(t.segments[1].seq, [1, 3, 5, 10, 10, 10, 4, 7, 7, 3, 5, 11, 11, 11])
        self.assertEqual(t.segments[2].seq, [1, 3, 5, 11, 11, 11, 4, 7, 2])
        self.assertEqual(t.segments[1].kinds[:8], [0]*8)
        # Advantage and normalization belong to the episode, never to a turn.
        for mode in ('episode', 'constant'):
            rows = rollout.build_training_rows(batch.trajectories, .7, mode)
            offset = 0
            for t in batch.trajectories:
                stop = offset + len(t.segments)
                mask = rows['train_mask'][offset:stop]
                self.assertTrue(torch.all(rows['adv'][offset:stop][mask] == t.adv))
                mass = sum(.7 if k == KIND_TXT else 1 for k in t.kinds if k)
                expected = 1 if mode == 'episode' else mass/rollout.cfg.max_seq_len
                self.assertAlmostEqual(float(rows['weight'][offset:stop].sum()), expected, places=6)
                offset = stop
            self.assertTrue(torch.all(rows['weight'][~rows['train_mask']] == 0))

    def test_caps_eos_padding_and_no_partial_image(self):
        for cap, reason, text in [(0, 'max_refinements', [3]),
                                  (3, 'reflection_limit', [7]*4)]:
            rollout = episode_rollout(reset_context_per_turn=True, max_refinements=cap)
            batch = rollout.run(ScriptedModel([('image', [[8]*3, [9]*3]),
                                               ('text', [text, text])]), PROMPTS)
            self.assertEqual([t.stop_reason for t in batch.trajectories], [reason]*2)
            self.assertTrue(all(len(t.images) == 1 for t in batch.trajectories))
        # Context reset must not replenish the logical episode budget.
        rollout = episode_rollout(reset_context_per_turn=True, max_seq_len=19)
        rollout.pad_id = rollout.eos_id
        batch = rollout.run(ScriptedModel([('image', [[8]*3, [9]*3]),
            ('text', [[2], [3]]), ('image', [[10]*3]), ('text', [[7, 3]])]), PROMPTS)
        self.assertEqual([t.stop_reason for t in batch.trajectories], ['eos', 'max_seq_len'])
        self.assertEqual(len(batch.trajectories[1].images), 2)
        rows = rollout.build_training_rows(batch.trajectories, 1)
        self.assertEqual(int(rows['train_mask'].sum()), sum(sum(k != 0 for k in t.kinds) for t in batch.trajectories))
        self.assertTrue(rows['train_mask'][0, 8])  # EOS == padding is an action
        self.assertFalse(rows['train_mask'][0, -1])  # padding is not
        model = ScriptedModel([])
        model.config.max_position_embeddings = 8
        with self.assertRaisesRegex(ValueError, 'cannot fit'):
            rollout.run(model, PROMPTS)

    def test_reward_only_scores_last_images_and_credits_earlier_actions(self):
        from types import SimpleNamespace
        rollout, batch = fixture()
        decoded = []
        def decode(codes):
            decoded.append(codes)
            return [tuple(codes[0])]
        def score(images, questions):
            self.assertEqual(images, [(0, 0, 0), (3, 3, 3)])
            return [.9, .1], [.8, .05], [[.9], [.1]]
        trainer.score_rollouts(batch, SimpleNamespace(score_images=score),
                               SimpleNamespace(decode_codes=decode))
        self.assertEqual(decoded, [[[0]*3], [[3]*3]])
        trainer.assign_advantages(batch, GRPOConfig())
        rows = rollout.build_training_rows(batch.trajectories, 1)
        self.assertAlmostEqual(batch.trajectories[1].adv, -.4)
        # Both this episode's draft and its final EOS get terminal credit.
        self.assertAlmostEqual(float(rows['adv'][1, 3]), -.4)
        self.assertAlmostEqual(float(rows['adv'][-1, 8]), -.4)
        self.assertTrue(rows['train_mask'][1, 3])
        self.assertTrue(rows['train_mask'][-1, 8])

    def test_actual_sampling_probabilities_match_segment_training(self):
        model = new_model().eval()
        rollout = episode_rollout(reset_context_per_turn=True, reflect_tokens=32)
        generate = model.generate
        captured = {}
        def capture(**kwargs):
            self.assertNotIn('past_key_values', kwargs)
            result = generate(**kwargs, return_dict_in_generate=True, output_scores=True)
            n = kwargs['input_ids'].shape[1]
            for row in range(len(result.sequences)):
                prefix = kwargs['input_ids'][row][kwargs['attention_mask'][row].bool()].tolist()
                for j, token in enumerate(result.sequences[row, n:].tolist()):
                    prefix.append(token)
                    captured[tuple(prefix)] = result.scores[j][row].float().log_softmax(-1)[token]
                    if kwargs['eos_token_id'] is not None and token in kwargs['eos_token_id']:
                        break
            return result.sequences
        # Deterministically find a genuine sampled multi-refinement trajectory.
        for seed in range(10):
            torch.manual_seed(seed)
            captured.clear()
            with patch.object(model, 'generate', side_effect=capture):
                batch = rollout.run(model, PROMPTS)
            if max(len(t.images) for t in batch.trajectories) >= 3:
                break
        self.assertGreaterEqual(max(len(t.images) for t in batch.trajectories), 3)
        rows = rollout.build_training_rows(batch.trajectories, 1)
        expected = []
        for ids, mask in zip(rows['input_ids'], rows['train_mask']):
            for i in mask.nonzero().flatten().tolist():
                expected.append(captured[tuple(ids[:i+1].tolist())])
        actual = masked_token_logprobs(model, rows['input_ids'], rows['attention_mask'],
                                       rows['train_mask'], rows['pos_kind'], 8, 16)
        torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-5, atol=1e-6)

    def test_microbatch_invariance_and_episode_denominator(self):
        for mode in ('episode', 'constant'):
            models, metrics = [], []
            for micro in (1, 2, 8):
                model = new_model()
                rollout, batch = fixture()
                metrics.append(update(model, rollout, batch, micro, mode))
                models.append(model)
            for model in models[1:]:
                for p, q in zip(models[0].parameters(), model.parameters()):
                    torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
            for m in metrics:
                self.assertAlmostEqual(m['loss'], m['pg_loss'] + .01*m['kl'], places=6)
                for key in ('loss', 'pg_loss', 'kl', 'clip_frac', 'approx_kl_old'):
                    self.assertAlmostEqual(m[key], metrics[0][key], places=6)

    def test_cli_metadata_and_legacy_resume(self):
        required = ['train', '--model_name_or_path', 'unused', '--data_path', 'unused', '--output_dir', 'unused']
        for extra, expected in [([], False), (['--reset_context_per_turn'], True),
            (['--reset_context_per_turn', '--no-reset_context_per_turn'], False)]:
            with patch('sys.argv', required + extra):
                self.assertEqual(trainer.parse_args().reset_context_per_turn, expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            for saved in (None, {}, {'reset_context_per_turn': True}):
                state = {'objective': trainer.RL_OBJECTIVE}
                if saved is not None:
                    state['rl_method'] = dict(length_normalization='episode', normalization_constant=None,
                                              kl_gradient_correction=False, **saved)
                path.write_text(json.dumps(state))
                for requested in (False, True):
                    args = policy_args(reset_context_per_turn=requested)
                    if requested == bool(saved):
                        trainer.validate_resume_checkpoint(directory, args)
                    else:
                        with self.assertRaisesRegex(ValueError, 'RL method settings differ'):
                            trainer.validate_resume_checkpoint(directory, args)
                self.assertEqual(checkpoint_resets_context({'checkpoint_state': state}), bool(saved))
        with self.assertRaises(ValueError):
            checkpoint_resets_context({'checkpoint_state': {'rl_method': {'reset_context_per_turn': 'false'}}})

    def test_distributed_unequal_segment_counts(self):
        with tempfile.TemporaryDirectory(prefix='tar-reset-dist-') as directory:
            mp.spawn(distributed_worker, args=(directory,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
