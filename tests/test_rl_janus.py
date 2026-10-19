"""Adversarial checks for --objective janus_one_iteration (draft / verdict / final rewards)."""
import copy
import itertools
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llava.train.rl import train_grpo as trainer
from llava.train.rl.grpo import GRPOConfig, KIND_IMG, KIND_NONE, KIND_TXT, grpo_loss, masked_token_logprobs, normalize_groups
from llava.train.rl.rollout import (RolloutBatch, SEGMENT_DRAFT, SEGMENT_FINAL, SEGMENT_REFLECTION,
                                    looks_good_verdict)
from test_rl_episodes import ScriptedModel
from test_rl_policy import small_rollout, policy_args, tiny_base, get_peft_model, LoraConfig
from test_rl_reset_context import new_model, update

# id -> text. 2 = EOS, 3 = <im_start>, 4 = <im_end>, 5 = <S0>, 8..15 image tokens.
TEXT = {0: '', 1: '<p>', 2: '<|im_end|>', 3: '<im_start>', 4: '<im_end>', 5: '<S0>', 6: '\n', 7: ' x',
        16: ' Correction: looks good.', 17: ' Correction: add a dog', 18: ' Self-reflect: looks good but',
        19: '.\n'}
SPECIAL = {0, 2, 3, 4, 5}
ENCODE = {'\n<im_start><S0>': [6, 3, 5], '<im_start><S0>': [3, 5], '\n': [6]}
PROMPTS = [{'prompt': 'two red dogs', 'vqa_list': [('How many dogs?', 'two'), ('Are the dogs red?', 'yes')]}]


class FakeTok:
    def decode(self, ids, skip_special_tokens=True):
        return ''.join('' if (skip_special_tokens and i in SPECIAL) else TEXT.get(i, f'<I{i}>') for i in ids)

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=list(ENCODE[text]))


def janus_rollout(num_rollouts=6, reflect_tokens=4):
    rollout = small_rollout()
    rollout.cfg.num_rollouts = num_rollouts
    rollout.cfg.max_seq_len = 100
    rollout.cfg.max_refinements = 1
    rollout.cfg.reflect_tokens = reflect_tokens
    rollout.reflect_splice = [4, 6]
    rollout.image_splice = [5]
    rollout.make_prefix = lambda prompt: [1, 7]
    rollout.tok = FakeTok()
    return rollout


# e0 looks good+EOS | e1 correction+<im_start> | e2 correction+".\n"+EOS | e3 looks good+<im_start>
# e4 correction capped (no stop token) | e5 "looks good but ... Correction: add" + <im_start>
CALLS = [('image', [[8] * 3, [9] * 3, [10] * 3, [11] * 3, [12] * 3, [13] * 3]),
         ('text', [[16, 2], [17, 3], [17, 19, 2], [16, 3], [17, 7, 7, 7], [18, 17, 3]]),
         ('image', [[13] * 3, [14] * 3, [15] * 3, [13, 14, 15]])]


def fixture(calls=CALLS, num_rollouts=6, rule='correction'):
    rollout = janus_rollout(num_rollouts)
    model = ScriptedModel(calls)
    batch = rollout.run_one_iteration(model, PROMPTS, verdict_rule=rule)
    assert model.calls == []
    return rollout, batch


def score(batch, ams):
    """Fake decoder/judge: ``ams`` gives one AM per rendered image, in render order."""
    decoded = []
    def decode(codes):
        decoded.append(codes[0])
        return [object()]
    queue = list(ams)
    def judge(images, questions):
        assert len(images) == len(queue)
        return queue, [a / 2 for a in queue], [[a, a] for a in queue]
    trainer.score_one_iteration(batch, SimpleNamespace(score_images=judge), SimpleNamespace(decode_codes=decode))
    return decoded


class JanusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_verdict_rules(self):
        self.assertEqual(looks_good_verdict('Self-reflect: fine. Correction: looks good.'), (True, True))
        self.assertEqual(looks_good_verdict('Self-reflect: looks good but no dog. Correction: add a dog'), (False, True))
        self.assertEqual(looks_good_verdict('Self-reflect: looks good but no dog. Correction: add a dog',
                                            'anywhere'), (True, True))
        self.assertEqual(looks_good_verdict('It Looks Good'), (True, False))  # no clause: whole text
        with self.assertRaises(ValueError):
            looks_good_verdict('x', 'typo')

    def test_rollout_paths_contexts_and_independence(self):
        rollout, batch = fixture()
        e = batch.trajectories
        self.assertEqual([t.stop_reason for t in e],
                         ['looks_good', 'refined', 'refined', 'looks_good', 'refined', 'refined'])
        self.assertEqual([t.one_iter.mismatch for t in e],
                         [None, None, 'correction_eos', 'looks_good_im_start', 'cap', None])
        self.assertEqual([t.one_iter.stop_token_agrees for t in e], [True, True, False, False, False, True])
        self.assertEqual([t.one_iter.verdict_disagreement for t in e], [False] * 5 + [True])
        self.assertEqual([len(t.images) for t in e], [1, 2, 2, 1, 2, 2])
        draft_ctx = [1, 7]
        self.assertTrue(all(t.one_iter.draft_context == draft_ctx for t in e))
        self.assertEqual(e[1].one_iter.reflect_context, [1, 7, 9, 9, 9, 4, 6])
        self.assertEqual(e[1].one_iter.refine_context, [1, 7, 9, 9, 9, 4, 6, 17, 3, 5])
        self.assertTrue(e[1].one_iter.packed)
        # EOS dropped; the text already ends with a newline -> no extra newline.
        self.assertEqual(e[2].one_iter.refine_context, [1, 7, 10, 10, 10, 4, 6, 17, 19, 3, 5])
        self.assertFalse(e[2].one_iter.packed)
        # Capped reflection: nothing dropped, newline + <im_start><S0> spliced.
        self.assertEqual(e[4].one_iter.refine_context, [1, 7, 12, 12, 12, 4, 6, 17, 7, 7, 7, 6, 3, 5])
        self.assertEqual(e[1].one_iter.refine, [13] * 3)
        self.assertEqual(e[5].images, [[5] * 3, [5, 6, 7]])
        for a, b in itertools.combinations(e, 2):
            self.assertIsNot(a.seq, b.seq)
            self.assertIsNot(a.one_iter.reflect_context, b.one_iter.reflect_context)
        self.assertEqual(len(e[1].seq), len(e[1].kinds))
        # The anywhere rule keeps e5's draft instead.
        _, other = fixture(calls=[CALLS[0], CALLS[1], ('image', [[13] * 3, [14] * 3, [15] * 3])], rule='anywhere')
        self.assertEqual(other.trajectories[5].stop_reason, 'looks_good')

    def test_scoring_rewards_and_janus_advantages(self):
        rollout, batch = fixture()
        # Render order: e0 d | e1 d,r | e2 d,r | e3 d | e4 d,r | e5 d,r
        ams = [.9, .3, .5, .2, .6, .8, .4, .1, .7, .35]
        decoded = score(batch, ams)
        self.assertEqual(len(decoded), 10)
        e = batch.trajectories
        self.assertIs(e[0].final_image, e[0].scored_images[0])
        self.assertEqual(len(e[1].scored_images), 2)
        drafts = [.9, .3, .2, .8, .4, .7]
        finals = [.9, .5, .6, .8, .1, .35]
        verdicts = [1, 0, 0, 1, 0, 0]
        for t, d, f, y in zip(e, drafts, finals, verdicts):
            self.assertAlmostEqual(t.one_iter.reward_draft, d)
            self.assertAlmostEqual(t.one_iter.reward_final, f)
            self.assertAlmostEqual(t.one_iter.reward_verdict, 1 - abs(d - y))
            self.assertAlmostEqual(t.am, f)
        counts = trainer.assign_janus_advantages(batch, GRPOConfig(adv_norm='std'))
        self.assertEqual(counts, dict(groups=1, zero_var_draft=0, zero_var_verdict=0, zero_var_final=0))
        for name, rewards in (('draft', drafts), ('final', finals),
                              ('verdict', [1 - abs(d - y) for d, y in zip(drafts, verdicts)])):
            r = torch.tensor(rewards, dtype=torch.float64)
            expected = (r - r.mean()) / (r.std() + 1e-4)  # Janus: unbiased std
            got = torch.tensor([getattr(t.one_iter, f'adv_{name}') for t in e], dtype=torch.float64)
            torch.testing.assert_close(got, expected)

    def test_rows_weights_masks_and_janus_loss(self):
        rollout, batch = fixture()
        score(batch, [.9, .3, .5, .2, .6, .8, .4, .1, .7, .35])
        trainer.assign_janus_advantages(batch, GRPOConfig(adv_norm='std'))
        rows = rollout.build_janus_rows(batch.trajectories)
        # Rows: kept episodes 2 each (e0, e3), packed e1/e5 1 each, forced e2 2, capped e4 1 (context extends).
        self.assertEqual(len(rows['input_ids']), 2 + 2 + 1 + 1 + 2 + 1)
        self.assertAlmostEqual(float(rows['weight'].sum()), 6.0, places=5)
        for k in (SEGMENT_DRAFT, SEGMENT_REFLECTION, SEGMENT_FINAL):
            self.assertAlmostEqual(float(rows['weight'][rows['pos_segment'] == k].sum()), 2.0, places=5)
        self.assertTrue(torch.all(rows['weight'][~rows['train_mask']] == 0))
        # Exactly two stop tokens lose their PG: e2's EOS and e3's <im_start>.
        masked = (rows['pg_mask'] == 0) & rows['train_mask']
        self.assertEqual(int(masked.sum()), 2)
        self.assertEqual(sorted(rows['input_ids'][masked].tolist()), [2, 3])
        # Per-episode check against a direct Janus compute_loss on the same token values.
        cfg = GRPOConfig(kl_coef=.04)
        torch.manual_seed(5)
        total, janus = 0.0, 0.0
        for t in batch.trajectories:
            r = rollout.build_janus_rows([t])
            sel = r['train_mask']
            n = int(sel.sum())
            lp, old, ref = torch.randn(n) * .1 - 1, torch.randn(n) * .1 - 1, torch.randn(n) * .1 - 1
            loss, _ = grpo_loss(lp, old, ref, r['adv'][sel], r['weight'][sel], r['pos_kind'][sel], 1, cfg,
                                pg_mask=r['pg_mask'][sel], segments=r['pos_segment'][sel])
            total += float(loss)
            ratio = (lp - old).exp()
            adv = r['adv'][sel]
            pg = -torch.min(ratio * adv, ratio.clamp(.8, 1.2) * adv) * r['pg_mask'][sel]
            d = ref - lp
            per_token = pg + .04 * (d.exp() - d - 1)
            seg = r['pos_segment'][sel]
            janus += sum(float(per_token[seg == k].mean()) for k in (1, 2, 3)) / 3
        self.assertAlmostEqual(total, janus, places=5)
        # A kept draft's tokens carry KL weight 2/3 (draft + final segment), as in Janus.
        r0 = rollout.build_janus_rows([batch.trajectories[0]])
        self.assertAlmostEqual(float(r0['weight'][r0['pos_kind'] == KIND_IMG].sum()), 2 / 3, places=6)

    def test_stats_have_fixed_keys_without_refinements(self):
        calls = [('image', [[8] * 3, [9] * 3]), ('text', [[16, 2], [16, 2]])]
        _, kept = fixture(calls=calls, num_rollouts=2)
        score(kept, [.5, .7])
        _, mixed = fixture()
        score(mixed, [.9, .3, .5, .2, .6, .8, .4, .1, .7, .35])
        a, b = trainer.janus_stats(kept, 'x/'), trainer.janus_stats(mixed, 'x/')
        self.assertEqual(set(a), set(b))
        f = trainer.finalize_janus_stats(a, 'x/')
        self.assertEqual((f['x/looks_good_rate'], f['x/delta_am_refined']), (1.0, 0.0))
        g = trainer.finalize_janus_stats(b, 'x/')
        self.assertAlmostEqual(g['x/mismatch_cap_rate'], 1 / 6)
        self.assertAlmostEqual(g['x/delta_am_refined'], ((.5 - .3) + (.6 - .2) + (.1 - .4) + (.35 - .7)) / 4)

    def test_normalize_groups_options_keep_legacy(self):
        r, g = [.1, .4, .9, .5], [0, 0, 0, 0]
        legacy, _ = normalize_groups(r, g, GRPOConfig(adv_norm='std'))
        t = torch.tensor(r, dtype=torch.float64)
        torch.testing.assert_close(torch.tensor(legacy, dtype=torch.float64),
                                   (t - t.mean()) / (t.std(unbiased=False) + 1e-4), rtol=0, atol=0)
        floored, _ = normalize_groups([.5, .51], [0, 0], GRPOConfig(adv_norm='std', adv_std_floor=.04))
        self.assertAlmostEqual(floored[1], .005 / (.04 + 1e-4), places=9)

    def test_sampling_likelihood_matches_janus_rows(self):
        torch.manual_seed(0)
        model = new_model().eval()
        rollout = janus_rollout(num_rollouts=8, reflect_tokens=3)
        generate = model.generate
        captured = {}
        def capture(**kwargs):
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
        seen = set()
        for seed in range(40):
            torch.manual_seed(seed)
            captured.clear()
            with patch.object(model, 'generate', side_effect=capture):
                batch = rollout.run_one_iteration(model, PROMPTS)
            kinds = {('kept' if t.one_iter.verdict else ('packed' if t.one_iter.packed else 'forced'))
                     for t in batch.trajectories}
            for t in batch.trajectories:
                t.one_iter.adv_draft = t.one_iter.adv_verdict = t.one_iter.adv_final = .1
            rows = rollout.build_janus_rows(batch.trajectories)
            expected = [captured[tuple(ids[:i + 1].tolist())]
                        for ids, mask in zip(rows['input_ids'], rows['train_mask'])
                        for i in mask.nonzero().flatten().tolist()]
            actual = masked_token_logprobs(model, rows['input_ids'], rows['attention_mask'],
                                           rows['train_mask'], rows['pos_kind'], 8, 16)
            torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-5, atol=1e-6)
            seen |= kinds
            if seen >= {'packed', 'forced'}:
                break
        self.assertTrue({'packed', 'forced'} <= seen, seen)

    def test_training_updates_microbatch_invariant_and_loss_identity(self):
        args = dict(objective='janus_one_iteration')
        models, metrics = [], []
        for micro in (1, 3, 16):
            rollout, batch = fixture()
            score(batch, [.9, .3, .5, .2, .6, .8, .4, .1, .7, .35])
            trainer.assign_janus_advantages(batch, GRPOConfig(adv_norm='std'))
            model = new_model()
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.2)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
            metrics.append(trainer.train_on_rollouts(
                model, rollout, batch, policy_args(train_micro_batch=micro, max_grad_norm=1e6,
                                                   reflect_token_weight=1.0, **args),
                GRPOConfig(adv_norm='std', kl_gradient_correction=True), opt, scheduler, torch.device('cpu')))
            models.append(model)
        for model in models[1:]:
            for p, q in zip(models[0].parameters(), model.parameters()):
                torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
        for m in metrics:
            self.assertAlmostEqual(m['loss'], m['pg_loss'] + .01 * m['kl'], places=6)
            self.assertAlmostEqual(m['kl'], m['kl_draft'] + m['kl_reflection'] + m['kl_final'], places=6)

    def test_cli_constraints_and_metadata(self):
        required = ['train', '--model_name_or_path', 'unused', '--data_path', 'unused', '--output_dir', 'unused']
        janus = ['--objective', 'janus_one_iteration', '--max_refinements', '1']
        with patch('sys.argv', required + janus):
            args = trainer.parse_args()
        self.assertEqual((args.objective, args.verdict_rule), ('janus_one_iteration', 'correction'))
        for bad in (['--objective', 'janus_one_iteration'], janus + ['--tree_branch', '2'],
                    janus + ['--text_policy_gradient', 'transitions'], janus + ['--reflect_token_weight', '2'],
                    janus + ['--length_normalization', 'constant'], ['--adv_std_floor', '-1']):
            with self.subTest(bad=bad), patch('sys.argv', required + bad), patch('sys.stderr'), \
                    self.assertRaises(SystemExit):
                trainer.parse_args()
        with tempfile.TemporaryDirectory() as directory:
            run = policy_args(output_dir=directory, save_total_limit=3, max_seq_len=32,
                              objective='janus_one_iteration', verdict_rule='correction', adv_std_floor=0.0)
            model = get_peft_model(tiny_base(), LoraConfig(r=2, target_modules=['q_proj'], task_type='CAUSAL_LM'))
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
            trainer.save_checkpoint(model, opt, scheduler, 1, run, torch.device('cpu'))
            ck = Path(directory) / 'checkpoint-1'
            saved = json.loads((ck / 'state.json').read_text())['rl_method']
            self.assertEqual((saved['objective'], saved['verdict_rule']), ('janus_one_iteration', 'correction'))
            trainer.validate_resume_checkpoint(ck, run)
            for key, value in (('objective', 'final_episode'), ('verdict_rule', 'anywhere'), ('adv_std_floor', .04)):
                changed = copy.copy(run)
                setattr(changed, key, value)
                with self.assertRaisesRegex(ValueError, 'RL method settings differ'):
                    trainer.validate_resume_checkpoint(ck, changed)
            # A checkpoint written before these keys existed resumes as the legacy objective.
            state = json.loads((ck / 'state.json').read_text())
            for key in ('objective', 'verdict_rule', 'adv_std_floor', 'tree_branch', 'text_policy_gradient',
                        'verdict_target', 'verdict_pass_threshold'):
                del state['rl_method'][key]
            (ck / 'state.json').write_text(json.dumps(state))
            trainer.validate_resume_checkpoint(ck, policy_args(max_seq_len=32))
            with self.assertRaisesRegex(ValueError, 'RL method settings differ'):
                trainer.validate_resume_checkpoint(ck, run)

    def test_distributed_kept_vs_refined_ranks_match_combined(self):
        with tempfile.TemporaryDirectory(prefix='tar-janus-dist-') as directory:
            mp.spawn(distributed_worker, args=(directory,), nprocs=2, join=True)


RANK_CALLS = [[('image', [[8] * 3, [9] * 3, [10] * 3]), ('text', [[16, 2], [16, 2], [16, 2]])],
              [('image', [[11] * 3, [12] * 3, [13] * 3]), ('text', [[17, 3], [17, 3], [17, 19, 2]]),
               ('image', [[14] * 3, [15] * 3, [13] * 3])]]
RANK_AMS = [[.2, .6, .9], [.3, .5, .4, .2, .8, .9]]


def rank_batch(rank):
    rollout, batch = fixture(calls=RANK_CALLS[rank], num_rollouts=3)
    score(batch, RANK_AMS[rank])
    trainer.assign_janus_advantages(batch, GRPOConfig(adv_norm='std'))
    return rollout, batch


def distributed_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + directory + '/rendezvous',
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        args = policy_args(train_micro_batch=1, max_grad_norm=1e6, reflect_token_weight=1.0,
                           objective='janus_one_iteration')
        cfg = GRPOConfig(adv_norm='std', kl_gradient_correction=True)
        def train(model, rollout, batch):
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.2)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
            return trainer.train_on_rollouts(model, rollout, batch, args, cfg, opt, scheduler, torch.device('cpu'))
        model = new_model()
        central = copy.deepcopy(model)
        rollout, batch = rank_batch(rank)
        metrics = train(model, rollout, batch)
        reduced = {}
        for key in ('loss', 'pg_loss', 'kl'):
            value = torch.tensor(metrics[key], dtype=torch.float64)
            dist.all_reduce(value)
            reduced[key] = float(value) / 2
        stats = trainer.reduce_sum(trainer.janus_stats(batch, 'x/'), torch.device('cpu'))
        if rank == 0:
            other = rank_batch(1)[1]
            combined = RolloutBatch(PROMPTS * 2, batch.trajectories + other.trajectories)
            with patch.object(trainer.dist, 'is_initialized', return_value=False):
                expected = train(central, rollout, combined)
            for p, q in zip(model.parameters(), central.parameters()):
                torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
            for key in reduced:
                torch.testing.assert_close(torch.tensor(reduced[key]), torch.tensor(expected[key]),
                                           rtol=1e-5, atol=1e-7)
            local = trainer.janus_stats(combined, 'x/')
            for key in local:
                assert abs(stats[key] - local[key]) < 1e-9, key
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
