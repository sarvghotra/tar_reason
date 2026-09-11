"""CPU regressions for distributed policy identity and on-policy RL sampling.

Run with: python -m unittest discover -s tests -p 'test_rl_policy.py' -v
No downloaded weights, tokenizer, dataset, or GPU is needed.
"""

import copy
import io
import math
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig, get_peft_model
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

from llava.train.rl import train_grpo as trainer
from llava.train.rl.grpo import GRPOConfig, KIND_IMG, KIND_NONE, KIND_TXT, masked_token_logprobs
from llava.train.rl.rollout import ImageVocabOnly, NoImageVocab, Trajectory, RolloutBatch, RolloutConfig, EpisodeRollout


def tiny_base():
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=24, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
        attention_dropout=0.0, pad_token_id=0, eos_token_id=2,
        attn_implementation="eager"))


def policy_args(path="unused", **overrides):
    values = dict(model_name_or_path=path, attn_implementation="eager", seed=42,
                  lora_r=2, lora_alpha=4, lora_dropout=0.0, gradient_checkpointing=True,
                  train_micro_batch=1, reflect_token_weight=0.7, num_ppo_epochs=2,
                  max_grad_norm=1.0)
    return SimpleNamespace(**(values | overrides))


def small_rollout():
    # Tokenizer-independent sampling fixture: exercise the production padding,
    # generate calls, and training-row construction with a small image length.
    rollout = EpisodeRollout.__new__(EpisodeRollout)
    rollout.cfg = RolloutConfig(gen_seq_len=3, reflect_tokens=4)
    rollout.device = torch.device("cpu")
    rollout.pad_id = 0
    rollout.img_start, rollout.img_end = 8, 16
    rollout.eos_id, rollout.im_start_tok = 2, 3
    rollout.text_stop_ids = [2, 3]
    rollout.image_proc = [ImageVocabOnly(8, 16)]
    rollout.text_proc = [NoImageVocab(8, 16)]
    return rollout


def training_batch(rank):
    nodes = [Trajectory(0, [1, 4 + rank, 8 + rank, 9, 5],
                  [KIND_NONE, KIND_NONE, KIND_IMG, KIND_IMG, KIND_TXT],
                  adv=1.0 if rank == 0 else -0.6),
             Trajectory(0, [1, 10, 6], [KIND_NONE, KIND_IMG, KIND_TXT],
                  adv=-0.5 if rank == 0 else 0.9)]
    return RolloutBatch([{"prompt": "tiny"}], nodes)


def assert_synced(model):
    for parameter in model.parameters():
        if parameter.requires_grad:
            other = parameter.detach().clone()
            dist.broadcast(other, src=0)
            torch.testing.assert_close(parameter, other, rtol=0, atol=0)


def distributed_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + directory + "/rendezvous",
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(1234 + rank * 999)
        args = policy_args(directory + "/base")
        model = trainer.load_policy(args, torch.device("cpu"))
        assert_synced(model)
        # Force a divergence even with the common initialization seed. The
        # broadcast must repair real parameter state, not merely reset RNGs.
        with torch.no_grad():
            if rank == 1:
                for p in model.parameters():
                    if p.requires_grad:
                        p.add_(0.125)
        trainer.synchronize_trainable_parameters(model)
        assert_synced(model)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

        def optim(m):
            o = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.01)
            return o, torch.optim.lr_scheduler.LambdaLR(o, lambda step: 1.0 / (step + 1))

        optimizer, scheduler = optim(model)
        rollout, batch = small_rollout(), training_batch(rank)
        for step in range(2):
            torch.manual_seed(500 + rank * 10 + step)
            metrics = trainer.train_on_rollouts(model, rollout, batch, args, GRPOConfig(),
                                            optimizer, scheduler, torch.device("cpu"))
            assert all(math.isfinite(v) for v in metrics.values()), metrics
            assert_synced(model)
        assert any(not torch.equal(p, initial[n]) for n, p in model.named_parameters() if p.requires_grad)
        for n, p in model.named_parameters():
            if n in frozen:
                torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)

        if rank == 0:
            model.save_pretrained(directory + "/adapter")
            torch.save(optimizer.state_dict(), directory + "/optimizer.pt")
            torch.save(scheduler.state_dict(), directory + "/scheduler.pt")
        dist.barrier()
        resumed = trainer.load_policy(args, torch.device("cpu"), directory + "/adapter")
        resumed_optimizer, resumed_scheduler = optim(resumed)
        resumed_optimizer.load_state_dict(torch.load(directory + "/optimizer.pt"))
        resumed_scheduler.load_state_dict(torch.load(directory + "/scheduler.pt"))
        assert_synced(resumed)
        # Resumption must take the same next step as the uninterrupted run.
        for m, o, s in [(model, optimizer, scheduler), (resumed, resumed_optimizer, resumed_scheduler)]:
            trainer.train_on_rollouts(m, rollout, batch, args, GRPOConfig(), o, s, torch.device("cpu"))
            assert_synced(m)
        for p, q in zip(model.parameters(), resumed.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class RLPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initialization_is_independent_of_incoming_rng(self):
        torch.manual_seed(17)
        base = tiny_base()
        models = []
        with patch.object(trainer.Qwen2ForCausalLM, "from_pretrained", side_effect=lambda *a, **k: copy.deepcopy(base)):
            for seed in (100, 200):
                torch.manual_seed(seed)
                models.append(trainer.load_policy(policy_args(), torch.device("cpu")))
        for p, q in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_sampling_overrides_rejected_before_model_loading(self):
        required = ["train_grpo", "--model_name_or_path", "missing", "--data_path", "missing", "--output_dir", "missing"]
        for name, value in [("img_top_k", "1"), ("img_top_p", "0.95"),
                            ("reflect_top_k", "2"), ("reflect_top_p", "0.9"),
                            ("img_temperature", "0.7"), ("reflect_temperature", "2"),
                            ("img_temperature", "nan"), ("lora_dropout", "0.1")]:
            with self.subTest(name=name, value=value), patch("sys.argv", required + ["--" + name, value]):
                error = io.StringIO()
                with redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                    trainer.parse_args()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("RL training requires", error.getvalue())
        with patch("sys.argv", required):
            args = trainer.parse_args()
        self.assertEqual((args.img_top_k, args.img_top_p, args.reflect_top_p), (0, 1.0, 1.0))

    def test_policy_dropout_rejected_including_resumed_adapter(self):
        base = tiny_base()
        with patch.object(trainer.Qwen2ForCausalLM, "from_pretrained", side_effect=lambda *a, **k: copy.deepcopy(base)):
            with self.assertRaisesRegex(ValueError, "dropout must be zero"):
                trainer.load_policy(policy_args(lora_dropout=0.1), torch.device("cpu"))
            base.config.attention_dropout = 0.1
            with self.assertRaisesRegex(ValueError, "dropout must be zero"):
                trainer.load_policy(policy_args(), torch.device("cpu"))
        with tempfile.TemporaryDirectory(prefix="tar-dropout-test-") as directory:
            tiny_base().save_pretrained(directory + "/base")
            adapter = get_peft_model(tiny_base(), LoraConfig(r=2, lora_alpha=4, lora_dropout=0.1,
                                     target_modules=["q_proj"], task_type="CAUSAL_LM"))
            adapter.save_pretrained(directory + "/adapter")
            with self.assertRaisesRegex(ValueError, "dropout must be zero"):
                trainer.load_policy(policy_args(directory + "/base"), torch.device("cpu"), directory + "/adapter")

    def test_training_rejects_incompatible_rollout(self):
        for override in ({"img_top_k": 2}, {"reflect_sample": False}, {"reflect_top_p": 0.95}):
            rollout = small_rollout()
            for name, value in override.items():
                setattr(rollout.cfg, name, value)
            with self.subTest(override=override), self.assertRaises(ValueError):
                trainer.train_on_rollouts(None, rollout, None, None, None, None, None, None)

    def test_generate_probabilities_match_loss_with_padding_and_hostile_defaults(self):
        torch.manual_seed(72)
        model = get_peft_model(tiny_base(), LoraConfig(r=2, lora_alpha=4,
                              target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        # Nonzero adapters ensure the comparison includes the policy adapter.
        with torch.no_grad():
            model.get_base_model().lm_head.weight.mul_(10)
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(0, 0.1)
        model.eval()
        rollout = small_rollout()
        for hostile in (False, True):
            model.generation_config = GenerationConfig(
                do_sample=True, top_k=1, top_p=0.1, temperature=0.3,
                repetition_penalty=2.0, no_repeat_ngram_size=1,
                suppress_tokens=[8, 4] if hostile else None,
                min_p=0.9 if hostile else None,
                forced_eos_token_id=8 if hostile else None,
                eos_token_id=8 if hostile else 2, pad_token_id=0)
            # Unmodified neutral defaults still have to agree with the loss.
            if not hostile:
                model.generation_config = GenerationConfig(pad_token_id=0, eos_token_id=2)
            for kind, sample in [(KIND_IMG, rollout._sample_images), (KIND_TXT, rollout._sample_reflections)]:
                prefixes = [[1, 4, 5], [1]]
                captured = []
                generate = model.generate

                def capture(**kwargs):
                    result = generate(**kwargs, return_dict_in_generate=True, output_scores=True)
                    captured.append(result)
                    return result.sequences

                with self.subTest(hostile=hostile, kind=kind), patch.object(model, "generate", side_effect=capture):
                    samples = sample(model, prefixes)
                    result = captured[0]
                    expected = []
                    leaves = []
                    for i, (prefix, tokens) in enumerate(zip(prefixes, samples)):
                        for t, token in enumerate(tokens):
                            expected.append(result.scores[t][i].float().log_softmax(-1)[token])
                        leaves.append(Trajectory(i, prefix + tokens,
                                           [KIND_NONE] * len(prefix) + [kind] * len(tokens)))
                    self.assertTrue(expected, "fixture must score at least one sampled token")
                    rows = rollout.build_training_rows(leaves, 1.0)
                    model.train()  # Same mode as the actual PPO forward.
                    with torch.no_grad():
                        actual = masked_token_logprobs(model, rows["input_ids"], rows["attention_mask"],
                                                       rows["train_mask"], rows["pos_kind"], 8, 16)
                    model.eval()
                    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-5, atol=1e-6)

    def test_sampled_eos_is_retained_without_padding(self):
        rollout = small_rollout()
        fake = SimpleNamespace(generate=lambda **kw: torch.cat((kw["input_ids"], torch.tensor([[2, 7, 7]])), dim=1))
        self.assertEqual(rollout._sample_reflections(fake, [[1, 4]]), [[2]])

    def test_two_rank_updates_and_adapter_resume(self):
        with tempfile.TemporaryDirectory(prefix="tar-rl-test-") as directory:
            torch.manual_seed(51)
            tiny_base().save_pretrained(directory + "/base")
            mp.spawn(distributed_worker, args=(directory,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
