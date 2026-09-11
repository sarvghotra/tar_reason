"""GPU integration smoke: real Qwen pixels, then SFT -> renderer -> Qwen reward.

Run with tar/bin/python inside a GPU batch or interactive allocation after cluster_env.sh.
Saves evidence under results/evaluations/qwen_reward_smoke; no optimizer updates.
"""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from transformers import AutoTokenizer

from llava.train.rl.grpo import GRPOConfig
from llava.train.rl.pixel_reward import QwenPixelReward
from llava.train.rl.rollout import EpisodeRollout, RolloutConfig
from llava.train.rl.train_grpo import (
    ImageDecoder, assign_advantages, load_policy, save_visual_evaluation, score_rollouts,
)


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/evaluations/qwen_reward_smoke"
    output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "Run this test inside a GPU allocation."
    torch.cuda.set_device(0)
    torch.set_num_threads(8)
    device = torch.device("cuda:0")
    model_path = os.environ.get("REWARD_MODEL_PATH", str(root / "reward_model"))
    judge_python = os.environ.get("REWARD_PYTHON", str(root / "qwen_reward/bin/python"))
    started = time.time()
    judge = QwenPixelReward(judge_python, model_path, device)
    try:
        questions = [("Is the image red?", "Yes"), ("Is the image blue?", "Yes")]
        images = [Image.new("RGB", (512, 512), color) for color in ("red", "blue")]
        am, gm, scores = judge.score_images(images, [questions, questions])
        assert scores[0][0] > scores[0][1], scores
        assert scores[1][1] > scores[1][0], scores
        _, _, repeated = judge.score_images(images[:1], [questions])
        assert all(abs(a-b) < 1e-5 for a, b in zip(scores[0], repeated[0])), (scores, repeated)
        for color, image in zip(("red", "blue"), images):
            image.save(output / f"fixture-{color}.png")
        report = dict(gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                      judge_model=model_path, color_question_scores=scores,
                      color_am=am, color_gm=gm, repeat_scores=repeated,
                      job_id=os.environ.get("SLURM_JOB_ID"))
        (output / "judge_smoke.json").write_text(json.dumps(report, indent=2))
        print("Real Qwen color discrimination and repeatability passed.", flush=True)

        models = root / "models"
        policy_path = os.environ.get("SFT_MODEL", str(root / "sft_model"))
        args = SimpleNamespace(model_name_or_path=policy_path, seed=421, lora_r=64,
                               lora_alpha=128, lora_dropout=0.0, gradient_checkpointing=True,
                               attn_implementation="flash_attention_2", ar_path=os.environ.get("AR_MODEL", str(models / "tar/ar_dtok_lp_512px.pth")),
                               encoder_path=os.environ.get("VISION_MODEL", str(models / "tar/ta_tok.pth")),
                               decoder_path=os.environ.get("DECODER", str(models / "tar/vq_ds16_t2i.pt")),
                               gen_seq_len=729, scale=0, cfg_scale=4.0)
        policy = load_policy(args, device)
        tokenizer = AutoTokenizer.from_pretrained(policy_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        img_start = tokenizer.convert_tokens_to_ids("<I0>")
        img_count = sum(t.startswith("<I") and t[2:-1].isdigit() for t in tokenizer.get_vocab())
        rollout = EpisodeRollout(tokenizer, RolloutConfig(num_rollouts=2, max_refinements=0,
                                  reflect_tokens=32, gen_batch_size=2), img_start, img_count, device)
        decoder = ImageDecoder(args, device)
        prompts = [dict(prompt="a red square on a white background", vqa_list=[
            ("Is there a red square in the image?", "Yes"),
            ("Is the background white?", "Yes")])]
        torch.manual_seed(421)
        batch = rollout.run(policy, prompts)
        score_rollouts(batch, judge, decoder)
        assign_advantages(batch, GRPOConfig())
        report["episode_rewards"] = [t.reward for t in batch.trajectories]
        report["episode_advantages"] = [t.adv for t in batch.trajectories]
        assert abs(sum(report["episode_advantages"])) < 1e-6
        report["elapsed_seconds"] = time.time() - started
        save_visual_evaluation(decoder, [(prompts[0]["prompt"], t) for t in batch.trajectories],
                               str(output), 0, 421, {})
        (output / "integration_smoke.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        print("SFT -> semantic tokens -> rendered pixels -> Qwen AM -> episode advantages passed.", flush=True)
    finally:
        judge.close()


if __name__ == "__main__":
    main()
