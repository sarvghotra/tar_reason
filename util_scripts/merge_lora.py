"""Merge a GRPO LoRA checkpoint into its base weights and save a plain
checkpoint, so the merged policy can be the base (and the KL reference,
adapters off) of the next RL stage.

    python util_scripts/merge_lora.py \
        --base output_dir/slf_ref_edit_t21/checkpoint-17000 \
        --adapter output_dir/rl_ft_oracle_t8/checkpoint-500 \
        --out output_dir/rl_ft_oracle_t8/merged-500

The tokenizer is copied from --base (train_grpo.py loads it from
--model_name_or_path). ~15 GB of CPU RAM for the 7B model; a GPU is not
needed. The merged weights round-trip through bf16, like the training forward.
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen2ForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    assert not os.path.exists(a.out), f"{a.out} exists"

    base = Qwen2ForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16).to(a.device)
    model = PeftModel.from_pretrained(base, a.adapter)
    with open(os.path.join(a.adapter, "adapter_config.json")) as f:
        cfg = json.load(f)
    print(f"merging LoRA r={cfg.get('r')} alpha={cfg.get('lora_alpha')} on {cfg.get('target_modules')}")
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
    with open(os.path.join(a.out, "merged_from.json"), "w") as f:
        json.dump({"base": os.path.abspath(a.base), "adapter": os.path.abspath(a.adapter)}, f, indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
