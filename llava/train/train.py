import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import deepspeed
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

from llava.data import make_supervised_data_module
from llava.model import LlavaQwenForCausalLM
from llava.train.llava_trainer import LLaVATrainer
from llava.utils import rank0_print

torch.multiprocessing.set_sharing_strategy("file_system")

local_rank = None

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})
    mm_tunable_parts: Optional[str] = field(default="mm_language_model")
    version: Optional[str] = field(default="v0")
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    mm_use_im_start_end: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)
    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)
    delay_load: Optional[bool] = field(default=True)
    num_image_tokens: Optional[int] = field(default=-1)
    image_token_format: str = field(default="<I{}>")
    num_scale_tokens: Optional[int] = field(default=-1)
    scale_token_format: str = field(default="<S{}>")
    load_embeddings_from_vision: Optional[bool] = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to validation data, using the same format as data_path."},
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    dataset_cls: str = field(default="llava")
    parquet_interleave_shards: int = field(
        default=3,
        metadata={"help": "Number of parquet shards to interleave per DataLoader worker."},
    )
    parquet_batch_size: int = field(
        default=16,
        metadata={"help": "Rows retained per active parquet shard while streaming."},
    )
    dataset_seed: int = field(
        default=0,
        metadata={"help": "Base seed for training data order and sampling. Each entry of a "
                          "weighted mixture derives a distinct seed from this, so vary it across "
                          "runs — otherwise every run replays the identical per-data_type sample "
                          "stream and a `ratio` change only alters how fast each stream is "
                          "consumed. Validation data is unaffected and stays comparable."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})
    dispatch_batches: Optional[bool] = field(default=None)
    split_batches: Optional[bool] = field(default=None)

    # LoRA arguments
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=128)
    lora_alpha: int = field(default=256)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated module names to apply LoRA to. If unset, all language-model Linear layers are targeted."},
    )
    lora_self_attn_only: bool = field(
        default=False,
        metadata={"help": "When True, restrict LoRA to self-attention projections (q/k/v/o_proj) in the LLM, "
                          "excluding vision tower, embed_tokens, and lm_head."},
    )
    weights_only_save_steps: Optional[int] = field(
        default=None,
        metadata={"help": "If set, every N steps save a lightweight checkpoint containing only the model "
                          "weights (no optimizer/scheduler/RNG/trainer global state) under the `weights_only/` "
                          "sub-directory of the output dir. Useful to keep many eval-able snapshots cheaply."},
    )


def find_self_attn_linear_names(model, exclude_keywords=("vision_tower", "embed_tokens", "lm_head")):
    """Return full paths of self-attention Linear layers in the LLM only.

    Targets q/k/v/o projections (any Linear whose parent path contains 'attn'),
    skipping the vision tower, token embeddings, and lm_head."""
    lora_module_names = set()
    for name, module in model.named_modules():
        if any(kw in name for kw in exclude_keywords):
            continue
        if "attn" not in name:
            continue
        if isinstance(module, torch.nn.Linear):
            lora_module_names.add(name)
    return sorted(lora_module_names)


def find_all_linear_names(model, exclude_keywords=("vision_tower", "embed_tokens", "lm_head")):
    """Collect the names of all nn.Linear modules eligible for LoRA, skipping the
    vision tower (TA-Tok) and the lm_head (kept as a full module_to_save).

    Returns the *full* module paths rather than leaf names. PEFT matches list
    entries by suffix, and the language model and vision tower share leaf names
    (e.g. ``q_proj``), so returning leaf names would re-add LoRA to the excluded
    vision tower. Full paths force an exact match and keep the exclusion intact."""
    lora_module_names = set()
    for name, module in model.named_modules():
        if any(kw in name for kw in exclude_keywords):
            continue
        if isinstance(module, torch.nn.Linear):
            lora_module_names.add(name)
    return sorted(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def get_model(model_args, training_args):
    customized_kwargs = {}
    overwrite_config = {}

    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )

    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)
        customized_kwargs["config"] = cfg_pretrained

    model = LlavaQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=False,
        **customized_kwargs)
    return model


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    model = get_model(model_args, training_args)
    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right")
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio

        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end

        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

        ### Deciding train which part of the model
        rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
        model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
        # Set the entire model to not require gradients by default
        model.requires_grad_(False)
        vision_tower.requires_grad_(False)
        vision_tower.eval()
        # Parse the mm_tunable_parts to decide which parts to unfreeze
        tunable_parts = model_args.mm_tunable_parts.split(",")
        if "mm_vision_tower" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" in name:
                    param.requires_grad_(True)
        if "mm_language_model_wo_embed" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" not in name and "embed_tokens" not in name and 'lm_head' not in name:
                    param.requires_grad_(True)
        if "mm_language_model" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" not in name:
                    param.requires_grad_(True)
        if 'mm_embedding' in tunable_parts:
            for name, param in model.named_parameters():
                if "embed_tokens" in name or 'lm_head' in name:
                    param.requires_grad_(True)

        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model

            # Keep the (image/scale) token embeddings and lm_head as full,
            # non-LoRA trainable modules whenever the tunable-parts config marks
            # them trainable -- the discrete image tokens cannot be learned via a
            # low-rank adapter on a frozen embedding table.
            modules_to_save = sorted({
                "embed_tokens" if "embed_tokens" in name else "lm_head"
                for name, param in model.named_parameters()
                if param.requires_grad and ("embed_tokens" in name or "lm_head" in name)
            })

            if training_args.lora_target_modules:
                target_modules = [m.strip() for m in training_args.lora_target_modules.split(",") if m.strip()]
            elif training_args.lora_self_attn_only:
                target_modules = find_self_attn_linear_names(model)
            else:
                target_modules = find_all_linear_names(model)

            rank0_print(f"Adding LoRA adapters (r={training_args.lora_r}, alpha={training_args.lora_alpha}) "
                        f"to: {target_modules}")
            if modules_to_save:
                rank0_print(f"Keeping fully trainable (modules_to_save): {modules_to_save}")

            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                target_modules=target_modules,
                modules_to_save=modules_to_save or None,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
        trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
        for name, p in model.named_parameters():
            if p.requires_grad:
                rank0_print(f"Trainable parameter: {name}")
        rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
        rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")

        rank0_print(f"Total parameters: {total_params}")
        rank0_print(f"Trainable parameters: {trainable_params}")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
