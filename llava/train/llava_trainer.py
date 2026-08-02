from datetime import timedelta
from typing import List, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin, InitProcessGroupKwargs
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import Trainer
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_accelerate_available,
    is_datasets_available,
    is_sagemaker_mp_enabled,
    logger,
)
from transformers.trainer_pt_utils import (
    get_length_grouped_indices as get_length_grouped_indices_hf,
)
from transformers.trainer_utils import seed_worker

if is_datasets_available():
    import datasets

from llava.utils import rank0_print


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult=8, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i : i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # FIXME: Hard code to avoid last batch mixed with different modalities
    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        variable_length: bool = False,
        group_by_modality: bool = False,
        group_by_modality_auto: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.variable_length = variable_length
        self.group_by_modality = group_by_modality
        self.group_by_modality_auto = group_by_modality_auto

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.variable_length:
            assert not self.group_by_modality, "Variable length grouping is not supported with modality grouping."
            indices = get_variable_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            if self.group_by_modality:
                indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            elif self.group_by_modality_auto:
                indices = get_modality_length_grouped_indices_auto(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            else:
                indices = get_length_grouped_indices_auto_single(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The type sets are read from the dataset configs so every rank performs
        # reductions in the same order (avoiding collective deadlocks).
        self._loss_type_names = self._discover_data_types(self.train_dataset)
        self._eval_loss_type_names = self._discover_data_types(self.eval_dataset)

        # Training accumulators cover one logging window. Evaluation
        # accumulators cover one complete validation pass.
        self._per_type_loss_sum = {t: 0.0 for t in self._loss_type_names}
        self._per_type_loss_count = {t: 0.0 for t in self._loss_type_names}
        self._eval_per_type_loss_sum = {
            t: 0.0 for t in self._eval_loss_type_names
        }
        self._eval_per_type_loss_count = {
            t: 0.0 for t in self._eval_loss_type_names
        }
        self._collecting_eval_per_type_loss = False

    def _discover_data_types(self, dataset):
        types = set()
        datasets_ = getattr(dataset, "datasets", None)
        if datasets_ is not None:
            for ds in datasets_:
                dt = getattr(ds, "data_type", None)
                if dt is not None:
                    types.add(dt)
        else:
            dt = getattr(dataset, "data_type", None)
            if dt is not None:
                types.add(dt)
        return sorted(types)

    def _reset_eval_per_type_loss(self):
        for dt in self._eval_loss_type_names:
            self._eval_per_type_loss_sum[dt] = 0.0
            self._eval_per_type_loss_count[dt] = 0.0

    def _collect_eval_per_type_metrics(self, metric_key_prefix):
        import torch.distributed as dist

        metrics = {}
        distributed = dist.is_available() and dist.is_initialized()
        total_loss_sum = torch.zeros(
            (), dtype=torch.float32, device=self.args.device
        )
        total_count = torch.zeros(
            (), dtype=torch.float32, device=self.args.device
        )
        for dt in self._eval_loss_type_names:
            loss_sum = torch.as_tensor(
                self._eval_per_type_loss_sum[dt],
                dtype=torch.float32,
                device=self.args.device,
            )
            count = torch.as_tensor(
                self._eval_per_type_loss_count[dt],
                dtype=torch.float32,
                device=self.args.device,
            )
            if distributed:
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            total_loss_sum += loss_sum
            total_count += count
            if count.item() > 0:
                metrics[f"{metric_key_prefix}_loss_{dt}"] = (
                    loss_sum / count
                ).item()
        if total_count.item() > 0:
            # Trainer's generic iterable-dataset loss can overweight a final
            # partial batch. The model-provided sums/counts yield the exact
            # token-weighted validation loss instead.
            metrics[f"{metric_key_prefix}_loss"] = (
                total_loss_sum / total_count
            ).item()
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Always fetch the model outputs so we can harvest the per-type loss that
        # the model attaches for logging, then honor the caller's return_outputs.
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        per_type_loss = getattr(outputs, "per_type_loss", None)
        if per_type_loss:
            if self._collecting_eval_per_type_loss:
                for dt, (loss_sum, count) in per_type_loss.items():
                    if dt not in self._eval_per_type_loss_sum:
                        continue
                    self._eval_per_type_loss_sum[dt] += loss_sum.detach()
                    self._eval_per_type_loss_count[dt] += count.detach()
            elif model.training:
                for dt, (loss_sum, count) in per_type_loss.items():
                    if dt not in self._per_type_loss_sum:
                        continue
                    self._per_type_loss_sum[dt] += loss_sum.detach()
                    self._per_type_loss_count[dt] += count.detach()
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        self._reset_eval_per_type_loss()
        self._collecting_eval_per_type_loss = True
        try:
            output = super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self._collecting_eval_per_type_loss = False

        output.metrics.update(
            self._collect_eval_per_type_metrics(metric_key_prefix)
        )
        return output

    def log(self, logs, *args, **kwargs):
        # Inject per-data-type train losses alongside the standard "loss" log.
        # `_per_type_*` hold device tensors (or 0.0 if a type saw no samples this
        # window); reduce across ranks in the fixed type order so all processes
        # issue matching collectives.
        if self._loss_type_names and "loss" in logs:
            import torch.distributed as dist

            distributed = dist.is_available() and dist.is_initialized()
            for dt in self._loss_type_names:
                loss_sum = self._per_type_loss_sum[dt]
                count = self._per_type_loss_count[dt]
                loss_sum = torch.as_tensor(loss_sum, dtype=torch.float32, device=self.args.device)
                count = torch.as_tensor(count, dtype=torch.float32, device=self.args.device)
                if distributed:
                    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(count, op=dist.ReduceOp.SUM)
                if count.item() > 0:
                    logs[f"loss_{dt}"] = round((loss_sum / count).item(), 4)
                # reset accumulator for the next logging window
                self._per_type_loss_sum[dt] = 0.0
                self._per_type_loss_count[dt] = 0.0
        super().log(logs, *args, **kwargs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        # ZeRO-1/2 replicate the full model on every rank, so HF/accelerate's
        # get_state_dict() builds a full CPU copy of the model on ALL ranks even
        # though only the main process (`should_save`) writes it. With N ranks on
        # one node that redundant work is an Nx host-RAM spike at checkpoint time,
        # which OOM-kills a dataloader worker (surfacing as the misleading
        # "DataLoader worker is killed by signal: Killed" during the save).
        # Build/write the HF state dict only on the saving rank; DeepSpeed's
        # sharded native checkpoint (written by all ranks in
        # _save_optimizer_and_scheduler) still handles resume.
        #
        # For ZeRO-1/2 get_state_dict() is purely local, so skipping it on
        # non-saving ranks involves no collective and cannot deadlock. ZeRO-3
        # gathers the sharded params via an all-rank collective, so never skip
        # there.
        if self.is_deepspeed_enabled and not self.args.should_save:
            try:
                zero_stage = self.accelerator.state.deepspeed_plugin.zero_stage
            except AttributeError:
                zero_stage = 0
            if zero_stage != 3:
                return
        super().save_model(output_dir, _internal_call=_internal_call)

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        # The model's large temporary logits/CE buffers can leave almost the
        # entire GPU reserved in PyTorch's caching allocator after backward.
        # NCCL allocates its logging-collective buffers outside that allocator,
        # so the first loss all_gather may otherwise fail even though the
        # temporary tensors are no longer live. Release only unused cached
        # blocks before entering the logging collectives.
        if self.control.should_log and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Standard HF logging/eval/checkpointing (full checkpoint with optimizer,
        # scheduler, RNG and trainer global state when control.should_save).
        super()._maybe_log_save_evaluate(*args, **kwargs)
        # In addition, periodically drop a lightweight weights-only snapshot under
        # `weights_only/` so we can keep many eval-able checkpoints cheaply without
        # the (large) global training state.
        self._maybe_save_weights_only()

    def _maybe_save_weights_only(self):
        save_steps = getattr(self.args, "weights_only_save_steps", None)
        if not save_steps:
            return
        step = self.state.global_step
        if step == 0 or step % save_steps != 0:
            return
        # Avoid writing the same snapshot twice when this is hit by both the
        # step-level and epoch-level _maybe_log_save_evaluate calls.
        if getattr(self, "_last_weights_only_step", None) == step:
            return
        self._last_weights_only_step = step

        import os
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        run_dir = self._get_output_dir(trial=None)
        weights_only_dir = os.path.join(run_dir, "weights_only", f"{PREFIX_CHECKPOINT_DIR}-{step}")
        rank0_print(f"Saving weights-only checkpoint (no global state) to {weights_only_dir}")
        # save_model writes only the model weights/config (and tokenizer); it does
        # not save optimizer/scheduler/RNG/trainer state.
        self.save_model(weights_only_dir, _internal_call=True)

    def _load_rng_state(self, checkpoint):
        # PyTorch's weights_only=True (the new default) refuses to deserialize
        # numpy arrays that are embedded in RNG state files saved by older versions.
        # The RNG state file is written by the training process itself and is trusted.
        import os
        import numpy as np

        rng_file = os.path.join(checkpoint, f"rng_state_{self.args.process_index}.pth")
        if not os.path.isfile(rng_file):
            rng_file = os.path.join(checkpoint, "rng_state.pth")
        if not os.path.isfile(rng_file):
            return

        checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        torch.set_rng_state(checkpoint_rng_state["cpu"])
        if "cuda" in checkpoint_rng_state:
            if isinstance(checkpoint_rng_state["cuda"], list):
                for i, state in enumerate(checkpoint_rng_state["cuda"]):
                    torch.cuda.set_rng_state(state)
            else:
                torch.cuda.set_rng_state(checkpoint_rng_state["cuda"])
        if "numpy" in checkpoint_rng_state:
            np.random.set_state(checkpoint_rng_state["numpy"])
        if "python" in checkpoint_rng_state:
            import random
            random.setstate(checkpoint_rng_state["python"])

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        rank0_print("Setting NCCL timeout to INF to avoid running errors.")

        try:
            from accelerate import DataLoaderConfiguration

            # Define DataLoaderConfiguration
            dataloader_config = DataLoaderConfiguration(
                dispatch_batches=self.args.dispatch_batches,
                split_batches=self.args.split_batches
            )
            self.accelerator = Accelerator(
                dataloader_config=dataloader_config, deepspeed_plugin=self.args.deepspeed_plugin, gradient_accumulation_plugin=gradient_accumulation_plugin, kwargs_handlers=[accelerator_kwargs]
            )
        except:
            # create accelerator object
            self.accelerator = Accelerator(
                dispatch_batches=self.args.dispatch_batches, split_batches=self.args.split_batches, deepspeed_plugin=self.args.deepspeed_plugin, gradient_accumulation_plugin=gradient_accumulation_plugin, kwargs_handlers=[accelerator_kwargs]
            )
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get("limit_all_gathers", fsdp_plugin.limit_all_gathers)
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get("activation_checkpointing", fsdp_plugin.activation_checkpointing)
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError("The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg " "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic " "when using FSDP.")

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
            )
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality=True,
            )
        elif self.args.group_by_modality_length_auto:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality_auto=True,
            )
        elif self.args.group_by_varlen:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                # self.args.train_batch_size, # TODO: seems that we should have gradient_accumulation_steps
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                variable_length=True,
            )
        else:
            return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

        return dataloader

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper["vision_tower"] = self.args.mm_vision_tower_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
