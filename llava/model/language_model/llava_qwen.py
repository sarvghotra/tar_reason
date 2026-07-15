from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.constants import IGNORE_INDEX
from llava.model.llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"

class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)

class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        data_types: Optional[List[str]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        # Without per-type logging, defer entirely to the base model (which
        # computes the training loss in its usual single fp32 softmax).
        if data_types is None or labels is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Per-type logging path: have the base model return only logits (labels
        # left None so it skips its own loss), then compute the causal-LM CE ONCE
        # ourselves. That single reduction="none" pass yields both the training
        # loss and the per-data-type breakdown, so this adds ~zero memory/compute
        # over the base model's own loss instead of a redundant second softmax.
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        loss, per_type_loss = self._loss_with_per_type(outputs.logits, labels, data_types)
        outputs["loss"] = loss
        outputs["per_type_loss"] = per_type_loss
        return outputs

    def _loss_with_per_type(self, logits, labels, data_types):
        """Compute the shifted causal-LM cross-entropy once and return
        ``(loss, {data_type: (loss_sum, token_count)})``.

        ``loss`` is the mean CE over the non-ignored target tokens, which is
        bit-for-bit what Qwen2's default loss produces here (``fixed_cross_entropy``
        with ``num_items_in_batch=None`` → ``reduction="mean"``; the model does not
        receive ``num_items_in_batch``). The per-type dict is derived from the same
        per-token vector purely for logging, so there is no second softmax and no
        per-sample fp32 ``[seq_len, vocab]`` duplicates.
        """
        # Shift so tokens < n predict n (matches ForCausalLMLoss).
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        bsz, seq_m1 = shift_labels.shape

        flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        flat_labels = shift_labels.reshape(-1)
        sup_mask = flat_labels.ne(IGNORE_INDEX)

        if not torch.any(sup_mask):
            # No supervised tokens this micro-batch; keep a differentiable zero.
            return flat_logits.sum() * 0.0, {}

        # Gather only the supervised rows before upcasting: one fp32
        # [num_supervised, vocab] tensor (never the ignored-inclusive grid),
        # still differentiable for backprop.
        sel_logits = flat_logits[sup_mask].float()
        sel_labels = flat_labels[sup_mask]
        per_token = F.cross_entropy(sel_logits, sel_labels, reduction="none")

        loss = per_token.mean()
        per_type_loss = self._bucket_per_type(
            per_token.detach(), sup_mask, bsz, seq_m1, data_types
        )
        return loss, per_type_loss

    @torch.no_grad()
    def _bucket_per_type(self, per_token, sup_mask, bsz, seq_m1, data_types):
        """Bucket detached per-token losses by the sample's ``data_type``.
        ``data_types[i]`` labels row ``i`` of the (batch-ordered) shifted labels.
        """
        device = per_token.device
        # Sample index for each selected (supervised) token.
        sample_ids = (
            torch.arange(bsz, device=device)
            .unsqueeze(1)
            .expand(bsz, seq_m1)
            .reshape(-1)[sup_mask]
        )
        result: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for dt in sorted({t for t in data_types if t is not None}):
            members = torch.tensor(
                [i for i, t in enumerate(data_types) if t == dt], device=device
            )
            rows = torch.isin(sample_ids, members)
            count = rows.sum()
            if count == 0:
                continue
            result[dt] = (per_token[rows].sum(), count.to(per_token.dtype))
        return result

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
