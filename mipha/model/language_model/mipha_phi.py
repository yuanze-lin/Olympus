import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
    PhiModel, PhiPreTrainedModel

from transformers.modeling_outputs import CausalLMOutputWithPast
from ..mipha_arch import MiphaMetaModel, MiphaMetaForCausalLM
from transformers.utils import logging
# transformers >= 4.50 no longer gives PreTrainedModel a GenerationMixin base, so
# models overriding prepare_inputs_for_generation must inherit it explicitly or
# lose .generate(). Older versions expose it at the same path, so this is safe.
try:
    from transformers.generation import GenerationMixin
except ImportError:  # transformers < 4.30
    from transformers.generation.utils import GenerationMixin

from .configuration_mipha import MiphaPhiConfig

logger = logging.get_logger(__name__)


def _past_length(past_key_values) -> int:
    """Number of cached positions, for legacy tuple caches and Cache objects alike.

    transformers >= 4.50 hands `generate()` an already-instantiated (empty)
    DynamicCache, so the old `if past_key_values:` truthiness test no longer
    distinguishes "first step" from "later step".
    """
    if past_key_values is None:
        return 0
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        try:
            return int(get_seq_length() or 0)
        except TypeError:
            return 0
    try:
        return int(past_key_values[0][0].shape[2])
    except (IndexError, AttributeError, TypeError):
        return 0


class MiphaPhiModel(MiphaMetaModel, PhiModel):
    config_class = MiphaPhiConfig

    def __init__(self, config):
        super(MiphaPhiModel, self).__init__(config)


class MiphaPhiForCausalLM(PhiPreTrainedModel, GenerationMixin, MiphaMetaForCausalLM):
    config_class = MiphaPhiConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super(PhiPreTrainedModel, self).__init__(config)
        self.model = MiphaPhiModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

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
            cache_position: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images)

        # From transformers 4.50 the Phi decoder no longer infers absolute
        # positions from the cache length, so they must be supplied explicitly.
        # Without this every decoding step after the first is treated as
        # position 0 and generation degenerates into repeated text.
        if position_ids is None:
            seq_len = (inputs_embeds if inputs_embeds is not None else input_ids).shape[1]
            past_len = _past_length(past_key_values)
            device = (inputs_embeds if inputs_embeds is not None else input_ids).device
            position_ids = torch.arange(past_len, past_len + seq_len,
                                        dtype=torch.long, device=device).unsqueeze(0)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
            cache_position=None, position_ids=None, **kwargs
    ):
        past_len = _past_length(past_key_values)
        if past_len > 0:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_len == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + input_ids.shape[1],
                dtype=torch.long, device=input_ids.device,
            ).unsqueeze(0)

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


AutoConfig.register("mipha_phi", MiphaPhiConfig)
AutoModelForCausalLM.register(MiphaPhiConfig, MiphaPhiForCausalLM)
