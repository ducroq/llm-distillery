"""
Model loading utilities for filters.

Handles compatibility issues with AutoModelForSequenceClassification not
supporting all model types (e.g., Gemma3TextConfig in transformers <4.56).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


def load_base_model_for_seq_cls(
    model_name: str,
    num_labels: int,
    problem_type: str = "regression",
    torch_dtype=None,
    **kwargs,
):
    """
    Load a base model for sequence classification.

    Tries AutoModelForSequenceClassification first. Falls back to a manual
    construction for model types not yet in the Auto mapping (e.g., gemma3_text).

    Args:
        model_name: HuggingFace model name or path
        num_labels: Number of output labels/dimensions
        problem_type: "regression" or "single_label_classification"
        torch_dtype: Optional dtype override
        **kwargs: Additional kwargs passed to from_pretrained

    Returns:
        A model compatible with PEFT wrapping
    """
    load_kwargs = {
        "num_labels": num_labels,
        "problem_type": problem_type,
        **kwargs,
    }
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **load_kwargs
        )
        return model
    except (ValueError, KeyError) as e:
        if "Gemma3TextConfig" not in str(e) and "gemma3_text" not in str(e):
            raise

    # Fallback: manually build Gemma3Text sequence classifier
    logger.info(
        f"AutoModelForSequenceClassification doesn't support {model_name} config. "
        f"Falling back to manual Gemma3Text classifier construction."
    )
    return _build_gemma3_text_classifier(model_name, num_labels, load_kwargs)


def _build_gemma3_text_classifier(model_name, num_labels, load_kwargs):
    """Build a Gemma3Text sequence classification model manually."""
    from transformers.models.gemma3 import Gemma3TextConfig, Gemma3TextModel
    from transformers.models.gemma3.modeling_gemma3 import Gemma3PreTrainedModel
    from transformers.modeling_outputs import SequenceClassifierOutputWithPast

    class Gemma3TextForSequenceClassification(Gemma3PreTrainedModel):
        """Gemma3 text-only model with a sequence classification head."""

        config_class = Gemma3TextConfig

        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.model = Gemma3TextModel(config)
            self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            transformer_outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0] if not return_dict else transformer_outputs.last_hidden_state
            logits = self.score(hidden_states)

            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = inputs_embeds.shape[0]

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                last_non_pad_token = -1
            elif input_ids is not None:
                non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
                token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
                last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
            else:
                last_non_pad_token = -1

            pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

            loss = None
            if labels is not None:
                if self.config.problem_type == "regression":
                    loss_fct = nn.MSELoss()
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

            if not return_dict:
                output = (pooled_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutputWithPast(
                loss=loss,
                logits=pooled_logits,
                past_key_values=transformer_outputs.past_key_values if hasattr(transformer_outputs, 'past_key_values') else None,
                hidden_states=transformer_outputs.hidden_states if hasattr(transformer_outputs, 'hidden_states') else None,
                attentions=transformer_outputs.attentions if hasattr(transformer_outputs, 'attentions') else None,
            )

    # Load config and set classification attributes
    config = Gemma3TextConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    config.problem_type = load_kwargs.get("problem_type", "regression")

    # Build model
    dtype_kwargs = {}
    if "torch_dtype" in load_kwargs:
        dtype_kwargs["torch_dtype"] = load_kwargs["torch_dtype"]

    model = Gemma3TextForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        **dtype_kwargs,
    )

    return model
