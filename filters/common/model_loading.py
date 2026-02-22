"""
Model loading utilities for filters.

Provides shared functions for loading fine-tuned LoRA models, both from local
files and from HuggingFace Hub. Also handles compatibility issues with
AutoModelForSequenceClassification not supporting all model types (e.g.,
Gemma3TextConfig in transformers <4.56).

Note: The Gemma3TextForSequenceClassification class is defined inside
_build_gemma3_text_classifier() because it needs Gemma3PreTrainedModel as its
base class for from_pretrained() to work. This means it cannot be pickled for
multiprocessing. This is acceptable for the current inference-only usage, but
if training or DataLoader workers ever need this class, it should be refactored
to use module-level imports with a try/except guard.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
            (device_map, revision, trust_remote_code, etc.)

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
        err_str = str(e)
        if "Gemma3TextConfig" not in err_str and "gemma3_text" not in err_str:
            raise
        logger.debug(
            f"AutoModelForSequenceClassification raised {type(e).__name__}: {e}. "
            "Activating Gemma3Text fallback."
        )

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
        """Gemma3 text-only model with a sequence classification head.

        Note: This class is NOT in the HF Auto mapping. Always use
        load_base_model_for_seq_cls() instead of Auto directly. See ADR-007.
        """

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
                # Guard against all-padding sequences (argmax returns 0 = meaningless)
                non_pad_counts = non_pad_mask.sum(-1)
                if (non_pad_counts == 0).any():
                    logger.warning(
                        "One or more sequences in the batch are entirely padding. "
                        "Scores for those items will be meaningless."
                    )
                token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
                last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
            else:
                # inputs_embeds path: cannot determine padding without input_ids
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

    # Separate dtype and passthrough kwargs from classification kwargs
    dtype_kwargs = {}
    if "torch_dtype" in load_kwargs:
        dtype_kwargs["torch_dtype"] = load_kwargs["torch_dtype"]

    passthrough_keys = {"device_map", "revision", "trust_remote_code", "low_cpu_mem_usage"}
    extra_kwargs = {k: v for k, v in load_kwargs.items() if k in passthrough_keys}

    model = Gemma3TextForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        **dtype_kwargs,
        **extra_kwargs,
    )

    return model


def load_lora_local(
    model_path: Path,
    num_labels: int,
    device,
) -> Tuple:
    """
    Load a trained LoRA model from local files.

    Handles PeftConfig loading, key remapping (old-format adapters to
    get_peft_model's expected format), and device placement.

    Args:
        model_path: Path to model directory containing adapter_model.safetensors
        num_labels: Number of output labels/dimensions
        device: torch.device or string

    Returns:
        (model, tokenizer) tuple ready for inference

    Raises:
        FileNotFoundError: If model files not found
        RuntimeError: If model loading fails
    """
    from peft import PeftConfig, get_peft_model
    from safetensors.torch import load_file

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            f"Please ensure the model is trained and saved."
        )

    adapter_path = model_path / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter weights not found: {adapter_path}\n"
            f"Please ensure the model training completed successfully."
        )

    try:
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Device: {device}")

        # Load PEFT config
        peft_config = PeftConfig.from_pretrained(str(model_path))
        base_model_name = peft_config.base_model_name_or_path

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        base_model = load_base_model_for_seq_cls(
            base_model_name,
            num_labels=num_labels,
            problem_type="regression",
        )

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id

        # Create PEFT model
        model = get_peft_model(base_model, peft_config)

        # Load adapter weights
        adapter_state_dict = load_file(str(adapter_path))

        # Remap keys for get_peft_model() + load_state_dict() compatibility.
        # - LoRA keys: old format .lora_A.weight -> .lora_A.default.weight
        # - Score keys: always need modules_to_save.default prefix for this path
        needs_lora_remap = any(
            ".lora_A.weight" in k and ".lora_A.default.weight" not in k
            for k in adapter_state_dict.keys()
        )
        needs_score_remap = any(
            k in ("base_model.model.score.weight", "base_model.model.score.bias")
            for k in adapter_state_dict.keys()
        )

        if needs_lora_remap or needs_score_remap:
            remapped = {}
            for key, value in adapter_state_dict.items():
                new_key = key
                if needs_lora_remap and (".lora_A.weight" in key or ".lora_B.weight" in key):
                    new_key = key.replace(".lora_A.weight", ".lora_A.default.weight")
                    new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
                elif key == "base_model.model.score.weight":
                    new_key = "base_model.model.score.modules_to_save.default.weight"
                elif key == "base_model.model.score.bias":
                    new_key = "base_model.model.score.modules_to_save.default.bias"
                remapped[new_key] = value
            adapter_state_dict = remapped

        model.load_state_dict(adapter_state_dict, strict=False)
        model = model.to(device)
        model.eval()

        logger.info("Model loaded successfully")
        return model, tokenizer

    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {type(e).__name__}: {e}")


def load_lora_hub(
    repo_id: str,
    num_labels: int,
    device,
    token: Optional[str] = None,
    torch_dtype=None,
) -> Tuple:
    """
    Load a trained LoRA model from HuggingFace Hub.

    Handles adapter config download, base model loading, and PEFT model
    construction from Hub-hosted adapters.

    Args:
        repo_id: HuggingFace repo ID (e.g., "username/model-name")
        num_labels: Number of output labels/dimensions
        device: torch.device or string
        token: HuggingFace token (required for private repos)
        torch_dtype: Model dtype override (e.g., torch.float16)

    Returns:
        (model, tokenizer) tuple ready for inference

    Raises:
        RuntimeError: If model loading fails
    """
    from peft import PeftModel
    from huggingface_hub import hf_hub_download

    try:
        logger.info(f"Loading model from HuggingFace Hub: {repo_id}")
        logger.info(f"Device: {device}")

        # Download and load adapter config
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="adapter_config.json",
            token=token,
        )

        with open(config_path, "r") as f:
            adapter_config = json.load(f)

        base_model_name = adapter_config["base_model_name_or_path"]
        logger.info(f"Base model: {base_model_name}")

        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            token=token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        logger.info("Loading base model...")
        base_model = load_base_model_for_seq_cls(
            base_model_name,
            num_labels=num_labels,
            problem_type="regression",
            torch_dtype=torch_dtype,
        )

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id

        # Load PEFT model from hub
        logger.info("Loading LoRA adapter from Hub...")
        model = PeftModel.from_pretrained(
            base_model,
            repo_id,
            token=token,
        )

        model = model.to(device)
        model.eval()

        logger.info("Model loaded successfully")
        return model, tokenizer

    except Exception as e:
        # Re-raise HuggingFace-specific errors as-is
        from huggingface_hub.errors import RepositoryNotFoundError, GatedRepoError
        if isinstance(e, (RepositoryNotFoundError, GatedRepoError)):
            raise
        raise RuntimeError(
            f"Failed to load model from Hub ({repo_id}): "
            f"{type(e).__name__}: {e}"
        )
