# Gemma-3-1B Model Loading & PEFT

## Auto Mapping Issue

`google/gemma-3-1b-pt` uses `Gemma3TextConfig` with `model_type: gemma3_text`, but `AutoModelForSequenceClassification` only maps `gemma3` (multimodal), not `gemma3_text`. Affects transformers 4.55.3.

**Solution**: `filters/common/model_loading.py` provides `load_base_model_for_seq_cls()`. It tries Auto first, falls back to building a custom `Gemma3TextForSequenceClassification` class using `Gemma3TextModel` + `nn.Linear` head.

**All Gemma-3 filters must use `load_base_model_for_seq_cls()` instead of `AutoModelForSequenceClassification`.**

## PEFT Adapter Key Formats

Two loading paths require DIFFERENT key formats:

| Path | LoRA keys | Score head key | How |
|------|-----------|----------------|-----|
| **Local** (`get_peft_model()` + `load_state_dict()`) | `.lora_A.default.weight` | `score.modules_to_save.default.weight` | inference.py remaps at load |
| **Hub** (`PeftModel.from_pretrained()`) | `.lora_A.weight` (OLD) | `score.weight` | PEFT handles internally |

**Rule**: Keep adapter files in OLD format for Hub compatibility. Local inference.py remaps as needed. Do NOT resave to new PEFT format — it breaks Hub loading.

## Hub Upload

Upload script (`scripts/deployment/upload_to_huggingface.py`) automatically verifies Hub loading after upload. If verification fails, check:
1. Adapter key format (must be OLD format)
2. `adapter_config.json` references correct base model
3. Base model accessible (or cached if `HF_HUB_OFFLINE=1`)

## GPU Server Specifics

- Use `torch_dtype=torch.float16` directly on gpu-server (not bfloat16)
- Base model `google/gemma-3-1b-pt` must be cached in `~/.cache/huggingface/hub/`
- `HF_HUB_OFFLINE=1` required — gpu-server can't resolve huggingface.co
