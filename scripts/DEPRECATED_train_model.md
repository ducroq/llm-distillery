# DEPRECATED: train_model.py

**Date**: 2025-11-09
**Status**: Deprecated in favor of `training/train.py`

## Why Deprecated?

`scripts/train_model.py` uses a **text generation approach** (via Unsloth) that expects data in prompt/completion format:

```json
{
  "prompt": "Score this article...",
  "completion": "ASSISTANT: [8, 7, 6, 8, 5, 7, 6, 8]"
}
```

However, our training data preparation (`prepare_training_data_tech_deployment.py`) creates **regression format**:

```json
{
  "id": "...",
  "title": "...",
  "content": "...",
  "labels": [8, 7, 6, 8, 5, 7, 6, 8],
  "dimension_names": [...]
}
```

## Decision

Per ADR `2025-11-09-model-output-format.md`, we decided on **score arrays only** (no text generation). Therefore:

- ✅ **Use**: `training/train.py` (regression-based training)
- ❌ **Deprecated**: `scripts/train_model.py` (text generation-based)

## Migration Path

If you need the Unsloth approach (for memory efficiency), you would need to:

1. Rewrite `prepare_training_data_tech_deployment.py` to output prompt/completion format
2. Update the ADR to change the decision from "score arrays only" to "text generation"
3. Accept the tradeoffs (risk of invalid JSON, slower inference)

For now, **use the regression approach** with `training/train.py`.

## Training Command

```bash
python -m training.train \
    --filter filters/sustainability_tech_deployment/v1 \
    --data-dir datasets/training/sustainability_tech_deployment \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5
```

## See Also

- ADR: `docs/decisions/2025-11-09-model-output-format.md`
- Training script: `training/train.py`
- Data prep: `scripts/prepare_training_data_tech_deployment.py`
