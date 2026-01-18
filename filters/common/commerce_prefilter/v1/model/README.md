# Commerce Prefilter Model

This directory contains the trained DistilBERT model for commerce/promotional content detection.

## Files (after training)

- `config.json` - Model configuration
- `model.safetensors` - Model weights
- `tokenizer.json` - Tokenizer vocabulary
- `tokenizer_config.json` - Tokenizer configuration
- `training_config.json` - Training parameters and metrics

## Training

Train the model with:

```bash
python -m filters.common.commerce_prefilter.training.train \
    --data-dir datasets/commerce_prefilter/splits/ \
    --output filters/common/commerce_prefilter/v1/model/
```

## Model Details

| Attribute | Value |
|-----------|-------|
| Base model | `distilbert-base-uncased` |
| Parameters | ~66M |
| Output | Single sigmoid score (0-1) |
| Inference speed | <50ms (CPU) |
| Model size | ~250MB |

## Note

Model files are gitignored. After training, the model can be:
1. Used locally from this directory
2. Uploaded to HuggingFace Hub for sharing
