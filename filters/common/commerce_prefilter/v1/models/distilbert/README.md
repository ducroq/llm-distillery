# Commerce Prefilter - DistilBERT Model

Selected model for commerce/promotional content detection.

## Model Details

| Attribute | Value |
|-----------|-------|
| Base model | `distilbert-base-multilingual-cased` |
| Parameters | 135M |
| Languages | 104 |
| Model size | ~516MB |
| Classification | 2-label softmax |
| Training data | 1,512 articles |
| Test F1 | 97.8% |
| Test Precision | 96.7% |
| Test Recall | 98.9% |

## Files

| File | Description |
|------|-------------|
| `model.safetensors` | Model weights (541MB) |
| `config.json` | Model configuration |
| `tokenizer.json` | Tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer settings |
| `vocab.txt` | Vocabulary file |
| `special_tokens_map.json` | Special token mappings |
| `training_args.bin` | Training arguments |
| `training_metrics.json` | Final validation metrics |
| `cpu_benchmark.json` | CPU inference benchmark |

## Input Format

```
[TITLE] {article_title} [CONTENT] {article_content}
```

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model
model_path = "filters/common/commerce_prefilter/v1/models/distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Format input
title = "Best Black Friday Deals on Electric Vehicles"
content = "Looking for the best EV deals this holiday season?..."
text = f"[TITLE] {title} [CONTENT] {content}"

# Predict
inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    commerce_score = probs[0, 1].item()  # Probability of commerce class
    is_commerce = commerce_score > 0.5

print(f"Commerce score: {commerce_score:.3f}")
print(f"Is commerce: {is_commerce}")
```

## Inference Speed

| Device | Average | P50 | P95 |
|--------|---------|-----|-----|
| GPU (RTX 4080) | 1.8ms | 1.4ms | 2.9ms |
| CPU (Windows) | 90.8ms | 100.5ms | 164.2ms |

## Training

Trained on remote GPU (jeroen@llm-distiller) with:
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5
- Max length: 512 tokens
- Early stopping: patience=2, metric=val_f1

See [../TRAINING_REPORT.md](../TRAINING_REPORT.md) for full details.

## Confusion Matrix (Test Set, n=190)

```
                    Predicted
                 Commerce  Journalism
Actual Commerce      88         1       (98.9% recall)
Actual Journalism     3        98       (97.0% specificity)
```

- **1 false negative:** Commerce passed as journalism
- **3 false positives:** Journalism blocked as commerce
