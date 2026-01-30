---
license: mit
language:
- en
- fr
- es
- de
- nl
- it
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B
- lora
- transformers
- cultural-discovery
- content-filter
library_name: peft
base_model: Qwen/Qwen2.5-1.5B
pipeline_tag: text-classification
---

# Cultural Discovery Filter v3

## Model Description

A fine-tuned **Qwen2.5-1.5B** model with LoRA adapters for multi-dimensional cultural discovery content scoring.

This model evaluates news articles across **5 dimensions** to identify content about cultural discoveries, cross-cultural connections, and heritage significance - separating meaningful cultural journalism from superficial tourism pieces or speculation.

**Key Innovation**: Merged training datasets from v1 (random sampling) and v2 (screening filter) to achieve both data quantity AND distribution quality, solving the regression-to-mean problem.

**Target Applications**: ovr.news (Wisdom tab), Busara

## Dimensions

The model scores articles on 5 dimensions:

### Discovery Dimensions (50% weight)
| Dimension | Weight | Question |
|-----------|--------|----------|
| **Discovery Novelty** | 25% | How new or unknown is this finding? |
| **Cross-Cultural Connection** | 25% | How does it bridge different peoples/civilizations? |

### Heritage & Resonance (35% weight)
| Dimension | Weight | Question |
|-----------|--------|----------|
| **Heritage Significance** | 20% | How culturally/historically important? |
| **Human Resonance** | 15% | Does it connect to lived human experience? |

### Assessment (15% weight)
| Dimension | Weight | Question |
|-----------|--------|----------|
| **Evidence Quality** | 15% | How well-researched and documented? |

## Performance

| Metric | Value |
|--------|-------|
| **Validation MAE** | **0.77** |
| Training MAE | 0.72 |
| Validation RMSE | 1.24 |

### Per-Dimension MAE (Validation)
| Dimension | MAE |
|-----------|-----|
| Discovery Novelty | 0.54 |
| Heritage Significance | 0.58 |
| Cross-Cultural Connection | 0.62 |
| Human Resonance | 0.77 |
| Evidence Quality | 1.36 |

### Tier-Level Performance (vs v1)
| Tier | v1 MAE | v3 MAE | Improvement |
|------|--------|--------|-------------|
| LOW (0-3.9) | 0.75 | 0.60 | +20% |
| MEDIUM (4-6.9) | 2.85 | 1.73 | **+39%** |
| HIGH (7-10) | 3.49 | 2.69 | **+23%** |

## Training Details

- **Base Model**: Qwen/Qwen2.5-1.5B
- **Training Mode**: Knowledge Distillation (from Gemini Flash oracle)
- **Adapter**: LoRA (r=16, α=32, 18.5M trainable params, 1.2% of model)
- **Training Samples**: 6,261
- **Validation Samples**: 783
- **Test Samples**: 783
- **Epochs**: 6 (best at epoch 4)
- **Batch Size**: 8
- **Learning Rate**: 2e-5
- **Max Length**: 512 tokens
- **Preprocessing**: Head+tail (256+256 tokens with " [...] " separator)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

# Load base model and LoRA adapter
base_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    num_labels=5,
    problem_type="regression"
)
model = PeftModel.from_pretrained(base_model, "YOUR_USERNAME/cultural-discovery-v3")
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/cultural-discovery-v3")

# Score an article
article = "Title: Ancient Silk Road Temple Reveals Buddhist-Zoroastrian Syncretism\n\nExcavations at a 4th-century temple..."
inputs = tokenizer(article, return_tensors="pt", max_length=512, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    scores = outputs.logits[0].numpy()

dimensions = ["discovery_novelty", "heritage_significance", "cross_cultural_connection",
              "human_resonance", "evidence_quality"]
weights = [0.25, 0.20, 0.25, 0.15, 0.15]

for dim, weight, score in zip(dimensions, weights, scores):
    print(f"{dim}: {score:.1f}")

# Compute weighted average
weighted_avg = sum(s * w for s, w in zip(scores, weights))
print(f"\nWeighted Average: {weighted_avg:.2f}")

# Assign tier
if weighted_avg >= 7.0:
    tier = "HIGH"
elif weighted_avg >= 4.0:
    tier = "MEDIUM"
else:
    tier = "LOW"
print(f"Tier: {tier}")
```

## Head+Tail Preprocessing

**Important**: This model was trained with head+tail preprocessing. For best results, apply the same preprocessing at inference:

```python
def extract_head_tail(text, tokenizer, head_tokens=256, tail_tokens=256, separator=" [...] "):
    """Extract first and last tokens from text."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= head_tokens + tail_tokens:
        return text
    head = tokens[:head_tokens]
    tail = tokens[-tail_tokens:]
    return tokenizer.decode(head) + separator + tokenizer.decode(tail)
```

## Gatekeeper Rule

**Evidence Quality < 3 → Overall score capped at 3.0**

Speculation without documentation cannot claim cultural significance.

## Tier Definitions

| Tier | Score Range | Description |
|------|-------------|-------------|
| **HIGH** | ≥ 7.0 | Significant discovery or deep cross-cultural insight |
| **MEDIUM** | ≥ 4.0 | Meaningful cultural content with some discovery value |
| **LOW** | < 4.0 | Superficial, speculative, or single-culture content |

## Limitations

- Trained on multilingual news articles (majority English, with French, Spanish, German, Dutch, Italian)
- MAE of ~0.77 means predictions within ±0.8 of oracle on average
- `evidence_quality` dimension has highest error (1.36 MAE) - abstract qualities are harder to learn
- Model still under-predicts high scores (bias -2.27 for articles scoring 7-10)
- Limited high-tier training data (1.9% of dataset)

## Version History

| Version | MAE | Training Data | Key Change |
|---------|-----|---------------|------------|
| v1 | 0.82 | 4,996 (random) | Regression-to-mean problem |
| v2 | 1.47 | 2,919 (screened) | Insufficient data |
| **v3** | **0.77** | **7,827 (merged)** | Best of both worlds |

## License

MIT

## Citation

```bibtex
@misc{cultural_discovery_v3,
  title={Cultural Discovery Filter v3},
  author={LLM Distillery},
  year={2026},
  url={https://huggingface.co/YOUR_USERNAME/cultural-discovery-v3}
}
```

### Framework versions

- PEFT 0.18.1
- Transformers 4.47.0
- PyTorch 2.5.1
