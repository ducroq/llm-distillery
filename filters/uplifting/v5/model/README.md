---
license: mit
language: en
tags:
- text-classification
- content-filtering
- multi-dimensional-scoring
- knowledge-distillation
- uplifting-content
- news-analysis
library_name: peft
base_model: Qwen/Qwen2.5-1.5B
pipeline_tag: text-classification
---

# Uplifting Content Filter v5

## Model Description

A fine-tuned **Qwen2.5-1.5B** model with LoRA adapters for multi-dimensional uplifting content scoring.

This model evaluates news articles across **6 orthogonal dimensions** to identify genuinely uplifting content with documented positive outcomes - not just feel-good stories or speculation.

**Key Innovation**: Uses an orthogonal dimension framework (inspired by LCSA methodology) to avoid the high correlation issues found in previous versions.

## Dimensions

The model scores articles on 6 dimensions:

### Impact Domains (WHAT kind of uplift)
| Dimension | Weight | Question |
|-----------|--------|----------|
| **Human Wellbeing Impact** | 25% | Health, safety, livelihoods improved? |
| **Social Cohesion Impact** | 15% | Communities strengthened, solidarity built? |
| **Justice & Rights Impact** | 10% | Wrongs addressed, rights expanded? |

### Assessment Dimensions (HOW real/accessible)
| Dimension | Weight | Question |
|-----------|--------|----------|
| **Evidence Level** | 20% | Documented outcomes or speculation? |
| **Benefit Distribution** | 20% | Who benefits? Elite → Universal? |
| **Change Durability** | 10% | Temporary relief → Systemic change? |

## Performance

| Metric | Value |
|--------|-------|
| **Validation MAE** | **0.681** |
| Training MAE | 0.637 |
| Validation RMSE | 0.880 |

### Per-Dimension MAE (Validation)
| Dimension | MAE |
|-----------|-----|
| Human Wellbeing Impact | 0.686 |
| Social Cohesion Impact | 0.704 |
| Justice Rights Impact | 0.619 |
| Evidence Level | 0.636 |
| Benefit Distribution | 0.792 |
| Change Durability | 0.648 |

## Training Details

- **Base Model**: Qwen/Qwen2.5-1.5B
- **Training Mode**: Knowledge Distillation (from Gemini Flash oracle)
- **Adapter**: LoRA (18.5M trainable params, 1.2% of model)
- **Training Samples**: 7,999
- **Validation Samples**: 1,000
- **Epochs**: 3
- **Batch Size**: 8
- **Learning Rate**: 2e-5
- **Max Length**: 512 tokens

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

# Load base model and LoRA adapter
base_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    num_labels=6,
    problem_type="regression"
)
model = PeftModel.from_pretrained(base_model, "nexusmind/uplifting-filter-v5")
tokenizer = AutoTokenizer.from_pretrained("nexusmind/uplifting-filter-v5")

# Score an article
article = "Title: Community garden feeds 500 families\n\nA new community garden..."
inputs = tokenizer(article, return_tensors="pt", max_length=512, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    scores = outputs.logits[0].numpy()

dimensions = ["human_wellbeing_impact", "social_cohesion_impact", "justice_rights_impact",
              "evidence_level", "benefit_distribution", "change_durability"]

for dim, score in zip(dimensions, scores):
    print(f"{dim}: {score:.1f}")
```

## Gatekeeper Rule

**Evidence Level < 3 → Overall score capped at 3.0**

Speculation without documented outcomes cannot be truly uplifting.

## Limitations

- Trained on English news articles only
- MAE of ~0.68 means predictions within ±0.7 of oracle on average
- `benefit_distribution` dimension has highest error (0.79 MAE)
- Model focuses on documented outcomes, not emotional tone

## License

MIT

## Citation

```bibtex
@misc{uplifting_filter_v5,
  title={Uplifting Content Filter v5},
  author={NexusMind},
  year={2025},
  url={https://huggingface.co/nexusmind/uplifting-filter-v5}
}
```
