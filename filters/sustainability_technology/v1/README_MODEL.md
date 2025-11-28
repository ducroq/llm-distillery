---
license: mit
language: en
tags:
- text-classification
- content-filtering
- multi-dimensional-scoring
- knowledge-distillation
library_name: transformers
pipeline_tag: text-classification
---

# jeergrvgreg/sustainability-technology-v1

## Model Description

This model is a fine-tuned version of [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
for multi-dimensional content scoring using the **sustainability_technology** filter.

The model was trained using **knowledge distillation** from Gemini Flash, learning to replicate
its judgment patterns on content evaluation.

**Filter Focus**: LCSA-based climate tech assessment

## Intended Use

This model scores articles across 6 semantic dimensions:

- **Technology Readiness Level** (weight: 0.15): Deployment stage (TRL 1-9)
- **Technical Performance** (weight: 0.15): Real-world reliability and efficiency
- **Economic Competitiveness** (weight: 0.20): Life Cycle Cost (LCC) competitiveness
- **Life Cycle Environmental Impact** (weight: 0.30): Holistic environmental assessment
- **Social Equity Impact** (weight: 0.10): Jobs, ethics, equitable access
- **Governance Systemic Impact** (weight: 0.10): Systemic disruption potential


## Training Data

- **Training samples**: 7,990
- **Validation samples**: 999
- **Oracle**: Gemini Flash (for ground truth generation)
- **Quality threshold**: Articles with quality_score >= 0.7

## Training Procedure

### Model Architecture

- **Base model**: Qwen/Qwen2.5-1.5B
- **Parameters**: 1,562,197,504
- **Task**: Multi-dimensional regression (8 outputs)
- **Input**: Article title + content (max 512 tokens)
- **Output**: 8 continuous scores (0-10 range)

### Training Configuration

- **Epochs**: 3
- **Batch size**: 8
- **Learning rate**: 2e-05
- **Optimizer**: AdamW
- **Loss function**: Mean Squared Error (MSE)
- **Gradient checkpointing**: Enabled

## Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| Validation MAE | 0.7116 |
| Training MAE | 0.6388 |
| Validation RMSE | 1.0188 |
| Training RMSE | 0.8950 |

### Per-Dimension Performance (Validation MAE)

| Dimension | MAE |
|-----------|-----|
| Technology Readiness Level | 0.7534 |
| Technical Performance | 0.6854 |
| Economic Competitiveness | 0.6557 |
| Life Cycle Environmental Impact | 0.5570 |
| Social Equity Impact | 0.6978 |
| Governance Systemic Impact | 0.9206 |


## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "jeergrvgreg/sustainability-technology-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prepare input
article = {
    "title": "Example Article Title",
    "content": "Article content here..."
}

text = f"{article['title']}\n\n{article['content']}"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    scores = outputs.logits[0].numpy()

# Dimension names
dimensions = ['technology_readiness_level', 'technical_performance', 'economic_competitiveness', 'life_cycle_environmental_impact', 'social_equity_impact', 'governance_systemic_impact']

# Print scores
for dim, score in zip(dimensions, scores):
    print(f"{dim}: {score:.2f}")
```

## Limitations

- Model was trained on English news articles
- Performance may vary on other content types
- Validation MAE of 0.7116 indicates ~0.8 point average error on 0-10 scale
- Some overfitting observed (train/val gap: 0.07)

## Ethical Considerations

This model evaluates content based on specific semantic dimensions. Users should:
- Understand the filter's focus and biases
- Not use as sole decision-maker for content moderation
- Regularly evaluate model performance on their specific use case
- Be aware that automated scoring may miss nuance

## Citation

If you use this model, please cite:

```bibtex
@misc{sustainability_technology_filter_v1.0,
  title={Sustainability_Technology Content Filter},
  author={Your Name},
  year={2025},
  url={https://huggingface.co/jeergrvgreg/sustainability-technology-v1}
}
```

## Model Card Contact

For questions or feedback about this model, please open an issue in the repository.
