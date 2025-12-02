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

# jeergrvgreg/investment-risk-filter-v5

## Model Description

This model is a fine-tuned version of [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
for multi-dimensional content scoring using the **investment-risk** filter.

The model was trained using **knowledge distillation** from Gemini Flash, learning to replicate
its judgment patterns on content evaluation.

**Filter Focus**: Capital preservation filter with orthogonal dimensions

## Intended Use

This model scores articles across 6 semantic dimensions:

- **Risk Domain Type** (weight: 0.20): WHERE in the financial system is the risk?
- **Severity Magnitude** (weight: 0.25): HOW BAD could this be if it materializes?
- **Materialization Timeline** (weight: 0.15): WHEN would the impact materialize?
- **Evidence Quality** (weight: 0.15): HOW well documented is this risk signal?
- **Impact Breadth** (weight: 0.15): WHO is affected by this risk?
- **Retail Actionability** (weight: 0.10): CAN a hobby investor (10K-500K) respond meaningfully?


## Training Data

- **Training samples**: 8,157
- **Validation samples**: 1,020
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

- **Epochs**: 8
- **Batch size**: 8
- **Learning rate**: 2e-05
- **Optimizer**: AdamW
- **Loss function**: Mean Squared Error (MSE)
- **Gradient checkpointing**: Enabled

## Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| Validation MAE | 0.4782 |
| Training MAE | 0.2377 |
| Validation RMSE | 0.9112 |
| Training RMSE | 0.3864 |

### Per-Dimension Performance (Validation MAE)

| Dimension | MAE |
|-----------|-----|
| Risk Domain Type | 0.4018 |
| Severity Magnitude | 0.3250 |
| Materialization Timeline | 0.7149 |
| Evidence Quality | 0.5521 |
| Impact Breadth | 0.4953 |
| Retail Actionability | 0.3799 |


## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "jeergrvgreg/investment-risk-filter-v5"
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
dimensions = ['risk_domain_type', 'severity_magnitude', 'materialization_timeline', 'evidence_quality', 'impact_breadth', 'retail_actionability']

# Print scores
for dim, score in zip(dimensions, scores):
    print(f"{dim}: {score:.2f}")
```

## Limitations

- Model was trained on English news articles
- Performance may vary on other content types
- Validation MAE of 0.4782 indicates ~0.8 point average error on 0-10 scale
- Some overfitting observed (train/val gap: 0.24)

## Ethical Considerations

This model evaluates content based on specific semantic dimensions. Users should:
- Understand the filter's focus and biases
- Not use as sole decision-maker for content moderation
- Regularly evaluate model performance on their specific use case
- Be aware that automated scoring may miss nuance

## Citation

If you use this model, please cite:

```bibtex
@misc{investment-risk_filter_v5.0,
  title={Investment-Risk Content Filter},
  author={Your Name},
  year={2025},
  url={https://huggingface.co/jeergrvgreg/investment-risk-filter-v5}
}
```

## Model Card Contact

For questions or feedback about this model, please open an issue in the repository.
