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

# jeergrvgreg/cultural-discovery-v3

## Model Description

This model is a fine-tuned version of [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
for multi-dimensional content scoring using the **cultural-discovery** filter.

The model was trained using **knowledge distillation** from Gemini Flash, learning to replicate
its judgment patterns on content evaluation.

**Filter Focus**: DISCOVERIES about art, culture, history AND CONNECTIONS between peoples/civilizations

## Intended Use

This model scores articles across 5 semantic dimensions:

- **Discovery Novelty** (weight: 0.25): New finding, revelation, or insight about art/culture/history
- **Heritage Significance** (weight: 0.20): Cultural or historical importance of the subject matter
- **Cross Cultural Connection** (weight: 0.25): Bridges between different peoples, traditions, or civilizations
- **Human Resonance** (weight: 0.15): Connects to lived human experience, not just dry facts
- **Evidence Quality** (weight: 0.15): How well-researched and documented is the content?


## Training Data

- **Training samples**: 6,261
- **Validation samples**: 783
- **Oracle**: Gemini Flash (for ground truth generation)
- **Quality threshold**: Articles with quality_score >= 0.7

## Training Procedure

### Model Architecture

- **Base model**: Qwen/Qwen2.5-1.5B
- **Parameters**: 1,562,194,432
- **Task**: Multi-dimensional regression (8 outputs)
- **Input**: Article title + content (max 512 tokens)
- **Output**: 8 continuous scores (0-10 range)

### Training Configuration

- **Epochs**: 6
- **Batch size**: 8
- **Learning rate**: 2e-05
- **Optimizer**: AdamW
- **Loss function**: Mean Squared Error (MSE)
- **Gradient checkpointing**: Enabled

## Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| Validation MAE | 0.7879 |
| Training MAE | 0.5709 |
| Validation RMSE | 1.2743 |
| Training RMSE | 0.9201 |

### Per-Dimension Performance (Validation MAE)

| Dimension | MAE |
|-----------|-----|
| Discovery Novelty | 0.5419 |
| Heritage Significance | 0.5994 |
| Cross Cultural Connection | 0.6357 |
| Human Resonance | 0.7832 |
| Evidence Quality | 1.3790 |


## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "jeergrvgreg/cultural-discovery-v3"
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
dimensions = ['discovery_novelty', 'heritage_significance', 'cross_cultural_connection', 'human_resonance', 'evidence_quality']

# Print scores
for dim, score in zip(dimensions, scores):
    print(f"{dim}: {score:.2f}")
```

## Limitations

- Model was trained on English news articles
- Performance may vary on other content types
- Validation MAE of 0.7879 indicates ~0.8 point average error on 0-10 scale
- Some overfitting observed (train/val gap: 0.22)

## Ethical Considerations

This model evaluates content based on specific semantic dimensions. Users should:
- Understand the filter's focus and biases
- Not use as sole decision-maker for content moderation
- Regularly evaluate model performance on their specific use case
- Be aware that automated scoring may miss nuance

## Citation

If you use this model, please cite:

```bibtex
@misc{cultural-discovery_filter_v3.0,
  title={Cultural-Discovery Content Filter},
  author={Your Name},
  year={2025},
  url={https://huggingface.co/jeergrvgreg/cultural-discovery-v3}
}
```

## Model Card Contact

For questions or feedback about this model, please open an issue in the repository.
