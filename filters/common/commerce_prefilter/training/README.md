# Commerce Prefilter Training Pipeline

## Overview

Training pipeline for the Commerce Prefilter SLM. Trains multilingual encoder models for binary classification of commerce vs journalism content.

**Status:** COMPLETE (2026-01-17)
**Selected Model:** distilbert-base-multilingual-cased (97.8% F1)

## Scripts

| Script | Purpose |
|--------|---------|
| `train_encoder.py` | Train encoder models (DistilBERT, MiniLM, XLM-R) |
| `train_qwen_lora.py` | Train Qwen with LoRA adapter |
| `evaluate_models.py` | Evaluate all models on test set |
| `benchmark_inference.py` | Benchmark CPU inference speed |

## Training Commands

Training was done on remote GPU (jeroen@llm-distiller, RTX 4080):

```bash
# Train encoder models
python train_encoder.py --model distilbert-base-multilingual-cased --output models/distilbert --data-dir splits
python train_encoder.py --model microsoft/Multilingual-MiniLM-L12-H384 --output models/minilm --data-dir splits
python train_encoder.py --model xlm-roberta-base --output models/xlm-roberta --data-dir splits

# Train Qwen with LoRA
python train_qwen_lora.py --output models/qwen-lora --data-dir splits

# Evaluate all models
python evaluate_models.py --test-data splits/test.jsonl --models-dir models --output results
```

## Data Format

### Input (splits/*.jsonl)
```json
{"title": "Article Title", "content": "Article content...", "label": 1, "commerce_score": 8.5}
{"title": "Article Title", "content": "Article content...", "label": 0, "commerce_score": 1.2}
```

**Labels:**
- `1` = Commerce (oracle score >= 7.0)
- `0` = Journalism (oracle score < 3.0)

**Input format for model:**
```
[TITLE] {title} [CONTENT] {content}
```

### Training Splits

| Split | Total | Commerce | Journalism |
|-------|-------|----------|------------|
| Train | 1,512 | 711 (47%) | 801 (53%) |
| Val | 189 | 89 (47%) | 100 (53%) |
| Test | 190 | 89 (47%) | 101 (53%) |

## Hyperparameters

### Encoder Models

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Max length | 512 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Classification | 2-label softmax |
| Precision | FP16 |

### Qwen LoRA

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 1e-4 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Precision | BF16 |

## Results

| Model | F1 | Precision | Recall | GPU Inference |
|-------|-----|-----------|--------|---------------|
| **distilbert** | **97.8%** | 96.7% | 98.9% | **1.8ms** |
| xlm-roberta | 97.2% | 95.7% | 98.9% | 3.9ms |
| minilm | 95.6% | 93.5% | 97.8% | 3.8ms |
| qwen-lora | 95.0% | 93.5% | 96.6% | 17.3ms |

See [TRAINING_PLAN.md](TRAINING_PLAN.md) for detailed plan and [../v1/TRAINING_REPORT.md](../v1/TRAINING_REPORT.md) for full report.
