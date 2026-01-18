# Commerce Prefilter SLM - Training Plan

**Date:** 2026-01-17
**Status:** APPROVED - Ready for execution

---

## Overview

Train 4 multilingual model candidates for commerce/promotional content detection. Select the best model based on accuracy and inference speed.

---

## Model Candidates

| Model | Params | Type | Fine-tuning |
|-------|--------|------|-------------|
| distilbert-base-multilingual-cased | 135M | Encoder | Classification head |
| microsoft/Multilingual-MiniLM-L12-H384 | 118M | Encoder | Classification head |
| xlm-roberta-base | 270M | Encoder | Classification head |
| Qwen/Qwen2.5-0.5B | 500M | Decoder | LoRA (PEFT) |

---

## Training Data

| Split | Total | Commerce | Journalism | File |
|-------|-------|----------|------------|------|
| Train | 1,512 | 711 (47%) | 801 (53%) | `splits/train.jsonl` |
| Val | 189 | 89 (47%) | 100 (53%) | `splits/val.jsonl` |
| Test | 190 | 89 (47%) | 101 (53%) | `splits/test.jsonl` |

**Labels:**
- `1` = Commerce (oracle score >= 7.0)
- `0` = Journalism (oracle score < 3.0)

---

## Input Format

**Format: Title + Content**

```
[TITLE] {article_title} [CONTENT] {article_content}
```

Example:
```
[TITLE] Best Black Friday Deals on Electric Vehicles [CONTENT] Looking for the best EV deals this holiday season? We've rounded up...
```

**Rationale:**
- Title contains strong commerce signals
- Avoids overfitting to URL patterns
- Model learns content-based signals

---

## Hyperparameters

### Encoder Models (DistilBERT, MiniLM, XLM-RoBERTa)

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Max length | 512 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Early stopping | patience=2, metric=val_f1 |
| Optimizer | AdamW |

### Decoder Model (Qwen 0.5B with LoRA)

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 8 (smaller due to size) |
| Learning rate | 1e-4 (higher for LoRA) |
| Max length | 512 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA target modules | q_proj, v_proj |
| Early stopping | patience=2, metric=val_f1 |

---

## Success Criteria

| Metric | Minimum | Target |
|--------|---------|--------|
| F1 Score | 90% | 95% |
| Precision | 90% | 95% |
| Recall | 85% | 90% |
| Inference (CPU) | <100ms | <50ms |

**Decision rules:**
- If multiple models meet target: choose fastest
- If no model meets minimum: need more data or different approach
- Trade-off: prioritize precision over recall (avoid blocking journalism)

---

## Infrastructure

| Task | Location | Hardware |
|------|----------|----------|
| Training | jeroen@llm-distiller | GPU |
| Inference benchmark | Local (Windows) | CPU |
| Production | Local/Server | CPU |

---

## Execution Steps

### Step 1: Sync Data to Remote
```bash
# From local machine
scp -r datasets/training/commerce_prefilter_v1/splits jeroen@llm-distiller:~/commerce_prefilter/
```

### Step 2: Install Dependencies (Remote)
```bash
# On llm-distiller
pip install transformers datasets peft accelerate scikit-learn
```

### Step 3: Train Encoder Models (Remote)
```bash
# Train each encoder model
python train_encoder.py --model distilbert-base-multilingual-cased --output models/distilbert
python train_encoder.py --model microsoft/Multilingual-MiniLM-L12-H384 --output models/minilm
python train_encoder.py --model xlm-roberta-base --output models/xlm-roberta
```

### Step 4: Train Qwen with LoRA (Remote)
```bash
python train_qwen_lora.py --output models/qwen-lora
```

### Step 5: Evaluate All Models (Remote)
```bash
python evaluate_models.py --test-data splits/test.jsonl --output results/
```

### Step 6: Sync Models Back to Local
```bash
# From local machine
scp -r jeroen@llm-distiller:~/commerce_prefilter/models filters/common/commerce_prefilter/v1/
```

### Step 7: Benchmark Inference Speed (Local)
```bash
python benchmark_inference.py --models-dir v1/models --output benchmark_results.json
```

### Step 8: Select Best Model
Based on F1 score and inference speed, update `config.yaml` with chosen model.

---

## Output Artifacts

```
filters/common/commerce_prefilter/v1/
├── models/
│   ├── distilbert/           # Full model
│   ├── minilm/               # Full model
│   ├── xlm-roberta/          # Full model
│   └── qwen-lora/            # Base + LoRA adapter
├── results/
│   ├── training_metrics.json # Loss/accuracy curves
│   ├── evaluation_results.json # Test set metrics
│   └── benchmark_results.json # Inference speed
└── config.yaml               # Updated with best model
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GPU OOM | Reduce batch size, use gradient accumulation |
| Slow training | Use mixed precision (fp16) |
| Poor accuracy | Add more training data, adjust thresholds |
| Slow inference | Choose smaller model, quantization |

---

## Timeline

| Step | Estimated Time |
|------|----------------|
| Sync data | 5 min |
| Train 3 encoders | 30-60 min total |
| Train Qwen LoRA | 30-60 min |
| Evaluate | 10 min |
| Sync back | 5 min |
| Benchmark | 10 min |
| **Total** | **~2 hours** |

---

*Plan finalized: 2026-01-17*
