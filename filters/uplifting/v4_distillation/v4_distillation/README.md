# Uplifting v4 - Knowledge Distillation Model

**Training Date:** 2025-11-16
**Model:** Qwen 2.5-1.5B + LoRA
**Training Mode:** Knowledge Distillation
**Status:** ✅ Production Ready

---

## Model Performance

**Validation Results:**
- **MAE**: 1.00 (target: <1.2) ✅
- **RMSE**: 1.32
- **Train/Val Gap**: 0.22 (28% overfitting - moderate)

**Per-Dimension Validation MAE:**
| Dimension | Weight | Val MAE | Status |
|-----------|--------|---------|--------|
| collective_benefit | 38% | 0.87 | ✅ Excellent |
| progress | 19% | 0.87 | ✅ Excellent |
| agency | 14% | 0.90 | ✅ Excellent |
| innovation | 8% | 0.94 | ✅ Excellent |
| justice | 3% | 1.05 | ✅ Good |
| wonder | 5% | 1.12 | ✅ Good |
| resilience | 3% | 1.14 | ✅ Good |
| connection | 10% | 1.15 | ✅ Good |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2.5-1.5B |
| Trainable Params | 18.5M (1.18% of total) |
| Epochs | 5 |
| Batch Size | 8 |
| Learning Rate | 2e-5 |
| Max Length | 512 tokens |
| Training Examples | 3,778 |
| Validation Examples | 472 |
| Training Time | ~2.5 hours |

---

## Directory Contents

```
filters/uplifting/v4_distillation/
├── model/                           # Trained model (deploy this)
│   ├── adapter_model.safetensors    # LoRA weights (70MB)
│   ├── adapter_config.json
│   ├── tokenizer files...
│   └── README.md
├── training_history.json            # Epoch-by-epoch metrics
├── training_metadata.json           # Training configuration
└── training_reports/                # Analysis and visualizations
    ├── uplifting_v4.0_knowledge_distillation_report.docx
    ├── overall_metrics.png
    ├── per_dimension_mae.png
    ├── loss_curves.png
    └── training_summary.txt
```

---

## Usage

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    num_labels=8,  # 8 dimensions
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "filters/uplifting/v4_distillation/model"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "filters/uplifting/v4_distillation/model"
)

# Inference
inputs = tokenizer(
    "Article title. Article content...",
    max_length=512,
    truncation=True,
    return_tensors="pt"
)

outputs = model(**inputs)
scores = outputs.logits[0].tolist()  # [agency, progress, collective_benefit, ...]
```

### Dimension Order

Scores are returned in this order:
1. agency (14% weight)
2. progress (19% weight)
3. collective_benefit (38% weight) - **gatekeeper dimension**
4. connection (10% weight)
5. innovation (8% weight)
6. justice (3% weight)
7. resilience (3% weight)
8. wonder (5% weight)

---

## Model Architecture

**Knowledge Distillation Mode:**
- Input: Article title + content (max 512 tokens)
- No prompt included (vs instruction tuning which includes reasoning prompt)
- Direct regression to oracle scores
- Faster inference, better accuracy

**LoRA Configuration:**
- Rank: 8
- Alpha: 16
- Target modules: All linear layers
- Trainable parameters: 1.18% of base model

---

## Training Progression

| Epoch | Train MAE | Val MAE | Improvement |
|-------|-----------|---------|-------------|
| 1 | 2.24 | 1.66 | Baseline |
| 2 | 1.41 | 1.18 | -29% |
| 3 | 1.05 | 1.05 | -11% |
| 4 | 0.88 | 1.01 | -4% |
| 5 | 0.78 | 1.00 | -1% |

Best validation MAE achieved at epoch 5: **1.00**

---

## Comparison to Investment-Risk v2

| Metric | Investment-Risk v2 | Uplifting v4 | Difference |
|--------|-------------------|--------------|------------|
| Validation MAE | 0.67 | 1.00 | +49% worse |
| Dimensions | 5 | 8 | +60% more |
| Training Examples | 4,120 | 3,778 | -8% fewer |
| Train/Val Gap | 7% | 28% | Higher overfitting |

**Why is uplifting worse?**
- More dimensions (8 vs 5) = more complex task
- More subjective dimensions (wonder, connection) vs objective financial metrics
- Both models meet production threshold (<1.2 MAE)

---

## Production Deployment

### Deployment Checklist

- [x] Validation MAE <1.2 ✅
- [x] All dimensions MAE <1.2 ✅
- [x] No training instabilities ✅
- [x] Model size reasonable (70MB LoRA weights) ✅
- [ ] Test set evaluation (recommended before deployment)
- [ ] Production monitoring setup

### Deployment Command

```bash
# Copy model to production server
scp -r filters/uplifting/v4_distillation/model/ production-server:/models/uplifting_v4/

# Or commit to repository
git add filters/uplifting/v4_distillation/
git commit -m "Add trained uplifting v4 distillation model (1.00 MAE)"
git push
```

---

## Quality Assessment

### ✅ Strengths
1. Meets production threshold (1.00 < 1.2 MAE)
2. High-weight dimensions perform best (collective_benefit, progress, agency all <0.9 MAE)
3. Smooth convergence, no training instabilities
4. Efficient model size (18.5M trainable params)

### ⚠️ Areas for Improvement
1. Moderate overfitting (28% train/val gap vs 7% for investment-risk)
2. Subjective dimensions harder to predict (wonder, connection, resilience >1.1 MAE)
3. Required more epochs than investment-risk (5 vs 3)

### Recommended Next Steps (Optional)
1. **Test set evaluation**: Validate generalization on held-out test set
2. **More data**: Collect 5,000 more articles if MAE <0.8 needed
3. **Regularization**: Add dropout if overfitting worsens in production

---

## Reports and Documentation

- **Training Report**: `filters/uplifting/v4/training_report.md`
- **Word Report**: `training_reports/uplifting_v4.0_knowledge_distillation_report.docx`
- **Visualizations**: `training_reports/*.png`
- **Ground Truth Quality**: `filters/uplifting/v4/ground_truth_quality_report.md`
- **Training Guide**: `filters/uplifting/v4/TRAINING_GUIDE.md`

---

## Metadata

**Filter Version**: 4.0
**Training Date**: 2025-11-16
**Oracle**: Gemini Flash 1.5 (batch API)
**Training Examples**: 3,778 articles
**Validation Examples**: 472 articles
**Seed**: 42 (reproducible)

**Best Validation MAE**: 1.00
**Status**: ✅ **PRODUCTION READY**
