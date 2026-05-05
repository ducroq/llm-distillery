# Investment-Risk v2 - Instruction Tuning Model

**Training Mode:** Instruction Tuning
**Base Model:** Qwen/Qwen2.5-1.5B
**Training Date:** 2025-11-16
**Status:** ⚠️ EXPERIMENTAL (Not recommended for production)

---

## Executive Summary

This package contains the instruction tuning model that was trained as a comparison to knowledge distillation. While it successfully learned the task, it **significantly underperformed** compared to the distillation approach.

**Final Results:**
- **Validation MAE:** 1.4157 (target: <1.0) ❌
- **Validation RMSE:** 1.7693
- **Training Duration:** 3 epochs
- **Dataset:** 4,118 training / 515 validation examples

**Comparison:** Knowledge distillation achieved 0.6711 MAE (52.6% better)

---

## Package Contents

### Model Files
- `model/` - Trained LoRA adapter weights and tokenizer
- `training_history.json` - Epoch-by-epoch training metrics
- `training_metadata.json` - Training configuration and metadata

### Training Reports
Located in `training_reports/`:

- `investment-risk_v2_instruction_training_report.docx` - Comprehensive Word report
- `overall_metrics.png` - MAE/RMSE learning curves
- `per_dimension_mae.png` - 8 dimension learning curves
- `loss_curves.png` - Training/validation loss
- `training_summary.txt` - Quick text summary

**Note:** For comparison analysis, see `../v2_distillation/training_reports/comparison_report.md`

---

## Why This Model Underperformed

### Performance Gap Analysis

| Dimension | This Model | Distillation | Gap |
|-----------|------------|--------------|-----|
| Macro Risk Severity | 1.6431 | 0.6955 | +136% worse |
| Credit Market Stress | 1.0053 | 0.5597 | +80% worse |
| Market Sentiment Extremes | 1.1219 | 0.5906 | +90% worse |
| Valuation Risk | 1.2658 | 0.6384 | +98% worse |
| Policy Regulatory Risk | 1.6673 | 0.7363 | +127% worse |
| Systemic Risk | 1.3044 | 0.6366 | +105% worse |
| Evidence Quality | 1.7990 | 0.8622 | +109% worse |
| Actionability | 1.5191 | 0.6493 | +134% worse |

### Root Causes

1. **Harder Learning Task:** Generate reasoning + scores vs just predict scores
2. **More Complex:** 1024 token context vs 512 tokens
3. **Prompt Overhead:** Instruction prompt takes valuable context
4. **Regression Mismatch:** Reasoning is better for classification, not regression
5. **Less Training Signal:** Model tries to learn two tasks (reasoning + scoring)

---

## Training Configuration

```json
{
  "training_mode": "instruction_tuning",
  "model": "Qwen/Qwen2.5-1.5B",
  "parameters": 1,562,203,648,
  "trainable_parameters": 18,477,056,
  "epochs": 3,
  "batch_size": 8,
  "learning_rate": 2e-05,
  "max_length": 1024,
  "include_prompt": true
}
```

### What is Instruction Tuning?

In this mode, the student model learns to generate reasoning and scores:

**Input:** Instruction prompt + Article title + content (max 1024 tokens)
**Output:** Reasoning paragraph + 8 dimension scores (JSON format)
**Loss:** Mean Squared Error on scores + token prediction loss on reasoning

The model tries to mimic the oracle's reasoning process, but this added complexity hurts performance for this regression task.

---

## When to Use Instruction Tuning

While this model underperformed for investment-risk scoring, instruction tuning can be valuable when:

1. **Interpretability is Critical** - You need to see the model's reasoning
2. **Multi-Task Learning** - Model needs to do multiple things beyond scoring
3. **Few-Shot Adaptation** - Model needs to adapt to new dimensions quickly
4. **Human Review** - Humans need to validate the reasoning, not just scores
5. **Classification Tasks** - Better for discrete categories than continuous scores

**For this use case:** Knowledge distillation is clearly superior.

---

## Lessons Learned

### What Worked
- ✅ Model successfully learned the instruction format
- ✅ Training converged without issues
- ✅ Negative train/val gap (-0.0377) shows good generalization

### What Didn't Work
- ❌ Accuracy far below target (1.42 vs <1.0)
- ❌ Lost all 8 dimensions to distillation
- ❌ 2x context length for worse results
- ❌ Reasoning didn't improve score prediction

### Recommendation
**Do not use this model for production.** Use the knowledge distillation model instead (`../v2_distillation/`), which achieved 52.6% better validation MAE.

---

## Experimental Value

This model serves as an important **negative result** that validates the training strategy decision:

1. **Confirms:** Direct score learning > reasoning-based learning for regression
2. **Validates:** Simpler is better for this task
3. **Informs:** Future filter training should use distillation by default
4. **Documents:** What doesn't work (important for institutional knowledge)

---

## Related Documentation

- **Filter Specification:** `../v2/config.yaml`
- **Oracle Calibration:** `../v2/ground_truth_quality_report.md`
- **Package Validation:** `../v2/package_validation.md`
- **Winning Model:** `../v2_distillation/` ⭐ **USE THIS INSTEAD**
- **Comparison Analysis:** `../v2_distillation/training_reports/comparison_report.md`

---

**Status:** Archived for comparison purposes only. Not recommended for production use.
