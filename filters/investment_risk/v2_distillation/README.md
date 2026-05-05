# Investment-Risk v2 - Knowledge Distillation Model

**Training Mode:** Knowledge Distillation
**Base Model:** Qwen/Qwen2.5-1.5B
**Training Date:** 2025-11-16
**Status:** ✅ PRODUCTION READY

---

## Executive Summary

This package contains the **winning model** from the training mode comparison. Knowledge distillation significantly outperformed instruction tuning with **52.6% better validation MAE**.

**Final Results:**
- **Validation MAE:** 0.6711 (target: <1.0) ✅
- **Validation RMSE:** 0.9303
- **Training Duration:** 3 epochs
- **Dataset:** 4,118 training / 515 validation examples

---

## Package Contents

### Model Files
- `model/` - Trained LoRA adapter weights and tokenizer
- `training_history.json` - Epoch-by-epoch training metrics
- `training_metadata.json` - Training configuration and metadata

### Training Reports
Located in `training_reports/`:

#### Individual Training Analysis
- `investment-risk_v2_distillation_training_report.docx` - Comprehensive Word report
- `overall_metrics.png` - MAE/RMSE learning curves
- `per_dimension_mae.png` - 8 dimension learning curves
- `loss_curves.png` - Training/validation loss
- `training_summary.txt` - Quick text summary

#### Comparison Analysis (vs Instruction Tuning)
- `comparison_report.md` - Full comparison analysis with recommendations
- `mode_comparison_mae.png` - Side-by-side MAE comparison
- `mode_comparison_per_dimension.png` - Per-dimension performance bars
- `mode_comparison_improvement.png` - Improvement percentage breakdown

---

## Key Findings

### Why Knowledge Distillation Won

1. **Superior Accuracy:** 52.6% better validation MAE (0.67 vs 1.42)
2. **All Dimensions:** Won 8/8 dimensions (44-58% improvement each)
3. **More Efficient:** 512 tokens vs 1024 tokens max length
4. **Better for Regression:** Direct score learning > reasoning-based learning
5. **Lower Inference Cost:** Faster and cheaper in production

### Performance by Dimension

| Dimension | Val MAE | vs Instruction | Improvement |
|-----------|---------|----------------|-------------|
| Macro Risk Severity | 0.6955 | 1.6431 | +57.7% |
| Credit Market Stress | 0.5597 | 1.0053 | +44.3% |
| Market Sentiment Extremes | 0.5906 | 1.1219 | +47.4% |
| Valuation Risk | 0.6384 | 1.2658 | +49.6% |
| Policy Regulatory Risk | 0.7363 | 1.6673 | +55.8% |
| Systemic Risk | 0.6366 | 1.3044 | +51.2% |
| Evidence Quality | 0.8622 | 1.7990 | +52.1% |
| Actionability | 0.6493 | 1.5191 | +57.3% |

---

## Training Configuration

```json
{
  "training_mode": "knowledge_distillation",
  "model": "Qwen/Qwen2.5-1.5B",
  "parameters": 1,562,203,648,
  "trainable_parameters": 18,477,056,
  "epochs": 3,
  "batch_size": 8,
  "learning_rate": 2e-05,
  "max_length": 512,
  "include_prompt": false
}
```

### What is Knowledge Distillation?

In this mode, the student model learns directly from the oracle's numerical scores:

**Input:** Article title + content (max 512 tokens)
**Output:** 8 dimension scores (0-10 scale)
**Loss:** Mean Squared Error between student and oracle scores

The model learns to predict scores without seeing the oracle's reasoning, making it:
- Faster (no prompt overhead)
- More accurate (direct score learning)
- Cheaper (shorter context)

---

## Comparison vs Instruction Tuning

See `training_reports/comparison_report.md` for full analysis.

**Quick Summary:**
- ✅ **Knowledge Distillation:** 0.6711 MAE, 512 tokens, no prompt
- ⚠️ **Instruction Tuning:** 1.4157 MAE, 1024 tokens, includes reasoning prompt

The instruction tuning model was trained to generate reasoning + scores, but performed significantly worse. For this regression task, direct score learning is more effective.

---

## Production Deployment

### Recommendation: ✅ USE THIS MODEL

**Reasons:**
1. Best accuracy (0.67 MAE vs target <1.0)
2. Most efficient (512 token limit)
3. Fastest inference
4. Lowest cost
5. Won all 8 dimensions

### Inference Pipeline

```
Article → Prefilter → Model (512 tokens) → 8 Scores → Tier Classification
```

### Expected Performance
- **Throughput:** ~1000 articles/hour
- **Latency:** <50ms per article
- **Cost:** $0 (local inference)
- **Accuracy:** 0.67 MAE vs oracle

---

## Next Steps

1. **Deploy to Production** - Model is ready
2. **Monitor Live Performance** - Track prediction quality on real traffic
3. **Collect Feedback** - Identify edge cases and failure modes
4. **Plan v3** - Consider larger model (3B/7B) if more accuracy needed

---

## Related Documentation

- **Filter Specification:** `../v2/config.yaml`
- **Oracle Calibration:** `../v2/ground_truth_quality_report.md`
- **Package Validation:** `../v2/package_validation.md`
- **Release Report:** `../v2/release_report.md`
- **Instruction Tuning Model:** `../v2_instruction/` (for comparison)

---

**For Questions:** See training reports in `training_reports/` directory
