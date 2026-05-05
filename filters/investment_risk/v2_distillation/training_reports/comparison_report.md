# Training Mode Comparison Report

## Executive Summary

**Winner:** Knowledge Distillation
**Improvement:** 52.6% better validation MAE

| Metric | Distillation | Instruction | Better |
|--------|--------------|-------------|--------|
| Val MAE | 0.6711 | 1.4157 | ✅ Distillation |
| Val RMSE | 0.9303 | 1.7693 | ✅ Distillation |
| Train/Val Gap | +0.0463 | -0.0377 | ✅ Instruction |

## Training Configuration Differences

| Setting | Distillation | Instruction |
|---------|--------------|-------------|
| Training Mode | knowledge_distillation | instruction_tuning |
| Include Prompt | False | True |
| Max Length | 512 tokens | 1024 tokens |
| Best Val MAE | 0.6711 | 1.3630 |

## Overall Metrics Comparison

### Final Epoch Results

#### Knowledge Distillation

- **Train MAE:** 0.6248
- **Val MAE:** 0.6711
- **Train RMSE:** 0.8415
- **Val RMSE:** 0.9303
- **Train/Val Gap:** +0.0463

#### Instruction Tuning

- **Train MAE:** 1.4534
- **Val MAE:** 1.4157
- **Train RMSE:** 1.7883
- **Val RMSE:** 1.7693
- **Train/Val Gap:** -0.0377

## Per-Dimension Analysis

| Dimension | Distillation Val MAE | Instruction Val MAE | Improvement % | Winner |
|-----------|---------------------|---------------------|---------------|--------|
| Macro Risk Severity | 0.6955 | 1.6431 | +57.7% | ✅ Distillation |
| Credit Market Stress | 0.5597 | 1.0053 | +44.3% | ✅ Distillation |
| Market Sentiment Extremes | 0.5906 | 1.1219 | +47.4% | ✅ Distillation |
| Valuation Risk | 0.6384 | 1.2658 | +49.6% | ✅ Distillation |
| Policy Regulatory Risk | 0.7363 | 1.6673 | +55.8% | ✅ Distillation |
| Systemic Risk | 0.6366 | 1.3044 | +51.2% | ✅ Distillation |
| Evidence Quality | 0.8622 | 1.7990 | +52.1% | ✅ Distillation |
| Actionability | 0.6493 | 1.5191 | +57.3% | ✅ Distillation |

## Key Findings

1. **Dimension Performance:** Distillation won 8/8 dimensions
2. **Overall Accuracy:** Distillation MAE is 57.3% better than instruction tuning
3. **Generalization:** Both models show good generalization
4. **Best Absolute Performance:** Knowledge Distillation achieved the lowest validation MAE (0.6711)

## Recommendations

### ✅ Use Knowledge Distillation for Production

**Reasons:**
- 57.3% better validation accuracy
- More efficient training (no prompt overhead)
- Lower inference cost (512 vs 1024 token limit)
- Direct score learning is more effective for this task

**When to consider Instruction Tuning:**
- If interpretability of reasoning is critical
- If you need the model to explain its scores
- For multi-task learning scenarios

## Visualizations

See generated plots:
- `mode_comparison_mae.png` - Overall MAE over epochs
- `mode_comparison_per_dimension.png` - Per-dimension performance
- `mode_comparison_improvement.png` - Improvement breakdown
