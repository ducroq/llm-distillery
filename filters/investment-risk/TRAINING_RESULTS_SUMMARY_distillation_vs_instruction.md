# Investment-Risk Filter - Training Results Summary

**Date:** 2025-11-16
**Objective:** Compare Knowledge Distillation vs Instruction Tuning for investment-risk scoring
**Result:** ✅ Knowledge Distillation wins decisively

---

## Quick Navigation

### ⭐ Production Model (RECOMMENDED)
📁 `filters/investment-risk/v2_distillation/`
- **Val MAE:** 0.6711 (52.6% better than instruction tuning)
- **Status:** Production Ready
- **See:** `v2_distillation/README.md`

### 📊 Comparison Analysis
📁 `filters/investment-risk/v2_distillation/training_reports/`
- **Main Report:** `comparison_report.md`
- **Visualizations:** `mode_comparison_*.png`

### ⚠️ Experimental Model (NOT RECOMMENDED)
📁 `filters/investment-risk/v2_instruction/`
- **Val MAE:** 1.4157 (underperformed significantly)
- **Status:** Archived for comparison only
- **See:** `v2_instruction/README.md`

---

## Results Summary

### Overall Performance

| Metric | Knowledge Distillation | Instruction Tuning | Winner |
|--------|----------------------|-------------------|--------|
| **Val MAE** | **0.6711** | 1.4157 | ✅ Distillation (52.6% better) |
| **Val RMSE** | **0.9303** | 1.7693 | ✅ Distillation (47.4% better) |
| **Max Tokens** | **512** | 1024 | ✅ Distillation (2x more efficient) |
| **Train/Val Gap** | +0.0463 | -0.0377 | ✅ Instruction (better generalization) |
| **Dimensions Won** | **8/8** | 0/8 | ✅ Distillation (clean sweep) |

### Per-Dimension Results

| Dimension | Distillation MAE | Instruction MAE | Improvement |
|-----------|-----------------|----------------|-------------|
| Macro Risk Severity | **0.6955** | 1.6431 | +57.7% |
| Credit Market Stress | **0.5597** | 1.0053 | +44.3% |
| Market Sentiment Extremes | **0.5906** | 1.1219 | +47.4% |
| Valuation Risk | **0.6384** | 1.2658 | +49.6% |
| Policy Regulatory Risk | **0.7363** | 1.6673 | +55.8% |
| Systemic Risk | **0.6366** | 1.3044 | +51.2% |
| Evidence Quality | **0.8622** | 1.7990 | +52.1% |
| Actionability | **0.6493** | 1.5191 | +57.3% |

---

## Directory Structure

```
filters/investment-risk/
├── v2/                          # Filter specification (oracle-based)
│   ├── config.yaml
│   ├── prefilter.py
│   ├── prompt-compressed.md
│   ├── ground_truth_quality_report.md
│   ├── package_validation.md
│   └── release_report.md
│
├── v2_distillation/             # ⭐ PRODUCTION MODEL
│   ├── README.md                # Start here
│   ├── model/                   # LoRA adapter weights
│   ├── training_history.json
│   ├── training_metadata.json
│   └── training_reports/
│       ├── investment-risk_v2_distillation_training_report.docx
│       ├── comparison_report.md              # Key comparison analysis
│       ├── overall_metrics.png
│       ├── per_dimension_mae.png
│       ├── loss_curves.png
│       ├── mode_comparison_mae.png           # Side-by-side comparison
│       ├── mode_comparison_per_dimension.png
│       ├── mode_comparison_improvement.png
│       └── training_summary.txt
│
└── v2_instruction/              # ⚠️ EXPERIMENTAL (not recommended)
    ├── README.md                # Why this didn't work
    ├── model/
    ├── training_history.json
    ├── training_metadata.json
    └── training_reports/
        ├── investment-risk_v2_instruction_training_report.docx
        ├── overall_metrics.png
        ├── per_dimension_mae.png
        ├── loss_curves.png
        └── training_summary.txt
```

---

## Key Takeaways

### ✅ What Worked

1. **Knowledge Distillation is Superior for Regression**
   - Direct score learning beats reasoning-based learning
   - 52.6% better validation MAE
   - Won all 8 dimensions decisively

2. **Simpler is Better**
   - 512 tokens sufficient (vs 1024)
   - No prompt overhead needed
   - Faster inference, lower cost

3. **Training Infrastructure Works**
   - Both models trained successfully
   - Clean comparison methodology
   - Reproducible results

### ❌ What Didn't Work

1. **Instruction Tuning for Regression**
   - Adding reasoning didn't help score prediction
   - More complex doesn't mean better
   - 1024 token context was overkill

2. **Dual-Task Learning**
   - Learning reasoning + scores simultaneously hurt both
   - Split focus reduced score accuracy
   - Not worth the interpretability gain

### 📚 Lessons Learned

1. **Default to Distillation** - For future filters with regression scoring
2. **Test Both Modes** - Comparison validates the choice
3. **Document Failures** - Negative results are valuable
4. **Package Everything** - All artifacts stay with the filter

---

## Recommendation

### For Production: Use v2_distillation

**Deploy:** `filters/investment-risk/v2_distillation/model/`

**Reasons:**
- ✅ Meets accuracy target (0.67 MAE vs <1.0)
- ✅ Most efficient (512 tokens)
- ✅ Fastest inference
- ✅ Lowest cost
- ✅ Best performance across all dimensions

**Next Steps:**
1. Deploy model to production pipeline
2. Monitor live performance
3. Collect edge cases for v3
4. Consider larger model (3B/7B) if more accuracy needed

### For Research: Keep v2_instruction

**Archive:** `filters/investment-risk/v2_instruction/`

**Value:**
- Documents what doesn't work
- Validates training strategy
- Reference for future experiments
- Comparison baseline

---

## Tools Used

All training analysis tools are now updated to output to filter packages by default:

1. **`training/plot_learning_curves.py`** - Generate training visualizations
2. **`training/generate_training_report.py`** - Create Word reports
3. **`training/compare_training_modes.py`** - Compare distillation vs instruction tuning

These tools automatically detect filter directories and output results to `training_reports/` subdirectories.

---

## Related Reports

- **Oracle Calibration:** `v2/ground_truth_quality_report.md` (5,150 articles scored)
- **Package Validation:** `v2/package_validation.md` (90 article validation)
- **Release Report:** `v2/release_report.md` (production readiness)
- **Training Comparison:** `v2_distillation/training_reports/comparison_report.md` ⭐

---

**For Questions:** Start with `v2_distillation/README.md` or the comparison report.
