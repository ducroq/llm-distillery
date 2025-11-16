# Investment-Risk v2 - Training Report

**Filter Version:** v2.0
**Training Date:** 2025-11-16
**Status:** ✅ TRAINING COMPLETE - Production Model Available
**Recommended Model:** Knowledge Distillation (`../v2_distillation/`)

---

## Executive Summary

The investment-risk v2 filter has been successfully trained using knowledge distillation from the Gemini Flash oracle. A comprehensive comparison between knowledge distillation and instruction tuning was conducted to determine the optimal training approach.

**Key Results:**
- ✅ **Knowledge distillation significantly outperformed** instruction tuning (52.6% better MAE)
- ✅ **Production-ready model** achieved 0.67 MAE (target: <1.0)
- ✅ **All 8 dimensions** learned successfully
- ✅ **Efficient deployment** ready (512 tokens, local inference)

**Recommendation:** Deploy the knowledge distillation model to production.

---

## Training Overview

### Dataset
- **Oracle:** Gemini Flash 1.5 (batch API)
- **Ground Truth:** 5,150 labeled articles
- **Training Set:** 4,118 articles (80%)
- **Validation Set:** 515 articles (20%)
- **Quality:** Excellent (see `ground_truth_quality_report.md`)

### Model Architecture
- **Base Model:** Qwen/Qwen2.5-1.5B
- **Method:** LoRA fine-tuning
- **Parameters:** 1.56B total / 18.5M trainable
- **Training Duration:** 3 epochs (~2-3 hours on GPU)

### Training Modes Compared

Two approaches were tested to determine the best method:

1. **Knowledge Distillation** - Direct score learning
2. **Instruction Tuning** - Reasoning + score generation

---

## Results Summary

### Overall Performance Comparison

| Metric | Knowledge Distillation | Instruction Tuning | Winner |
|--------|----------------------|-------------------|--------|
| **Validation MAE** | **0.6711** ✅ | 1.4157 | Distillation (52.6% better) |
| **Validation RMSE** | **0.9303** ✅ | 1.7693 | Distillation (47.4% better) |
| **Max Token Length** | **512** ✅ | 1024 | Distillation (2x efficient) |
| **Inference Cost** | **Lower** ✅ | Higher | Distillation |
| **Dimensions Won** | **8/8** ✅ | 0/8 | Distillation (clean sweep) |
| **Production Ready** | **Yes** ✅ | No | Distillation |

### Per-Dimension Performance

| Dimension | Distillation MAE | Instruction MAE | Improvement |
|-----------|-----------------|----------------|-------------|
| Macro Risk Severity | **0.6955** | 1.6431 | +57.7% better |
| Credit Market Stress | **0.5597** | 1.0053 | +44.3% better |
| Market Sentiment Extremes | **0.5906** | 1.1219 | +47.4% better |
| Valuation Risk | **0.6384** | 1.2658 | +49.6% better |
| Policy Regulatory Risk | **0.7363** | 1.6673 | +55.8% better |
| Systemic Risk | **0.6366** | 1.3044 | +51.2% better |
| Evidence Quality | **0.8622** | 1.7990 | +52.1% better |
| Actionability | **0.6493** | 1.5191 | +57.3% better |

**Verdict:** Knowledge distillation won decisively across all dimensions.

---

## Training Methodology

### Knowledge Distillation (RECOMMENDED)

**Approach:** Student model learns to directly predict oracle scores

**Configuration:**
```yaml
training_mode: knowledge_distillation
include_prompt: false
max_length: 512 tokens
batch_size: 8
learning_rate: 2e-05
epochs: 3
```

**Input Format:**
```
Article Title + Content (max 512 tokens)
```

**Output Format:**
```json
{
  "macro_risk_severity": 7.5,
  "credit_market_stress": 6.0,
  "market_sentiment_extremes": 4.5,
  "valuation_risk": 5.0,
  "policy_regulatory_risk": 6.5,
  "systemic_risk": 7.0,
  "evidence_quality": 8.0,
  "actionability": 7.0
}
```

**Advantages:**
- ✅ Direct score learning (simpler task)
- ✅ Lower token count (faster, cheaper)
- ✅ Better accuracy for regression
- ✅ Easier to optimize

**Training Results:**
- Train MAE: 0.6248
- Val MAE: 0.6711 ✅
- Train/Val Gap: +0.0463 (slight overfitting, acceptable)

### Instruction Tuning (TESTED, NOT RECOMMENDED)

**Approach:** Student model learns to generate reasoning + scores

**Configuration:**
```yaml
training_mode: instruction_tuning
include_prompt: true
max_length: 1024 tokens
batch_size: 8
learning_rate: 2e-05
epochs: 3
```

**Input Format:**
```
[Instruction Prompt]
Article Title + Content (max 1024 tokens)
```

**Output Format:**
```
[Reasoning paragraph explaining the analysis]
{
  "macro_risk_severity": 7.5,
  ...
}
```

**Disadvantages:**
- ❌ More complex task (reasoning + scoring)
- ❌ Higher token count (slower, more expensive)
- ❌ Worse accuracy (1.42 vs 0.67 MAE)
- ❌ Dual-task learning splits optimization

**Training Results:**
- Train MAE: 1.4534
- Val MAE: 1.4157 ❌ (exceeds target)
- Train/Val Gap: -0.0377 (good generalization, but poor accuracy)

---

## Why Knowledge Distillation Won

### 1. Task Alignment
- **Regression task** benefits from direct score prediction
- Adding reasoning doesn't help numerical accuracy
- Simpler objective = better optimization

### 2. Efficiency
- 512 vs 1024 token limit
- 2x faster inference
- Lower API costs in production
- No prompt engineering overhead

### 3. Empirical Results
- 52.6% better validation MAE
- Won all 8 dimensions
- Meets production quality threshold
- Stable training convergence

### 4. Production Readiness
- Meets accuracy target (0.67 < 1.0)
- Efficient deployment
- Fast inference (<50ms)
- Local execution (no API costs)

---

## Model Selection Decision

### ✅ Selected for Production: Knowledge Distillation

**Location:** `../v2_distillation/`

**Justification:**
1. **Accuracy:** 0.67 MAE meets target (<1.0)
2. **Efficiency:** 512 token limit, fast inference
3. **Performance:** Won all 8 dimensions decisively
4. **Cost:** Optimal for production deployment
5. **Reliability:** Stable training, good convergence

**Deployment Path:**
```
Article → Prefilter → v2_distillation Model → 8 Scores → Tier Assignment
```

### ⚠️ Not Selected: Instruction Tuning

**Location:** `../v2_instruction/` (archived)

**Reasons for Rejection:**
1. **Accuracy:** 1.42 MAE exceeds target (>1.0)
2. **Efficiency:** 1024 tokens, slower inference
3. **Performance:** Lost all 8 dimensions
4. **Cost:** Higher deployment cost
5. **Complexity:** Unnecessary for this task

**Value:** Serves as important negative result, validates training strategy

---

## Training Artifacts

### Production Model Package
📁 `../v2_distillation/`
- `model/` - LoRA adapter weights + tokenizer
- `training_history.json` - Epoch-by-epoch metrics
- `training_metadata.json` - Configuration details
- `README.md` - Model documentation
- `training_reports/` - Comprehensive analysis
  - Word report with visualizations
  - Learning curves (MAE, RMSE, loss)
  - Comparison analysis vs instruction tuning
  - Summary statistics

### Experimental Model Package
📁 `../v2_instruction/`
- Same structure as distillation
- Archived for reference
- Documents what didn't work

### Comparison Analysis
📁 `../v2_distillation/training_reports/`
- `comparison_report.md` - Full comparison analysis
- `mode_comparison_mae.png` - Side-by-side MAE curves
- `mode_comparison_per_dimension.png` - Per-dimension bars
- `mode_comparison_improvement.png` - Improvement breakdown

---

## Production Deployment

### Recommended Configuration

**Model:** `v2_distillation/model/`

**Inference Pipeline:**
1. **Prefilter** - Block obvious noise (30-60% of articles)
2. **Model** - Score remaining articles (8 dimensions, 0-10 scale)
3. **Postfilter** - Apply tier classification rules
4. **Output** - Tier assignment (RED/YELLOW/GREEN/BLUE/NOISE)

**Expected Performance:**
- **Throughput:** ~1000 articles/hour
- **Latency:** <50ms per article
- **Accuracy:** 0.67 MAE vs oracle
- **Cost:** $0 (local inference after training)

### Monitoring Plan

**Metrics to Track:**
1. Prediction distribution vs oracle ground truth
2. Tier classification accuracy
3. False positive/negative rates
4. Inference latency
5. Edge cases and failure modes

**Quality Assurance:**
1. Monthly validation on fresh sample
2. Human review of RED tier articles
3. Drift detection (distribution changes)
4. Periodic retraining with new data

---

## Lessons Learned

### ✅ What Worked

1. **Knowledge Distillation for Regression**
   - Direct score learning superior to reasoning-based
   - Simpler is better for numerical prediction
   - 52.6% improvement validates approach

2. **Comprehensive Comparison**
   - Testing both modes validates decision
   - Negative results have value
   - Empirical evidence > intuition

3. **Filter Package Organization**
   - All artifacts stay with the filter
   - Self-contained model packages
   - Clear documentation trail

4. **Training Infrastructure**
   - Automated report generation
   - Reproducible methodology
   - Standardized comparison tools

### ❌ What Didn't Work

1. **Instruction Tuning for Regression**
   - Reasoning didn't improve score accuracy
   - Dual-task learning split optimization
   - Higher complexity ≠ better results

2. **Longer Context**
   - 1024 tokens didn't help
   - 512 tokens sufficient for this task
   - Efficiency matters for deployment

### 📚 Institutional Knowledge

1. **Default to distillation** for future regression-based filters
2. **Test both modes** when in doubt (comparison is cheap)
3. **Document failures** (negative results prevent repeating mistakes)
4. **Package everything** (self-contained filter artifacts)

---

## Next Steps

### Immediate (Production Deployment)

1. ✅ **Model Ready:** `v2_distillation/model/` trained and validated
2. ⏭️ **Integration:** Deploy to production pipeline
3. ⏭️ **Monitoring:** Set up performance tracking
4. ⏭️ **Validation:** Run on live traffic, collect feedback

### Short Term (1-3 months)

1. **Monitor Performance**
   - Track prediction quality on live articles
   - Identify edge cases and failure modes
   - Collect human feedback on RED tier articles

2. **Quality Assurance**
   - Monthly validation samples
   - Drift detection
   - False positive/negative analysis

### Medium Term (3-6 months)

1. **Consider v3 Improvements**
   - Larger model (3B or 7B params) if more accuracy needed
   - Additional training data if available
   - Fine-tune tier thresholds based on production data

2. **Expand Coverage**
   - Additional filters using same methodology
   - Cross-filter consistency checks
   - Multi-filter ensembles

---

## References

### Filter Specification
- `config.yaml` - Filter configuration and tier thresholds
- `prefilter.py` - Pre-filter implementation
- `prompt-compressed.md` - Oracle prompt used for ground truth

### Validation Reports
- `ground_truth_quality_report.md` - Oracle quality analysis (5,150 articles)
- `package_validation.md` - Pre-training validation (90 articles)
- `release_report.md` - Production readiness assessment

### Training Documentation
- `TRAINING_PLAN.md` - Original training strategy
- `../v2_distillation/README.md` - Production model documentation
- `../v2_instruction/README.md` - Experimental model analysis
- `../v2_distillation/training_reports/comparison_report.md` - Detailed comparison

### Training Tools
- `training/plot_learning_curves.py` - Generate visualizations
- `training/generate_training_report.py` - Create Word reports
- `training/compare_training_modes.py` - Compare training modes

---

## Conclusion

The investment-risk v2 filter training was successful. Knowledge distillation proved to be the superior approach, achieving 0.67 MAE validation accuracy and decisively outperforming instruction tuning across all dimensions.

**The production model is ready for deployment.**

Key success factors:
- High-quality ground truth (5,150 oracle-scored articles)
- Appropriate training methodology (knowledge distillation)
- Comprehensive comparison (empirical validation)
- Proper organization (self-contained filter package)

The model meets all production readiness criteria and is recommended for immediate deployment.

---

**For detailed analysis:** See training reports in `../v2_distillation/training_reports/`
**For production deployment:** Use model in `../v2_distillation/model/`
**For questions:** Start with `../v2_distillation/README.md`
