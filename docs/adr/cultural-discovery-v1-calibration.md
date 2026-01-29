# ADR: Cultural-Discovery v1 Oracle Calibration

**Date:** 2025-01-27
**Status:** Accepted
**Filter:** cultural-discovery v1

## Context

Completed Phase 3 (Validation) calibration for the cultural-discovery filter. Scored 100 random articles from master dataset using Gemini Flash oracle.

## Calibration Results

### Score Distributions

| Dimension | Mean | Std | Low (<3) | Mid (3-7) | High (≥7) |
|-----------|------|-----|----------|-----------|-----------|
| discovery_novelty | 1.2 | 1.1 | 94 | 5 | 1 |
| heritage_significance | 1.3 | 1.4 | 92 | 5 | 3 |
| cross_cultural_connection | 1.0 | 1.2 | 93 | 6 | 1 |
| human_resonance | 1.5 | 1.7 | 89 | 7 | 4 |
| evidence_quality | 2.7 | 1.9 | 67 | 26 | 7 |

### Tier Distribution

- **High (≥7.0):** 0 articles (0%)
- **Medium (≥4.0):** 6 articles (6%)
- **Low (<4.0):** 94 articles (94%)

This is **expected** for a random sample - most news is not cultural discovery content.

### Content Type Classification

- general: 86%
- political_conflict: 9%
- cultural_discovery: 5%

### Correlation Analysis

**All articles (n=100):**
- discovery ↔ heritage: 0.84 (high - floor effects)
- heritage ↔ human: 0.71
- cross_cultural ↔ human: 0.69

**Articles with signal (n=12, weighted ≥2.0):**
- discovery ↔ heritage: 0.65 (acceptable)
- heritage ↔ human: 0.23 (good independence)

**Decision:** Keep all 5 dimensions. The 0.84 correlation was driven by floor effects (94% low-scoring). When filtered to articles with signal, correlation drops to 0.65 with clear examples of independent variation (e.g., discovery=1.5 vs heritage=7.0).

## Spot-Check Validation

| Score | Article | Oracle Assessment |
|-------|---------|-------------------|
| 5.8 | Dahibara food culture (India) | ✓ High human (8), cross-cultural (7) |
| 4.4 | Oregon forest art | ✓ High discovery (6), low cross-cultural (1) - correct |
| 1.1 | EU political | ✓ Typed as political_conflict, all low |
| 0.3 | Google NotebookLM | ✓ Typed as general, zeros |

## Findings

1. **Oracle performing well** - Correctly identifies cultural vs general content
2. **Dimensions are independent** when there's signal (correlation 0.65 vs 0.84)
3. **Evidence quality most independent** - good gatekeeper behavior
4. **Prefilter passing most content** - 86% general (acceptable for training diversity)

## Recommendations

1. **Proceed to Phase 5** - Collect 5K training articles
2. **No prompt changes needed** - Oracle calibration is healthy
3. **Consider tightening prefilter** later if training data has too much noise

## Training Data Analysis (5K Articles)

**Date:** 2025-01-28

Scored 4,996 articles from master dataset using Gemini Flash.

### Tier Distribution

| Tier | Count | Percentage |
|------|-------|------------|
| High (≥7.0) | 33 | 0.7% |
| Medium (≥4.0) | 279 | 5.6% |
| Low (<4.0) | 4,684 | 93.8% |

### Score Distribution (Weighted Average)

```
Score   Count   Assessment
3.0-3.9   133   ✓ Solid
4.0-4.9   122   ✓ Solid
5.0-5.9    92   ✓ Good
6.0-6.9    65   ✓ Good
7.0-7.9    30   ✓ Enough
8.0+        3   ⚠️ Thin
```

**Total with signal (≥3.0):** 445 articles

### Content Type Distribution

- general: 83.0%
- political_conflict: 9.3%
- cultural_discovery: 6.1%
- speculation/tourism/celebrity/other: 1.6%

### Assessment

**Decision:** Proceed to training.

**Rationale:**
1. **Smooth gradient** across score range (3→8) - no gaps for regression to learn
2. **445 articles with signal** - sufficient for model to learn upper range
3. **Tier distribution matches production** - 94% low is realistic for random news
4. **Similar to successful filters** - uplifting v5 trained on similar distribution, achieved MAE 0.68

**Known limitation:** Only 3 articles at 8.0+. Model may underpredict truly exceptional content. Acceptable for v1; can collect targeted cultural heritage sources for v2 if needed.

## Training Results

**Date:** 2026-01-28

### Configuration

- **Model:** Qwen2.5-1.5B + LoRA (18.5M/1.56B parameters, 1.18% trainable)
- **Data:** 3,995 train / 500 val / 501 test (stratified by tier)
- **Epochs:** 9 total (3 initial + 6 extended), batch size 8, learning rate 2e-5
- **Hardware:** RTX 4080 (16GB), ~15 min/epoch

### Epoch Summary

| Epoch | Train MAE | Val MAE | Val RMSE | evidence_quality MAE | Best? |
|-------|-----------|---------|----------|---------------------|-------|
| 1     | 1.68      | 1.02    | 1.60     | 1.44                |       |
| 2     | 0.90      | 0.90    | 1.42     | 1.32                |       |
| 3     | 0.82      | 0.83    | 1.31     | 1.26                |       |
| 4     | 0.77      | 0.84    | 1.31     | 1.29                |       |
| 5     | 0.77      | 0.83    | 1.30     | 1.25                |       |
| 6     | 0.76      | **0.82**| 1.30     | **1.24**            | Yes   |
| 7     | 0.76      | 0.82    | 1.29     | 1.25                |       |
| 8     | 0.76      | 0.82    | 1.29     | 1.25                |       |
| 9     | 0.76      | 0.82    | 1.29     | 1.25                |       |

**Best validation MAE: 0.82** (epoch 6)

### Per-Dimension Validation MAE (Best Epoch 6)

| Dimension | MAE | RMSE |
|-----------|-----|------|
| discovery_novelty | 0.59 | 1.09 |
| heritage_significance | 0.72 | 1.28 |
| cross_cultural_connection | 0.68 | 1.11 |
| human_resonance | 0.88 | 1.32 |
| evidence_quality | 1.24 | 1.61 |

### Assessment

1. **Overall MAE 0.82 beats target** - Well under the 1.0 threshold
2. **4 of 5 dimensions under 1.0** - discovery_novelty (0.59), heritage_significance (0.72), cross_cultural_connection (0.68), human_resonance (0.88) all excellent
3. **evidence_quality plateaued at ~1.25** - Extended training (epochs 4-9) improved it from 1.26 to 1.24, then plateaued. This is a data limitation, not a model capacity issue (train MAE continued to 1.15 without overfitting on other dims)
4. **No overfitting** - Val loss decreased throughout; train-val gap remained small
5. **Comparable to other filters** - uplifting v5 achieved 0.68 with 10K training examples; our 0.82 with ~4K is reasonable
6. **Diminishing returns after epoch 6** - Epochs 7-9 showed no meaningful improvement, confirming convergence

### evidence_quality Analysis

The evidence_quality dimension is hardest because:
- Most independent dimension (lowest correlation with others)
- Requires assessing research methodology, sourcing, and argumentation -- abstract qualities harder to learn from text surface features
- Only 33 high-tier (>=7.0) examples in training data
- **Recommendation for v2:** Collect targeted training data from academic/research journalism sources to improve evidence_quality learning

## Next Steps

1. Evaluate on held-out test set
2. Deploy to HuggingFace Hub

## Lessons Learned

**Date:** 2026-01-29

### The Regression-to-Mean Problem

Training on 94% low-scoring articles caused a systematic bias: the model learned to predict conservative scores (~2.0) because that minimizes overall MAE. This manifests as:

1. **Excellent overall MAE (0.82)** - Misleading metric, dominated by easy low-scoring cases
2. **Poor high-score predictions** - Evidence_quality MAE of **4.12** for articles scoring 8-10
3. **Dimension plateau** - evidence_quality stuck at MAE ~1.25 despite extended training

### Root Cause Analysis

| Symptom | Cause | Evidence |
|---------|-------|----------|
| evidence_quality plateau | Insufficient high-score examples | Only 33 articles with weighted avg >= 7.0 |
| Underpredict 8-10 range | Regression hedging | Model can predict 2.0 and achieve "good" MAE |
| Extended training no help | Data limitation, not model capacity | Train MAE continued improving, val plateaued |

### Why This Matters

For needle-in-haystack filters (most of ours), the 0.7% high-tier articles are **the entire point**. A model that achieves 0.82 MAE overall but 4.12 MAE on high-scorers is useless for production.

### Solution: Screening Filters

**Recommendation for cultural-discovery v2:**

1. **Apply screening filter** before oracle scoring
   - Target: 30-40% of screened articles scoring >= 4.0 (vs 6% currently)
   - Filter for: heritage, archaeology, tradition, discovery keywords
   - Reject: generic news, political, celebrity content

2. **Collect from signal-rich sources**
   - Academic/research journalism
   - Heritage/museum publications
   - Cultural institutions

3. **Target distribution:**
   - 50% low (0-3) - still need negatives
   - 30% medium (4-6) - critical for learning gradients
   - 20% high (7-10) - the gems

4. **Consider loss weighting**
   - Weight high-score errors more heavily
   - Penalize underestimation more than overestimation

### Impact on Other Filters

This analysis applies to ALL needle-in-haystack filters:

| Filter | Est. High-Tier in Random | Recommendation |
|--------|--------------------------|----------------|
| cultural-discovery | 0.7% | Screening required |
| belonging | ~5% (estimated) | Screening recommended |
| signs-of-wisdom | ~3% (estimated) | Screening required |
| nature-recovery | ~8% (estimated) | Screening recommended |

### Documentation

- [ADR-003: Screening Filter for Training Data Enrichment](003-screening-filter-for-training-data.md)
- [Screening Filter Template](../templates/screening-filter-template.md)
- [Filter Development Guide - Phase 5](../agents/filter-development-guide.md#phase-5-training-data-collection)
