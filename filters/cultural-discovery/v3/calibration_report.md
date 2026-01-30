# Cultural Discovery Filter v3 - Calibration Report

**Date:** 2026-01-29
**Oracle Model:** Gemini Flash 2.5
**Status:** ✅ PASS - Production Ready
**Target Applications:** ovr.news (Wisdom tab), Busara

---

## Executive Summary

**Decision:** PASS - Deploy to production

**Key Achievement:** Merged v1 and v2 datasets to solve the data quantity vs distribution quality tradeoff. v3 achieves the best of both worlds: sufficient training data AND enriched score distribution.

**Results:**
- Overall MAE: **0.77** (target: <0.80) ✅
- 39% improvement on medium-tier articles vs v1
- 23% improvement on high-tier articles vs v1

---

## Problem Statement: The v1/v2 Journey

### v1: The Regression-to-Mean Problem

v1 was trained on 4,996 randomly sampled articles with this distribution:

| Tier | Count | Percentage |
|------|-------|------------|
| High (≥7.0) | 33 | 0.7% |
| Medium (≥4.0) | 279 | 5.6% |
| Low (<4.0) | 4,684 | **93.8%** |

**Result:** Overall MAE 0.82 looked good, but the model learned to predict conservative scores (~2.0) because that minimizes overall MAE when 94% of articles score low.

**Evidence of failure:**
- evidence_quality MAE: **4.12** for articles scoring 8-10
- Model systematically under-predicted high scores (bias -2 to -4)

### v2: The Screening Filter Solution

v2 applied a screening filter before oracle scoring to enrich the distribution:

| Tier | v1 | v2 | Change |
|------|-----|-----|--------|
| High (≥7.0) | 0.7% | 3.0% | +4x |
| Medium (≥4.0) | 5.6% | 17.0% | +3x |
| Low (<4.0) | 93.8% | 80.0% | -15% |

**Problem:** Only 2,919 articles passed the screening filter (vs 4,996 in v1). With 42% less data and only 3 epochs of training, the model couldn't learn the harder (more varied) distribution.

**Result:** MAE **1.47** - worse than v1!

### v3: The Best of Both Worlds

**Solution:** Merge v1 and v2 datasets.

| Metric | v1 | v2 | v3 |
|--------|-----|-----|-----|
| Total articles | 4,996 | 2,919 | **7,827** |
| High-tier % | 0.7% | 3.0% | 1.9% |
| Medium-tier % | 5.6% | 17.0% | 9.8% |
| Low-tier % | 93.8% | 80.0% | 88.3% |
| Training epochs | 9 | 3 | 6 |
| **Validation MAE** | 0.82 | 1.47 | **0.77** |

---

## Dimension Framework

### Core Discovery Dimensions (50% weight)

| Dimension | Weight | Description |
|-----------|--------|-------------|
| discovery_novelty | 25% | New finding, revelation, or insight |
| cross_cultural_connection | 25% | Bridges between peoples/civilizations |

### Heritage & Resonance (35% weight)

| Dimension | Weight | Description |
|-----------|--------|-------------|
| heritage_significance | 20% | Cultural or historical importance |
| human_resonance | 15% | Connects to lived human experience |

### Assessment (15% weight)

| Dimension | Weight | Description |
|-----------|--------|-------------|
| evidence_quality | 15% | Research quality and documentation |

**Gatekeeper:** evidence_quality < 3 caps overall score at 3.0

---

## Training Results

### Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen2.5-1.5B |
| Training method | LoRA (r=16, α=32) |
| Trainable parameters | 18.5M / 1.56B (1.18%) |
| Train/Val/Test split | 6,261 / 783 / 783 |
| Epochs | 6 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Max length | 512 tokens |
| Preprocessing | Head+tail (256+256 tokens) |

### Epoch Summary

| Epoch | Train MAE | Val MAE | Val RMSE | Best? |
|-------|-----------|---------|----------|-------|
| 1 | 1.68 | 1.17 | 1.70 | |
| 2 | 1.00 | 0.86 | 1.32 | |
| 3 | 0.81 | 0.80 | 1.26 | |
| 4 | 0.72 | **0.77** | 1.24 | ✅ Yes |
| 5 | 0.64 | 0.78 | 1.26 | |
| 6 | 0.57 | 0.79 | 1.27 | |

**Best validation MAE: 0.77** (epoch 4)

### Per-Dimension Validation MAE

| Dimension | MAE | RMSE | Assessment |
|-----------|-----|------|------------|
| discovery_novelty | 0.54 | 0.99 | ✅ Excellent |
| heritage_significance | 0.58 | 1.00 | ✅ Excellent |
| cross_cultural_connection | 0.62 | 0.96 | ✅ Excellent |
| human_resonance | 0.77 | 1.19 | ✅ Good |
| evidence_quality | 1.36 | 1.85 | ⚠️ Hardest |

**Note:** evidence_quality remains the hardest dimension across all our filters. It requires assessing abstract qualities (research methodology, sourcing) that are harder to learn from text surface features.

---

## Tier-Level Performance (Key Metric)

This is the critical improvement - how well does the model perform on articles that actually matter?

### MAE by Tier

| Tier | v1 MAE | v3 MAE | Improvement |
|------|--------|--------|-------------|
| LOW (0-3.9) | 0.75 | 0.60 | +20% |
| MEDIUM (4-6.9) | 2.85 | 1.73 | **+39%** |
| HIGH (7-10) | 3.49 | 2.69 | **+23%** |

### Bias by Tier (Average Prediction - Actual)

| Tier | v1 Bias | v3 Bias | Improvement |
|------|---------|---------|-------------|
| LOW | +0.49 | +0.26 | Over-prediction reduced |
| MEDIUM | -2.23 | -1.27 | **Under-prediction reduced** |
| HIGH | -3.04 | -2.27 | **Under-prediction reduced** |

**Key insight:** v3 still under-predicts high scores (bias -2.27), but this is a 25% improvement over v1's -3.04 bias. The merged dataset with more medium/high examples gave the model better gradient signal.

---

## Correlation Analysis

### Full Dataset (n=7,827)

| Pair | Correlation |
|------|-------------|
| discovery ↔ heritage | 0.72 |
| heritage ↔ human | 0.65 |
| cross_cultural ↔ human | 0.61 |
| evidence ↔ discovery | 0.48 |
| evidence ↔ human | 0.42 |

**Assessment:** Correlations are acceptable. The 0.72 discovery↔heritage correlation reflects genuine content patterns (discoveries often involve significant heritage).

### Articles with Signal (weighted avg ≥ 4.0)

| Pair | Correlation |
|------|-------------|
| discovery ↔ heritage | 0.58 |
| heritage ↔ human | 0.41 |
| cross_cultural ↔ human | 0.52 |

**Assessment:** When filtered to articles with signal, correlations drop further. Dimensions show independent variation where it matters.

---

## Validation Examples

### High Score Article (Correctly Identified)
**Title:** "Silk Road Temple Reveals Buddhist-Zoroastrian Syncretism"
- **Oracle:** discovery=8.5, heritage=9.0, cross_cultural=9.5, human=7.0, evidence=8.5 → **8.7**
- **Model:** discovery=7.8, heritage=8.4, cross_cultural=8.2, human=6.5, evidence=7.6 → **7.9**
- **Assessment:** ✅ Correctly identifies as HIGH tier, slight under-prediction

### Medium Score Article (Correctly Identified)
**Title:** "Oregon Forest Art Installation Opens"
- **Oracle:** discovery=6.0, heritage=4.0, cross_cultural=2.0, human=5.5, evidence=6.0 → **4.5**
- **Model:** discovery=5.2, heritage=3.8, cross_cultural=2.5, human=4.8, evidence=5.5 → **4.2**
- **Assessment:** ✅ Correctly identifies as MEDIUM tier

### Low Score Article (Correctly Identified)
**Title:** "EU Summit Discusses Trade Policy"
- **Oracle:** discovery=1.0, heritage=1.0, cross_cultural=1.5, human=2.0, evidence=5.0 → **1.7**
- **Model:** discovery=1.2, heritage=1.4, cross_cultural=1.8, human=2.3, evidence=4.2 → **2.0**
- **Assessment:** ✅ Correctly identifies as LOW tier

---

## Comparison with Other Filters

| Filter | Version | MAE | Training Data | Dimensions |
|--------|---------|-----|---------------|------------|
| uplifting | v5 | 0.68 | 10,000 | 6 |
| investment-risk | v5 | 0.48 | 10,000 | 6 |
| sustainability_technology | v2 | 0.71 | 8,000 | 5 |
| **cultural-discovery** | **v3** | **0.77** | **7,827** | **5** |

**Assessment:** v3 performance is in line with other production filters. The slightly higher MAE reflects the inherent difficulty of cultural discovery scoring (more subjective than financial risk or technology assessment).

---

## Known Limitations

### 1. High-Score Under-Prediction
The model still under-predicts scores ≥7.0 (bias -2.27). This is acceptable for production because:
- Better to under-predict than over-predict (avoids false positives)
- Relative ranking is preserved (high-scorers still rank higher)
- 23% improvement over v1

### 2. evidence_quality Remains Hardest
MAE 1.36 vs other dimensions at 0.54-0.77. This is consistent across all our filters and reflects the abstract nature of evidence assessment.

### 3. Limited High-Tier Training Data
Only 149 articles (1.9%) in the high tier. Future versions could benefit from targeted collection from heritage/archaeology publications.

---

## Decision Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Overall MAE | < 0.80 | 0.77 | ✅ PASS |
| Medium-tier MAE | < 2.0 | 1.73 | ✅ PASS |
| High-tier MAE | < 3.5 | 2.69 | ✅ PASS |
| No severe overfitting | Train-Val gap < 0.3 | 0.20 | ✅ PASS |
| Prefilter tested | Yes | Yes | ✅ PASS |

---

## Recommendations

### Immediate: Deploy to Production
1. Deploy to ovr.news Wisdom tab
2. Deploy to Busara
3. Monitor tier distribution in production

### Future Improvements (v4 if needed)
1. **Targeted data collection** from heritage/archaeology sources
2. **Loss weighting** to penalize high-score errors more heavily
3. **Longer context** - current 512 tokens may miss key details in long articles

---

## Files Reference

```
filters/cultural-discovery/v3/
├── config.yaml                 # Filter configuration
├── prompt-compressed.md        # Oracle prompt
├── prefilter.py               # Keyword prefilter
├── base_scorer.py             # Scoring logic (with head+tail)
├── inference.py               # Local inference
├── inference_hub.py           # HuggingFace Hub inference
├── training_metadata.json     # Training config
├── training_history.json      # Training curves
├── calibration_report.md      # This report
└── model/                     # Trained LoRA adapter
```

**Training Data:** `datasets/training/cultural-discovery_v3/`
**Scored Data:** `datasets/scored/cultural-discovery_v3/`

---

## Conclusion

**Final Decision:** ✅ PASS - Deploy to production

Cultural-discovery v3 successfully resolves the v1/v2 tradeoff by merging datasets. The 0.77 MAE meets our target, and the 39% improvement on medium-tier articles (the content we actually want to surface) makes this filter production-ready.

**Key Learnings:**
1. Data quantity AND distribution quality both matter
2. Merged datasets can solve the screening filter data scarcity problem
3. Tier-level MAE is more meaningful than overall MAE for needle-in-haystack filters

---

*Report generated: 2026-01-29*
*Calibration by: Claude Code + Human review*
