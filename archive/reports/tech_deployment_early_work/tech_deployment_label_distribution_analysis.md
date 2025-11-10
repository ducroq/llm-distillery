# Label Distribution & Coverage Analysis

**Filter**: sustainability_tech_deployment
**Total Labels**: 1,938
**Date**: 2025-11-08

---

## Executive Summary

⚠️ **ACCEPTABLE WITH CAVEATS**: The dataset has **severe class imbalance** but is still usable for training. The distribution reflects reality (most tech news is vaporware), but we should be aware of potential biases.

**Key Concerns**:
1. **82.5% vaporware** - Heavy skew toward low scores
2. **Only 1.4% deployed/proven** (28 samples) - Very few high-quality examples
3. **High-score scarcity** - Some dimensions have <50 examples of scores ≥8

**Recommendation**: **Proceed with training** but apply class weighting/sampling strategies to compensate for imbalance.

---

## Tier Distribution

| Tier | Count | Percentage | Training Concern |
|------|-------|------------|------------------|
| **Vaporware** (<4.0) | 1,598 | 82.5% | ⚠️ Dominant class - risk of model always predicting low |
| **Pilot** (4.0-5.9) | 187 | 9.6% | ✅ Adequate (>100 samples) |
| **Early Commercial** (6.0-7.9) | 125 | 6.4% | ✅ Adequate (>100 samples) |
| **Deployed/Proven** (≥8.0) | 28 | 1.4% | ❌ Severely underrepresented |

**Analysis**:
- **Expected distribution**: General tech news dataset naturally skews toward announcements, concepts, and early-stage projects
- **Oracle behavior**: Gemini Flash is appropriately conservative (not inflating scores)
- **Training risk**: Model may learn to predict "vaporware" for most inputs due to class dominance

---

## Overall Score Distribution

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean | 2.69 | Low average (vaporware range) |
| Median | 1.90 | Most articles score very low |
| Stdev | 1.93 | Moderate variance |
| Min | 1.00 | Floor score (complete vaporware) |
| Max | 9.55 | Strong deployment example exists |
| **Skewness** | **Right-skewed** | Long tail of high scorers |

**Score Range Breakdown**:
```
  0.0-3.9 (vaporware):        ████████████████████████████████ 82.5%
  4.0-5.9 (pilot):            ████ 9.6%
  6.0-7.9 (early commercial): ███ 6.4%
  8.0-10.0 (deployed/proven): █ 1.4%
```

---

## Dimension Score Distributions

### Summary Table

| Dimension | Mean | Median | Stdev | Min | Max | Range |
|-----------|------|--------|-------|-----|-----|-------|
| **technology_readiness** | 3.75 | 3.00 | 2.81 | 1.0 | 10.0 | ✅ Full |
| **deployment_maturity** | 3.36 | 3.00 | 2.60 | 1.0 | 10.0 | ✅ Full |
| **technology_performance** | 3.33 | 3.00 | 2.34 | 1.0 | 10.0 | ✅ Full |
| **supply_chain_maturity** | 2.97 | 1.00 | 2.57 | 1.0 | 10.0 | ✅ Full |
| **scale_of_deployment** | 2.93 | 1.00 | 2.60 | 1.0 | 10.0 | ✅ Full |
| **cost_trajectory** | 2.36 | 1.00 | 1.85 | 1.0 | 10.0 | ✅ Full |
| **proof_of_impact** | 2.28 | 1.00 | 1.65 | 1.0 | 9.0 | ⚠️ Max 9.0 |
| **market_penetration** | 2.25 | 1.00 | 2.00 | 1.0 | 10.0 | ✅ Full |

**Observations**:
- All dimensions have **full or near-full range** (1.0-10.0)
- **Highest variance**: `technology_readiness` (2.81 stdev) - good discrimination
- **Lowest variance**: `proof_of_impact` (1.65 stdev) - more conservative scoring
- **Gatekeeper dimensions**:
  - `deployment_maturity`: 3.36 mean (below 5.0 threshold - working as intended)
  - `proof_of_impact`: 2.28 mean (below 4.0 threshold - working as intended)

---

## Dimension Score Range Coverage

**Coverage by Range** (Low: 0-3, Mid: 4-7, High: 8-10):

| Dimension | Low (0-3) | Mid (4-7) | High (8-10) | Coverage Quality |
|-----------|-----------|-----------|-------------|------------------|
| **technology_readiness** | 1109 (57.2%) | 605 (31.2%) | 224 (11.6%) | ✅ EXCELLENT |
| **deployment_maturity** | 1183 (61.0%) | 637 (32.9%) | 118 (6.1%) | ✅ GOOD |
| **scale_of_deployment** | 1285 (66.3%) | 530 (27.3%) | 123 (6.3%) | ✅ GOOD |
| **supply_chain_maturity** | 1341 (69.2%) | 480 (24.8%) | 117 (6.0%) | ✅ GOOD |
| **technology_performance** | 1137 (58.7%) | 758 (39.1%) | 43 (2.2%) | ⚠️ LOW HIGH |
| **market_penetration** | 1586 (81.8%) | 298 (15.4%) | 54 (2.8%) | ⚠️ LOW HIGH |
| **cost_trajectory** | 1533 (79.1%) | 383 (19.8%) | 22 (1.1%) | ❌ VERY LOW HIGH |
| **proof_of_impact** | 1583 (81.7%) | 352 (18.2%) | 3 (0.2%) | ❌ ALMOST NO HIGH |

### Coverage Issues

**Critical Gaps** (high scores ≥8):
1. **proof_of_impact**: Only **3 examples** (0.2%) - Model will struggle to learn high scores
2. **cost_trajectory**: Only **22 examples** (1.1%) - Insufficient high-score training
3. **technology_performance**: Only **43 examples** (2.2%) - Below target (50+)
4. **market_penetration**: Only **54 examples** (2.8%) - Marginal

**Why This Matters**:
- **Regression task**: Model needs examples across full range to learn continuous scoring
- **High-score scarcity**: Model may never predict scores >8 for these dimensions
- **Gatekeeper impact**: `proof_of_impact` is a gatekeeper - weak high-score learning could cap all predictions

---

## Training Implications

### Strengths ✅

1. **Sufficient total samples**: 1,938 is adequate for fine-tuning 7B model
2. **Full dimensional range**: All dimensions have 1.0-10.0 examples (except proof_of_impact max 9.0)
3. **Mid-range coverage**: All dimensions have 298-758 mid-range (4-7) examples
4. **Appropriate conservatism**: Oracle isn't inflating scores (mean 2.69 reflects reality)

### Weaknesses ⚠️

1. **Extreme class imbalance**: 82.5% vaporware vs 1.4% deployed/proven
2. **High-score scarcity**: 4 dimensions have <50 high-score examples
3. **Gatekeeper weakness**: `proof_of_impact` only 3 examples ≥8 (critical dimension)
4. **Tier imbalance**: No "pilot_demonstration" tier (might be merged with vaporware)

### Risks 🚨

1. **Model bias toward low predictions**: May learn to always predict <4.0
2. **Poor high-score calibration**: May cap predictions at 6-7 even for truly deployed tech
3. **Gatekeeper underfitting**: May not learn when to allow high overall scores (proof_of_impact dependency)

---

## Mitigation Strategies

### Option 1: Proceed As-Is (Recommended for POC)

**Rationale**: This is a **proof-of-concept** for the distillation pipeline. We can tolerate some weaknesses to validate the approach.

**Actions**:
- Use **class weighting** during training (upweight rare classes)
- Apply **focal loss** to emphasize hard examples
- **Oversample** high-score examples (duplicate 8-10 scores 5-10x)
- Evaluate on **stratified holdout** (ensure all tiers represented)

**Expected Outcome**:
- Model will work well for majority class (vaporware detection)
- May struggle with edge cases (deployed tech discrimination)
- Good enough to demonstrate value and inform next iteration

### Option 2: Generate Additional High-Score Labels

**Process**:
1. Filter dataset for articles with keywords: "deployed", "commercial", "adoption", "market share", "revenue"
2. Run oracle on filtered set (targeting 500 additional labels)
3. Merge with existing 1,938 labels
4. Re-analyze distribution

**Cost**: ~$0.50 (500 × $0.001 Gemini Flash)

**Timeline**: ~30 minutes

**Expected Outcome**:
- Increase high-score examples from 28 to 100-150
- Better model calibration for deployed tech
- More balanced tier distribution

### Option 3: Synthetic Upsampling with Augmentation

**Process**:
1. Identify 28 high-scoring articles (≥8.0)
2. For each, generate 5 variations:
   - Paraphrase content (using Gemini Flash)
   - Re-label each variation
3. Add 140 synthetic high-score examples

**Cost**: ~$0.14 (140 × $0.001)

**Risks**:
- Synthetic data may not reflect real distribution
- Potential overfitting to paraphrased content

---

## Recommendation

### For POC (Current Phase): **Proceed with Option 1**

**Why**:
1. **Validate pipeline first**: Goal is to prove distillation works, not achieve perfect accuracy
2. **Cost-effective**: No additional oracle labeling required
3. **Class weighting works**: Proven technique for imbalanced datasets
4. **Learn from results**: Evaluation will reveal if we need Option 2

**Training Configuration**:
```python
# Apply class weights inversely proportional to frequency
class_weights = {
    'vaporware': 1.0,
    'pilot': 8.5,  # 82.5% / 9.6%
    'early_commercial': 12.9,  # 82.5% / 6.4%
    'deployed_proven': 58.9  # 82.5% / 1.4%
}

# Oversample high scores in training data
high_score_multiplier = 10  # Duplicate 28 deployed samples 10x → 280 examples
```

**Success Criteria**:
- **Overall accuracy**: ≥85% (was targeting 88%, acceptable to lower given imbalance)
- **Vaporware detection**: ≥90% (majority class)
- **Deployed detection**: ≥60% (minority class - lower threshold acceptable)
- **MAE per dimension**: ≤1.5 (was targeting ≤1.0, relaxed)

### If POC Succeeds: **Apply Option 2 for Production**

After validating the distillation approach works, generate 500 additional labels targeting high-scoring articles to improve production model quality.

---

## Conclusion

**Dataset Quality**: ⚠️ ACCEPTABLE (with mitigation)

**Coverage**: MIXED
- ✅ Sufficient total samples (1,938)
- ✅ Full dimensional range coverage
- ❌ Severe class imbalance (82.5% vaporware)
- ❌ High-score scarcity (1.4% deployed)

**Training Viability**: ✅ YES (with class weighting)

**Next Steps**:
1. ✅ Proceed to train/val split
2. ✅ Apply class weighting during training
3. ✅ Oversample high-score examples
4. Evaluate on stratified holdout set
5. If eval shows poor high-score performance → Option 2 (additional labeling)

---

**Analyst**: Claude (AI Assistant)
**Date**: 2025-11-08
