# Stratified Sampling Strategy for Ground Truth Generation

**Date**: 2025-11-08
**Filter**: sustainability_tech_deployment
**Principle**: Ground truth datasets must have **representative coverage** across the full output range

---

## Problem Statement

Initial random sampling from general tech news produced severe class imbalance:

| Tier | Count | Percentage | Issue |
|------|-------|------------|-------|
| Vaporware (<4.0) | 1,598 | 82.5% | ✅ Dominant (as expected) |
| Pilot (4.0-5.9) | 187 | 9.6% | ✅ Adequate |
| Early Commercial (6.0-7.9) | 125 | 6.4% | ✅ Adequate |
| **Deployed/Proven (≥8.0)** | **28** | **1.4%** | ❌ **SEVERELY underrepresented** |

**Training Risk**: Model will learn to predict low scores for everything due to majority class dominance.

**Root Cause**: Most tech news IS vaporware (announcements, concepts, early research). Random sampling reflects reality but doesn't support discriminative learning.

---

## Solution: Stratified Sampling

### Principle

**Ground truth generation ≠ natural distribution sampling**

- **Natural distribution**: What you encounter in the wild (82.5% vaporware)
- **Training distribution**: What the model needs to learn discrimination (balanced representation)

**Goal**: Ensure adequate examples across ALL tiers the model must predict.

### Two-Stage Approach

**Stage 1: Random Sampling** (COMPLETED)
- Sample: 2,000 articles randomly from 147K dataset
- Purpose: Capture natural distribution baseline
- Result: 1,938 labels (82.5% vaporware)

**Stage 2: Stratified Sampling** (IN PROGRESS)
- Sample: 500 articles with deployment signals (keyword-based targeting)
- Purpose: Boost underrepresented high-score tier
- Expected: ~100-200 high-scoring examples (vs current 28)

**Stage 3: Merge & Balance** (NEXT)
- Combine both datasets: 1,938 + 500 = ~2,400 labels
- Expected distribution:
  - Vaporware: ~70-75% (down from 82.5%)
  - Deployed/Proven: ~5-10% (up from 1.4%)

---

## Stratified Sampling Implementation

### Keyword-Based Targeting

Identify deployment-focused articles by keyword matching:

**Deployment Keywords** (requires ≥3 matches):
```python
deployment_keywords = [
    # Direct deployment
    'deployed', 'deployment', 'rollout', 'operational', 'in production',

    # Commercial terms
    'commercial', 'revenue', 'sales', 'sold', 'market share',

    # Adoption/scale
    'adoption', 'customers', 'installations', 'gigawatt', 'million units',

    # Manufacturing
    'manufacturing', 'factory', 'mass production', 'supply chain',

    # Market maturity
    'market leader', 'widespread use', 'standard', 'mainstream'
]
```

**Filtering Logic**:
1. Scan all 147K articles
2. Count deployment keywords in title + content
3. Select articles with ≥3 keyword matches
4. Sample 500 from candidates (prioritize high keyword counts)

**Result**: Found 1,000 candidates, sampled 500 (mean 3.7 keywords per article)

### Oracle Labeling

**Process** (running now - bash b19c31):
```bash
python -m ground_truth.batch_labeler \
  --filter filters/sustainability_tech_deployment/v1 \
  --source "datasets/curated/deployment_focused_500.jsonl" \
  --llm gemini-flash \
  --target-count 500 \
  --output-dir ground_truth/labeled/tech_deployment_supplemental
```

**Expected Duration**: ~25-30 minutes (3 sec/article × 500)
**Cost**: ~$0.50 (500 × $0.001 Gemini Flash)

---

## Expected Impact

### Before Stratified Sampling

**Dimension: proof_of_impact** (gatekeeper):
- Low scores (0-3): 1,583 (81.7%)
- Mid scores (4-7): 352 (18.2%)
- High scores (8-10): **3 (0.2%)** ← Critical gap

**Overall Scores**:
- Mean: 2.69
- Deployed/Proven (≥8.0): 28 (1.4%)

### After Stratified Sampling (Projected)

Assuming 500 deployment-focused articles yield:
- ~40% score ≥6.0 (200 articles) - early commercial/deployed
- ~20% score ≥8.0 (100 articles) - deployed/proven

**New Distribution**:
- Total: ~2,400 labels
- Deployed/Proven: 28 + 100 = **128 (5.3%)**
- Early Commercial: 125 + 100 = **225 (9.4%)**
- Pilot: 187 + 50 = **237 (9.9%)**
- Vaporware: 1,598 + 250 = **1,848 (77.0%)**

**Dimension: proof_of_impact** (projected):
- High scores (8-10): 3 + 50 = **53 (2.2%)** ← Much better!

**Critical Improvement**: All dimensions will have >50 high-score examples (vs <3 for some dimensions currently)

---

## Training Benefits

### With Stratified Sampling

✅ **Better high-score calibration**: Model sees enough examples to learn when to predict 8-10
✅ **Gatekeeper learning**: `proof_of_impact` has adequate high-score training data
✅ **Reduced class weighting**: Less extreme weighting needed (5.3% vs 1.4%)
✅ **Improved discrimination**: Model learns to differentiate deployed from pilot/vaporware

### Comparison

| Metric | Random Only | + Stratified | Improvement |
|--------|-------------|--------------|-------------|
| **Deployed examples** | 28 (1.4%) | 128 (5.3%) | **+357%** |
| **High proof_of_impact** | 3 (0.2%) | 53 (2.2%) | **+1,667%** |
| **Total labels** | 1,938 | 2,438 | +26% |
| **Cost** | $2.00 | $2.50 | +$0.50 |

---

## Validation After Merge

Once supplemental labeling completes, we will:

1. **Merge datasets**:
   ```bash
   cat ground_truth/labeled/tech_deployment/sustainability_tech_deployment/*.jsonl \
       ground_truth/labeled/tech_deployment_supplemental/*.jsonl \
       > ground_truth/labeled/tech_deployment_merged/all_labels.jsonl
   ```

2. **Re-analyze distribution**:
   - Run distribution analysis script
   - Verify deployed tier now 5-10%
   - Check all dimensions have >50 high-score examples

3. **Quality check**:
   - Spot-check 20 supplemental labels
   - Ensure keyword targeting didn't introduce bias
   - Verify scores align with expectations

4. **Proceed to training** if:
   - Deployed tier ≥5%
   - All dimensions have ≥50 high-score examples
   - Quality spot-check passes

---

## Lessons Learned

### Key Insight

**Random sampling assumes you want to model the natural distribution.**

For discriminative tasks (classification, regression with full range), you need **stratified sampling** to ensure model can learn ALL classes/ranges, not just the majority.

### When to Use Stratified Sampling

✅ **Classification with rare classes** - Ensure minority classes represented
✅ **Regression with skewed distribution** - Boost examples at low-density ranges
✅ **Multi-label tasks** - Ensure all label combinations represented
✅ **Imbalanced datasets** - Actively sample underrepresented regions

### When Random Sampling Is OK

✅ **Generative modeling** - Want to match natural distribution
✅ **Well-balanced datasets** - Natural distribution already covers range
✅ **Density estimation** - Explicitly modeling the distribution

---

## Replication for Other Filters

This stratified sampling approach will be applied to all 6 filters:

1. **Economic Viability**: Boost articles about profitability, cost reductions, commercial success
2. **Policy Effectiveness**: Boost articles about policy outcomes, measured impacts
3. **Nature Recovery**: Boost articles about ecosystem restoration, verified biodiversity gains
4. **Movement Growth**: Boost articles about behavior change, adoption metrics
5. **AI-Augmented Practice**: Boost articles about real workflow integration, productivity data

**Strategy per filter**:
- Start with 2,000 random samples (natural distribution)
- Add 500 targeted samples (stratified by filter-specific keywords)
- Target: 5-10% high-tier examples minimum

---

## References

- **Initial distribution**: `reports/tech_deployment_label_distribution_analysis.md`
- **Validation report**: `reports/tech_deployment_label_validation.md`
- **Keyword finder script**: `scripts/find_deployment_articles.py`
- **Curated sample**: `datasets/curated/deployment_focused_500.jsonl`

---

**Conclusion**: Stratified sampling is ESSENTIAL for creating high-quality ground truth that enables discriminative learning across the full output range. Random sampling alone is insufficient when the natural distribution is severely imbalanced.
