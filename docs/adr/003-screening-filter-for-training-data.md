# ADR-003: Screening Filter for Training Data Enrichment

**Date:** 2026-01-29
**Status:** Accepted
**Tags:** training, architecture, data-quality

## Context

During cultural-discovery v1 development, we discovered a systematic problem with training data collection that affects all "needle-in-haystack" filters. When scoring random articles from the master dataset:

- **94% of articles scored in the low tier** (weighted average < 4.0)
- **Only 0.7% scored in the high tier** (weighted average >= 7.0)
- **evidence_quality dimension plateaued at MAE ~1.25** despite extended training

The root cause: **regression-to-mean**. When training on 94% low-scoring articles, the model learns that predicting ~2.0 minimizes error. This is mathematically optimal for overall MAE but catastrophic for our actual goal: finding rare high-quality content.

Analysis of high-scoring articles (8-10 range) showed evidence_quality MAE of **4.12** - the model consistently under-predicted these gems.

## Decision

Introduce a **"screening filter"** step before oracle scoring during training data collection. This filter enriches the training distribution with signal-bearing content before expensive oracle scoring.

**Workflow change:**

```
CURRENT (flawed):
Raw articles ─────────────────────→ Oracle ─────→ Training
                                    (94% zeros)
                                    Model learns: "predict 2.0 for everything"

PROPOSED (screening filter):
Raw articles ──→ Screening Filter ──→ Oracle ──→ Training
                 (reject ~60-80%,     (maybe 40-50% zeros,
                  enriches signal)     richer gradient)
                                    Model learns: "distinguish 3 from 7"
```

## Rationale

### Why Traditional ML Wisdom Doesn't Apply

**Traditional wisdom says:** "Training and production distributions should match"

**Why that doesn't apply here:**

1. **We're doing regression, not classification**
   - Classification: Model learns P(class|features) - distribution matching ensures prior estimates are correct
   - Regression: Model learns f(features) → score - we need coverage across the score range, not distribution matching
   - A regression model trained only on scores 0-3 cannot extrapolate to 7-10

2. **Needle-in-haystack problem**
   - We're not modeling "what score would a random article get" (would require matching)
   - We're finding rare high-quality content (0.7% high-tier in production)
   - Production goal: minimize false negatives on gems, not model full distribution
   - Similar to fraud detection, anomaly detection, rare disease diagnosis

3. **Asymmetric error costs**
   - Missing a gem (false negative): HIGH cost - user never sees valuable content
   - Slight over-prediction on noise: LOW cost - postfilter catches it, no harm done
   - Training on production distribution optimizes for unweighted error, ignoring cost asymmetry

4. **Regression hedging behavior**
   - With 94% low scores, model can predict mean (~2.0) and achieve "good" overall MAE
   - This is optimal for squared/absolute error but useless for finding gems
   - Model learns to be conservative, never predicting 8-10

5. **Post-prefilter distribution at inference**
   - At inference time, articles already pass prefilter (enriched distribution)
   - Training should reflect post-prefilter distribution, not raw corpus
   - Screening filter aligns training with inference conditions

### Academic Support

This approach is established in ML under several names:
- **Stratified sampling for imbalanced regression** (Torgo et al., 2013)
- **SMOTE for regression** (SMOTER) - synthetic oversampling of rare regions
- **Importance sampling** - upweight rare regions of input space
- **Curriculum learning** - train on balanced data first

### The Key Insight

> **Screening is not cheating - it's acknowledging that our GOAL is finding needles, not modeling haystacks.**

## Consequences

### Positive

- **Better high-score predictions** - Model sees enough 7-10 examples to learn the upper range
- **Richer training gradient** - More signal in each batch, faster convergence
- **Aligns training with inference** - Training distribution matches post-prefilter inference distribution
- **Validates filter signal** - If screening finds nothing, filter may not work for this domain

### Negative

- **More oracle API calls** - Screening rejects ~60-80% of articles before oracle scoring
  - Mitigation: Screen 25K-50K articles to get ~5K-10K scored articles
  - Cost increase: ~2-3x in screening compute, but screening is cheap (rule-based)
- **Additional code to maintain** - One more component per filter
  - Mitigation: Can often reuse prefilter patterns
- **Requires tuning screening criteria** - Must balance enrichment vs over-filtering
  - Mitigation: Clear guidelines and validation metrics

### Neutral

- **Different from prefilter** - Screening is for training data enrichment, prefilter is for inference noise reduction
  - Screening: Aggressive (reject most), goal is to enrich signal
  - Prefilter: Conservative (pass most), goal is to avoid false negatives

## Alternatives Considered

### Loss Weighting

**Approach:** Weight high-score examples more heavily in loss function

**Why not sufficient alone:**
- Still limited by the 33 high-tier examples in cultural-discovery v1
- Can't learn from what you don't have
- Helps but doesn't solve the coverage problem

**Recommendation:** Use loss weighting AS WELL AS screening filter

### More Training Data (Random)

**Approach:** Score 20K articles instead of 5K

**Why diminishing returns:**
- 94% of additional articles are still low-scoring noise
- 20K articles → ~140 high-tier (vs 33 from 5K)
- 4x cost for 4x high-tier examples
- Screening is more cost-effective

### SMOTE for Regression (SMOTER)

**Approach:** Synthetically oversample high-scoring examples

**Why not preferred:**
- Interpolating between articles creates unrealistic training examples
- Text interpolation doesn't produce coherent articles
- May teach model spurious patterns

**Recommendation:** Prefer real data collection over synthesis

### Manual Curation

**Approach:** Manually find high-quality articles

**Why not scalable:**
- Requires domain expertise
- Time-consuming
- Hard to maintain consistency
- Doesn't generalize to new filters

## Lessons Learned: The Merging Strategy (2026-01-30)

### The v2 Failure

Cultural-discovery v2 applied screening successfully, enriching the distribution:
- v1 (random): 94% low, 5% medium, 1% high
- v2 (screened): 80% low, 17% medium, 3% high

**But v2 performed WORSE (MAE 1.47 vs v1's 0.82)!**

**Root cause:** Screening produced only 2,919 articles (vs v1's 4,996). The enriched distribution was harder to learn, and with 42% less data, the model couldn't converge.

### The v3 Solution: Merge Datasets

**Key insight:** You need BOTH sufficient volume AND better distribution. The solution is **additive, not reductive**.

```
Random data (v1)     →  Provides negatives (low-scoring coverage)
        +
Screened data (v2)   →  Provides positives (medium/high signal)
        =
Merged dataset (v3)  →  Best of both worlds
```

**Results:**

| Version | Data Size | Distribution | MAE | Medium-tier MAE | High-tier MAE |
|---------|-----------|--------------|-----|-----------------|---------------|
| v1 (random) | 4,996 | 94% low | 0.82 | 2.85 | 3.49 |
| v2 (screened) | 2,919 | 80% low | 1.47 | - | - |
| v3 (merged) | 7,827 | 88% low | **0.77** | **1.73** (-39%) | **2.69** (-23%) |

### Updated Workflow

```
RECOMMENDED (merged approach):

1. Existing data (random)  ──→  Keep as-is (provides negatives)

2. New screening pass      ──→  Screen 30K articles
         │
         ▼
   Screened articles (~5K) ──→  Score with oracle
         │
         ▼
3. Merge datasets          ──→  Deduplicate and combine
         │
         ▼
   Training dataset        ──→  ~8-10K articles with enriched distribution
```

### When to Use Merging vs Fresh Screening

| Scenario | Approach |
|----------|----------|
| **New filter** | Screen → Score → Train (single pass) |
| **Existing filter with poor high-tier** | Merge existing + screened |
| **Existing filter performing well** | No change needed |
| **Screening produces <3K articles** | Merge with random sample |

### Implications for Other Filters

Analysis of production filters (2026-01-30):

| Filter | LOW | MED | HIGH | Recommendation |
|--------|-----|-----|------|----------------|
| uplifting v5 | 68% | 31% | 0.1% (7) | **Screen+merge for v6** |
| investment-risk v5 | 86% | 14% | 0.1% (12) | Consider screen+merge |
| sustainability-tech v2 | - | - | - | Check distribution |
| cultural-discovery v3 | 88% | 10% | 2% | ✅ Already merged |

**Key metric:** If HIGH tier < 1% of training data, model likely under-predicts high scores.

## Implementation

See:
- `docs/templates/screening-filter-template.md` - Template for creating screening filters
- `docs/agents/filter-development-guide.md` Phase 5 - Updated workflow
- `docs/ARCHITECTURE.md` - Screening filter section

## References

- Cultural-discovery v1 calibration: `docs/adr/cultural-discovery-v1-calibration.md`
- Torgo et al. (2013) "SMOTE for Regression" - https://link.springer.com/chapter/10.1007/978-3-642-40988-2_29
- Branco et al. (2016) "A Survey of Predictive Modeling on Imbalanced Domains" - ACM Computing Surveys
