# ADR 005: Active Learning for Filter Improvement

**Date**: 2026-01-31
**Status**: Accepted
**Context**: Need systematic method to improve filter accuracy on boundary cases and find rare high-scoring articles

## Decision

Use **active learning** as the standard methodology for improving trained filters:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ACTIVE LEARNING LOOP                            │
│                                                                     │
│  Production    Screen by      Oracle      Merge with    Retrain    │
│   Model    →   Prediction  →  Score   →   Training  →   Model     │
│              (>= threshold)              Data                       │
│                                                                     │
│      ↑                                                    │        │
│      └────────────────────────────────────────────────────┘        │
│                         Repeat                                      │
└─────────────────────────────────────────────────────────────────────┘
```

Key parameters:
- **Screening threshold**: Model prediction >= 5.0 (adjustable per filter)
- **Batch size**: 500-1000 articles per iteration
- **Merge strategy**: Combine with existing training data, re-split

## Context

### The Problem

After initial training, filters have systematic weaknesses:
- **Regression to mean**: Models under-predict high scores due to skewed training distribution
- **Boundary cases**: Uncertain on articles at the edge of in-scope/out-of-scope
- **Needle-in-haystack**: HIGH-tier articles (score >= 7) are extremely rare (~0.1% of corpus)

Random sampling for additional training data is inefficient - most articles score LOW.

### Active Learning Solution

Use the model's own predictions to select which articles to label next:
- Model predicts >= 5.0 → likely MEDIUM-tier → worth oracle scoring
- Model predicts < 3.0 → likely LOW-tier → skip (already have plenty)
- Model predicts >= 6.5 → potential HIGH-tier → priority for needle hunting

This focuses oracle budget on:
1. Articles the model thinks are good (validate or correct)
2. Boundary cases (improve discrimination)
3. Potential needles (find rare HIGH-tier)

## Evidence: Uplifting v6

### Process

1. Filtered production MEDIUM-tier output: 4,531 articles
2. Screened by model prediction >= 5.0: 1,355 articles
3. Manual curation (removed commerce): 496 articles
4. Oracle scored: 495 articles

### Results

| Metric | Value |
|--------|-------|
| Model predicted average | 5.50 |
| Oracle scored average | 5.86 |
| Calibration error | +0.36 (model slightly under-predicts) |
| HIGH-tier found | 0 |
| MEDIUM-tier found | 495 (100%) |

### Training Data Impact

| Dataset | LOW (<4) | MEDIUM (4-7) | HIGH (>=7) |
|---------|----------|--------------|------------|
| v5 (before) | 68.4% | 31.5% | 0.1% (7) |
| v6 (after) | 65.3% | 34.6% | 0.1% (8) |

**Key finding**: Active learning enriches MEDIUM tier effectively (+3.1%), but does NOT find HIGH-tier needles.

## Rationale

### Why Active Learning Works

1. **Efficient oracle usage**: Only score articles likely to be informative
2. **Self-improving**: Each iteration teaches model about its blind spots
3. **Calibration check**: Compare prediction vs oracle to detect drift
4. **No keyword engineering**: Model learns what "uncertain" looks like

### Why It Doesn't Find Needles

HIGH-tier articles are rare by definition. Active learning with threshold >= 5.0 finds MEDIUM-tier boundary cases, not needles scoring 7+.

To find needles, need:
- Higher threshold (>= 6.5) with lower yield
- Targeted sources (positive_news_the_better_india had HIGHs in v5)
- More volume (screen 50K articles to find a few HIGHs)

### Comparison with Alternatives

| Method | Finds MEDIUM | Finds HIGH | Oracle Cost |
|--------|--------------|------------|-------------|
| Random sampling | ❌ Inefficient | ❌ Rare | High |
| Keyword screening | ✅ OK | ❌ Misses semantic needles | Medium |
| **Active learning** | ✅ Efficient | ❌ Need higher threshold | Low |
| Targeted source collection | ❌ Not systematic | ✅ Best for needles | Medium |

## Implementation

### Standard Workflow

```python
# 1. Filter production output by model prediction
candidates = [a for a in production_output if a['predicted_score'] >= 5.0]

# 2. Optional: manual curation (remove obvious commerce/noise)
curated = manual_review(candidates)

# 3. Oracle score
scored = oracle.batch_score(curated, filter='uplifting')

# 4. Merge with existing training data
merged = existing_training + scored

# 5. Re-split (maintain stratification)
train, val, test = stratified_split(merged, ratios=[0.8, 0.1, 0.1])

# 6. Retrain model
train_model(train, val)
```

### Threshold Guidelines

| Goal | Threshold | Expected Yield |
|------|-----------|----------------|
| MEDIUM enrichment | >= 5.0 | ~80% MEDIUM, ~20% LOW |
| Boundary cases | >= 4.5 | ~60% MEDIUM, ~40% LOW |
| Needle hunting | >= 6.0 | <5% HIGH, ~70% MEDIUM |
| Aggressive needle hunting | >= 6.5 | <10% HIGH (if lucky) |

### Iteration Frequency

- **Per filter version**: At least one active learning round before training
- **Ongoing**: Monthly for production filters to catch drift
- **Needle hunting**: Continuous background process for HIGH-tier collection

## Ongoing: The Needle Hunt

HIGH-tier articles (score >= 7) remain extremely rare. Active learning alone won't find them efficiently.

**Needle hunting strategy:**

1. **Higher threshold screening**: Predict >= 6.5 from production
2. **Targeted sources**: Focus on sources that yielded HIGHs before
   - uplifting: `positive_news_the_better_india`, `positive_news_upworthy`
   - sustainability_tech: research news, academic sources
3. **Volume**: Screen large batches (10K+) to find a few needles
4. **Patience**: Collect needles over time, don't expect them in every batch

**Goal**: Collect 50+ HIGH-tier articles per filter for future versions.

## Consequences

### Positive

- Systematic method for filter improvement
- Efficient oracle usage (10x better than random)
- Self-documenting (model predictions vs oracle scores)
- Catches boundary cases and calibration drift

### Negative

- Doesn't efficiently find HIGH-tier needles
- Requires production deployment first (chicken-and-egg for new filters)
- Manual curation step needed to remove commerce/noise

### Trade-offs Accepted

- Accept that needle hunting requires separate, targeted effort
- Accept ~30-40% of screened articles may be out-of-scope (valuable negative examples per ADR-004)

## References

- Uplifting v6 PLAN: `filters/uplifting/v6/PLAN.md`
- Sustainability_tech active learning: `datasets/scored/sustainability_tech_active_learning/`
- ADR-004: Commerce as universal noise (zeros are valuable)
- IDEAS.md: Original "Active Learning Loop" concept
