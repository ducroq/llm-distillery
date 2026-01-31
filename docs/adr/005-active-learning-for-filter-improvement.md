# ADR 005: Active Learning for Filter Improvement

**Date**: 2026-01-31
**Status**: Accepted
**Context**: Need systematic method to improve filter accuracy, especially for rare high-scoring articles

## Decision

Use **active learning** to continuously improve filter training data:

1. **Model-guided sample selection**: Use production model's predictions to identify candidates for oracle scoring
2. **Iterative enrichment**: Merge oracle-scored articles with existing training data, retrain
3. **Ongoing needle search**: Continue searching for rare HIGH-tier articles across versions

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACTIVE LEARNING LOOP                         │
│                                                                 │
│  Production    Screen by      Oracle      Merge with    Retrain │
│  Model      →  prediction  →  Score   →   training  →  Model   │
│  Output        >= threshold   batch       data          vN+1    │
│                                                                 │
│       ↑                                                    │    │
│       └────────────────────────────────────────────────────┘    │
│                         (repeat)                                │
└─────────────────────────────────────────────────────────────────┘
```

## Context

### The Problem: Needle in Haystack

Production article distributions are heavily skewed:

| Tier | uplifting v5 | sustainability_tech v1 |
|------|--------------|------------------------|
| LOW (<4) | 68.4% | ~75% |
| MEDIUM (4-7) | 31.5% | ~25% |
| HIGH (≥7) | **0.07%** (7 articles) | **<0.1%** |

Training on random samples means:
- Model sees mostly LOW-tier articles
- Insufficient gradient signal for MEDIUM/HIGH boundaries
- Model regresses to mean, under-predicts high scores

### The Solution: Model-Guided Sampling

Instead of random sampling, use the model's own predictions to find interesting cases:

| Threshold | What It Finds |
|-----------|---------------|
| predicted ≥ 4.0 | Boundary between LOW and MEDIUM |
| predicted ≥ 5.0 | Confident MEDIUM (enrichment) |
| predicted ≥ 5.5 | Upper MEDIUM, potential HIGH candidates |
| predicted ≥ 6.5 | Rare HIGH candidates (needle search) |

## Implementation

### Phase 1: MEDIUM-Tier Enrichment (Implemented)

**uplifting v6 example:**

1. Filter production output: 4,531 MEDIUM-tier articles
2. Screen by predicted ≥ 5.0: 1,355 candidates
3. Manual curation (remove commerce): 496 articles
4. Oracle score: 495 articles scored
5. Result: All MEDIUM (5.52-6.93), zero HIGH
6. Merge with v5: 10,495 total articles
7. MEDIUM tier increased: 31.5% → 34.6%

**sustainability_technology example:**

1. Screen production MEDIUM-tier by predicted ≥ 5.0: 119 candidates
2. Oracle score: 83 scored (24 filtered by prefilter)
3. Result: 50 in-scope (MEDIUM), 33 out-of-scope (zeros)
4. Both are valuable: positives enrich MEDIUM, zeros teach scope

### Phase 2: HIGH-Tier Needle Search (Ongoing)

HIGH-tier articles are extremely rare (~0.1% of corpus). Active learning alone doesn't find them because:
- Model predicts based on what it learned (mostly MEDIUM)
- True HIGH articles may look different than training distribution

**Strategies for finding needles:**

1. **Raise threshold**: Screen predicted ≥ 6.0 or ≥ 6.5
2. **Target sources**: Focus on signal-rich sources (positive_news_the_better_india, upworthy)
3. **Cross-filter discovery**: HIGH in one filter may surface needles for another
4. **Human curation**: Flag exceptional articles during manual review
5. **Keyword boost**: Combine model prediction with keyword signals

**Target**: Collect 50+ HIGH-tier articles for next major version

### The Feedback Loop

```
v5 model (7 HIGH)
    → active learning
    → v6 data (8 HIGH, more MEDIUM)
    → v6 model (better calibrated)
    → active learning with higher threshold
    → v7 data (target: 50+ HIGH)
    → v7 model (better HIGH-tier accuracy)
```

Each iteration:
- Model gets better at scoring MEDIUM/HIGH boundary
- Higher threshold becomes meaningful
- More HIGH candidates surface

## Rationale

### Why Active Learning Works

1. **Efficient labeling**: Oracle scores articles the model is uncertain about, not random noise
2. **Boundary cases**: Finds articles near decision thresholds
3. **Self-improving**: Better model → better candidate selection → better training data

### Why It Doesn't Find Needles Directly

Active learning finds what the model **thinks** might be high-scoring. But:
- Model trained on 0.07% HIGH can't reliably predict HIGH
- Predicted 5.5 often means oracle 5.5 (well-calibrated for MEDIUM)
- True HIGH articles may have features not in training data

**Solution**: Combine active learning with targeted source collection for HIGH-tier.

### Connection to ADR-004

Per ADR-004, oracle "zeros" are valuable negative training examples:
- Active learning may surface out-of-scope articles
- These zeros teach the model filter-specific scope
- Don't discard them - include in training data

## Consequences

### Positive

- **Systematic improvement**: Clear methodology for each filter version
- **Efficient oracle use**: Score candidates, not random articles
- **Compounding returns**: Each version enables better active learning
- **MEDIUM-tier enrichment**: Proven to work (31.5% → 34.6%)

### Negative

- **Doesn't find needles directly**: HIGH-tier requires additional strategies
- **Model bias**: Can only find what model predicts might be good
- **Diminishing returns**: Each round finds fewer new boundary cases

### Ongoing Commitment

**We commit to continuous needle search:**
- Every filter version includes active learning round
- Track HIGH-tier count as key metric
- Target sources known to contain HIGH-tier content
- Celebrate and analyze every HIGH-tier article found

## Workflow

### For Each Filter Version

1. **Baseline**: Note current tier distribution
2. **Screen**: Filter production output by prediction threshold
3. **Curate**: Remove obvious noise (commerce, duplicates)
4. **Score**: Oracle score candidates in batches
5. **Analyze**: Distribution of scores, any HIGH found?
6. **Merge**: Combine with existing training data
7. **Validate**: Check for duplicates, format consistency
8. **Train**: New model version
9. **Evaluate**: Compare tier-level MAE
10. **Repeat**: Raise threshold, continue needle search

### Thresholds by Goal

| Goal | Threshold | Expected Yield |
|------|-----------|----------------|
| Broad enrichment | ≥ 4.0 | Many MEDIUM, some LOW |
| MEDIUM enrichment | ≥ 5.0 | Mostly MEDIUM |
| HIGH candidates | ≥ 5.5 | Upper MEDIUM, rare HIGH |
| Needle search | ≥ 6.0 | Few candidates, potential HIGH |

## References

- uplifting v6 PLAN: `filters/uplifting/v6/PLAN.md`
- ADR-004 (zeros as training data): `docs/adr/004-universal-noise-prefilter.md`
- ROADMAP (active learning backlog): `docs/ROADMAP.md`
- Training data: `datasets/training/uplifting_v6/`
