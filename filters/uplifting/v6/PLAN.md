# Uplifting v6: Active Learning from Production

**Date:** 2026-01-31
**Status:** TRAINING DATA READY
**Approach:** Active learning from production filter output

---

## What We Did (vs Original Plan)

Originally planned to use keyword-based screening. Instead, we used **active learning**:

```
Production filter output → Screen by model prediction ≥5.0 → Oracle score → Merge with v5
```

This is more elegant because:
1. Model's own predictions select hard examples
2. No keyword engineering required
3. Finds articles the model is uncertain about

---

## Results

### Data Collection

| Step | Count |
|------|-------|
| Production MEDIUM-tier articles | 4,531 |
| Filtered by predicted ≥5.0 | 1,355 |
| After manual curation (removed commerce) | 496 |
| After oracle scoring | 495 |

### Distribution Analysis

All 495 articles scored in **MEDIUM tier** (5.52-6.93). Zero HIGH articles found.

This reveals: TRUE HIGH articles (≥7) are extremely rare in general news corpus.

### v6 Training Data

| Dataset | LOW (<4) | MEDIUM (4-7) | HIGH (≥7) |
|---------|----------|--------------|-----------|
| v5 | 68.4% | 31.5% | 0.1% (7) |
| **v6** | 65.3% | 34.6% | 0.1% (8) |

**Files:** `datasets/training/uplifting_v6/` (10,495 articles)

---

## Key Learnings

1. **Model is well-calibrated** - Predicted 5.5, oracle scored 5.86
2. **HIGH articles are needle-in-haystack** - Only 7-8 in 10K articles
3. **Active learning works for MEDIUM enrichment** - But doesn't find HIGH
4. **Need targeted sources for HIGH** - positive_news_the_better_india had v5 HIGHs

---

## Next Steps

### Immediate: Train v6 Model

```bash
# Copy v5 architecture
# Train on v6 dataset
# Compare tier-level MAE
```

### Ongoing: Active Learning for HIGH Tier

Added to backlog - continue collecting HIGH candidates:

1. Monitor production filter for predicted ≥6.5
2. Curate from positive news sources
3. Goal: 50+ HIGH articles for v7

---

## Files

```
filters/uplifting/v6/
├── PLAN.md              # This file
datasets/training/uplifting_v6/
├── train.jsonl          # 8,396 articles
├── val.jsonl            # 1,049 articles
└── test.jsonl           # 1,050 articles
datasets/curation/uplifting_high_candidates/
├── uplifting_high_candidates_20260131.jsonl  # 496 candidates
└── test_sample_50.jsonl                       # Test batch
datasets/scored/uplifting_active_learning/
└── uplifting/scored_batch_*.jsonl            # 495 scored
```

---

*Updated: 2026-01-31*
