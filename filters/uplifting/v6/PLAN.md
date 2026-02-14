# Uplifting v6: Active Learning from Production

**Date:** 2026-01-31
**Status:** BLOCKED — Collecting HIGH-tier data (8 articles is not enough)
**Approach:** Active learning from production filter output + targeted HIGH collection

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

## Prompt Fixes for v6

### Fix: Sensational Crime / Individual Court Cases (2026-02-14)

**Problem:** Articles about individual criminal convictions leak into medium tier as "uplifting" because:
- `justice_rights_impact` scores 7-8 (conviction = "accountability achieved")
- `evidence_level` scores high (court rulings are well-documented)
- `human_wellbeing_impact` scores moderate ("safety improved")
- `change_durability` scores moderate (sentences are "durable")

These are sensational crime news, not solutions journalism. A single criminal getting convicted is not systemic change.

**Root cause:** No content-type cap for crime reporting, and the justice dimension doesn't distinguish individual sentencing from systemic reform.

**Proposed fix (3 changes to prompt):**

**1. Add to OUT OF SCOPE list:**
```
- **Individual criminal cases** (single arrests, sentencing, convictions) unless systemic reform or landmark ruling
```

**2. Add content-type cap (Section 3: Pre-Classification Step):**
```
**E) INDIVIDUAL CRIMINAL CASE?** Single arrest, trial, conviction, sentencing of individual(s)?
   - If YES and NOT (systemic reform | class action | landmark/precedent-setting ruling | policy change):
   - → FLAG "individual_crime" → **max_score = 3.0**
```

**3. Add critical filter to justice_rights_impact dimension:**
```
**CRITICAL FILTERS - Score 0-2 if:**
- Problem identification only (no action toward justice)
- Corporate accountability theater (PR without consequences)
- Speculation about future justice ("could lead to reform")
- Individual criminal sentencing without systemic impact (single convictions, arrests, sentences)
```

**Distinction preserved:** Landmark rulings, class actions, systemic reform, and policy changes still score normally. Only individual "criminal X convicted" stories are capped.

---

## Blocker: HIGH-Tier Data Collection (2026-02-14)

**Decision:** Do NOT train v6 until HIGH-tier representation is addressed. With only 8 HIGH articles out of 10,495 (0.08%), the model cannot learn the upper score range. This is the same regression-to-mean problem documented in ADR-003 for cultural-discovery.

### Why This Matters

- Model learns to never predict above ~6.5 because it almost never sees 7+
- The 495 active-learning articles enriched MEDIUM but found zero HIGH
- General news corpus simply doesn't contain enough genuinely uplifting content at the 7+ level
- Training now would waste compute and produce a model with the same HIGH-tier blind spot as v5

### Collection Strategy for HIGH-Tier Articles

**Target:** 50-100 HIGH-tier articles (≥7.0 weighted average) before training.

**Approach 1: Targeted source scraping**
- Sources that produced v5's 7 HIGH articles (check which sources they came from)
- Positive news outlets: The Better India, Upworthy, Reasons to be Cheerful, Yes! Magazine, Positive News
- Solutions Journalism Network tagged stories
- Score with updated v6 prompt (with crime fix), keep only ≥7.0

**Approach 2: Active learning with higher threshold**
- Run production filter on new articles, screen predicted ≥6.0
- Oracle score with v6 prompt
- These won't all be HIGH, but may catch some near the boundary

**Approach 3: Manual curation + oracle scoring**
- Manually identify articles that look genuinely transformative
- Oracle score to confirm ≥7.0
- Labor-intensive but highest precision

**Recommended:** Start with Approach 1 (targeted sources), supplement with Approach 2.

### Training Plan (once data is collected)

1. Apply v6 prompt fixes (crime cap, etc.)
2. Re-score existing training data with updated prompt (at least a sample to check impact)
3. Merge: v5 data + 495 active learning + new HIGH-tier articles
4. Train on merged dataset
5. Evaluate tier-level MAE, especially HIGH-tier performance

### Success Criteria

- At least 50 HIGH-tier articles in training set (currently 8)
- HIGH-tier MAE improvement over v5 (v5 HIGH MAE unknown — measure as baseline)
- No regression on LOW/MEDIUM performance

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

*Updated: 2026-02-14*
