---
status: Superseded
date: 2026-02-24
deciders: [Jeroen, Claude]
superseded_by: ADR-014
---

# ADR-001: Cross-Filter Score Normalization

## Context

NexusMind runs three content filters (sustainability_technology, uplifting, cultural-discovery) that each produce a weighted average score on a 0-10 scale. In production across 11,733 articles, scores were not balanced:

| Filter | Weighted Avg | IQR |
|--------|-------------|-----|
| Cultural-discovery | 5.60 | 4.98-6.40 |
| Uplifting | 4.73 | 4.31-5.31 |
| Sustainability tech | 4.51 | 4.22-4.78 |

This ~1.1 point gap makes cross-filter comparison misleading: a cultural-discovery 5.6 and a sustainability 4.5 represent similar quality relative to their domains, but appear different on the dashboard.

## Root Cause

Each filter was trained using oracle labels (Gemini Flash) with different scoring prompts. The oracle produced systematically different score distributions per domain. Isotonic calibration faithfully preserved these distributions, resulting in different effective score ranges:

| Filter | Theoretical Max Weighted Avg | Effective Range |
|--------|------------------------------|-----------------|
| Sustainability tech | 7.39 | Compressed (econ_comp maxes at 6.2) |
| Uplifting | 7.41 | Compressed (most dims max at 7.0) |
| Cultural-discovery | 8.80 | Wide (heritage/evidence reach 9.5) |

None of the filters use the full 0-10 scale. Sustainability and uplifting top out around 7.4, while cultural-discovery reaches 8.8 -- a 1.4 point ceiling gap.

A secondary factor is prefilter selection bias: cultural-discovery's aggressive prefilter (~15% pass rate) only lets strong candidates through, inflating its average compared to sustainability (~25% pass rate).

## Decision

Apply a post-calibration linear scale factor to each filter's weighted average, computed as `10.0 / theoretical_max`, so that the theoretical maximum for every filter reaches 10.0:

| Filter | Scale Factor | Derivation | Avg Before | Avg After |
|--------|-------------|------------|------------|-----------|
| Sustainability tech | 1.35 | 10.0 / 7.39 | 4.51 | 6.09 |
| Uplifting | 1.35 | 10.0 / 7.41 | 4.73 | 6.39 |
| Cultural-discovery | 1.14 | 10.0 / 8.80 | 5.60 | 6.38 |

The normalization is applied after the gatekeeper caps but before tier assignment. Gatekeeper cap values were adjusted so that scaled caps remain below their respective tier thresholds:

| Filter | Old Cap | New Cap | Scaled Cap | Tier Boundary |
|--------|---------|---------|------------|---------------|
| Sustainability tech | 2.9 | 2.2 | 2.97 | < medium (3.0) |
| Uplifting | 3.0 | 2.9 | 3.92 | < medium (4.0) |
| Cultural-discovery | 3.0 | 3.0 | 3.42 | < medium (4.0) |

## Alternatives Considered

### 1. Adjust tier thresholds per filter (no score change)
- Raise cultural-discovery thresholds (high: 8.0, medium: 5.0) to match its wider range
- **Rejected**: Dashboard still shows different raw numbers; doesn't solve the "below average on 0-10" perception

### 2. Re-label all training data with aligned oracle prompts
- Re-run Gemini Flash with standardized scoring guidelines across all domains
- Retrain all LoRAs, recalibrate, validate, redeploy
- **Rejected**: Expensive pipeline work for all 3 filters, ~$15-25 in API costs plus days of engineering

### 3. Z-score normalization (mean=5.0, std=1.5)
- Normalize each filter to a standard distribution
- **Rejected**: Changes distribution shape, not just location/scale; harder to reason about

### 4. Do nothing
- **Rejected**: Scores averaging 4.5 on a 0-10 scale for prefiltered (already-relevant) content looks broken

## Consequences

### Positive
- All three filters now center around 6.0-6.4, which is intuitive on a 0-10 scale
- Cross-filter weighted averages are directly comparable
- Tier thresholds (7.0 for high, 4.0 for medium) now represent meaningful fractions of the full scale
- No retraining needed; dimension scores and calibration are untouched

### Negative
- Historical scores in existing filtered JSONL files are on the old scale; new runs will produce higher numbers
- The scale factor must be recomputed when calibration data changes (e.g., after retraining)
- Individual dimension scores are normalized by the same `SCORE_SCALE_FACTOR` as the weighted average (changed 2026-03-04, Contract B v1.3.0)

### Deployment
- gpu-server needs updated filter files: `scp -r filters/ gpu-server:~/NexusMind/filters/` + service restart
- sadalsuud: `git pull` (picks up changes automatically on next timer run)
- No config changes needed; scale factors are class constants in each filter's `base_scorer.py`

## Files Changed

- `filters/common/filter_base_scorer.py` -- `SCORE_SCALE_FACTOR` constant + normalization in `_process_raw_scores()`
- `filters/common/hybrid_scorer.py` -- Apply same factor to stage 1 (embedding probe) scores
- `filters/sustainability_technology/v3/base_scorer.py` -- `SCORE_SCALE_FACTOR = 1.35`, cap 2.9 -> 2.2
- `filters/uplifting/v6/base_scorer.py` -- `SCORE_SCALE_FACTOR = 1.35`, cap 3.0 -> 2.9
- `filters/cultural-discovery/v4/base_scorer.py` -- `SCORE_SCALE_FACTOR = 1.14`, cap unchanged
