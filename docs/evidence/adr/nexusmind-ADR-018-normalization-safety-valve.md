---
status: Accepted
date: 2026-04-15
deciders: [Jeroen]
amends: ADR-014
---

# ADR-018: Normalization Safety Valve and Scale Factor Fallback

## Decision

Percentile normalization (ADR-014) requires a minimum of 200 articles in the CDF to be applied. Below that threshold, the scorer falls back to the pre-ADR-014 linear `score_scale_factor` from `config.yaml`. If neither is available, raw scores pass through unchanged.

## Context

### The incident

nature_recovery's normalization.json was fitted on only 24 articles. Because 98.6% of production articles score below 1.0, a raw score of 3.8 landed at the 99.5th percentile, normalizing to 9.17. This made casino articles, hippo culling, and road construction appear as top recovery content on ovr.news (#167).

### Why ADR-014 alone was insufficient

ADR-014 assumed "thin but sufficient for a monotonic lookup" at 48 articles. This proved wrong:

| Sample size | CDF quality | Consequence |
|-------------|------------|-------------|
| 73,986 (uplifting v7) | Dense, smooth | Normalization works as intended |
| 4,826 (belonging v1) | Adequate | Normalization works |
| 24 (nature_recovery v1) | Noise | A raw 3.8 maps to 9.17 — garbage inflation |

The fundamental issue: percentile normalization is distribution-agnostic (good), but CDF estimation is sample-size-dependent. With 24 points, the empirical CDF is unreliable — small changes in the sample shift percentile ranks dramatically.

### The compounding failure

When ADR-014 shipped, `score_scale_factor` was removed from the scorer code (#112). This created a gap: if normalization was disabled for any reason, scores passed through completely unscaled. For nature_recovery (calibrated max 6.52), no article could reach ovr.news's 5.0 display threshold — the lens went empty.

## Implementation

### Scoring pipeline (filter_base_scorer.py)

```
raw weighted average
  ↓
if normalization loaded AND n_articles >= 200:
  → percentile normalization (ADR-014)
elif score_scale_factor != 1.0 in config.yaml:
  → linear stretch: min(10.0, raw * scale_factor)
else:
  → raw score passthrough
```

### The 200-article threshold

200 is the minimum for a CDF with 1% granularity (each article represents ~0.5 percentile points). Below this, the mapping is dominated by sampling noise. The threshold is a class constant (`_MIN_NORMALIZATION_ARTICLES`) — adjustable if experience shows a different cutoff is better.

### Affected filters (as of 2026-04-15)

| Filter | n_articles | Normalization | Fallback |
|--------|-----------|---------------|----------|
| uplifting v7 | 73,986 | Percentile | — |
| sustainability_tech v3 | 15,641 | Percentile | — |
| cultural-discovery v4 | 8,740 | Percentile | — |
| belonging v1 | 4,826 | Percentile | — |
| investment_risk v6 | — (no file) | **No normalization.json** | scale_factor 1.2652 |
| foresight v1 | 623 | Percentile | — |
| nature_recovery v1 | 24 | **Disabled** | scale_factor 1.5345 |

### Recovery path

When a filter accumulates 200+ articles with `raw_weighted_average` in its production JSONL, refit normalization via llm-distillery. The safety valve automatically re-enables percentile normalization once the CDF is dense enough.

## Consequences

**Positive:**
- nature_recovery no longer inflates garbage to 9.17
- New filters with thin data degrade gracefully instead of producing misleading scores
- The pre-ADR-014 scale factor was already calibrated per filter (10.0 / theoretical_max) — it's an honest, if crude, stretch
- Self-healing: as filters accumulate data, they graduate from scale factor to percentile normalization automatically

**Negative:**
- nature_recovery scores are now linear (not percentile-ranked), so cross-filter comparison with uplifting is imperfect — a 7.0 on nature_recovery doesn't mean the same thing as 7.0 on uplifting
- The root problem (nature_recovery model quality) remains — the scale factor makes scores honest, not good. Tracked in llm-distillery#41.

## Amendment 2026-05-19: `raw_min` guard for oracle-biased fits

A second failure mode of the same shape as #167 surfaced via #205. `foresight` v1's `normalization.json` was fit on 623 articles (above the n_articles threshold) — but the sample was oracle-biased, drawn from already-filtered output rather than a representative production slice. Its `stats.raw_min` was 5.01, so production articles scoring `raw < 5.0` clamped to `wavg ≈ 0` via `np.interp`'s edge behavior. Live since 2026-04-06; 2026-05-08 production case: raw 4.60 → wavg 0.02 (tier "low") despite "medium" verdict on 5 other filters.

The n_articles gate alone couldn't catch this — the count was healthy but the **distribution** wasn't representative.

**New guard:** `MAX_NORMALIZATION_RAW_MIN = 4.5` in `src/scoring/production_scorer.py`. If `stats.raw_min` exceeds this (or is non-finite), reject the fit and fall through to `score_scale_factor`. The threshold sits in the gap between the well-fit cluster (~4.0 across belonging / cultural_discovery / sustainability_technology / uplifting) and the broken outlier (foresight at 5.01). Boundary is strict-greater-than: `raw_min == 4.5` is accepted.

Foresight currently routes through this fall-through path — wavg = raw × 1.2529, tier-medium content surfaces correctly. The artifact-side refit is tracked in [llm-distillery#64](https://github.com/ducroq/llm-distillery/issues/64).

## References

- ADR-014: Percentile normalization (amended by this ADR)
- #167: Percentile normalization inflates garbage scores on skewed filters (article-count failure)
- #205: foresight CDF fitted on raw_min=5.01 — distribution failure
- llm-distillery#41: nature_recovery model needs retraining
- llm-distillery#64: foresight normalization refit
- `src/scoring/production_scorer.py`: `MIN_NORMALIZATION_ARTICLES`, `MAX_NORMALIZATION_RAW_MIN`, `_load_normalization`
