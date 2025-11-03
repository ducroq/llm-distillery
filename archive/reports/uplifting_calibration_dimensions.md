# Model Calibration Report: Uplifting

**Generated**: 2025-10-31 15:05:09
**Prompt**: `filters\uplifting\v1\prompt-compressed.md`
**Sample Size**: 92 articles (labeled by both models)
**Models**: Claude 3.5 Sonnet vs Gemini 1.5 Pro

---

## Pre-filter Statistics

- **Total Articles Sampled**: 92
- **Passed Pre-filter**: 92 (100.0%)
- **Blocked by Pre-filter**: 0 (0.0%)


---

## Executive Summary

- **Tier Distribution Difference**: 102.0%
- **Average Score Difference**: 1.60
- **Claude Average Score**: 4.26
- **Gemini Average Score**: 5.85

**Recommendation**: Models differ significantly. **Use Claude** or refine prompt for better Gemini consistency.

---

## Tier Distribution Comparison

| Tier | Claude | Gemini | Difference |
|------|--------|--------|------------|
| Connection | 52.2% | 24.0% | -28.2% |
| Impact | 0.0% | 51.0% | +51.0% |
| Not Uplifting | 47.8% | 25.0% | -22.8% |

---

## Score Statistics

| Metric | Claude | Gemini | Difference |
|--------|--------|--------|------------|
| Average | 4.26 | 5.85 | +1.60 |
| Median | 5.11 | 7.09 | +1.98 |
| Min | 0.00 | 0.00 | +0.00 |
| Max | 7.60 | 8.90 | +1.30 |

---

## Cost Analysis (5,000 articles)

| Model | Cost per Article | Total Cost | Savings |
|-------|------------------|------------|---------|
| Claude 3.5 Sonnet | $0.009 | $45.00 | - |
| Gemini 1.5 Pro | $0.00018 | $0.90 | $44.10 (96%) |

*Gemini pricing assumes Cloud Billing enabled (Tier 1: 150 RPM)*

---

## Sample Article Comparisons

Comparing 92 Claude articles vs 100 Gemini articles

**Matched articles**: 0 (analyzed successfully by both models)

Showing 5 articles with largest score disagreement:

---

## Next Steps

1. Review the tier distribution and score statistics
2. Examine sample articles with large disagreements
3. If distributions are similar, proceed with Gemini for cost savings
4. If distributions differ, consider:
   - Refining the prompt for better clarity
   - Adding more examples to the prompt
   - Using Claude for ground truth (higher cost but higher quality)
