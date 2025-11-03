# Model Calibration Report: Uplifting

**Generated**: 2025-10-31 14:18:07
**Prompt**: `filters\uplifting\v1\prompt-compressed.md`
**Sample Size**: 100 articles (labeled by both models)
**Models**: Claude 3.5 Sonnet vs Gemini 1.5 Pro

---

## Pre-filter Statistics

- **Total Articles Sampled**: 100
- **Passed Pre-filter**: 95 (95.0%)
- **Blocked by Pre-filter**: 5 (5.0%)

**Block Reasons**:

- corporate_finance: 3 (60.0%)
- military_security: 2 (40.0%)

**Cost Savings**: Pre-filter blocked 5 articles, saving expensive LLM API calls.
For ground truth generation (5,000 articles), this would save ~250 API calls.

---

## Executive Summary

- **Tier Distribution Difference**: 102.0%
- **Average Score Difference**: 0.56
- **Claude Average Score**: 5.85
- **Gemini Average Score**: 5.29

**Recommendation**: Models differ significantly. **Use Claude** or refine prompt for better Gemini consistency.

---

## Tier Distribution Comparison

| Tier | Claude | Gemini | Difference |
|------|--------|--------|------------|
| Connection | 24.0% | 68.5% | +44.5% |
| Impact | 51.0% | 0.0% | -51.0% |
| Not Uplifting | 25.0% | 31.5% | +6.5% |

---

## Score Statistics

| Metric | Claude | Gemini | Difference |
|--------|--------|--------|------------|
| Average | 5.85 | 5.29 | -0.56 |
| Median | 7.09 | 5.48 | -1.61 |
| Min | 0.00 | 0.00 | +0.00 |
| Max | 8.90 | 7.77 | -1.13 |

---

## Cost Analysis (5,000 articles)

| Model | Cost per Article | Total Cost | Savings |
|-------|------------------|------------|---------|
| Claude 3.5 Sonnet | $0.009 | $45.00 | - |
| Gemini 1.5 Pro | $0.00018 | $0.90 | $44.10 (96%) |

*Gemini pricing assumes Cloud Billing enabled (Tier 1: 150 RPM)*

---

## Sample Article Comparisons

Comparing 100 Claude articles vs 92 Gemini articles

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
