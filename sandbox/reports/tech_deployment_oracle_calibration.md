# Model Calibration Report: Sustainability Tech Deployment

**Generated**: 2025-11-08 16:23:16
**Prompt**: `filters\sustainability_tech_deployment\v1\prompt-compressed.md`
**Sample Size**: 1 articles (labeled by both models)
**Models**: Claude 3.5 Sonnet vs Gemini 1.5 Pro

---

## Pre-filter Statistics

- **Total Articles Sampled**: 20
- **Passed Pre-filter**: 1 (5.0%)
- **Blocked by Pre-filter**: 19 (95.0%)

**Block Reasons**:

- not_sustainability_topic: 19 (100.0%)

**Cost Savings**: Pre-filter blocked 19 articles, saving expensive LLM API calls.
For ground truth generation (5,000 articles), this would save ~4750 API calls.

---

## Executive Summary

- **Tier Distribution Difference**: 0.0%
- **Average Score Difference**: 0.00
- **Claude Average Score**: 0.00
- **Gemini Average Score**: 0.00

**Recommendation**: Models show very similar results. **Use Gemini** for large-scale labeling (50x cheaper).

---

## Tier Distribution Comparison

| Tier | Claude | Gemini | Difference |
|------|--------|--------|------------|
| Unknown | 100.0% | 100.0% | +0.0% |

---

## Score Statistics

| Metric | Claude | Gemini | Difference |
|--------|--------|--------|------------|
| Average | 0.00 | 0.00 | +0.00 |
| Median | 0.00 | 0.00 | +0.00 |
| Min | 0.00 | 0.00 | +0.00 |
| Max | 0.00 | 0.00 | +0.00 |

---

## Cost Analysis (5,000 articles)

| Model | Cost per Article | Total Cost | Savings |
|-------|------------------|------------|---------|
| Claude 3.5 Sonnet | $0.009 | $45.00 | - |
| Gemini 1.5 Pro | $0.00018 | $0.90 | $44.10 (96%) |

*Gemini pricing assumes Cloud Billing enabled (Tier 1: 150 RPM)*

---

## Sample Article Comparisons

Comparing 1 Claude articles vs 1 Gemini articles

**Matched articles**: 1 (analyzed successfully by both models)

Showing 5 articles with largest score disagreement:

### Sample 1 - Disagreement: 0.00

**Title**: Overshoot: Exploring the implications of meeting 1.5C climate goal ‘from above’

**Excerpt**: The first-ever international conference on the contentious topic of “overshoot” was held last week in... The post Overshoot: Exploring the implications of meeting 1.5C climate goal ‘from above’ appear...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 0.00 | unknown | ... |
| Gemini | 0.00 | unknown | ... |

---

## Next Steps

1. Review the tier distribution and score statistics
2. Examine sample articles with large disagreements
3. If distributions are similar, proceed with Gemini for cost savings
4. If distributions differ, consider:
   - Refining the prompt for better clarity
   - Adding more examples to the prompt
   - Using Claude for ground truth (higher cost but higher quality)
