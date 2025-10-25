# Model Calibration Report: Uplifting

**Generated**: 2025-10-25 17:32:42
**Prompt**: `prompts/uplifting.md`
**Sample Size**: 10 articles (labeled by both models)
**Models**: Claude 3.5 Sonnet vs Gemini 1.5 Pro

---

## Executive Summary

- **Tier Distribution Difference**: 26.7%
- **Average Score Difference**: 1.23
- **Claude Average Score**: 5.81
- **Gemini Average Score**: 4.57

**Recommendation**: Models differ significantly. **Use Claude** or refine prompt for better Gemini consistency.

---

## Tier Distribution Comparison

| Tier | Claude | Gemini | Difference |
|------|--------|--------|------------|
| Connection | 40.0% | 33.3% | -6.7% |
| Impact | 40.0% | 33.3% | -6.7% |
| Not Uplifting | 20.0% | 33.3% | +13.3% |

---

## Score Statistics

| Metric | Claude | Gemini | Difference |
|--------|--------|--------|------------|
| Average | 5.81 | 4.57 | -1.23 |
| Median | 6.37 | 5.07 | -1.30 |
| Min | 1.27 | 0.86 | -0.41 |
| Max | 7.92 | 7.79 | -0.13 |

---

## Cost Analysis (5,000 articles)

| Model | Cost per Article | Total Cost | Savings |
|-------|------------------|------------|---------|
| Claude 3.5 Sonnet | $0.009 | $45.00 | - |
| Gemini 1.5 Pro | $0.00018 | $0.90 | $44.10 (96%) |

*Gemini pricing assumes Cloud Billing enabled (Tier 1: 150 RPM)*

---

## Sample Article Comparisons

Showing 5 articles with largest score disagreement:

### Sample 1 - Disagreement: 1.20

**Title**: Your Gut Gas Could Be Making You Absorb More Calories

**Excerpt**: A little-known microbe in your gut produces methane and may help your body extract more calories from food, according to a study led by Arizona State University. Deep inside your gut lives a vast comm...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 6.27 | connection | Scientists are expanding human knowledge about how our bodies interact with gut microbes in ways tha... |
| Gemini | 5.07 | connection | This article describes a scientific discovery that advances human understanding of the gut microbiom... |

### Sample 2 - Disagreement: 0.61

**Title**: Energy Independence with Home Batteries

**Excerpt**: There is no denying that the Australian federal government’s Cheaper Home Battery program has been a resounding success. With over 40,000 batteries installed in the first 8 weeks of the program, Austr...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 7.18 | impact | 40,000 Australian households are actively taking control of their energy independence through home b... |
| Gemini | 7.79 | impact | A government program has successfully enabled over 40,000 households to install home batteries, a ta... |

### Sample 3 - Disagreement: 0.41

**Title**: It’s troll vs. troll in Netflix’s Troll 2 trailer

**Excerpt**: Norwegian director Roar Uthaug's sequel to his 2022 film Troll knows to not take itself too seriously....

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 1.27 | not_uplifting | This is primarily a commercial entertainment product announcement with minimal direct impact on huma... |
| Gemini | 0.86 | not_uplifting | The article describes the creation of a commercial entertainment product, a sequel to a film. This p... |

---

## Next Steps

1. Review the tier distribution and score statistics
2. Examine sample articles with large disagreements
3. If distributions are similar, proceed with Gemini for cost savings
4. If distributions differ, consider:
   - Refining the prompt for better clarity
   - Adding more examples to the prompt
   - Using Claude for ground truth (higher cost but higher quality)
