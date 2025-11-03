# Model Calibration Report: Uplifting

**Generated**: 2025-10-31 14:18:05
**Prompt**: `filters\uplifting\v1\prompt-compressed.md`
**Sample Size**: 92 articles (labeled by both models)
**Models**: Claude 3.5 Sonnet vs Gemini 1.5 Pro

---

## Pre-filter Statistics

- **Total Articles Sampled**: 93
- **Passed Pre-filter**: 88 (94.6%)
- **Blocked by Pre-filter**: 5 (5.4%)

**Block Reasons**:

- corporate_finance: 3 (60.0%)
- military_security: 2 (40.0%)

**Cost Savings**: Pre-filter blocked 5 articles, saving expensive LLM API calls.
For ground truth generation (5,000 articles), this would save ~268 API calls.

---

## Executive Summary

- **Tier Distribution Difference**: 32.6%
- **Average Score Difference**: 1.03
- **Claude Average Score**: 5.29
- **Gemini Average Score**: 4.26

**Recommendation**: Models differ significantly. **Use Claude** or refine prompt for better Gemini consistency.

---

## Tier Distribution Comparison

| Tier | Claude | Gemini | Difference |
|------|--------|--------|------------|
| Connection | 68.5% | 52.2% | -16.3% |
| Not Uplifting | 31.5% | 47.8% | +16.3% |

---

## Score Statistics

| Metric | Claude | Gemini | Difference |
|--------|--------|--------|------------|
| Average | 5.29 | 4.26 | -1.03 |
| Median | 5.48 | 5.11 | -0.37 |
| Min | 0.00 | 0.00 | +0.00 |
| Max | 7.77 | 7.60 | -0.17 |

---

## Cost Analysis (5,000 articles)

| Model | Cost per Article | Total Cost | Savings |
|-------|------------------|------------|---------|
| Claude 3.5 Sonnet | $0.009 | $45.00 | - |
| Gemini 1.5 Pro | $0.00018 | $0.90 | $44.10 (96%) |

*Gemini pricing assumes Cloud Billing enabled (Tier 1: 150 RPM)*

---

## Sample Article Comparisons

Comparing 92 Claude articles vs 92 Gemini articles

**Matched articles**: 92 (analyzed successfully by both models)

Showing 5 articles with largest score disagreement:

### Sample 1 - Disagreement: 4.69

**Title**: Los países en desarrollo pagan demasiado por endeudarse

**Excerpt**: Si los recursos de las instituciones financieras internacionales siguen destinándose al reembolso de los acreedores comerciales en lugar de a inversiones nacionales, aumentará la probabilidad de un im...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 5.07 | connection | The article highlights a potential problem where resources are being diverted from national investme... |
| Gemini | 0.38 | not_uplifting | The article describes a negative financial trend where resources in developing countries are diverte... |

### Sample 2 - Disagreement: 4.67

**Title**: How Growth Experts Can Adapt to AI Without Losing Their Soul

**Excerpt**: “The real problem of humanity,” biologist E.O. Wilson once said, “is that we have Paleolithic emotions, medieval institutions, and godlike…Continue reading on Medium »...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 5.48 | connection | The article discusses how growth experts (presumably in business) can adapt to AI without losing the... |
| Gemini | 0.81 | not_uplifting | The article is a piece of commentary or advice for a specific professional group ('Growth Experts').... |

### Sample 3 - Disagreement: 4.50

**Title**: Barclays’ $800 Million Fintech Gamble Marks A Pivot From Wall Street To Main Street

**Excerpt**: The British bank’s purchase of fintech Best Egg signals a deeper push into U.S. consumer lending as part of its broader plan to bolster returns....

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 4.88 | not_uplifting | Barclays' investment in Best Egg indicates a potential expansion of consumer lending in the U.S., wh... |
| Gemini | 0.38 | not_uplifting | The article describes a corporate acquisition intended to increase profits for a bank. The action is... |

### Sample 4 - Disagreement: 4.42

**Title**: Zijn nog geen regels voor: politie Californië kan zelfrijdende auto niet bekeuren

**Excerpt**: Een auto in Californië maakte voor de ogen van agenten een illegale U-bocht. Toen zij de bestuurder wilden bekeuren, ontdekten de agenten dat er niemand achter het stuur zat. Het ging namelijk om een ...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 5.19 | connection | A self-driving car (Waymo) is operating in a real-world environment, demonstrating progress in auton... |
| Gemini | 0.77 | not_uplifting | The article describes a technological novelty creating a regulatory problem. It highlights a failure... |

### Sample 5 - Disagreement: 4.25

**Title**: Claude Haiku 4.5 vs. GLM

**Excerpt**: Article URL: https://blog.kilocode.ai/p/mini-models-battle-claude-haiku-45 Comments URL: https://news.ycombinator.com/item?id=45741464 Points: 1 # Comments: 0...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 5.14 | connection | The article discusses the performance of AI models, Claude Haiku 4.5 and GLM. This represents progre... |
| Gemini | 0.89 | not_uplifting | The article is a technical comparison of two commercial AI models. This action serves a niche techni... |

---

## Next Steps

1. Review the tier distribution and score statistics
2. Examine sample articles with large disagreements
3. If distributions are similar, proceed with Gemini for cost savings
4. If distributions differ, consider:
   - Refining the prompt for better clarity
   - Adding more examples to the prompt
   - Using Claude for ground truth (higher cost but higher quality)
