# Model Calibration Report: Ai Augmented Practice

**Generated**: 2025-11-08 19:33:37
**Prompt**: `filters\ai_augmented_practice\v1\prompt-compressed.md`
**Sample Size**: 17 articles (labeled by both models)
**Models**: Claude 3.5 Sonnet vs Gemini 1.5 Pro

---

## Pre-filter Statistics

- **Total Articles Sampled**: 100
- **Passed Pre-filter**: 17 (17.0%)
- **Blocked by Pre-filter**: 83 (83.0%)

**Block Reasons**:

- not_ai_related: 80 (96.4%)
- model_benchmark: 2 (2.4%)
- generic_overview: 1 (1.2%)

**Cost Savings**: Pre-filter blocked 83 articles, saving expensive LLM API calls.
For ground truth generation (5,000 articles), this would save ~4150 API calls.

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

Comparing 17 Claude articles vs 17 Gemini articles

**Matched articles**: 17 (analyzed successfully by both models)

Showing 5 articles with largest score disagreement:

### Sample 1 - Disagreement: 0.00

**Title**: An AI dose engine for fast carbon ion treatment planning

**Excerpt**: arXiv:2510.11271v1 Announce Type: new Abstract: Monte Carlo (MC) simulations provide gold-standard accuracy for carbon ion therapy dose calculations but are computationally intensive. Analytical penci...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 0.00 | unknown | ... |
| Gemini | 0.00 | unknown | ... |

### Sample 2 - Disagreement: 0.00

**Title**: Manual2Skill: Learning to Read Manuals and Acquire Robotic Skills for Furniture Assembly Using Visio

**Excerpt**: arXiv:2502.10090v3 Announce Type: replace Abstract: Humans possess an extraordinary ability to understand and execute complex manipulation tasks by interpreting abstract instruction manuals. For robot...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 0.00 | unknown | ... |
| Gemini | 0.00 | unknown | ... |

### Sample 3 - Disagreement: 0.00

**Title**: Superposition Yields Robust Neural Scaling

**Excerpt**: arXiv:2505.10465v3 Announce Type: replace Abstract: The success of today's large language models (LLMs) depends on the observation that larger models perform better. However, the origin of this neural...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 0.00 | unknown | ... |
| Gemini | 0.00 | unknown | ... |

### Sample 4 - Disagreement: 0.00

**Title**: Matryoshka Pilot: Learning to Drive Black

**Excerpt**: arXiv:2410.20749v2 Announce Type: replace Abstract: Despite the impressive generative abilities of black-box large language models (LLMs), their inherent opacity hinders further advancements in capabi...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 0.00 | unknown | ... |
| Gemini | 0.00 | unknown | ... |

### Sample 5 - Disagreement: 0.00

**Title**: OraPlan

**Excerpt**: arXiv:2510.23870v1 Announce Type: new Abstract: We present OraPlan-SQL, our system for the Archer NL2SQL Evaluation Challenge 2025, a bilingual benchmark requiring complex reasoning such as arithmetic...

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
