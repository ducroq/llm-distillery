# Calibration Reports

This directory contains model calibration reports comparing Claude vs Gemini for each semantic filter.

## Purpose

Calibration reports help you decide which LLM to use for large-scale labeling by comparing:
- Tier distributions
- Score statistics
- Quality adherence
- Cost tradeoffs

## Generating Reports

```bash
python -m ground_truth.calibrate_models \
    --prompt prompts/<filter>.md \
    --source ../content-aggregator/data/collected/articles.jsonl \
    --sample-size 100 \
    --output reports/<filter>_calibration.md
```

## Example Reports

- `sustainability_calibration.md` - Claude vs Gemini on sustainability filter
- `uplifting_calibration.md` - Claude vs Gemini on uplifting filter

## What's Included

Each calibration report contains:

1. **Executive Summary**
   - Tier distribution difference
   - Average score difference
   - Recommendation (which model to use)

2. **Tier Distribution Comparison**
   - Side-by-side tier percentages
   - Difference calculations

3. **Score Statistics**
   - Average, median, min, max scores
   - Comparison between models

4. **Cost Analysis**
   - Claude cost ($0.009/article)
   - Gemini cost ($0.00018/article)
   - Potential savings (usually ~96%)

5. **Sample Article Comparisons**
   - 5 articles with largest score disagreements
   - Shows both models' reasoning

6. **Next Steps**
   - Actionable guidance based on results

## Decision Matrix

| Tier Difference | Score Difference | Recommendation |
|----------------|------------------|----------------|
| < 10% | < 0.5 | Use Gemini (50x cheaper, very similar) |
| 10-20% | 0.5-1.0 | Use Gemini but spot-check with Claude |
| > 20% | > 1.0 | Use Claude or refine prompt |

## Best Practice

**Always calibrate before large-scale labeling!**

10 minutes testing 100 articles can save you $40+ on a 5,000-article batch by helping you choose the right model.
