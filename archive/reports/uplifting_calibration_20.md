# Model Calibration Report: Uplifting

**Generated**: 2025-10-25 21:57:22
**Prompt**: `prompts/uplifting.md`
**Sample Size**: 20 articles (labeled by both models)
**Models**: Claude 3.5 Sonnet vs Gemini 1.5 Pro

---

## Executive Summary

- **Tier Distribution Difference**: 50.0%
- **Average Score Difference**: 1.14
- **Claude Average Score**: 5.14
- **Gemini Average Score**: 4.01

**Recommendation**: Models differ significantly. **Use Claude** or refine prompt for better Gemini consistency.

---

## Tier Distribution Comparison

| Tier | Claude | Gemini | Difference |
|------|--------|--------|------------|
| Connection | 30.0% | 40.0% | +10.0% |
| Impact | 40.0% | 15.0% | -25.0% |
| Not Uplifting | 30.0% | 45.0% | +15.0% |

---

## Score Statistics

| Metric | Claude | Gemini | Difference |
|--------|--------|--------|------------|
| Average | 5.14 | 4.01 | -1.14 |
| Median | 6.14 | 4.89 | -1.25 |
| Min | 0.22 | 0.00 | -0.22 |
| Max | 8.90 | 8.26 | -0.64 |

---

## Cost Analysis (5,000 articles)

| Model | Cost per Article | Total Cost | Savings |
|-------|------------------|------------|---------|
| Claude 3.5 Sonnet | $0.009 | $45.00 | - |
| Gemini 1.5 Pro | $0.00018 | $0.90 | $44.10 (96%) |

*Gemini pricing assumes Cloud Billing enabled (Tier 1: 150 RPM)*

---

## Sample Article Comparisons

Comparing 20 Claude articles vs 20 Gemini articles

**Matched articles**: 20 (analyzed successfully by both models)

Showing 5 articles with largest score disagreement:

### Sample 1 - Disagreement: 5.57

**Title**: Désirée Nick will Ignoranten das wahre Berlin nahebringen

**Excerpt**: »Haben Sie Berlin überhaupt begriffen?« Désirée Nick, bekannt aus dem Dschungelcamp, wirbt gewohnt ruppig für die Hauptstadt – und sagt auch Ortsfremden wie Markus Söder, wo es in der Metropole langge...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 6.14 | connection | A cultural figure is taking active steps to bridge understanding between locals and visitors in Berl... |
| Gemini | 0.57 | not_uplifting | The article describes a celebrity promoting her personal, provocative view of a city's culture. This... |

### Sample 2 - Disagreement: 3.74

**Title**: This 16

**Excerpt**: Isaque Carvalho Borges experiences the urban heat island effect in his home of Palmas, Brazil, and he wants to do something about it...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 5.92 | connection | A young person is taking initiative to address urban heat island effects that impact their community... |
| Gemini | 2.18 | not_uplifting | The story identifies an individual who is motivated to take action on a significant environmental pr... |

### Sample 3 - Disagreement: 3.04

**Title**: Samsung is working on XR smart glasses with Warby Parker and Gentle Monster

**Excerpt**: As part of its Galaxy XR headset presentation, Samsung also briefly teased another wearable product. It's working in collaboration with two eyewear companies, Warby Parker and Gentle Monster, on AI-po...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 5.17 | connection | Multiple companies are collaborating to create accessible smart glasses that integrate AI assistance... |
| Gemini | 2.13 | not_uplifting | The article describes a corporate collaboration to develop a new consumer technology product, AI-pow... |

### Sample 4 - Disagreement: 1.51

**Title**: Microsoft increases the price of Xbox dev kits by $500

**Excerpt**: Players aren't the only ones facing higher price tags from Xbox. According to a report by The Verge, Microsoft has upped the cost of the Xbox Development Kit from $1,500 to $2,000. That's a 33 percent...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 1.51 | not_uplifting | This is primarily a story about price increases affecting game developers and consumers, with limite... |
| Gemini | 0.00 | not_uplifting | The article reports on a corporation increasing the price of essential hardware for game developers,... |

### Sample 5 - Disagreement: 1.39

**Title**: Workers and Employers Face Higher Health Insurance Costs

**Excerpt**: Article URL: https://www.nytimes.com/2025/10/22/health/workers-and-employers-face-higher-health-insurance-costs.html Comments URL: https://news.ycombinator.com/item?id=45667730 Points: 1 # Comments: 0...

| Model | Score | Tier | Reasoning |
|-------|-------|------|-----------|
| Claude | 1.39 | not_uplifting | The article describes rising healthcare costs affecting both employers and workers, indicating syste... |
| Gemini | 0.00 | not_uplifting | The article describes a negative economic event where the cost of health insurance is increasing for... |

---

## Next Steps

1. Review the tier distribution and score statistics
2. Examine sample articles with large disagreements
3. If distributions are similar, proceed with Gemini for cost savings
4. If distributions differ, consider:
   - Refining the prompt for better clarity
   - Adding more examples to the prompt
   - Using Claude for ground truth (higher cost but higher quality)
