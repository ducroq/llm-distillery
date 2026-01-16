# Prefilter Evaluation Report

**Date:** 2026-01-16
**Filter:** sustainability_technology v2
**Prefilter Version:** v2.1

## Executive Summary

Comprehensive evaluation of the sustainability_technology prefilter revealed that **v2.1 performs well** with 88.2% FP block rate and 89.0% TP pass rate. Attempts to improve FP blocking through additional AI/ML patterns (v2.2) caused unacceptable TP regression without improving FP block rate.

**Recommendation:** Keep v2.1 prefilter. The remaining 12% FPs are edge cases best handled by the oracle.

## Test Data

| Dataset | Count | Source |
|---------|-------|--------|
| False Positives | 271 | Manually reviewed from production |
| True Positives | 300 | Sampled from medium-tier (excluding FPs) |

Test data frozen in:
- `ground_truth/` - 271 FPs
- `true_positives/frozen_true_positives.json` - 300 TPs

## Results

### Prefilter v2.1 Performance

```
                    Blocked    |    Passed
                 --------------+--------------
    FP (block)  |     239      |       32      |  n=271
    TP (pass)   |      33      |      267      |  n=300
```

| Metric | Value |
|--------|-------|
| FP Block Rate | **88.2%** |
| TP Pass Rate | **89.0%** |
| Accuracy | 88.6% |

### v2.2 Experiment (Rejected)

Added expanded AI/ML patterns targeting the 65% of FPs that were ML papers.

| Metric | v2.1 | v2.2 | Change |
|--------|------|------|--------|
| FP Block Rate | 88.2% | 88.2% | +0.0% |
| TP Pass Rate | 89.0% | 76.3% | **-12.7%** |

**Rejected:** No FP improvement, significant TP regression.

## FP Analysis

### Category Breakdown

| Category | Count | % |
|----------|-------|---|
| ai_ml_infrastructure | 176 | 64.9% |
| developer_tutorial | 28 | 10.3% |
| healthcare | 20 | 7.4% |
| reddit_programming | 14 | 5.2% |
| smartphone_review | 9 | 3.3% |
| consumer_electronics | 8 | 3.0% |
| Other | 16 | 5.9% |

### Why 32 FPs Still Pass

The remaining FPs have characteristics that make them hard to distinguish from legitimate sustainability content at the keyword level:

1. **Misleading keywords**: "efficient", "sustainable", "resource" in ML context
2. **Sustainability-adjacent topics**: Healthcare, infrastructure optimization
3. **Edge cases**: Articles that mention sustainability briefly but aren't about it

Examples:
- "Sustainable Dialogue Breakdown Management" - "sustainable" means maintainable
- "Resource-Efficient Arrhythmia Detection" - "resource" means compute, not natural

## Alternative Approaches Evaluated

### 1. Clickbait Filter (Rejected)

Tested `valurank/distilroberta-clickbait` model.

**Result:** Model classified nearly all content (including academic papers) as clickbait. Wrong domain - trained on sensational news, not technical content.

### 2. Semantic Prefilter (Not Recommended for This Use Case)

Tested BART zero-shot classification at threshold 0.35.

| Metric | Value |
|--------|-------|
| FP Block Rate | 80.4% |
| TP Pass Rate | **44.7%** |

**Result:** Too aggressive. Blocks more TPs than FPs. The zero-shot classifier can't distinguish between ML papers about sustainability vs ML papers that use sustainability-like terminology.

## Conclusions

1. **Prefilter v2.1 is near-optimal** for keyword-based filtering
2. **Further improvement requires oracle refinement**, not prefilter expansion
3. **The 12% remaining FPs** are semantically ambiguous and need LLM judgment
4. **Model-based prefilters** (clickbait, semantic) don't improve the trade-off

## Recommendations

1. **Keep v2.1 prefilter** - It achieves good balance (88% FP block, 89% TP pass)
2. **Focus on oracle prompt** - Add disambiguation for ML terminology:
   - "communication-efficient" ≠ energy efficiency
   - "sustainable" in ML ≠ environmental sustainability
   - "resource-efficient" (compute) ≠ natural resources
3. **Accept the 12% FP leak** - These cost oracle inference but are correctly handled

## Files

| File | Description |
|------|-------------|
| `compare_prefilters.py` | A/B test script for prefilter versions |
| `true_positives/frozen_true_positives.json` | 300 frozen TPs for testing |
| `ground_truth/` | 271 manually identified FPs |
