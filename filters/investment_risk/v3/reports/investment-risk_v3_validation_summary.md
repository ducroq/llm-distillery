# Investment-Risk v3 Validation Summary

## Overview
Analyzed 4,654 scored articles from `datasets/scored/investment-risk_v3/investment-risk/` using the v3 postfilter classification logic.

## Key Findings

### Tier Distribution

| Tier | Count | Percentage | Expected Range | Status |
|------|-------|------------|----------------|--------|
| RED | 69 | 1.48% | 5-10% | **BELOW EXPECTED** |
| YELLOW | 550 | 11.82% | 10-20% | WITHIN EXPECTED |
| GREEN | 0 | 0.00% | 5-10% | **BELOW EXPECTED** |
| BLUE | 915 | 19.66% | 20-30% | **BELOW EXPECTED** |
| NOISE | 3,120 | 67.04% | 40-50% | **ABOVE EXPECTED** |

### Actionable Signal Rate
- **Actionable**: 619 articles (13.30%) - RED + YELLOW + GREEN combined
- **Non-actionable**: 4,035 articles (86.70%) - BLUE + NOISE combined

## Issues Identified

### 1. Very Low RED Tier Rate (1.48%)
The RED tier is catching only 69 articles out of 4,654, which is significantly below the expected 5-10% range. This suggests:
- Thresholds for RED tier may be too strict
- Oracle may not be scoring macro/credit/systemic risks high enough
- Missing high-risk content that should trigger RED classification

**Sample RED articles:**
- "Denmark reports repeated Russian naval provocations in its straits" (macro=7, systemic=6)
- "What Gaza looks like today, after two years of war" (macro=7, systemic=7)
- Geopolitical tensions and conflict scenarios are being caught, but general market risk signals may be missed

### 2. Zero GREEN Tier Articles (0.00%)
No articles were classified as GREEN (value opportunities). This indicates:
- Oracle may not be identifying fear/sentiment extremes (need sentiment >= 7)
- Or valuation risk is not being scored low enough (need valuation <= 3)
- May indicate a gap in the input data - not enough contrarian/value opportunity content

**GREEN tier requirements:**
- Sentiment extremes >= 7 (fear level)
- Valuation risk <= 3 (cheap)
- Evidence quality >= 6
- Actionability >= 5

### 3. High NOISE Rate (67.04%)
Two-thirds of articles are classified as NOISE, significantly above the 40-50% expected range. This is driven by:
- Low evidence quality scores (< 4) across many articles
- Academic papers (arxiv), car reviews, tech content getting scored as low quality
- Oracle may be too aggressive in filtering out content

**NOISE trigger:** Evidence quality < 4

### 4. Low Actionable Signal Rate (13.30%)
Only 13.30% of articles are actionable (RED + YELLOW + GREEN), which is at the lower end of what would be useful for a production system.

## Root Cause Analysis

### Oracle Scoring Patterns
Based on the examples, the oracle appears to be:

1. **Conservative on Risk Dimensions**: Most articles score 1-3 on macro risk, credit stress, and systemic risk, even when discussing economic topics
2. **Strict on Evidence Quality**: Many articles get evidence scores of 2-3, immediately classifying them as NOISE
3. **Rare High Scores**: Very few articles receive scores of 7+ on any dimension

### Input Data Characteristics
The scored dataset contains:
- Scientific papers (arxiv) - appropriately filtered as NOISE
- General news content - mixed scoring
- Financial/economic content - some captured in YELLOW tier
- Limited high-risk content - explains low RED rate
- Limited contrarian/value content - explains zero GREEN rate

## Recommendations

### Immediate Actions

1. **Review RED Thresholds**
   - Current: macro >= 7 OR credit >= 7 OR systemic >= 8
   - Consider: macro >= 6 OR credit >= 6 OR systemic >= 7
   - This would capture more warning signals earlier

2. **Investigate GREEN Tier Absence**
   - Review oracle scoring patterns for sentiment extremes
   - Check if input data contains contrarian/value opportunity content
   - Consider adjusting GREEN thresholds:
     - Sentiment >= 6 (instead of 7)
     - Valuation <= 4 (instead of 3)

3. **Reduce NOISE Over-filtering**
   - Current: evidence < 4 triggers NOISE
   - Consider: evidence < 3 triggers NOISE
   - This would allow more borderline content into BLUE tier

### Longer-term Improvements

1. **Oracle Calibration**
   - Review oracle prompt and examples to encourage more varied scoring
   - Ensure oracle is not being too conservative on risk dimensions
   - Add more examples of high-risk scenarios in the oracle training

2. **Input Data Curation**
   - Pre-filter obvious NOISE sources (arxiv, automotive, etc.) before scoring
   - Focus oracle capacity on genuinely financial/economic content
   - This would improve signal-to-noise ratio

3. **Threshold Optimization**
   - Conduct A/B testing with different threshold configurations
   - Analyze false positives/negatives with domain experts
   - Optimize for desired actionable signal rate (target 20-30%)

## Validation Assessment

**Overall Rating:** NEEDS REVIEW

- Only 1 out of 5 tiers within expected ranges
- Multiple red flags detected
- Distribution significantly deviates from expected patterns

**However:**
- YELLOW tier is performing well (11.82% - within range)
- BLUE tier is close to expected range (19.66% vs 20-30%)
- Classification logic appears to be working correctly
- The issue is primarily with oracle scoring patterns, not postfilter logic

## Next Steps

1. Review oracle scoring examples for RED tier scenarios
2. Investigate why GREEN tier has zero hits
3. Consider adjusting NOISE threshold (evidence < 3 instead of < 4)
4. Test with a smaller, curated dataset of known high-risk content
5. Re-run validation after any threshold adjustments

---

**Generated:** 2025-11-18
**Dataset:** investment-risk_v3 (4,654 articles)
**Filter:** investment-risk v3 postfilter
**Validation Script:** validate_investment_risk_v3.py
