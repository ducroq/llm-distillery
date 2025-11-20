# Investment-Risk v3 Filter - Complete Validation Report

**Date:** 2025-11-18
**Dataset:** 4,654 scored articles from `datasets/scored/investment-risk_v3/investment-risk/`
**Filter Version:** v3
**Oracle Model:** gemini-flash-api-batch

---

## Executive Summary

The investment-risk v3 filter was validated by analyzing 4,654 scored articles and classifying them using the v3 postfilter logic. The analysis reveals **significant issues** with oracle scoring patterns that result in suboptimal tier distribution.

**Key Findings:**
- Only **1.48%** of articles classified as RED (expected 5-10%)
- **Zero** articles classified as GREEN (expected 5-10%)
- **67.04%** of articles classified as NOISE (expected 40-50%)
- Overall actionable signal rate: **13.30%** (below optimal range)

**Root Cause:** Oracle is scoring dimensions too conservatively, with most articles receiving scores of 1-3 across all risk dimensions, preventing them from reaching action thresholds.

**Recommendation:** **NEEDS REVIEW** - Oracle calibration required before production deployment.

---

## Tier Distribution Analysis

### Overall Results

| Tier | Count | Percentage | Expected Range | Status |
|------|-------|------------|----------------|--------|
| **RED** | 69 | 1.48% | 5-10% | BELOW EXPECTED |
| **YELLOW** | 550 | 11.82% | 10-20% | ✓ WITHIN EXPECTED |
| **GREEN** | 0 | 0.00% | 5-10% | BELOW EXPECTED |
| **BLUE** | 915 | 19.66% | 20-30% | SLIGHTLY BELOW |
| **NOISE** | 3,120 | 67.04% | 40-50% | ABOVE EXPECTED |

### Actionable Signal Rate

- **Actionable** (RED + YELLOW + GREEN): 619 articles (13.30%)
- **Non-actionable** (BLUE + NOISE): 4,035 articles (86.70%)

**Assessment:** Only 1 out of 5 tiers within expected ranges.

---

## Score Distribution Analysis

### Overall Dimensional Statistics

| Dimension | Mean | Median | Min | Max | StdDev |
|-----------|------|--------|-----|-----|--------|
| macro_risk_severity | 2.25 | 2.00 | 0 | 8 | 1.54 |
| credit_market_stress | 1.82 | 2.00 | 0 | 7 | 0.96 |
| market_sentiment_extremes | 2.01 | 2.00 | 0 | 7 | 1.19 |
| valuation_risk | 1.97 | 2.00 | 0 | 8 | 1.15 |
| policy_regulatory_risk | 2.50 | 2.00 | 0 | 9 | 1.84 |
| systemic_risk | 2.11 | 2.00 | 0 | 8 | 1.34 |
| evidence_quality | 3.47 | 3.00 | 0 | 7 | 1.66 |
| actionability | 2.30 | 2.00 | 0 | 7 | 1.52 |

**Key Observation:** All risk dimensions are heavily skewed low (mean < 3), indicating oracle conservatism.

### Critical Insights

#### 1. Low Variance Across Dimensions
The following dimensions show very low variance, suggesting oracle under-utilization:
- credit_market_stress: stdev=0.96
- market_sentiment_extremes: stdev=1.19
- valuation_risk: stdev=1.15
- systemic_risk: stdev=1.34

#### 2. Score Concentration at Low Values
Looking at the most common scores:
- **45-46%** of articles score **1** on most dimensions
- **13-29%** of articles score **2** on most dimensions
- Only **1-2%** score 7 or higher on risk dimensions

#### 3. Evidence Quality Distribution
- **57.76%** of articles have evidence quality < 4 (automatically classified as NOISE)
- Only **34.14%** have evidence quality >= 5 (eligible for actionable tiers)
- This is the primary driver of high NOISE rate

---

## Tier-Specific Score Analysis

### RED Tier (69 articles, 1.48%)
**Average Scores:**
- Macro risk: 7.01
- Credit stress: 4.55
- Systemic risk: 5.97
- Evidence quality: 6.25
- Actionability: 5.68

**Assessment:** Articles that do reach RED tier are appropriately classified with high macro/systemic risk.

**Example:**
- "Denmark reports repeated Russian naval provocations in its straits" (macro=7, systemic=6)
- "What Gaza looks like today, after two years of war" (macro=7, systemic=7)

### YELLOW Tier (550 articles, 11.82%)
**Average Scores:**
- Macro risk: 5.25
- Credit stress: 3.28
- Valuation risk: 3.65
- Evidence quality: 5.93
- Actionability: 5.01

**Assessment:** Performing well - capturing moderate risk signals appropriately.

**Example:**
- "More than 2,700 stores are closing across the US this year" (macro=5, evidence=6)
- "Price and Volume Divergence in China's Real Estate Markets" (macro=6, credit=6, evidence=7)

### GREEN Tier (0 articles, 0.00%)
**Why zero hits?**

GREEN tier requires:
- Sentiment extremes >= 7 (fear level)
- Valuation risk <= 3 (cheap)
- Evidence quality >= 6
- Actionability >= 5

**Analysis:**
- Only **6 articles** (0.13%) have sentiment >= 7
- **4,172 articles** (89.64%) have valuation <= 3
- **Zero articles** meet BOTH conditions simultaneously

**Root Cause:** Oracle is not identifying sentiment extremes, even when valuations are scored low.

### BLUE Tier (915 articles, 19.66%)
**Average Scores:**
- Evidence quality: 5.26
- Actionability: 3.50
- All risk dimensions: 2-3 range

**Assessment:** Capturing educational content appropriately - decent quality but not urgent.

### NOISE Tier (3,120 articles, 67.04%)
**Average Scores:**
- Evidence quality: 2.45
- All dimensions: 1-1.5 range

**Assessment:** Dominated by low evidence quality scores. Many legitimate news articles are being scored too low.

---

## Critical Threshold Analysis

### RED Tier Thresholds
Current requirements: macro >= 7 OR credit >= 7 OR systemic >= 8

**Reality:**
- Macro risk >= 7: **68 articles** (1.46%)
- Credit stress >= 7: **2 articles** (0.04%)
- Systemic risk >= 8: **1 article** (0.02%)

**Problem:** Almost no articles meet these thresholds because oracle rarely scores above 6.

### GREEN Tier Thresholds
Current requirements: sentiment >= 7 AND valuation <= 3

**Reality:**
- Sentiment >= 7: **6 articles** (0.13%)
- Valuation <= 3: **4,172 articles** (89.64%)
- **Both conditions: 0 articles**

**Problem:** Oracle is not identifying extreme fear/sentiment, making GREEN tier impossible to trigger.

### Evidence Quality Gate
Current: Evidence < 4 triggers NOISE

**Reality:**
- **2,688 articles** (57.76%) have evidence < 4 and are classified as NOISE
- This is the single largest factor in NOISE classification

---

## Example Articles by Tier

### RED Examples

**Example 1: Geopolitical Risk**
- Title: "Denmark reports repeated Russian naval provocations in its straits"
- Source: global_news_reuters
- Scores: macro=7, credit=4, systemic=6, evidence=7, actionability=6
- Signal Strength: 5.15
- Reason: High risk signal (macro=7.0, credit=4.0, systemic=6.0)

**Example 2: Conflict Zone**
- Title: "What Gaza looks like today, after two years of war"
- Source: global_news_reuters
- Scores: macro=7, credit=5, systemic=7, evidence=7, actionability=6
- Signal Strength: 5.90
- Reason: High risk signal (macro=7.0, credit=5.0, systemic=7.0)

### YELLOW Examples

**Example 1: Economic Weakness**
- Title: "More than 2,700 stores are closing across the US this year"
- Source: professional_business_business_insider
- Scores: macro=5, credit=3, valuation=5, evidence=6, actionability=5
- Signal Strength: 4.10
- Reason: Warning signal (macro=5.0, credit=3.0, valuation=5.0)

**Example 2: Structural Risk**
- Title: "Price and Volume Divergence in China's Real Estate Markets"
- Source: economics_nber_working_papers
- Scores: macro=6, credit=6, valuation=6, evidence=7, actionability=5
- Signal Strength: 5.80
- Reason: Warning signal (macro=6.0, credit=6.0, valuation=6.0)

### BLUE Examples

**Example 1: Tech Analysis**
- Title: "Security and AI's potential to protect"
- Source: industry_intelligence_mckinsey_insights
- Scores: All dimensions=3, evidence=6, actionability=4
- Signal Strength: 3.20
- Reason: Educational content (evidence=6.0, actionability=4.0)

### NOISE Examples

**Example 1: Academic Paper**
- Title: "Topics in Probability, Parametric Estimation and Stochastic Calculus"
- Source: science_arxiv_math
- Scores: All dimensions=1, evidence=2, actionability=1
- Signal Strength: 0.00
- Reason: Low evidence quality (2.0 < 4)

**Example 2: Car News**
- Title: "Iconic Nissan Skyline set for 2027 rebirth"
- Source: automotive_autoexpress_uk
- Scores: All dimensions=2, evidence=3, actionability=2
- Signal Strength: 0.00
- Reason: Low evidence quality (3.0 < 4)

---

## Red Flags Identified

### 1. Actionable Signal Rate Too Low (13.30%)
With only 13.30% of articles being actionable, the filter may not provide sufficient signals for investment decision-making.

### 2. RED Tier Rate Critically Low (1.48%)
At 1.48%, the RED tier is missing most high-risk scenarios. Expected rate should be 5-10%.

### 3. NOISE Rate Excessively High (67.04%)
Two-thirds of all articles are classified as noise, suggesting:
- Oracle is too strict on evidence quality
- Many legitimate articles are being filtered out
- Potential loss of valuable signals

### 4. GREEN Tier Complete Absence (0.00%)
Zero articles identified as value opportunities indicates:
- Oracle not detecting market fear/sentiment extremes
- Missing contrarian signals
- Potential gap in input data sources

---

## Root Cause Analysis

### Oracle Scoring Behavior

The oracle (gemini-flash-api-batch) exhibits the following patterns:

1. **Conservative Risk Scoring:**
   - Scores 1-3 for most articles across all risk dimensions
   - Rarely assigns scores of 7 or higher
   - Only extreme scenarios (war, geopolitical crises) get high scores

2. **Strict Evidence Quality Standards:**
   - 57.76% of articles scored below 4 on evidence quality
   - Academic papers, news aggregation, opinion pieces scored very low
   - May be filtering out legitimate financial news

3. **Poor Sentiment Detection:**
   - Only 0.13% of articles scored 7+ on sentiment extremes
   - Not identifying market fear, panic, or euphoria signals
   - Critical gap for contrarian value identification

4. **Valuation Paradox:**
   - 89.64% of articles scored 3 or below on valuation risk
   - Most things scored as "cheap" or "fairly valued"
   - But combined with low sentiment scores, prevents GREEN classification

### Input Data Characteristics

The scored dataset includes:
- **Science/Academic:** arxiv papers (appropriately filtered as NOISE)
- **General News:** Reuters, NPR, El Pais (mixed results)
- **Financial:** Business Insider, InfoMoney, NBER (better signal capture)
- **Niche:** Automotive, utilities, Dutch news (low relevance)

**Observation:** Dataset contains significant non-financial content that dilutes signal quality.

---

## Recommendations

### Immediate Actions (Before Production)

#### 1. Oracle Calibration - HIGH PRIORITY
The oracle needs recalibration to:
- Score risk dimensions more aggressively (wider distribution)
- Identify sentiment extremes more effectively
- Apply more nuanced evidence quality scoring

**Specific Changes:**
- Review oracle prompt for risk dimension examples
- Add examples of sentiment extremes (fear, panic, euphoria)
- Clarify evidence quality criteria (distinguish news quality vs relevance)

#### 2. Threshold Adjustments - MEDIUM PRIORITY
Consider lowering some thresholds to capture more signals:

**RED Tier:**
- Current: macro >= 7 OR credit >= 7 OR systemic >= 8
- Proposed: macro >= 6 OR credit >= 6 OR systemic >= 7
- Impact: Would increase RED rate from 1.48% to ~4-5%

**GREEN Tier:**
- Current: sentiment >= 7 AND valuation <= 3
- Proposed: sentiment >= 6 AND valuation <= 4
- Impact: Would enable GREEN tier to capture some signals

**NOISE Gate:**
- Current: evidence < 4
- Proposed: evidence < 3
- Impact: Would reduce NOISE rate from 67% to ~43%

#### 3. Input Data Pre-filtering - MEDIUM PRIORITY
Add prefiltering to remove obvious noise before oracle scoring:
- Filter out arxiv/academic papers
- Filter out automotive/sports/entertainment sources
- Focus oracle capacity on financial/economic content
- **Expected Impact:** Reduce NOISE rate, increase signal density

### Longer-term Improvements

#### 4. Oracle Model Upgrade - CONSIDER
Evaluate whether gemini-flash is appropriate for this task:
- Consider gemini-pro or claude-sonnet for better reasoning
- Test with different temperature settings
- Compare scoring variance across models

#### 5. Two-Stage Filtering - CONSIDER
Implement pre-filter + post-filter approach:
- **Stage 1:** Fast relevance filter (remove obvious noise)
- **Stage 2:** Deep oracle scoring on remaining articles
- **Benefit:** Better resource allocation, higher signal quality

#### 6. Calibration Dataset - RECOMMENDED
Create a gold-standard calibration set:
- 100-200 manually labeled articles with expert scores
- Use for oracle prompt engineering
- Regular validation against expert scores
- Track oracle drift over time

#### 7. Threshold Optimization - RECOMMENDED
Run systematic optimization:
- Test multiple threshold configurations
- Measure precision/recall for each tier
- Optimize for target actionable signal rate (20-30%)
- A/B test with domain experts

---

## Production Readiness Assessment

### Current Status: NOT READY FOR PRODUCTION

**Reasons:**
1. RED tier severely under-performing (1.48% vs 5-10% expected)
2. GREEN tier completely non-functional (0% vs 5-10% expected)
3. NOISE rate too high (67% vs 40-50% expected)
4. Actionable signal rate below useful threshold (13% vs 20-30% target)

### Required Before Production:

**Must Have:**
- [ ] Oracle recalibration to achieve better score distributions
- [ ] GREEN tier functionality restored (at least 3-5% hit rate)
- [ ] RED tier rate increased to 4-6% minimum
- [ ] NOISE rate reduced below 55%

**Should Have:**
- [ ] Input data pre-filtering implemented
- [ ] Threshold optimization completed
- [ ] Expert validation on sample of 100 articles

**Nice to Have:**
- [ ] Calibration dataset created
- [ ] Two-stage filtering implemented
- [ ] Alternative oracle model tested

---

## Validation Methodology

### Data Collection
- Source: `datasets/scored/investment-risk_v3/investment-risk/scored_batch_*.jsonl`
- Total articles: 4,654
- Batch files: 94
- Date range: September-November 2025

### Classification Process
- Postfilter: `filters/investment-risk/v3/postfilter.py`
- Method: `InvestmentRiskPostFilter.classify()`
- Tier logic: Based on dimensional scores and thresholds

### Analysis Scripts
1. `validate_investment_risk_v3.py` - Main validation and tier distribution
2. `analyze_v3_score_distributions.py` - Dimensional score analysis

### Validation Outputs
- `investment-risk_v3_validation_report.txt` - Detailed tier examples
- `investment-risk_v3_score_distributions.txt` - Statistical analysis
- `investment-risk_v3_validation_summary.md` - Summary findings
- `investment-risk_v3_VALIDATION_REPORT.md` - This comprehensive report

---

## Conclusion

The investment-risk v3 postfilter logic is **correctly implemented** and functioning as designed. However, the **oracle scoring patterns** are preventing the filter from achieving its intended distribution.

**The core issue is not the postfilter, but the oracle's conservative scoring behavior.**

With oracle recalibration and threshold adjustments, this filter has the potential to provide valuable investment risk signals. However, in its current state, it is **not recommended for production use** until the identified issues are addressed.

### Next Steps:
1. Review oracle prompt and examples for risk scoring
2. Investigate sentiment extreme detection failure
3. Implement input data pre-filtering
4. Test threshold adjustments with sample data
5. Re-validate after changes

---

**Report Generated:** 2025-11-18
**Validation Scripts:** Available in repository root
**Full Data:** 4,654 articles analyzed across 94 batch files
