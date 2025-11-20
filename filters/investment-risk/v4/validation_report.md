# Investment-Risk v4 Validation Report

**Date**: 2025-11-18
**Test Sample**: 100 articles (random sample from master_dataset)
**Oracle**: Gemini Flash 1.5
**Purpose**: Validate v4 calibration fixes compared to v3 validation (4,654 articles)

---

## Executive Summary

✅ **PARTIAL SUCCESS**: v4 calibration improved evidence_quality and NOISE rate significantly, but dimensional risk scores remain too conservative.

### Key Findings

1. **NOISE rate improved**: 50% (v4) vs 67% (v3) → **17 percentage point improvement** ✅
2. **Evidence quality improved**: mean 3.86 (v4) vs 2.50 (v3) → **+1.36 improvement** ✅
3. **RED tier still too low**: 2% (v4) vs 1.48% (v3) → minimal improvement ⚠️
4. **GREEN tier still broken**: 0% (v4) vs 0% (v3) → no improvement ❌
5. **Risk dimensions barely improved**: Most scores still clustering in 1-3 range ⚠️

---

## Tier Distribution Analysis

| Tier   | v4 Count | v4 % | v3 % (4,654 articles) | Target | Status |
|--------|----------|------|----------------------|--------|--------|
| **RED**    | 2        | 2.0% | 1.48%                | 5-10%  | ⚠️ Improved but below target |
| **YELLOW** | 12       | 12.0% | -                    | 15-25% | ✅ Within range |
| **GREEN**  | 0        | 0.0% | 0%                   | 5-10%  | ❌ Still broken |
| **BLUE**   | 36       | 36.0% | -                    | 15-25% | ⚠️ Too high |
| **NOISE**  | 50       | 50.0% | 67%                  | 40-50% | ✅ Much improved! |

**Interpretation**:
- **NOISE reduction** (67% → 50%): Evidence quality calibration worked well
- **RED still too low** (1.48% → 2%): Only marginal improvement despite calibration examples
- **GREEN completely missing**: Valuation + sentiment extreme logic not triggering
- **BLUE too high** (36%): Articles passing evidence threshold but not meeting tier criteria

---

## Dimensional Score Analysis

### v3 vs v4 Comparison

| Dimension | v3 Mean | v4 Mean | Change | 7+ Scores | Status |
|-----------|---------|---------|--------|-----------|--------|
| **macro_risk_severity** | 1.82 | 2.30 | **+0.48** | 2 (2%) | ⚠️ Slight improvement |
| **credit_market_stress** | 1.92 | 1.85 | -0.07 | 0 (0%) | ❌ Worse |
| **market_sentiment_extremes** | 2.12 | 1.92 | -0.20 | 0 (0%) | ❌ Worse |
| **valuation_risk** | 2.18 | 1.92 | -0.26 | 0 (0%) | ❌ Worse |
| **policy_regulatory_risk** | 2.14 | 2.55 | **+0.41** | 4 (4%) | ⚠️ Improved |
| **systemic_risk** | 1.98 | 2.17 | +0.19 | 0 (0%) | ⚠️ Slight improvement |
| **evidence_quality** | 2.50 | 3.86 | **+1.36** ✅ | 1 (1%) | ✅ Big improvement! |
| **actionability** | 2.36 | 2.30 | -0.06 | 0 (0%) | ❌ Slightly worse |

### Distribution Analysis

```
Dimension                      Mean   7+ scores  4-6 scores  <4 scores
------------------------------------------------------------------
macro_risk_severity            2.30      2         13          85
credit_market_stress           1.85      0          8          92
market_sentiment_extremes      1.92      0          6          94
valuation_risk                 1.92      0          9          91
policy_regulatory_risk         2.55      4         22          74
systemic_risk                  2.17      0         17          83
evidence_quality               3.86      1         49          50
actionability                  2.30      0         17          83
```

**Key Observations**:
- **85-94% of scores still < 4**: Oracle not using full 0-10 scale despite calibration examples
- **Only 2 articles scored macro >= 7**: Calibration examples (2008 crisis, COVID) didn't help enough
- **0 articles scored sentiment >= 7**: VIX/panic examples had no effect
- **Evidence quality**: 50% of articles now >= 4 (v3: 42%), showing relaxed criteria worked

---

## Calibration Effectiveness Assessment

### What Worked ✅

1. **Evidence quality criteria relaxation**
   - Added examples: "WSJ/Bloomberg/FT analysis is 6-8 evidence quality"
   - Result: Mean jumped from 2.50 → 3.86 (+1.36)
   - 50% of articles now pass evidence >= 4 threshold (was blocking 57.76% in v3)

2. **NOISE rate reduction**
   - Combination of better evidence scoring + prefilter
   - Result: 67% → 50% NOISE rate (17 point improvement)

### What Didn't Work ❌

1. **High-risk scenario examples**
   - Added examples: "2008 crisis = 9-10 macro, COVID = 8-9, energy crisis = 7-8"
   - Result: macro mean only 2.30, only 2 articles >= 7
   - **Conclusion**: Examples alone insufficient to shift oracle behavior

2. **Sentiment extreme examples**
   - Added examples: "VIX >30 = 7+, panic = 8-9, euphoria = 9-10"
   - Result: sentiment mean 1.92, ZERO articles >= 7
   - **Conclusion**: Oracle not applying sentiment scale correctly

3. **Calibration guidance header**
   - Added: "USE THE FULL 0-10 SCALE", "Score 7-10 for genuine risks"
   - Result: 85-94% of scores still < 4
   - **Conclusion**: Top-level guidance insufficient

---

## Root Cause Analysis

### Why is the Oracle Still Scoring Conservatively?

1. **Test sample bias**: 100 articles may not contain high-risk content
   - Random sample from Oct-Nov 2025 (no major financial crisis during this period)
   - Most articles are tech/science (arxiv, github, dev.to, electrek)
   - v3 validation was also random sample, so comparison is fair

2. **Calibration examples too abstract**:
   - Oracle sees "2008 crisis = 9-10" but current articles aren't crises
   - No concrete examples of "what scores 5-6" (moderate risk)
   - Missing examples for current/recent events (2024-2025)

3. **Conservative anchoring**:
   - Oracle may be interpreting current low-risk environment correctly
   - Scoring 2-3 for "no immediate risk" is technically accurate
   - Problem: Target distribution assumes 5-10% RED even in normal times

4. **Dimension interaction effects**:
   - Evidence quality improved → more articles pass to tier classification
   - But risk dimensions didn't improve → articles land in BLUE instead of RED/YELLOW
   - Result: BLUE inflated (36%), RED still low (2%)

---

## Sample Analysis

### RED Tier Articles (2%)

The 2 articles that scored RED likely had:
- macro_risk >= 7 OR credit_market >= 7 OR systemic >= 8
- evidence >= 5
- actionability >= 5

Without seeing article content, likely candidates:
- Financial policy articles (Fed, ECB, banking stress)
- Economic recession warnings
- Geopolitical risk with financial contagion

### BLUE Tier Articles (36%)

36 articles passed evidence threshold (4+) but didn't trigger RED/YELLOW/GREEN:
- Risk scores in 2-4 range (not high enough for RED/YELLOW)
- Valuation not extreme enough for GREEN
- Educational/framework content
- Historical analysis without immediate action

This suggests **calibration is working for evidence, but risk thresholds are too strict**.

---

## Recommendations

### Option A: Accept Current Distribution (Recommended if sample is representative)

If Oct-Nov 2025 genuinely had low financial risk:
- **Current scoring may be accurate** (mean 2-3 for low-risk environment)
- **Proceed to full 5K training run** with v4
- **Revalidate on known high-risk articles** (2008 crisis articles, COVID crash articles)
- **Accept that RED rate varies with actual market risk** (2% in calm periods, 10%+ in crisis)

### Option B: Further Calibration (If we need higher RED/YELLOW rates)

Add more concrete examples to prompt:
```markdown
**SCORING CALIBRATION (2024-2025 examples)**:

**Macro Risk 7-8**:
- "Fed holds rates at 5.5% amid sticky inflation, unemployment rising to 4.2%"
- "China property sector contagion spreading to regional banks, $2T debt concerns"

**Macro Risk 5-6**:
- "Yield curve inversion persists for 6 months, recession odds at 40%"
- "Energy prices surge 30% on geopolitical tensions, inflation concerns return"

**Sentiment 7-8**:
- "VIX spikes to 32 as tech selloff accelerates, put/call ratio at 1.2"
- "Market breadth negative: 80% of stocks below 50-day MA, fear gauge elevated"
```

### Option C: Adjust Tier Thresholds (Easiest fix)

Lower thresholds in postfilter (no retraining needed):
```yaml
RED:
  condition: "(macro >= 6 OR credit >= 6 OR systemic >= 7) AND evidence >= 5"  # Was: 7/7/8

YELLOW:
  condition: "(macro >= 4 OR credit >= 4 OR valuation >= 6) AND evidence >= 4"  # Was: 5-6/5-6/7-8
```

**Trade-off**: More RED/YELLOW signals, but potentially more false positives.

---

## Next Steps

### Immediate (Before 5K training run)

1. **Validate on known high-risk content**:
   - Create test set: 20 articles from 2008 crisis, 20 from COVID crash, 20 from energy crisis
   - Score with v4 oracle
   - Check if oracle gives 7-10 scores for historical crises

2. **If historical validation passes → Proceed with v4 as-is**
   - Accept that 2% RED rate is correct for Oct-Nov 2025 (low-risk period)
   - Train on 5K articles
   - Monitor tier distribution in production (will vary with market conditions)

3. **If historical validation fails → Create v5 with Option B fixes**
   - Add concrete 2024-2025 examples to each dimension
   - Add scoring rubric with current event anchors
   - Revalidate on 100-article sample

### Before Production Deployment

1. **Validate on time-series data**:
   - Score articles from Jan 2020 (pre-COVID), March 2020 (crash), Sept 2008 (Lehman)
   - Verify RED rate correlates with actual market stress

2. **Calibrate postfilter thresholds**:
   - Based on production feedback
   - Adjust RED/YELLOW/GREEN thresholds without retraining

3. **Monitor tier distribution drift**:
   - Track RED rate over time
   - Alert if RED rate deviates significantly from market risk indicators (VIX, credit spreads)

---

## Conclusion

**v4 calibration was PARTIALLY successful:**
- ✅ Evidence quality improved significantly (+1.36)
- ✅ NOISE rate reduced to target range (50%)
- ⚠️ Risk dimensions showed minimal improvement
- ❌ RED tier still below target (2% vs 5-10%)
- ❌ GREEN tier still non-functional (0%)

**Primary issue**: Oracle scoring conservatively is likely **correct behavior for Oct-Nov 2025** (low-risk period). The v3/v4 validation samples may not contain enough genuinely high-risk content to trigger RED signals.

**Recommended path**:
1. Validate v4 on historical high-risk articles (2008, 2020)
2. If historical validation passes → proceed with v4 for 5K training
3. Accept that RED rate will vary with actual market conditions (2-15% range)
4. Use postfilter threshold adjustments (not retraining) to tune sensitivity

---

**Validation completed**: 2025-11-18
**Sample size**: 100 articles
**Oracle**: Gemini Flash 1.5
**Next**: Historical validation on known high-risk content
