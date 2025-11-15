# Investment Risk v2 Filter - Live Validation Report

**Date**: 2025-11-15
**Validator**: Claude Code (Automated)
**Oracle**: Gemini Flash 1.5
**Data Source**: `master_dataset_20251026_20251029.jsonl` (fresh, unused data)
**Sample Size**: 30 articles (random sample, seed=42)

---

## Executive Summary

### VERDICT: **ISSUES FOUND** - False Positive Rate of 10%

The investment-risk v2 filter with inline filters pattern successfully processed all 30 fresh articles with 100% scoring success rate. However, the filter exhibits a **10% false positive rate**, incorrectly classifying 3 academic research papers as YELLOW signals when they should be NOISE.

### Key Findings

✅ **Strengths:**
- 100% scoring success rate (30/30 articles)
- Excellent NOISE detection for off-topic content (sports, food, entertainment, healthcare)
- Legitimate macro risk signals correctly identified (Turkish political risk, US beef prices/policy, utility regulation)
- All dimensional scores within valid 0-10 range
- No false negatives detected

❌ **Issues:**
- **10% false positive rate** (3/30 articles)
- Academic research papers bypass inline filters
- Filter fails to recognize purely theoretical/academic content despite inline filter instructions

---

## Detailed Results

### Success Rate: 30/30 (100%)

All articles were successfully scored by the oracle. No API failures, JSON parsing errors, or timeout issues.

**Processing stats:**
- Average processing time: 2.88s per article
- Total batches: 3 (batch size: 10)
- Retries needed: 0

### Signal Tier Distribution

| Tier   | Count | Percentage | Notes |
|--------|-------|------------|-------|
| RED    | 0     | 0.0%       | No crisis signals in sample |
| YELLOW | 7     | 23.3%      | **4 legitimate, 3 false positives** |
| GREEN  | 0     | 0.0%       | No buying opportunities in sample |
| BLUE   | 3     | 10.0%      | Educational/political context |
| NOISE  | 20    | 66.7%      | Correctly filtered off-topic content |

**False Positive Rate**: 10.0% (3/30 articles)
**True Signal Rate**: 13.3% (4/30 legitimate YELLOW signals)

### Dimensional Score Statistics

All dimensional scores are within valid range [0-10]:

| Dimension                   | Mean | Std Dev | Min | Max | Variance |
|-----------------------------|------|---------|-----|-----|----------|
| macro_risk_severity         | 1.8  | 2.0     | 0   | 6   | Good     |
| credit_market_stress        | 1.3  | 1.2     | 0   | 4   | Good     |
| market_sentiment_extremes   | 1.3  | 1.2     | 0   | 4   | Good     |
| valuation_risk              | 1.5  | 1.5     | 0   | 4   | Good     |
| policy_regulatory_risk      | 1.8  | 2.1     | 0   | 7   | Good     |
| systemic_risk               | 1.7  | 1.9     | 0   | 6   | Good     |
| evidence_quality            | 3.4  | 2.0     | 0   | 7   | Good     |
| actionability               | 1.7  | 1.8     | 0   | 5   | Good     |

**Interpretation**: Low mean scores expected for random article sample. Good variance indicates filter is discriminating between content types.

---

## Quality Checks

### ✅ Off-Topic Articles Correctly Identified as NOISE

Sample of correctly filtered NOISE articles:
1. **Mathematics paper** - "Solvability of The Operator Equations..." (arxiv_math) - All dims scored 0-2
2. **Sports** - "PSV'er Saibari beleefde geweldige middag..." (dutch soccer) - Macro=1, Credit=1
3. **Food blog** - "Menú semanal de El Comidista" (Spanish recipes) - All dims scored 0
4. **Entertainment** - "Luis Tosar" actor interview (El Pais) - Macro=1, Credit=1
5. **Healthcare** - "Pancreatic cancer surgeon" personal story (STAT News) - All dims scored 0
6. **Film review** - "Nino" French debut film (NRC) - All dims scored 0
7. **Refugee integration** - "Samen werken aan snelle participatie" (VluchtelingenWerk) - Macro=1, Credit=1
8. **Pets** - "Haustier" pet health problems (Spiegel) - Macro=1, Credit=1

**Verdict**: ✅ Excellent NOISE filtering for clearly off-topic content

### ❌ Academic Papers Incorrectly Scored as YELLOW

**FALSE POSITIVE #1**: Chinese stock market correlation networks (arxiv)
- **Title**: "The local Gaussian correlation networks among return tails in the Chinese stock market"
- **Signal Tier**: YELLOW (should be NOISE)
- **Scores**: Macro=5, Credit=3, Policy=3, **Systemic=6**, Evidence=7, Actionability=4
- **Reasoning**: "The article suggests that negative tail correlations in the Chinese stock market are sensitive to risk, indicating potential systemic fragility..."
- **Issue**: Academic research paper on statistical methods. No actionable investment signal. Filter failed to recognize purely theoretical content.

**FALSE POSITIVE #2**: LLM inference reproducibility (science_arxiv_cs)
- **Title**: "Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference"
- **Signal Tier**: YELLOW (should be NOISE)
- **Scores**: **Macro=5**, Credit=3, Policy=3, **Systemic=5**, Evidence=7, Actionability=4
- **Reasoning**: "The article highlights a potential fragility in LLM inference reproducibility... could impact reliability of AI-driven investment strategies..."
- **Issue**: Computer science research paper. No direct financial market impact. Filter incorrectly extrapolated academic research to investment risk.

**FALSE POSITIVE #3**: Electrical sensors paper (science_mdpi_sensors)
- **Title**: "Sensors, Vol. 25, Pages 6594: Sensing and Analyzing Partial Discharge Phenomenology..."
- **Signal Tier**: YELLOW (should be NOISE)
- **Scores**: **Macro=5**, Credit=3, Policy=3, **Systemic=5**, Evidence=6, Actionability=4
- **Reasoning**: "The article discusses the impact of distorted AC waveforms on electrical asset components, potentially leading to premature failures..."
- **Issue**: Engineering research paper. Industry-specific technical issue, not financial system contagion.

**Common Pattern in False Positives**:
- All are academic/research papers (arxiv, MDPI)
- All scored Macro=5 and Systemic=5-6 (triggering YELLOW threshold)
- Oracle incorrectly interpreted academic research as actionable risk signals
- Inline filters failed to catch "academic research without market impact"

### ✅ Legitimate YELLOW Signals Correctly Identified

**LEGITIMATE YELLOW #1**: Turkish political risk (global_news_reuters)
- **Title**: "Jailed Istanbul mayor hit with new 'espionage' arrest order as opposition crackdown deepens"
- **Scores**: **Macro=6**, Credit=4, **Policy=7**, **Systemic=5**, Evidence=6, Actionability=5
- **Reasoning**: "The deepening crackdown on opposition in Istanbul signals increased political and regulatory risk. This instability can negatively impact investor confidence and economic stability, particularly in emerging markets."
- **Verdict**: ✅ Correct - Political instability in emerging market (Turkey) is actionable macro risk

**LEGITIMATE YELLOW #2**: US commodity prices and policy (newsapi_general)
- **Title**: "US beef prices are soaring. Will Trump's plans lower them?"
- **Scores**: **Macro=5**, Credit=3, **Policy=6**, Systemic=3, Evidence=6, Actionability=5
- **Reasoning**: "Rising beef prices and ranchers' criticism of Trump's plans signal potential policy error risk. This could impact the commodities market..."
- **Verdict**: ✅ Correct - Commodity price inflation + policy uncertainty is legitimate signal

**LEGITIMATE YELLOW #3**: Italian cyberattack geopolitical risk (italian_wired_italia)
- **Title**: "Hacking Team, torna lo spyware italiano in un cyberattacco alla Russia"
- **Scores**: **Macro=5**, Credit=3, **Policy=5**, **Systemic=5**, Evidence=6, Actionability=4
- **Reasoning**: "The article discusses a cyberattack linked to a company with a history of state-sponsored hacking, potentially escalating geopolitical tensions..."
- **Verdict**: ✅ Correct - Geopolitical cyber risk is monitored signal

**LEGITIMATE YELLOW #4**: US utility regulation changes (energy_utilities_utility_dive)
- **Title**: "Maryland should resist rewriting utility monopoly laws"
- **Scores**: **Macro=5**, Credit=3, **Policy=7**, Systemic=4, Evidence=6, Actionability=5
- **Reasoning**: "The article discusses potential changes to utility monopoly laws in Maryland, introducing policy/regulatory risk. This uncertainty can impact equity valuations..."
- **Verdict**: ✅ Correct - Regulatory risk to utility sector is actionable

### ✅ BLUE Signals Correctly Identified

**BLUE #1**: US political context (global_news_spiegel)
- **Title**: "Dritte Amtszeit für Donald Trump? Mike Johnson sieht »keinen Weg« für erneute Kandidatur"
- **Scores**: Macro=3, Credit=2, Sentiment=2, Valuation=2, Policy=4, Systemic=2
- **Reasoning**: "The article discusses the potential for a third term for Donald Trump... potential long-term policy implications, but no immediate impact on capital preservation."
- **Verdict**: ✅ Correct - Educational/political context without immediate action

---

## Analysis: Why Are Academic Papers Bypassing Filters?

### Root Cause

The v2 inline filters include these instructions in each dimension:

```
**❌ CRITICAL FILTERS - If article is ANY of these, score 0-2:**
- Stock picking (individual companies, IPOs, earnings predictions, price targets)
- Industry news without financial contagion (gaming, entertainment, retail product launches)
- Political scandals/gossip without economic impact
- FOMO/speculation ("hot stocks", "buy now", "next big thing")
```

**Missing**: No explicit filter for "academic research papers" or "theoretical studies without actionable market signals"

### Why This Matters

Academic papers about financial markets (Chinese stock correlations, LLM risks, industrial sensors) can *sound* like they describe systemic risks, but they:
1. Are purely theoretical/academic
2. Have no immediate market impact
3. Are not actionable for hobby investors
4. Should be classified as NOISE (or at most BLUE for education)

The oracle (Gemini Flash) is interpreting theoretical research as actionable risk signals.

---

## Recommendations

### Immediate Actions

1. **Add Academic Filter to Inline Filters**
   - Add to each dimension's "CRITICAL FILTERS" section:
   ```
   - Academic research papers (arxiv, journals) without immediate market impact
   - Theoretical studies, statistical methods, simulations without real-world validation
   ```

2. **Add Actionability Gatekeeper**
   - Academic papers should always score **Actionability ≤ 2**
   - Add explicit check: "Is this actionable for €10K-€500K hobby investor? Or purely academic?"

3. **Update Validation Examples**
   - Add example of academic paper that should be NOISE:
   ```
   NOISE Example:
   Title: "Statistical Analysis of Stock Market Correlations Using Novel Method"
   Scores: All 0-2
   Reasoning: "Academic research paper with no actionable investment signal. Purely theoretical."
   ```

### Long-term Improvements

1. **Pre-filter enhancement**: Update `InvestmentRiskPreFilterV1` to block academic sources early
2. **Fine-tune with labeled examples**: Include academic false positives in training data
3. **Add source-based heuristics**: arxiv, MDPI, IEEE sources → likely academic → default NOISE unless explicit market impact

---

## Validation Examples by Tier

### YELLOW Example (Legitimate)

**Article**: "Jailed Istanbul mayor hit with new 'espionage' arrest order as opposition crackdown deepens"
**Source**: Reuters (global_news_reuters)
**Signal Tier**: YELLOW (strength: 5.0/10)

**Dimensional Scores**:
- Macro Risk Severity: 6
- Credit Market Stress: 4
- Market Sentiment Extremes: 3
- Valuation Risk: 4
- Policy/Regulatory Risk: 7
- Systemic Risk: 5
- Evidence Quality: 6
- Actionability: 5

**Reasoning**: "The deepening crackdown on opposition in Istanbul, as evidenced by the new arrest order for the jailed mayor, signals increased political and regulatory risk. This instability can negatively impact investor confidence and economic stability, particularly in emerging markets. Hobby investors should monitor the situation closely and consider rebalancing their portfolios to reduce exposure to Turkish assets or emerging markets generally."

**Verdict**: ✅ Correctly identified emerging market political risk

---

### BLUE Example

**Article**: "Dritte Amtszeit für Donald Trump? Mike Johnson sieht »keinen Weg« für erneute Kandidatur"
**Source**: Der Spiegel (global_news_spiegel)
**Signal Tier**: BLUE (strength: 3.0/10)

**Dimensional Scores**:
- Macro Risk Severity: 3
- Credit Market Stress: 2
- Market Sentiment Extremes: 2
- Valuation Risk: 2
- Policy/Regulatory Risk: 4
- Systemic Risk: 2
- Evidence Quality: 6
- Actionability: 3

**Reasoning**: "The article discusses the potential for a third term for Donald Trump, which is being dismissed by a leading Republican. This has potential long-term policy implications, but no immediate impact on capital preservation. Monitor the political landscape for potential policy shifts, but no immediate action is needed."

**Verdict**: ✅ Correctly classified as educational context

---

### NOISE Example (Correct)

**Article**: "Menú semanal de El Comidista (27 de octubre a 2 de noviembre)"
**Source**: El Pais (global_news_el_pais)
**Signal Tier**: NOISE (strength: 1.0/10)

**Dimensional Scores**:
- All dimensions: 0-1

**Reasoning**: "This article is a weekly menu from a food blog. It has no financial or economic content and is therefore irrelevant for investment risk analysis."

**Verdict**: ✅ Correctly filtered as off-topic

---

### NOISE Example (False Positive - Should Be NOISE)

**Article**: "The local Gaussian correlation networks among return tails in the Chinese stock market"
**Source**: arxiv (arxiv)
**Signal Tier**: YELLOW ❌ (strength: 6.0/10)

**Dimensional Scores**:
- Macro Risk Severity: 5 ❌
- Credit Market Stress: 3
- Market Sentiment Extremes: 3
- Valuation Risk: 3
- Policy/Regulatory Risk: 3
- Systemic Risk: 6 ❌
- Evidence Quality: 7
- Actionability: 4

**Reasoning**: "The article suggests that negative tail correlations in the Chinese stock market are sensitive to risk, indicating potential systemic fragility. While not a crisis, this warrants monitoring and potentially reducing risk assets. The evidence quality is moderate, based on a new statistical method."

**Why This Is Wrong**:
- This is an academic research paper about statistical methods
- It describes a theoretical model, not an actual market condition
- No actionable signal for hobby investors
- Should be scored as NOISE or BLUE (educational)

**Verdict**: ❌ False positive - academic paper misclassified as actionable signal

---

## Final Verdict

### ISSUES FOUND

**Success Rate**: 30/30 scored (100%)
**False Positive Rate**: 10% (3/30 articles)
**False Negative Rate**: 0% (no missed signals detected)

**Issues**:
1. ❌ Academic research papers bypass inline filters (10% false positive rate)
2. ❌ Oracle incorrectly interprets theoretical research as actionable signals
3. ❌ Missing explicit "academic/theoretical content" filter in inline filters

**Strengths**:
1. ✅ Excellent NOISE detection for off-topic content
2. ✅ Legitimate macro risk signals correctly identified
3. ✅ All dimensional scores valid and sensible
4. ✅ No false negatives detected
5. ✅ Stock-picking filter working well (no false passes)

**Recommendation**: **DO NOT DEPLOY** without fixing academic paper false positive issue. The 10% false positive rate is unacceptable for production use.

**Next Steps**:
1. Add explicit academic research filter to inline filters
2. Re-run validation on same 30 articles with updated prompt
3. Target: Reduce false positive rate from 10% to <3%
4. After fix, expand validation to 100-200 articles for confidence

---

## Appendix: Data Files

**Validation data**:
- Sample: `C:\local_dev\llm-distillery\sandbox\investment_risk_v2_fresh_validation\fresh_sample_30.jsonl`
- Scored batches: `C:\local_dev\llm-distillery\sandbox\investment_risk_v2_fresh_validation\investment-risk\scored_batch_*.jsonl`
- Session logs: `C:\local_dev\llm-distillery\sandbox\investment_risk_v2_fresh_validation\investment-risk\distillation.log`

**Validation stats**:
- `validation_metrics.txt` - Summary statistics
- `session_summary.json` - Processing metadata

**Filter config**:
- Filter: `C:\local_dev\llm-distillery\filters\investment-risk\v2\`
- Prompt: `prompt-compressed.md` (v2.0-compressed-inline-filters)
- Config: `config.yaml`
