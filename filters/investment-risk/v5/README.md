# Investment Risk Filter v5 - Orthogonal Dimension Framework

**Status:** Phase 3 Complete - Ready for Training Data Generation
**Created:** 2025-11-30
**Philosophy:** "You can't predict crashes, but you can prepare for them."

## Overview

Version 5 is a complete redesign of the investment-risk filter to fix the severe dimension correlation problem discovered in v4.

### Problem with v4

The v4 dimensions had catastrophic correlations - the oracle was effectively rating on 2-3 concepts instead of 8:

| Metric | v4 Result | Target |
|--------|-----------|--------|
| PC1 Variance | **87%** | <60% |
| Max correlation | **0.97** | <0.75 |
| Pairs > 0.70 | **27** | 0 |
| Pairs > 0.85 | **15** | 0 |

Key problematic correlations:
- macro_risk ↔ systemic_risk: **0.97** (essentially identical!)
- credit_stress ↔ systemic_risk: **0.96**
- credit_stress ↔ macro_risk: **0.95**

**Root cause:** All v4 dimensions answer the SAME question ("how risky is this?") in slightly different ways. When SVB fails, ALL dimensions go high together.

### Solution in v5

Inspired by the successful **uplifting v5** redesign (which fixed 0.97 → 0.76 correlation) and **sustainability_technology LCSA framework**, v5 uses 6 **orthogonal dimensions** that answer **DIFFERENT QUESTIONS**:

## Dimension Framework

### Risk Characterization (WHAT kind of risk)

| Dimension | Weight | Question |
|-----------|--------|----------|
| **Risk Domain Type** | 20% | WHERE in the financial system? (stock → sector → asset class → multi-asset → core system) |
| **Severity Magnitude** | 25% | HOW BAD if it materializes? (routine → minor → moderate → severe → catastrophic) |
| **Materialization Timeline** | 15% | WHEN would impact hit? (priced in → long-term → medium → short → immediate) |

### Assessment Dimensions (HOW real/actionable)

| Dimension | Weight | Question |
|-----------|--------|----------|
| **Evidence Quality** | 15% | HOW documented? (speculation → opinion → journalism → official → academic) **[GATEKEEPER]** |
| **Impact Breadth** | 15% | WHO is affected? (individual → sector → regional → national → global) |
| **Retail Actionability** | 10% | CAN hobby investors respond? (institutional → complex → moderate → simple → immediate) |

### Why These Are Orthogonal

The key insight: these dimensions can vary **independently**:

| Scenario | Risk Domain | Severity | Timeline | Evidence | Breadth | Actionability |
|----------|-------------|----------|----------|----------|---------|---------------|
| **Climate financial risk** | 8 (systemic) | **9** (catastrophic) | **2** (5+ years) | 9 | 9 | **2** (can't act now) |
| **SVB bank failure** | 9 (banking) | 8 | **9** (unfolding) | 8 | 7 | **8** (clear action) |
| **Fed meeting preview** | **9** (central bank) | **2** (routine) | **9** (tomorrow) | 7 | 8 | 6 |
| **Academic EMH paper** | 6 | 3 | **1** (historical) | **9** (rigorous) | 5 | **1** (not actionable) |
| **GTA 6 game delay** | **1** (not financial) | 1 | 8 | 7 | 1 | **0** |
| **Crypto FOMO tweet** | 2 | 3 | 9 | **1** (speculation) | 2 | 2 |

These examples show:
- **Severity ≠ Timeline**: Climate risk is severe (9) but distant (2); Fed meeting is minor (2) but immediate (9)
- **Evidence ≠ Actionability**: Academic papers have high evidence (9) but low actionability (1)
- **Domain ≠ Severity**: Fed meeting is core system (9) but routine severity (2)
- **Breadth ≠ Actionability**: Climate has global breadth (9) but low actionability (2)

## Gatekeeper Rule

**Evidence Quality < 4 → cap overall signal at 3.0**

Speculation without documentation cannot drive portfolio decisions.

## Content Type Flags

| Content Type | Effect | Example |
|--------------|--------|---------|
| Stock Picking | Cap risk_domain = 2, breadth = 2 | "Buy AAPL", price targets |
| FOMO/Speculation | Cap all dimensions at 2 | Meme stocks, crypto pumping |
| Affiliate Marketing | Cap all dimensions at 2 | "Use my broker link" |
| Clickbait | Cap evidence at 3 | "Market CRASH coming!" |
| Academic Research | Cap actionability at 2 | IMF working papers |
| Non-Financial | Cap risk_domain at 1 | Gaming, entertainment |

## Files

```
filters/investment-risk/v5/
├── README.md                 # This file
├── config.yaml               # Filter configuration
├── prompt-compressed.md      # Oracle prompt (calibrated)
└── prefilter.py              # Source-based prefilter (54% block rate)
```

## Calibration Results (Phase 2 - PASSED)

Oracle calibration on 100 random articles (seed=44):

### Content Classification
- **87% correctly classified as noise** (non-financial content)
- **13% identified as financial content** (macro_risk, geopolitical, policy, credit)

### Dimension Orthogonality (Financial Content Only, n=13)

| Correlation Pair | v4 (broken) | v5 Result | Target | Status |
|------------------|-------------|-----------|--------|--------|
| risk_domain ↔ severity | 0.97 | **0.60** | <0.75 | ✅ PASS |
| severity ↔ timeline | 0.89 | **0.10** | different | ✅ EXCELLENT |
| evidence ↔ actionability | 0.82 | **-0.17** | different | ✅ EXCELLENT |
| timeline ↔ actionability | 0.85 | **0.60** | <0.75 | ✅ PASS |

### Key Validation Examples
- **"El Ibex se aleja"** (market news): severity=3, timeline=9 (low severity, immediate)
- **"When In Doubt, Abstain"** (academic): evidence=7, actionability=1 (high evidence, not actionable)
- **"COP30: Climate crisis"**: severity=7, timeline=6, breadth=9 (global systemic risk)

**Conclusion:** Dimensions vary independently among financial content. High correlations in full dataset (0.93) were due to 87% noise articles all scoring 0-2.

## Development Phases

1. **Phase 1: Design** ✅ COMPLETE
   - Analyzed v4 correlation crisis
   - Designed orthogonal dimension framework
   - Created config.yaml and prompt-compressed.md

2. **Phase 2: Oracle Calibration** ✅ COMPLETE
   - Scored 100 articles with Gemini Flash
   - Validated dimension independence (max correlation 0.60 among financial content)
   - Fixed placeholder bug that prevented article injection

3. **Phase 3: Prefilter Development** ✅ COMPLETE
   - Created prefilter.py with source-based blocking
   - Block rate: 54% (arxiv, github, dev_to, science news, etc.)
   - False negative rate: 8% (1/13 financial articles blocked - low actionability arxiv paper)

4. **Phase 4: Training Data Generation**
   - Score 5,000+ articles
   - Validate correlation targets at scale

5. **Phase 5: Model Training**
   - Train Qwen2.5-1.5B with LoRA
   - Target MAE < 0.80

6. **Phase 6: Deployment**
   - Create inference.py
   - Deploy to HuggingFace Hub

## Comparison with Other Filters

| Aspect | uplifting v5 | sustainability_tech v1 | investment-risk v5 |
|--------|--------------|------------------------|-------------------|
| Dimensions | 6 | 6 | 6 |
| Core concept | Wellbeing impact | LCSA sustainability | Capital preservation |
| Assessment | Evidence, Distribution, Durability | TRL, Performance, Economics | Evidence, Breadth, Actionability |
| Gatekeeper | Evidence < 3 → cap 3.0 | TRL < 3 → cap 2.9 | Evidence < 4 → cap 3.0 |
| Framework | Orthogonal Impact | LCSA Lifecycle | Orthogonal Risk |

## Changelog

### v5.0 (2025-11-30)
- Complete dimension redesign to fix v4 correlation crisis (0.97 → target 0.75)
- 3 risk characterization + 3 assessment dimensions (6 total, all orthogonal)
- Key insight: Severity ≠ Timeline (severe risks can be distant)
- Evidence-based gatekeeper (speculation capped at 3.0)
- Inspired by uplifting v5 and sustainability_technology LCSA framework
