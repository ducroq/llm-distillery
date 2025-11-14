# Investment Risk Filter v2.0

**Purpose**: Identify investment risk signals for defense-first portfolio management focused on capital preservation, not speculation.

**Philosophy**: "You can't predict crashes, but you can prepare for them."

**Created**: 2025-10-30
**Updated**: 2025-11-14 (v2.0 - inline filters pattern)

---

## Version 2.0 Changes

**What changed**: Applied inline filters pattern to reduce false positives
**Result**: False positive rate reduced from 50-75% (v1) to 25-37% (v2)

**Key improvement**: Moved scope filters from top-level instructions into each dimension definition. This forces fast models (Gemini Flash) to check filters before scoring, preventing stock picking and gaming news from being classified as macro risk.

**Calibration**: Validated on 92 articles (47 calibration + 45 validation)

**See**: `calibration_report.md` for full details

---

## Overview

This filter evaluates content for **macro investment risk signals** relevant to hobby investors (€10K-€500K portfolios) practicing **defense-first portfolio management**.

### What This Filter IS
- ✅ Macro risk analysis (recession signals, credit crises, systemic risk)
- ✅ Capital preservation focused (when to reduce risk)
- ✅ Portfolio-level guidance (asset allocation, rebalancing)
- ✅ Evidence-based (Fed data, economic indicators, historical patterns)
- ✅ Time-appropriate (weeks/months, not day-trading)

### What This Filter is NOT
- ❌ Stock picking (individual stocks, buy recommendations)
- ❌ Speculation (FOMO, meme stocks, crypto pumping)
- ❌ Day-trading advice (technical analysis, price targets)
- ❌ Get-rich-quick schemes (guaranteed returns, insider tips)

---

## Signal Tiers

| Tier | Description | Example | Action |
|------|-------------|---------|--------|
| 🔴 **RED** | Act now - reduce risk immediately | Yield curve inversion + bank crisis | Increase cash, reduce risk assets |
| 🟡 **YELLOW** | Monitor closely - prepare for defense | Rising unemployment + credit stress | Review portfolio, prepare rebalancing |
| 🟢 **GREEN** | Consider buying - value emerging | Extreme fear + cheap valuations | Dollar-cost average, buy quality |
| 🔵 **BLUE** | Understand - no immediate action | Historical analysis, educational | Refine framework, no changes |
| ⚫ **NOISE** | Ignore completely | Stock tips, FOMO, pump-and-dump | Ignore |

---

## Pre-filter (Rule-based Blocking)

The pre-filter blocks obvious NOISE before LLM evaluation:

### Blocked Categories

1. **FOMO/Speculation** (8 patterns)
   - Hot stocks, meme stocks, crypto pumping
   - "Buy now", "don't miss out", "to the moon"
   - Get-rich-quick, guaranteed returns, 100x gains

2. **Stock Picking** (6 patterns, 6 exceptions)
   - Individual stock recommendations
   - Buy/sell ratings, price targets, earnings predictions
   - **Exception**: Allowed if macro context present (systemic risk, credit crisis, etc.)

3. **Affiliate/Conflict** (4 patterns)
   - "Sign up with this broker", promotional codes
   - Affiliate links, sponsored content
   - Discord/Telegram trading groups

4. **Clickbait** (5 patterns)
   - "Market CRASH coming!", "This ONE stock!"
   - Warren Buffett's secret, hidden gems
   - FOMO emojis (🚀💎🌙💰)

### Expected Performance
- **Pass rate**: 40-70% (conservative filtering)
- **Purpose**: Block speculation, pass macro analysis

---

## Scoring Dimensions

| Dimension | Weight | Description | Gatekeeper |
|-----------|--------|-------------|------------|
| **macro_risk_severity** | 25% | Systemic economic/financial risk | - |
| **credit_market_stress** | 20% | Credit market deterioration | - |
| **market_sentiment_extremes** | 15% | Panic or euphoria extremes | - |
| **valuation_risk** | 15% | Bubble or deep value territory | - |
| **policy_regulatory_risk** | 10% | Policy errors, regulatory changes | - |
| **systemic_risk** | 15% | Contagion, cascading failures | - |
| **evidence_quality** | 0%* | Data quality, source credibility | **✓ (≥5 for RED)** |
| **actionability** | 0%* | Hobby investor actionability | - |

\* evidence_quality is GATEKEEPER for RED tier (must be ≥5), not weighted in signal_strength
\* actionability is used for action_priority calculation, not signal_strength

---

## Calibration Status

### Pre-filter Calibration
- **Status**: ⏳ Pending
- **Sample size**: 500 articles recommended
- **Expected pass rate**: 40-70%
- **Command**:
  ```bash
  python -m ground_truth.calibrate_prefilter \
      --filter filters/investment-risk/v1 \
      --source datasets/raw/master_dataset_*.jsonl \
      --sample-size 500 \
      --output reports/investment_risk_v1_prefilter_calibration.md
  ```

### Oracle Calibration
- **Status**: ⏳ Pending
- **Sample size**: 100 articles recommended
- **Models to compare**: gemini-flash, gemini-pro, claude-sonnet
- **Command**:
  ```bash
  python -m ground_truth.calibrate_oracle \
      --filter filters/investment-risk/v1 \
      --source datasets/raw/master_dataset_*.jsonl \
      --sample-size 100 \
      --models gemini-flash,gemini-pro,claude-sonnet \
      --output reports/investment_risk_v1_oracle_calibration.md
  ```

---

## Ground Truth Generation

**Status**: ⏳ Not started

**Target**: 2,500 labeled articles

**Command**:
```bash
python -m ground_truth.batch_labeler \
    --filter filters/investment-risk/v1 \
    --source datasets/raw/master_dataset_*.jsonl \
    --target-labeled 2500 \
    --oracle gemini-flash \
    --output datasets/labeled/investment_risk_v1/
```

**Process**:
1. Stream through master datasets (99K articles)
2. Apply pre-filter (blocks 30-60% as NOISE)
3. Label passing articles with oracle (Gemini Flash)
4. Stop when 2,500 labeled articles collected

---

## Training (Planned)

**Model**: Qwen 2.5-7B

**Outputs** (multi-task):
- signal_tier (classification: RED/YELLOW/GREEN/BLUE/NOISE)
- signal_strength (regression: 0-10)
- 8 dimension scores (regression: 0-10 each)

**Target Performance**:
- Accuracy: ≥90% vs oracle
- Inference time: <50ms per article
- Cost: $0 per article (after training)

---

## Deployment (Planned)

**Inference Pipeline**:
1. Pre-filter (fast, local, free)
2. Qwen model (scores passing articles)

**Expected Performance**:
- Throughput: 1,000 articles/hour
- Latency: <50ms per article
- Cost: $0/article (vs $0.003 for API)

---

## Use Cases

1. **Portfolio Defense**: Identify systemic risks requiring immediate action
2. **Risk Monitoring**: Track credit market stress, policy errors, sentiment extremes
3. **Opportunity Identification**: Find value in fearful markets (GREEN signals)
4. **Education**: Learn from historical patterns and expert analysis (BLUE signals)
5. **Noise Filtering**: Block FOMO, stock tips, affiliate marketing (NOISE)

---

## Example Articles

### 🔴 RED FLAG (9.2/10)
**Title**: "Fed Emergency Meeting as Silicon Valley Bank Fails, FDIC Takes Control"
**Risk Indicators**: bank_stress_signals, credit_spread_widening, systemic_fragility
**Action**: Increase cash, reduce risk assets immediately
**Reasoning**: "Banking crisis unfolding with contagion. Emergency Fed/Treasury response indicates systemic risk."

### 🟡 YELLOW WARNING (6.5/10)
**Title**: "Unemployment Rises to 5.2% as Credit Spreads Widen to Levels Not Seen Since 2008"
**Risk Indicators**: recession_indicators_converging, credit_spread_widening
**Action**: Review portfolio, prepare defensive rebalancing
**Reasoning**: "Recession signals strengthening with labor market deterioration and credit stress. Monitor closely."

### 🟢 GREEN OPPORTUNITY (7.8/10)
**Title**: "VIX Surges to 45 as Quality Stocks Trade at 10-Year Valuation Lows"
**Risk Indicators**: extreme_sentiment, valuation_extreme
**Action**: Consider buying quality at discount, dollar-cost average
**Reasoning**: "Panic selling creating opportunity. Quality assets at deep value. Historical buying point."

### ⚫ NOISE (0.0/10)
**Title**: "🚀 THIS PENNY STOCK IS ABOUT TO EXPLODE!! 🚀"
**Flags**: speculation_noise, clickbait, affiliate_conflict
**Action**: Ignore completely
**Reasoning**: "Pure speculation with no macro analysis. Affiliate marketing. Red flags everywhere."

---

## Version History

- **v1.0** (2025-10-30): Initial implementation
  - Pre-filter with 4 blocking categories
  - 8 scoring dimensions
  - Signal tier classification (RED/YELLOW/GREEN/BLUE/NOISE)
  - Evidence quality gatekeeper for RED tier
  - Compressed prompt for Gemini Flash

---

## Next Steps

1. ⏳ Run pre-filter calibration (500 samples)
2. ⏳ Run oracle calibration (100 samples)
3. ⏳ Generate ground truth (2,500 labeled articles)
4. ⏳ Train Qwen 2.5-7B model
5. ⏳ Evaluate model vs oracle
6. ⏳ Deploy inference pipeline

---

**For complete workflow, see**: [Filter Development Guide](../../README.md)
