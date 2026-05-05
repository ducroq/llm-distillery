# Investment Risk Filter v4.0 - Oracle Calibration

**Version**: 4.0-calibrated
**Created**: 2025-11-17
**Forked from**: v3.0-harmonized

## What's New in v4.0

### Oracle Calibration Fixes

**Key Change**: Adjusted oracle prompt to score **less conservatively** based on v3 validation results (4,654 articles).

**v3 Validation Issues**:
- RED tier too low: 1.48% (target: 5-10%)
- GREEN tier non-functional: 0% (target: 5-10%)
- NOISE too high: 67% (target: 40-50%)
- Oracle scoring all dimensions too conservatively (means: 1.82-2.50)

**v4 Calibration Fixes**:
- Added high-risk scenario examples (2008 crisis, COVID crash, energy crises)
- Added sentiment extreme examples (market fear, panic, euphoria)
- Relaxed evidence quality criteria (was blocking 57.76% of articles)
- Clarified scoring scale usage (oracle now uses full 0-10 range)

### Oracle Output (v3.0)

```json
{
  "macro_risk_severity": <0-10>,
  "credit_market_stress": <0-10>,
  "market_sentiment_extremes": <0-10>,
  "valuation_risk": <0-10>,
  "policy_regulatory_risk": <0-10>,
  "systemic_risk": <0-10>,
  "evidence_quality": <0-10>,
  "actionability": <0-10>,

  "risk_indicators": {...},
  "asset_classes_affected": {...},
  "time_horizon": "...",
  "geographic_scope": [...],
  "recommended_actions": [...],
  "reasoning": "..."
}
// NO signal_tier field - computed by postfilter
```

### Postfilter Tier Classification

Signal tiers are computed from dimensional scores:

- **🔴 RED**: `macro_risk >= 7 OR credit >= 7 OR systemic >= 8` AND `evidence >= 5` AND `actionability >= 5`
- **🟡 YELLOW**: Risk scores 5-6, evidence >= 5, actionability >= 4
- **🟢 GREEN**: Valuation 0-3, sentiment extreme, opportunity signals
- **🔵 BLUE**: Educational, framework improvement, no immediate action
- **⚫ NOISE**: Stock picking, FOMO, affiliate content (filtered by prefilter)

## Differences from v3.0

| Aspect | v3.0 | v4.0 |
|--------|------|------|
| Oracle calibration | Too conservative (all dims mean <2.5) | Calibrated (using full 0-10 range) |
| RED tier rate | 1.48% | Target: 5-10% |
| GREEN tier rate | 0% (broken) | Target: 5-10% |
| NOISE rate | 67% (too strict) | Target: 40-50% |
| Prompt examples | Generic | High-risk scenarios, sentiment extremes |
| Evidence criteria | Strict (blocked 57.76%) | Relaxed (financial journalism acceptable) |
| Validation status | FAILED (4,654 articles) | Pending re-validation |

**Note**: v3.0 looked good on paper but failed validation. v4.0 incorporates learnings from 4,654-article validation to calibrate oracle scoring behavior.

## When to Use v4.0

- ✅ **New training runs**: Use v4.0 for oracle scoring (v3.0 deprecated)
- ✅ **Production deployment**: After v4.0 validation passes
- ❌ **v3.0 training data**: Discard - oracle was miscalibrated

## Training Dataset

**File**: `datasets/raw/investment-risk_v4_5k_mixed_30pct_synthetic.jsonl`

### Composition

- **Total**: 5,000 articles
- **Synthetic**: 1,500 articles (30%)
  - 300 crisis scenarios (2008, 2020, 2023 financial crises)
  - 400 moderate risk (inflation, geopolitical, recession signals)
  - 300 educational/frameworks (risk management, historical analysis)
  - 300 opportunity signals (extreme fear + valuation)
  - 200 noise (spam, clickbait, stock picking)
- **Real**: 3,500 articles (70%)
  - Random sample from master_dataset (Oct-Nov 2025)

### Why Partly Synthetic?

**Problem**: master_dataset contains only Oct-Nov 2025 articles (no major financial crisis during this period). Training exclusively on current low-risk articles would produce a student model unable to detect genuine crises.

**Solution**: Synthetic articles based on historical crises (2008 Lehman, 2020 COVID crash, 2023 SVB/Credit Suisse) ensure the model learns to recognize high-risk scenarios.

**Validation**: v4 oracle scored synthetic crisis articles correctly (all 6 crisis scenarios → RED tier, macro/credit/systemic = 8-10), confirming synthetic data quality.

## Training Target

- **Source dataset**: `datasets/raw/investment-risk_v4_5k_mixed_30pct_synthetic.jsonl`
- **Target samples**: 5,000 scored articles
- **Oracle**: Gemini Flash 1.5 / Claude Haiku
- **Expected cost**: ~$0.75 @ $0.00015/article (Gemini Flash)
- **Expected time**: ~4-5 hours
- **Training model**: Qwen2.5-7B or similar
- **Expected accuracy**: 92-96% on tier classification

## Quick Start

### Score Training Data

```bash
python -m ground_truth.batch_scorer \
  --filter filters/investment-risk/v4 \
  --source datasets/raw/investment-risk_v4_5k_mixed_30pct_synthetic.jsonl \
  --output-dir datasets/scored/investment-risk_v4_training \
  --llm gemini-flash \
  --batch-size 50 \
  --target-scored 5000
```

**Note**: No `--random-sample` flag needed - dataset is already prepared and shuffled.

### Train Distilled Model

```bash
python -m training.knowledge_distillation \
  --filter investment-risk \
  --version v4 \
  --scored-data datasets/scored/investment-risk_v4_training \
  --output-dir models/investment-risk_v4 \
  --base-model Qwen/Qwen2.5-7B \
  --epochs 3
```

## Files

- `prompt-compressed.md`: Oracle prompt (harmonized architecture)
- `config.yaml`: Filter configuration and tier definitions
- `prefilter.py`: Fast rule-based filter (blocks FOMO/speculation)
- `postfilter.py`: Tier classification from dimensional scores
- `README.md`: This file

## Philosophy

> "You can't predict crashes, but you can prepare for them."

This filter focuses on **capital preservation** and **macro risk signals**, not stock picking or speculation.
