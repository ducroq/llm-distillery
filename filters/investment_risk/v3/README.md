# Investment Risk Filter v3.0 - Harmonized Architecture

**Version**: 3.0-harmonized
**Created**: 2025-11-17
**Forked from**: v2.1-academic-filter

## What's New in v3.0

### Harmonized Architecture

**Key Change**: Oracle outputs **dimensional scores only**. Signal tier classification (RED/YELLOW/GREEN/BLUE/NOISE) is computed by **postfilter**, not by oracle.

**Why this matters**:
- **Flexible tier thresholds**: Adjust tier definitions without retraining
- **Clean knowledge distillation**: Student model learns dimensional scoring, not classification
- **Architectural consistency**: Matches uplifting v4 and sustainability_tech_innovation v1 patterns

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

## Differences from v2.1

| Aspect | v2.1 | v3.0 |
|--------|------|------|
| Oracle output | Dimensional scores only (already harmonized) | Same - no changes |
| Version label | 2.1-academic-filter | 3.0-harmonized |
| Training data | May have mixed approaches from earlier iterations | Clean slate - dimensional only |
| Purpose | Working version with inline filters | Harmonized version for distillation |

**Note**: v2.1 prompt was already harmonized (signal_tier removed from oracle output), but v3.0 provides a **clean fork** to ensure training data has no legacy classification artifacts.

## When to Use v3.0

- ✅ **New training runs**: Use v3.0 for fresh oracle scoring
- ✅ **Knowledge distillation**: Student models learn dimensional scoring
- ✅ **Production deployment**: Flexible tier thresholds via postfilter
- ❌ **Existing v2.x training data**: Don't mix with v3.0 (different lineage)

## Training Target

- **Target samples**: 5,000+ scored articles
- **Oracle**: Gemini Flash 1.5 / Claude Haiku
- **Expected cost**: ~$0.75 @ $0.00015/article (Gemini Flash)
- **Expected time**: ~2-3 hours
- **Training model**: Qwen2.5-7B or similar
- **Expected accuracy**: 92-96% on tier classification

## Quick Start

### Score Training Data

```bash
python -m ground_truth.batch_scorer \
  --filter filters/investment-risk/v3 \
  --source datasets/raw/master_dataset_20251010_20251114.jsonl \
  --output-dir datasets/scored/investment-risk_v3 \
  --llm gemini-flash \
  --batch-size 50 \
  --target-scored 5000 \
  --random-sample
```

### Train Distilled Model

```bash
python -m training.knowledge_distillation \
  --filter investment-risk \
  --version v3 \
  --scored-data datasets/scored/investment-risk_v3 \
  --output-dir models/investment-risk_v3 \
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
