# Uplifting Filter v5 - Orthogonal Dimension Framework

**Status:** ✅ Production Ready
**Training Data:** ✅ 10,000 articles scored (see datasets/training/uplifting_v5/)
**Created:** 2025-11-29
**Calibration:** ✅ PASSED (see calibration_report.md)

## Overview

Version 5 is a complete redesign of the uplifting filter to fix the high dimension correlation problem discovered in v4.

### Problem with v4

The v4 dimensions (`agency`, `progress`, `collective_benefit`) were highly correlated:
- agency ↔ progress: **0.97**
- agency ↔ collective_benefit: **0.94**
- progress ↔ collective_benefit: **0.93**

This meant the model was essentially learning 4-5 concepts instead of 8, wasting capacity and creating unstable training.

### Solution in v5

Inspired by the sustainability_technology LCSA framework, v5 uses **orthogonal dimensions** that answer **DIFFERENT QUESTIONS** about uplifting content:

## Dimension Framework

### Impact Domains (WHAT kind of uplift)

| Dimension | Weight | Question |
|-----------|--------|----------|
| **Human Wellbeing Impact** | 25% | Health, safety, livelihoods improved? |
| **Social Cohesion Impact** | 15% | Communities strengthened, solidarity built? |
| **Justice & Rights Impact** | 10% | Wrongs addressed, rights expanded? |

### Assessment Dimensions (HOW real/accessible)

| Dimension | Weight | Question |
|-----------|--------|----------|
| **Evidence Level** | 20% | Documented outcomes or speculation? (GATEKEEPER) |
| **Benefit Distribution** | 20% | Who benefits? Elite → Universal? |
| **Change Durability** | 10% | Temporary relief → Systemic change? |

### Why These Are Orthogonal

- High wellbeing impact (8) + elite-only distribution (2) = billionaire philanthropy
- High evidence (9) + temporary relief (2) = verified but one-time aid
- Community-led + proprietary (2) = local coop with closed practices
- Universal benefit (9) + speculation (2) = promising but unverified claims

## Gatekeeper Rule

**Evidence Level < 3 → cap overall score at 3.0**

Speculation without documented outcomes cannot be truly uplifting.

## Content Type Caps

| Content Type | Max Score | Exception |
|--------------|-----------|-----------|
| Corporate Finance | 2.0 | Worker coop, public benefit, open source |
| Military/Security | 4.0 | Peace process, disarmament, reconciliation |
| Pure Speculation | 3.0 | Documented pilot results |

## Files

```
filters/uplifting/v5/
├── README.md                 # This file
├── config.yaml               # Filter configuration
├── prompt-compressed.md      # Oracle prompt (validated)
├── prefilter.py              # Fast rule-based prefilter (10/10 tests pass)
├── calibration_report.md     # Phase 3 calibration results
├── inference.py              # Local model inference (production)
├── inference_hub.py          # HuggingFace Hub inference
├── model/                    # Trained LoRA adapter
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── tokenizer files
├── training_history.json     # Learning curves
└── training_metadata.json    # Training configuration
```

## Usage

### Local Inference (recommended for production)

```python
from filters.uplifting.v5.inference import UpliftingScorer

scorer = UpliftingScorer()
result = scorer.score_article({
    "title": "Community Garden Feeds 500 Families",
    "content": "A volunteer-led community garden..."
})

print(result['weighted_average'])  # 6.5
print(result['tier'])              # "medium"
```

### HuggingFace Hub Inference

```python
from filters.uplifting.v5.inference_hub import UpliftingScorerHub

scorer = UpliftingScorerHub(
    repo_id="jeergrvgreg/uplifting-filter-v5",
    token="hf_..."  # Only needed for private repos
)
result = scorer.score_article(article)
```

### CLI Usage

```bash
# Score articles from file
python -m filters.uplifting.v5.inference -i articles.jsonl -o results.jsonl

# Interactive demo
python -m filters.uplifting.v5.inference
```

## Prefilter

The v5 prefilter blocks:
- Corporate finance (unless worker coop/public benefit/open source)
- Military/security buildups (unless peace/demilitarization)
- Crime/violence news (unless reform, survivor stories, or positive resolution)
- Pure speculation (3+ speculative phrases with no outcome evidence)
- Academic preprints (arxiv, biorxiv, etc.)
- VC/startup news domains (TechCrunch, etc.)
- Code hosting domains (GitHub, etc.)

Run tests:
```bash
python -m filters.uplifting.v5.prefilter
```

## Progress

1. **Phase 3: Oracle Calibration** ✅ COMPLETE
   - Scored 100+ diverse articles with oracle
   - Dimensions verified orthogonal (max correlation 0.76 vs v4's 0.97)
   - Fixed placeholder bug and sharpened distribution definition
   - See `calibration_report.md` for details

2. **Phase 4: Prefilter Validation** ✅ COMPLETE
   - Prefilter tests: 10/10 passing
   - Blocks corporate finance, military, speculation, academic domains

3. **Phase 5: Training Data Generation** ✅ COMPLETE
   - Scored 10,000 articles (100% success rate, ~8.5 hours)
   - Created train/val/test splits (8000/1000/1000)
   - Final correlation: only 1 pair > 0.70 (wellbeing ↔ durability: 0.737)
   - Location: `datasets/training/uplifting_v5/`

4. **Phase 6: Model Training** ✅ COMPLETE
   - Trained Qwen2.5-1.5B with LoRA (18.5M params, 1.2% of model)
   - **Best Val MAE: 0.68** (target was < 1.0)
   - All dimensions under 0.80 MAE
   - Model saved: `filters/uplifting/v5/model/`

## Training Data Summary

| Split | Articles | Location |
|-------|----------|----------|
| Train | 8,000 | `datasets/training/uplifting_v5/train.jsonl` |
| Val | 1,000 | `datasets/training/uplifting_v5/val.jsonl` |
| Test | 1,000 | `datasets/training/uplifting_v5/test.jsonl` |

### Score Distributions

| Dimension | Mean | Std | Range |
|-----------|------|-----|-------|
| Human Wellbeing | 3.23 | 1.66 | 0-8 |
| Social Cohesion | 2.74 | 1.57 | 0-8 |
| Justice & Rights | 2.39 | 1.59 | 0-8 |
| Evidence Level | 4.58 | 1.43 | 0-8 |
| Benefit Distribution | 3.91 | 1.58 | 0-9 |
| Change Durability | 3.29 | 1.40 | 0-8 |

### At-Scale Correlation Validation (10,000 articles)

| Metric | Calibration (100) | At Scale (10,000) |
|--------|-------------------|-------------------|
| Max correlation | 0.76 | 0.737 |
| Pairs > 0.70 | 2 | 1 |

## Model Performance

| Dimension | Val MAE | Status |
|-----------|---------|--------|
| human_wellbeing_impact | 0.69 | ✅ |
| social_cohesion_impact | 0.70 | ✅ |
| justice_rights_impact | 0.62 | ✅ |
| evidence_level | 0.64 | ✅ |
| benefit_distribution | 0.79 | ✅ |
| change_durability | 0.65 | ✅ |
| **Overall** | **0.68** | ✅ |

Training: 3 epochs, Qwen2.5-1.5B + LoRA, knowledge distillation mode

## Calibration Results Summary

| Metric | v4 (broken) | v5 (fixed) |
|--------|-------------|------------|
| Max correlation | 0.97 | 0.76 |
| Pairs > 0.70 | 14 | 2 |
| Score range | 6-7.5 | 0-8 |
| Manual validation | N/A | 100% |

## Comparison with sustainability_technology

| Aspect | sustainability_technology | uplifting v5 |
|--------|---------------------------|--------------|
| Dimensions | 6 | 6 |
| Core concept | Environmental + Social + Governance sustainability | Wellbeing + Cohesion + Justice impact |
| Assessment | TRL, Performance, Economics | Evidence, Distribution, Durability |
| Gatekeeper | TRL < 3 → cap at 2.9 | Evidence < 3 → cap at 3.0 |
| Framework | LCSA (Life Cycle Sustainability Assessment) | Orthogonal Impact Assessment |

## Changelog

### v5.0 (2025-11-29)
- Complete dimension redesign to fix v4 correlation crisis (0.97 → 0.76)
- 3 impact domains + 3 assessment dimensions (6 total, all orthogonal)
- Evidence-based gatekeeper (speculation capped at 3.0)
- Speculation detection in prefilter
- Inspired by sustainability_technology LCSA framework
- Calibration: 3 iterations, 2 bugs fixed, manual validation 100%
