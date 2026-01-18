# Commerce Prefilter SLM

ML-based classifier for detecting commerce/promotional content. Replaces brittle regex patterns with a trained multilingual model.

## Status

| Phase | Status | Notes |
|-------|--------|-------|
| Oracle Calibration | COMPLETE | 100% accuracy on manual review |
| Full Batch Scoring | COMPLETE | 2,000 articles scored with Gemini Flash |
| Model Training | COMPLETE | DistilBERT selected (97.8% F1) |
| Backtesting | COMPLETE | Validated on 56K scored articles |
| Production | READY | Threshold 0.95 recommended |

### Training Results (2026-01-17)

| Model | F1 | Precision | Recall | Inference |
|-------|-----|-----------|--------|-----------|
| **DistilBERT** | **97.8%** | 96.7% | 98.9% | 1.8ms GPU |
| XLM-RoBERTa | 97.2% | 95.7% | 98.9% | 3.9ms GPU |
| MiniLM | 95.6% | 93.5% | 97.8% | 3.8ms GPU |
| Qwen LoRA | 95.0% | 93.5% | 96.6% | 17.3ms GPU |

### Backtest Results (2026-01-18)

Tested on 56,336 scored articles from NexusMind:
- **17.5% overall** flagged as commerce
- **sustainability_technology**: 660 commerce in high/medium tiers
- **Recommended threshold: 0.95** (catches 517 clear commerce, zero false positives)

## Documentation

All reports are in the `docs/` folder:

| Document | Description |
|----------|-------------|
| [BACKTEST_REPORT.md](docs/BACKTEST_REPORT.md) | Backtest results, threshold analysis, edge cases |
| [TRAINING_REPORT.md](docs/TRAINING_REPORT.md) | Model training results and metrics |
| [ORACLE_CALIBRATION.md](docs/ORACLE_CALIBRATION.md) | Oracle setup and calibration |
| [TRAINING_PLAN.md](docs/TRAINING_PLAN.md) | Initial training plan |

## Overview

This is a **cross-cutting prefilter** that benefits all domain filters by blocking promotional content before expensive LLM scoring.

**Problem**: Regex-based commerce detection is fragile:
- Hard to maintain growing pattern lists
- False positives on legitimate sustainability deals coverage
- False negatives on subtle promotional content

**Solution**: Fine-tuned multilingual classifier for binary classification:
- Fast CPU inference (~91ms per article)
- Multilingual support (handles Portuguese, French, Spanish, Dutch, etc.)
- Single score + tunable threshold
- Backward compatible integration

## Usage

```python
from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM

detector = CommercePrefilterSLM(threshold=0.95)

# Single article
result = detector.is_commerce(article)
# {"is_commerce": True, "score": 0.97, "inference_time_ms": 91}

# Batch processing
results = detector.batch_predict(articles)
```

### Integration with BasePreFilter

```python
from filters.common.base_prefilter import BasePreFilter

class MyPreFilter(BasePreFilter):
    def __init__(self, use_commerce_prefilter: bool = False):
        super().__init__(use_commerce_prefilter=use_commerce_prefilter)

    def apply_filter(self, article):
        # Commerce check happens automatically if enabled
        if self._commerce_detector:
            result = self._commerce_detector.is_commerce(article)
            if result["is_commerce"]:
                return (False, f"commerce_content_{result['score']:.2f}")

        # Continue with domain-specific filtering...
```

## Model Details

### Selected Model: distilbert-base-multilingual-cased

| Attribute | Value |
|-----------|-------|
| Base model | `distilbert-base-multilingual-cased` |
| Parameters | 135M |
| Languages | 104 |
| Model size | ~516MB |
| Output | Softmax probability (0-1) |
| Default threshold | 0.95 |
| Inference speed | ~91ms CPU, 1.8ms GPU |
| F1 Score | 97.8% |
| Precision | 96.7% |
| Recall | 98.9% |

### Threshold Recommendations

| Use Case | Threshold | Effect |
|----------|-----------|--------|
| High precision | 0.95 | Blocks clear commerce, no false positives |
| Balanced | 0.90 | Catches more, some borderline cases |
| High recall | 0.85 | Aggressive, may block product-adjacent journalism |

## Training Data

**Total:** 1,891 articles (after discarding 109 ambiguous samples)

### Commerce examples (889 articles, label=1):
- Electrek "Green Deals" column
- Affiliate content with discount codes
- PR Newswire product announcements
- Black Friday / Prime Day / Cyber Monday deals
- Product comparison shopping guides

### Journalism examples (1,002 articles, label=0):
- Reuters, AP, BBC news
- Nature, Science journals
- EPA, government reports
- High-tier articles from existing filters

### Splits
| Split | Total | Commerce | Journalism |
|-------|-------|----------|------------|
| Train | 1,512 | 711 (47%) | 801 (53%) |
| Val | 189 | 89 (47%) | 100 (53%) |
| Test | 190 | 89 (47%) | 101 (53%) |

## Project Structure

```
filters/common/commerce_prefilter/
├── README.md                       # This file
├── docs/                           # All documentation
│   ├── BACKTEST_REPORT.md          # Backtest results & threshold analysis
│   ├── TRAINING_REPORT.md          # Model training results
│   ├── ORACLE_CALIBRATION.md       # Oracle calibration
│   └── TRAINING_PLAN.md            # Initial planning
├── v1/
│   ├── config.yaml                 # Configuration
│   ├── prompt.md                   # Oracle prompt for Gemini
│   ├── oracle.py                   # Oracle implementation
│   ├── inference.py                # CommercePrefilterSLM class
│   ├── models/distilbert/          # Trained model (~516MB)
│   │   ├── model.safetensors
│   │   ├── README.md               # Model card
│   │   └── ...
│   └── results/
│       └── evaluation_results.json
└── training/
    ├── README.md                   # Training scripts docs
    ├── backtest.py                 # Backtest on scored articles
    ├── train_encoder.py            # Encoder model training
    ├── train_qwen_lora.py          # Qwen LoRA training
    ├── evaluate_models.py          # Test set evaluation
    ├── backtest_results.json       # Raw backtest data
    ├── sust_tech_commerce_leakage.csv  # Flagged articles
    └── splits/                     # Train/val/test data
```

## Success Criteria

| Metric | Target | Achieved |
|--------|--------|----------|
| F1 Score | >95% | **97.8%** |
| Precision | >95% | **96.7%** |
| Recall | >90% | **98.9%** |
| Inference (CPU) | <100ms | **~91ms** |
| Backtest validation | - | **17.5% commerce detected** |
