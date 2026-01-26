# Commerce Prefilter v2

Embedding-based commerce detection using frozen sentence-transformers + MLP classifier.

## Key Differences from v1

| Aspect | v1 (DistilBERT) | v2 (Embedding + MLP) |
|--------|-----------------|----------------------|
| Architecture | Fine-tuned DistilBERT | Frozen embedder + MLP |
| F1 Score | 97.8% | 97.8% |
| Model size | ~516MB | ~420MB (embedder) + <1MB (MLP) |
| Context window | 512 tokens | 128 tokens |
| Training | GPU, hours | CPU, minutes |
| Maintenance | Retrain full model | Retrain only MLP |

## Why 128 Tokens is OK

Commerce signals are **front-loaded**:
- Titles contain "deal", "discount", "sale"
- First paragraph mentions prices
- Affiliate disclaimers appear early

Despite shorter context, v2 achieves identical accuracy to v1.

## Usage

```python
from filters.common.commerce_prefilter.v2 import CommercePrefilterV2

detector = CommercePrefilterV2(threshold=0.95)

# Single article
result = detector.is_commerce(article)
# {"is_commerce": True, "score": 0.97, "version": "v2"}

# Batch processing
results = detector.batch_predict(articles)
```

## Files

```
v2/
├── __init__.py           # Package init
├── config.yaml           # Configuration
├── inference.py          # CommercePrefilterV2 class
├── README.md             # This file
└── models/
    ├── mlp_classifier.pkl    # Trained MLP (~100KB)
    ├── scaler.pkl            # StandardScaler (~5KB)
    └── training_config.json  # Training metadata
```

## Performance

| Metric | Value |
|--------|-------|
| F1 Score | 97.8% |
| Precision | 96.7% |
| Recall | 98.9% |
| Inference (single) | ~100ms CPU (mostly embedding) |
| Inference (batch) | ~50ms/article CPU |

## Threshold Recommendations

| Use Case | Threshold | Effect |
|----------|-----------|--------|
| High precision | 0.95 | Blocks clear commerce, minimal false positives |
| Balanced | 0.90 | Catches more, some borderline cases |
| High recall | 0.85 | Aggressive, may block product-adjacent journalism |

## Training

Models were trained on the same dataset as v1:
- 1,701 training samples (train + val)
- 190 test samples
- Binary labels: 0=journalism, 1=commerce

See `../docs/V2_DESIGN.md` for full design documentation.
