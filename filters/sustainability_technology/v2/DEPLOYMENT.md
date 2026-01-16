# Sustainability Technology v1 - Deployment Guide

**Status**: ✅ Production Ready
**Date**: 2025-11-27

---

## Quick Start

### 1. Python API

```python
from filters.sustainability_technology.v1.inference import SustainabilityTechnologyScorer

# Initialize (loads model once)
scorer = SustainabilityTechnologyScorer()

# Score a single article
article = {
    "title": "New Solar Panel Technology Achieves 30% Efficiency",
    "content": "Researchers have developed a new perovskite-silicon tandem..."
}

result = scorer.score_article(article)
print(f"Tier: {result['tier']}")  # e.g., "medium_high"
print(f"Score: {result['weighted_average']:.2f}")  # e.g., 5.42
```

### 2. Batch Processing

```python
# Score many articles efficiently
articles = [{"title": "...", "content": "..."}, ...]
results = scorer.score_batch(articles, batch_size=16)

for article, result in zip(articles, results):
    if result['passed_prefilter']:
        print(f"{article['title']}: {result['tier']}")
```

### 3. CLI

```bash
# Score articles from JSONL file
python -m filters.sustainability_technology.v1.inference \
    --input articles.jsonl \
    --output results.jsonl \
    --batch-size 16
```

---

## Pipeline Architecture

```
Article → Prefilter → Model → Postfilter → Tier
           ↓            ↓          ↓
        (fast)      (GPU)    (gatekeepers)
```

### Prefilter (Optional, Recommended)
- **Purpose**: Skip irrelevant articles before GPU inference
- **Method**: Keyword matching for sustainability topics
- **Speed**: ~100K articles/second
- **To disable**: `SustainabilityTechnologyScorer(use_prefilter=False)`

### Model
- **Base**: Qwen/Qwen2.5-1.5B
- **Adapter**: LoRA (18.5M params)
- **Input**: Title + Content (max 512 tokens)
- **Output**: 6 dimension scores (0-10)

### Postfilter (Gatekeepers)
- **TRL Gatekeeper**: If `technology_readiness_level < 3.0`, overall score capped at 2.9
- **Rationale**: Lab-only tech can't achieve high sustainability scores

---

## Output Format

```python
{
    "passed_prefilter": True,           # Did article pass prefilter?
    "prefilter_reason": None,           # Reason if blocked (e.g., "not_sustainability_topic")
    "scores": {                         # Per-dimension scores (0-10)
        "technology_readiness_level": 6.5,
        "technical_performance": 7.2,
        "economic_competitiveness": 5.8,
        "life_cycle_environmental_impact": 6.9,
        "social_equity_impact": 4.5,
        "governance_systemic_impact": 5.1
    },
    "weighted_average": 6.21,           # Weighted score (after gatekeepers)
    "tier": "medium_high",              # Assigned tier
    "tier_description": "Commercial deployment, good sustainability",
    "gatekeeper_applied": False         # Was TRL gatekeeper triggered?
}
```

---

## Tier Definitions

| Tier | Threshold | Description |
|------|-----------|-------------|
| `high_sustainability` | ≥7.0 | Mass deployed, proven sustainable, competitive |
| `medium_high` | ≥5.0 | Commercial deployment, good sustainability |
| `medium` | ≥3.0 | Pilot/early commercial, mixed profile |
| `low` | <3.0 | Lab stage or poor sustainability performance |

---

## Dimension Weights

| Dimension | Weight | Description |
|-----------|--------|-------------|
| technology_readiness_level | 15% | TRL 1-9 scale |
| technical_performance | 15% | Real-world reliability/efficiency |
| economic_competitiveness | 20% | Life Cycle Cost competitiveness |
| life_cycle_environmental_impact | 30% | Holistic environmental assessment |
| social_equity_impact | 10% | Jobs, ethics, equitable access |
| governance_systemic_impact | 10% | Systemic disruption potential |

---

## Hardware Requirements

### Minimum (CPU only)
- RAM: 8GB
- Latency: ~500ms/article
- Throughput: ~2 articles/second

### Recommended (GPU)
- VRAM: 4GB (FP16)
- Latency: ~15ms/article
- Throughput: ~50-100 articles/second (batched)

### Production (Multi-GPU)
- Scale horizontally with multiple workers
- Use batch inference for maximum throughput

---

## Dependencies

```bash
pip install torch transformers peft safetensors pyyaml
```

Minimum versions:
- `torch>=2.0`
- `transformers>=4.35`
- `peft>=0.6`

---

## Hugging Face Deployment

Upload to Hugging Face for easy sharing:

```bash
python scripts/deployment/upload_to_huggingface.py \
    --filter filters/sustainability_technology/v1 \
    --repo-name your-username/sustainability-technology-v1 \
    --private
```

Then load from anywhere:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "your-username/sustainability-technology-v1"
)
```

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Test MAE | 0.690 |
| Test RMSE | 0.970 |
| All dimensions MAE | < 1.0 |
| Overfitting | None detected |

### Per-Dimension MAE (Test Set)

| Dimension | MAE |
|-----------|-----|
| life_cycle_environmental_impact | 0.562 |
| technical_performance | 0.667 |
| economic_competitiveness | 0.667 |
| social_equity_impact | 0.690 |
| technology_readiness_level | 0.707 |
| governance_systemic_impact | 0.850 |

---

## Monitoring & Maintenance

### Recommended Metrics to Track

1. **Score distribution** - Should match training distribution
2. **Tier distribution** - ~0.3% high, ~7.7% medium-high, ~19.3% medium
3. **Prefilter pass rate** - ~19% of random articles
4. **Inference latency** - P50, P95, P99

### Drift Detection

If production scores diverge significantly from benchmarks:
1. Check input data quality
2. Verify model loading
3. Consider retraining on recent data

---

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size
- Use FP16: `model.half()`
- Use CPU: `SustainabilityTechnologyScorer(device='cpu')`

### Low throughput
- Enable batching with `score_batch()`
- Use GPU if available
- Consider disabling prefilter for pre-filtered data

### Unexpected scores
- Check input format (need `title` and `content` keys)
- Verify content is in English
- Check for truncation (>512 tokens)

---

## Files Reference

```
filters/sustainability_technology/v1/
├── config.yaml              # Filter configuration
├── inference.py             # Production inference module
├── prefilter.py             # Keyword prefilter
├── oracle.py                # Oracle scoring (for ground truth)
├── DEPLOYMENT.md            # This file
├── model/
│   ├── adapter_model.safetensors  # Trained weights
│   ├── adapter_config.json        # LoRA config
│   └── tokenizer.json             # Tokenizer
├── benchmarks/
│   ├── test_set_results.json      # Benchmark metrics
│   └── test_set_predictions.json  # Detailed predictions
└── reports/
    └── TRAINING_REPORT.md         # Training summary
```

---

## Support

For issues or questions:
1. Check the training report: `reports/TRAINING_REPORT.md`
2. Review benchmark results: `benchmarks/test_set_results.json`
3. Contact the filter maintainer
