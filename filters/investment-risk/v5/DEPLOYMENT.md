# Investment Risk v5 - Deployment Guide

**Status**: Production Ready
**Date**: 2025-12-02

---

## Quick Start

### 1. Python API

```python
# Note: Use importlib due to hyphen in path
import importlib.util
from pathlib import Path

# Load inference module
inference_path = Path("filters/investment-risk/v5/inference.py")
spec = importlib.util.spec_from_file_location("inference", inference_path)
inference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_module)

# Initialize (loads model once)
scorer = inference_module.InvestmentRiskScorer()

# Score a single article
article = {
    "title": "Fed Signals Extended Rate Pause Amid Inflation Concerns",
    "content": "The Federal Reserve indicated Wednesday that interest rates..."
}

result = scorer.score_article(article)
print(f"Signal: {result['tier']}")  # e.g., "YELLOW"
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
python filters/investment-risk/v5/inference.py \
    --input articles.jsonl \
    --output results.jsonl \
    --batch-size 16
```

---

## Pipeline Architecture

```
Article → Prefilter → Model → Postfilter → Signal Tier
           ↓            ↓          ↓
        (fast)      (GPU)    (gatekeepers)
```

### Prefilter (Optional, Recommended)
- **Purpose**: Skip non-financial content before GPU inference
- **Method**: Source filtering + keyword matching for investment topics
- **Speed**: ~100K articles/second
- **To disable**: `InvestmentRiskScorer(use_prefilter=False)`

### Model
- **Base**: Qwen/Qwen2.5-1.5B
- **Adapter**: LoRA (18.5M trainable params)
- **Input**: Title + Content (max 512 tokens)
- **Output**: 6 dimension scores (0-10)

### Postfilter (Gatekeepers)
- **Evidence Gatekeeper**: If `evidence_quality < 4.0`, overall score capped at 3.0
- **Rationale**: Speculation/opinion without data should not trigger high-priority signals

---

## Output Format

```python
{
    "passed_prefilter": True,           # Did article pass prefilter?
    "prefilter_reason": None,           # Reason if blocked
    "scores": {                         # Per-dimension scores (0-10)
        "risk_domain_type": 7.5,        # WHERE in financial system?
        "severity_magnitude": 6.8,      # HOW BAD if materialized?
        "materialization_timeline": 8.2, # WHEN would impact hit?
        "evidence_quality": 6.5,        # HOW documented?
        "impact_breadth": 7.0,          # WHO is affected?
        "retail_actionability": 5.5     # CAN hobby investors respond?
    },
    "weighted_average": 6.85,           # Weighted score (after gatekeepers)
    "tier": "YELLOW",                   # Signal tier
    "tier_description": "Monitor closely - prepare for defense",
    "gatekeeper_applied": False         # Was evidence gatekeeper triggered?
}
```

---

## Signal Tier Definitions

| Tier | Condition | Description | Action |
|------|-----------|-------------|--------|
| **RED** | risk_domain≥7 AND severity≥7 AND timeline≥7 AND evidence≥5 | Systemic risk unfolding | Act now - reduce exposure |
| **YELLOW** | (severity≥5 OR risk_domain≥6) AND evidence≥4 AND timeline≥5 | Elevated risk signal | Monitor closely |
| **GREEN** | severity≥6 AND timeline≤4 AND evidence≥6 AND actionability≥5 | Counter-cyclical opportunity | Consider accumulating |
| **BLUE** | evidence≥7 AND actionability≤3 | Educational content | Understand, no action |
| **NOISE** | risk_domain≤3 OR evidence<3 | Not investment-relevant | Ignore |

---

## Dimension Weights

| Dimension | Weight | Question Answered |
|-----------|--------|-------------------|
| risk_domain_type | 20% | WHERE in the financial system? |
| severity_magnitude | 25% | HOW BAD could this be? |
| materialization_timeline | 15% | WHEN would impact hit? |
| evidence_quality | 15% | HOW well documented? (GATEKEEPER) |
| impact_breadth | 15% | WHO is affected? |
| retail_actionability | 10% | CAN a hobby investor respond? |

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

### Upload Model

```bash
python scripts/deployment/upload_to_huggingface.py \
    --filter filters/investment-risk/v5 \
    --repo-name your-username/investment-risk-filter-v5 \
    --private
```

### Load from Hub

```python
from filters.investment_risk.v5.inference_hub import InvestmentRiskScorerHub

scorer = InvestmentRiskScorerHub(
    repo_id="your-username/investment-risk-filter-v5",
    token="hf_..."  # Only needed for private repos
)
```

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Validation MAE | 0.477 |
| **Test MAE** | **0.484** |
| Test RMSE | 0.941 |
| Overfitting | None detected (test ≈ val) |

### Per-Dimension MAE (Test Set)

| Dimension | MAE | RMSE |
|-----------|-----|------|
| severity_magnitude | 0.315 | 0.640 |
| retail_actionability | 0.367 | 0.650 |
| risk_domain_type | 0.400 | 0.892 |
| impact_breadth | 0.497 | 0.783 |
| evidence_quality | 0.586 | 0.901 |
| materialization_timeline | 0.741 | 1.501 |

All dimensions under 0.80 MAE target.

---

## Monitoring & Maintenance

### Recommended Metrics to Track

1. **Signal distribution** - Should see mostly NOISE, some YELLOW, rare RED
2. **Gatekeeper trigger rate** - How often evidence_quality < 4 caps scores
3. **Prefilter pass rate** - Expected ~40-60% pass rate
4. **Inference latency** - P50, P95, P99

### Drift Detection

If production scores diverge significantly from benchmarks:
1. Check input data quality (financial news vs general)
2. Verify model loading (missing weights warning is OK)
3. Consider retraining on recent financial data

---

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size
- Use FP16: `model.half()`
- Use CPU: `InvestmentRiskScorer(device='cpu')`

### Low throughput
- Enable batching with `score_batch()`
- Use GPU if available
- Consider disabling prefilter for pre-filtered data

### Unexpected scores
- Check input format (need `title` and `content` keys)
- Verify content is financial news (not entertainment/gaming)
- Check for truncation (>512 tokens)

### Import errors (hyphen in path)
- Use `importlib.util` to load modules
- See Quick Start example above

---

## Files Reference

```
filters/investment-risk/v5/
├── config.yaml              # Filter configuration
├── inference.py             # Production inference module
├── inference_hub.py         # HuggingFace Hub inference
├── prefilter.py             # Source/keyword prefilter
├── test_inference.py        # Unit tests
├── prompt-compressed.md     # Oracle prompt
├── DEPLOYMENT.md            # This file
├── README.md                # Filter overview
├── model/
│   ├── adapter_model.safetensors  # Trained LoRA weights (74MB)
│   ├── adapter_config.json        # LoRA config
│   └── tokenizer.json             # Tokenizer
├── benchmarks/
│   ├── test_set_results.json      # Benchmark metrics
│   └── test_set_predictions.json  # Detailed predictions
├── training_history.json    # Training curves
└── training_metadata.json   # Training configuration
```

---

## Philosophy

> "You can't predict crashes, but you can prepare for them."

This filter is designed for **capital preservation**, not alpha generation:
- Prioritizes **avoiding losses** over finding gains
- Rewards **documented evidence** over speculation
- Values **actionability** for retail investors
- Uses **orthogonal dimensions** to avoid correlation collapse

---

## Support

For issues or questions:
1. Check training metadata: `training_metadata.json`
2. Review benchmark results: `benchmarks/test_set_results.json`
3. Run unit tests: `python filters/investment-risk/v5/test_inference.py`
