# Change Request: Commerce Prefilter Integration

**CR ID:** CR-2026-001
**Date:** 2026-01-18
**Requested by:** LLM Distillery Team
**Target System:** NexusMind
**Priority:** Medium
**Estimated Effort:** 2-4 hours

---

## Summary

Add a commerce/promotional content prefilter to the NexusMind pipeline. This ML-based filter runs once per article before domain filters, blocking deals, affiliate content, and product promotions.

## Problem Statement

Backtesting on 56,336 scored articles revealed that **17.5% of articles** passing through to domain filters are commerce/promotional content:

- Deal roundups ("Save $500 on Solar Panels")
- Affiliate content with discount codes
- Product launch announcements
- Gift guides and shopping recommendations

This content:
1. Pollutes high/medium quality tiers
2. Wastes inference resources on non-journalism
3. Reduces signal-to-noise ratio for end users

## Proposed Solution

Integrate a pre-trained DistilBERT classifier that detects commerce content with:
- **97.8% F1 score** (96.7% precision, 98.9% recall)
- **~91ms inference** per article on CPU
- **Multilingual support** (104 languages)
- **Zero false positives** on high-tier journalism at threshold 0.95

### Pipeline Change

```
BEFORE:
RSS → Load Articles → Filter 1 → Filter 2 → Filter 3 → Output

AFTER:
RSS → Load Articles → [Commerce Prefilter] → Filter 1 → Filter 2 → Filter 3 → Output
                             ↓
                      prefiltered_out/commerce/
```

## Technical Specification

### 1. New Module

**Source:** `llm-distillery/filters/common/commerce_prefilter/v1/`
**Destination:** `NexusMind/filters/common/commerce_prefilter/v1/`

Files to copy:
```
commerce_prefilter/
├── __init__.py
└── v1/
    ├── __init__.py
    ├── inference.py          # CommercePrefilterSLM class
    └── models/
        └── distilbert/       # Pre-trained model (~516MB)
            ├── config.json
            ├── model.safetensors
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── special_tokens_map.json
            └── vocab.txt
```

### 2. Pipeline Integration

**File:** `scripts/main.py`
**Location:** `NexusMindPipeline.load_articles()` method

#### Add global singleton

```python
# At module level
_commerce_detector = None

def get_commerce_detector(threshold: float = 0.95):
    """Lazy load commerce detector singleton."""
    global _commerce_detector
    if _commerce_detector is None:
        from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM
        _commerce_detector = CommercePrefilterSLM(threshold=threshold)
    return _commerce_detector
```

#### Modify load_articles()

Insert after deduplication, before returning articles:

```python
# Commerce prefilter (runs once per article, before all domain filters)
commerce_enabled = self.config.get("pipeline.commerce_prefilter.enabled", True)
commerce_threshold = self.config.get("pipeline.commerce_prefilter.threshold", 0.95)

if commerce_enabled:
    commerce_detector = get_commerce_detector(threshold=commerce_threshold)
    self.logger.info(f"Commerce prefilter enabled (threshold: {commerce_threshold})")

    articles_filtered = []
    commerce_blocked = 0

    for article in articles:
        result = commerce_detector.is_commerce(article)
        if result["is_commerce"]:
            commerce_blocked += 1
            # Optional: save blocked articles for review
            if self.config.get("pipeline.commerce_prefilter.save_blocked", True):
                self._save_commerce_blocked(article, result["score"])
        else:
            articles_filtered.append(article)

    self.logger.info(f"Commerce prefilter: {commerce_blocked} blocked, {len(articles_filtered)} passed")
    stats["commerce_blocked"] = commerce_blocked
    articles = articles_filtered
```

#### Add helper method (optional)

```python
def _save_commerce_blocked(self, article: Dict, score: float):
    """Save commerce-blocked articles for review."""
    output_dir = self.filtered_dir / "prefiltered_out" / "commerce"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = output_dir / f"commerce_{timestamp}.jsonl"

    article["_commerce_score"] = score
    article["_blocked_at"] = datetime.now().isoformat()

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(article, ensure_ascii=False) + "\n")
```

### 3. Configuration

**File:** `config/app.yaml`

```yaml
pipeline:
  commerce_prefilter:
    enabled: true
    threshold: 0.95
    save_blocked: true
```

## API Reference

### CommercePrefilterSLM

```python
from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM

# Initialize
detector = CommercePrefilterSLM(threshold=0.95)

# Single article
result = detector.is_commerce(article)
# Returns: {"is_commerce": bool, "score": float, "inference_time_ms": float}

# Batch processing
results = detector.batch_predict(articles, batch_size=32)
# Returns: List[{"is_commerce": bool, "score": float, "inference_time_ms": float}]
```

### Input Format

Article dict with `title` and `content` (or `text`) fields:

```python
article = {
    "title": "Green Deals: Save $500 on Solar Panels",
    "content": "Limited time offer ends today..."
}
```

### Threshold Guidelines

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| 0.95 | ~99% | ~79% | Production (recommended) |
| 0.90 | ~97% | ~85% | Balanced |
| 0.85 | ~95% | ~90% | High recall |

## Testing Requirements

### Unit Tests

```python
def test_commerce_detector_loads():
    detector = CommercePrefilterSLM(threshold=0.95)
    assert detector.device in ["cpu", "cuda"]

def test_commerce_detection():
    detector = CommercePrefilterSLM(threshold=0.95)

    # Should detect commerce
    result = detector.is_commerce({
        "title": "Save 50% on Solar Panels Today",
        "content": "Use code SOLAR50 for discount..."
    })
    assert result["is_commerce"] == True
    assert result["score"] > 0.9

def test_journalism_passes():
    detector = CommercePrefilterSLM(threshold=0.95)

    # Should pass journalism
    result = detector.is_commerce({
        "title": "MIT Researchers Develop New Solar Cell",
        "content": "Scientists at MIT have achieved..."
    })
    assert result["is_commerce"] == False
    assert result["score"] < 0.5
```

### Integration Tests

```bash
# Test with small batch
python scripts/main.py --max-items 50 --dry-run

# Verify logs show:
# INFO: Commerce prefilter enabled (threshold: 0.95)
# INFO: Commerce prefilter: X blocked, Y passed
```

## Rollback Plan

### Quick Disable

```yaml
# config/app.yaml
pipeline:
  commerce_prefilter:
    enabled: false
```

### Full Removal

1. Remove commerce check from `load_articles()`
2. Remove `get_commerce_detector()` function
3. Optionally remove `filters/common/commerce_prefilter/` directory

## Expected Impact

| Metric | Current | After Integration |
|--------|---------|-------------------|
| Commerce in high/medium tiers | 15-20% | <1% |
| Articles per pipeline run | ~500 | ~425 |
| Pipeline runtime | baseline | +~40s (one-time model load) |
| Per-article overhead | 0ms | ~91ms |

### By Filter (based on backtest)

| Filter | Commerce Blocked | Impact |
|--------|------------------|--------|
| sustainability_technology | ~75/500 (15%) | Cleaner tech journalism |
| uplifting | ~100/500 (20%) | Less deal content |
| investment_risk | ~80/500 (16%) | Reduced noise |

## Dependencies

### Python Packages

```
torch>=2.0.0
transformers>=4.30.0
```

These should already be installed for existing SLM inference.

### Model Files

~516MB model directory must be copied to NexusMind. Consider:
- Git LFS for version control
- Or exclude from git and sync separately

## Documentation

Full documentation available in llm-distillery:

| Document | Path |
|----------|------|
| Backtest Report | `filters/common/commerce_prefilter/docs/BACKTEST_REPORT.md` |
| Training Report | `filters/common/commerce_prefilter/docs/TRAINING_REPORT.md` |
| Deployment Plan | `filters/common/commerce_prefilter/docs/DEPLOYMENT_PLAN.md` |
| Main README | `filters/common/commerce_prefilter/README.md` |

## Sign-off

| Role | Name | Date | Approved |
|------|------|------|----------|
| Requester | LLM Distillery Team | 2026-01-18 | ✓ |
| NexusMind Lead | | | |
| Code Review | | | |
| QA | | | |

---

## Appendix: Backtest Evidence

### Sample Commerce Detected (score >= 0.95)

| Score | Title | Source |
|-------|-------|--------|
| 0.996 | "Apple Watch Series 11 drops $100 to all-time low" | ai_engadget |
| 0.996 | "Anker SOLIX New Year Sale takes up to 65% off" | automotive_electrek |
| 0.996 | "5 presentes tech de Natal que parecem caros" | portuguese_canaltech |
| 0.996 | "Una tablet Xiaomi por menos de 95 euros" | spanish_xataka |

### Sample Journalism Passed (score < 0.5)

| Score | Title | Source |
|-------|-------|--------|
| 0.039 | "German company to sell more refurbished appliances" | deutsche_welle |
| 0.015 | "BYD announces new battery technology" | reuters |
| 0.008 | "EU proposes new renewable energy targets" | bbc |

### Model Comparison

| Model | F1 | High-Tier FP | Medium-Tier Blocked |
|-------|-----|--------------|---------------------|
| **DistilBERT** | **97.8%** | **0/3** | **517 (79%)** |
| MiniLM | 95.6% | 0/3 | 0 (0%) |
| XLM-RoBERTa | 97.2% | 1/3 | 520 (79%) |

DistilBERT selected for optimal precision/recall balance.
