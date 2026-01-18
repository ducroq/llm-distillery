# Commerce Prefilter - Deployment Plan

**Date:** 2026-01-18
**Target:** NexusMind production pipeline
**Model:** DistilBERT @ threshold 0.95

## Architecture

Commerce prefilter runs **ONCE** per article, before any domain filters:

```
Current Pipeline:
RSS → Load Articles → Filter 1 → Filter 2 → Filter 3 → Output

New Pipeline:
RSS → Load Articles → [COMMERCE PREFILTER] → Filter 1 → Filter 2 → Filter 3 → Output
                            ↓
                      prefiltered_out/commerce/
```

**Key principle:** Single commerce check per article, not per filter.

## Integration Point

**File:** `C:\local_dev\NexusMind\scripts\main.py`
**Method:** `NexusMindPipeline.load_articles()` (lines 369-473)

This is the central point where articles are loaded before ANY filter runs.

## Prerequisites

- [x] Model trained and validated (DistilBERT, 97.8% F1)
- [x] Backtest completed (517/657 commerce caught, 0 false positives)
- [x] Model synced locally in llm-distillery
- [ ] Copy commerce prefilter module to NexusMind
- [ ] Update NexusMind pipeline code
- [ ] Add configuration

## Deployment Steps

### Phase 1: Copy Commerce Module to NexusMind

```bash
# Create directory structure in NexusMind
mkdir -p C:\local_dev\NexusMind\filters\common\commerce_prefilter\v1\models

# Copy inference module
cp C:\local_dev\llm-distillery\filters\common\commerce_prefilter\v1\inference.py \
   C:\local_dev\NexusMind\filters\common\commerce_prefilter\v1\

cp C:\local_dev\llm-distillery\filters\common\commerce_prefilter\v1\__init__.py \
   C:\local_dev\NexusMind\filters\common\commerce_prefilter\v1\

# Copy trained model (~516MB)
cp -r C:\local_dev\llm-distillery\filters\common\commerce_prefilter\v1\models\distilbert \
   C:\local_dev\NexusMind\filters\common\commerce_prefilter\v1\models\

# Create __init__.py files
touch C:\local_dev\NexusMind\filters\common\__init__.py
touch C:\local_dev\NexusMind\filters\common\commerce_prefilter\__init__.py
```

### Phase 2: Update NexusMind Pipeline

**File:** `C:\local_dev\NexusMind\scripts\main.py`

#### 2.1 Add import at top of file

```python
# Commerce prefilter (optional, lazy loaded)
_commerce_detector = None
```

#### 2.2 Add helper function

```python
def get_commerce_detector(threshold: float = 0.95):
    """Lazy load commerce detector singleton."""
    global _commerce_detector
    if _commerce_detector is None:
        from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM
        _commerce_detector = CommercePrefilterSLM(threshold=threshold)
    return _commerce_detector
```

#### 2.3 Modify load_articles() method

In `NexusMindPipeline.load_articles()`, add commerce check after deduplication:

```python
def load_articles(self, input_files: List[Path], filter_name: str) -> Tuple[List[Dict], Dict, Set[str]]:
    # ... existing code for loading and deduplication ...

    # NEW: Commerce prefilter (runs once per article)
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
                # Optionally save to prefiltered_out/commerce/
                self._save_commerce_blocked(article, result["score"])
            else:
                articles_filtered.append(article)

        self.logger.info(f"Commerce prefilter: {commerce_blocked} blocked, {len(articles_filtered)} passed")
        stats["commerce_blocked"] = commerce_blocked
        articles = articles_filtered

    return articles, stats, processed_ids
```

#### 2.4 Add method to save blocked articles (optional)

```python
def _save_commerce_blocked(self, article: Dict, score: float):
    """Save commerce-blocked articles for review."""
    output_dir = self.filtered_dir / "prefiltered_out" / "commerce"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"commerce_{timestamp}.jsonl"

    article["commerce_score"] = score
    article["blocked_at"] = datetime.now().isoformat()

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(article, ensure_ascii=False) + "\n")
```

### Phase 3: Add Configuration

**File:** `C:\local_dev\NexusMind\config\app.yaml`

```yaml
pipeline:
  commerce_prefilter:
    enabled: true
    threshold: 0.95
    save_blocked: true  # Save blocked articles for review
```

### Phase 4: Test Locally

```bash
cd C:\local_dev\NexusMind

# Test commerce detector loads
python -c "
from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM
detector = CommercePrefilterSLM(threshold=0.95)
print(f'Loaded. Device: {detector.device}')

# Test on sample article
result = detector.is_commerce({
    'title': 'Green Deals: Save \$500 on Solar Panels',
    'content': 'Limited time offer ends today!'
})
print(f'Commerce test: {result}')
"

# Run pipeline with small batch
python scripts/main.py --max-items 50 --dry-run
```

### Phase 5: Deploy to Production

```bash
# Commit changes
cd C:\local_dev\NexusMind
git add filters/common/commerce_prefilter/
git add scripts/main.py
git add config/app.yaml
git commit -m "Add commerce prefilter to pipeline

- Runs ONCE per article before all domain filters
- DistilBERT model, 97.8% F1, threshold 0.95
- Blocks promotional/deals content
- Saves blocked articles for review"

git push
```

### Phase 6: Monitor

Check logs for:
```
INFO: Commerce prefilter enabled (threshold: 0.95)
INFO: Commerce prefilter: 75 blocked, 425 passed
```

Check output:
```
C:\local_dev\NexusMind\data\filtered\prefiltered_out\commerce\
```

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Commerce in high/medium tiers | ~15-20% | <1% |
| Articles processed per run | ~500 | ~425 (75 blocked) |
| Quality of high tier | Mixed | Pure journalism |

## Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Enable/disable commerce prefilter |
| `threshold` | `0.95` | Score threshold (higher = stricter) |
| `save_blocked` | `true` | Save blocked articles for review |

### Threshold Guidelines

| Threshold | Behavior |
|-----------|----------|
| 0.95 | High precision - only obvious commerce blocked |
| 0.90 | Balanced - catches more, rare false positives |
| 0.85 | High recall - aggressive, may block product journalism |

## Rollback

To disable without removing code:

```yaml
# config/app.yaml
pipeline:
  commerce_prefilter:
    enabled: false
```

Or remove the commerce check from `load_articles()`.

## Files Summary

| Location | File | Change |
|----------|------|--------|
| NexusMind | `filters/common/commerce_prefilter/v1/` | New module (copy from llm-distillery) |
| NexusMind | `scripts/main.py` | Add commerce check in load_articles() |
| NexusMind | `config/app.yaml` | Add commerce_prefilter config section |

## Success Criteria

- [ ] Commerce detector loads without error
- [ ] Pipeline runs with commerce prefilter enabled
- [ ] ~15-20% of articles blocked as commerce
- [ ] Zero false positives in high tier (spot check)
- [ ] Blocked articles saved for review
