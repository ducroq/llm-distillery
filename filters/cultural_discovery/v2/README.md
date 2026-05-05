# Cultural Discovery Filter v2

**Version:** 2.0
**Created:** 2026-01-29
**Status:** In Development

## Purpose

Rate content for cultural discovery and cross-cultural connection value. Surfaces articles about art, culture, history discoveries and connections between peoples/civilizations.

## v2 Changes from v1

### Problem Addressed: Regression-to-Mean

v1 achieved overall MAE 0.82 but showed poor performance on high-scoring articles:
- 94% of training data scored < 4.0 (low tier)
- Only 0.7% scored >= 7.0 (high tier)
- evidence_quality MAE of **4.12** for articles scoring 8-10
- Model learned to predict ~2.0 for everything (hedging behavior)

### Solution: Screening Filter

v2 applies a **screening filter** before oracle scoring to enrich the training distribution:

| Score Range | v1 Random | v2 Screened |
|-------------|-----------|-------------|
| Low (0-3) | ~94% | ~50-60% |
| Medium (4-6) | ~5% | ~30-35% |
| High (7-10) | ~1% | ~10-15% |

### Other Improvements

- Head+tail preprocessing (256+256 tokens) for context window optimization
- Same prompt and prefilter as v1 (validated to be effective)

## Files

| File | Purpose |
|------|---------|
| `config.yaml` | Filter configuration, dimensions, tiers |
| `prompt-compressed.md` | Oracle prompt (same as v1) |
| `prefilter.py` | Inference prefilter - noise reduction (same as v1) |
| `screening_filter.py` | **NEW** - Training data enrichment filter |

## Dimensions (5 total, 0-10 scale)

| Dimension | Weight | Category |
|-----------|--------|----------|
| discovery_novelty | 25% | Discovery |
| heritage_significance | 20% | Discovery |
| cross_cultural_connection | 25% | Connection |
| human_resonance | 15% | Connection |
| evidence_quality | 15% | Assessment (gatekeeper) |

## Tiers

- **High (>= 7.0):** Significant discovery or deep cross-cultural insight
- **Medium (>= 4.0):** Meaningful cultural content with some value
- **Low (< 4.0):** Superficial or single-culture content

## Gatekeepers

- **evidence_quality < 3:** Caps overall score at 3.0 (misinformation prevention)

## Usage

### Screening (Training Data Collection)

```bash
# Screen raw articles
python filters/cultural-discovery/v2/screening_filter.py \
    --input datasets/raw/master_dataset.jsonl \
    --output sandbox/screened_cultural_discovery.jsonl \
    --target 10000 \
    --verbose

# Score screened articles
python -m ground_truth.batch_scorer \
    --filter filters/cultural-discovery/v2 \
    --source sandbox/screened_cultural_discovery.jsonl \
    --output-dir datasets/scored/cultural-discovery-v2 \
    --llm gemini-flash \
    --random-sample
```

### Inference

```python
from filters.cultural_discovery.v2.inference import CulturalDiscoveryScorer

scorer = CulturalDiscoveryScorer()
result = scorer.score_article({
    "title": "Maya Temple Discovery Reveals Cross-Cultural Trade",
    "content": "Archaeologists have unearthed..."
})
print(f"Score: {result['weighted_average']:.2f}, Tier: {result['tier']}")
```

## Training Results

| Metric | v1 | v2 (target) |
|--------|----|----|
| Overall MAE | 0.82 | < 0.75 |
| High-score MAE (8-10) | 4.12 | < 1.5 |
| evidence_quality MAE | 1.24 | < 1.0 |

## References

- [ADR-003: Screening Filter Methodology](../../../docs/adr/003-screening-filter-for-training-data.md)
- [Cultural Discovery v1 Calibration](../../../docs/adr/cultural-discovery-v1-calibration.md)
- [Screening Filter Template](../../../docs/templates/screening-filter-template.md)
