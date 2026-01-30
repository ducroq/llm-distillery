# Cultural Discovery Filter v3 - Merged Training Data

**Status:** Production Ready
**Created:** 2026-01-29
**Philosophy:** "Culture is humanity's shared inheritance - discovery reveals connections."

## Overview

Version 3 merges the training datasets from v1 and v2 to achieve better performance across all tiers. The key insight was that neither v1 nor v2 alone had sufficient data to learn the score distribution effectively.

### Problem with v1 and v2

| Version | Training Examples | Distribution | Val MAE | Issue |
|---------|-------------------|--------------|---------|-------|
| **v1** | 3,995 | 94% low, 5% med, 1% high | 0.82 | Regression-to-mean on high scores |
| **v2** | 2,333 | 80% low, 17% med, 3% high | 1.47 | Enriched distribution but insufficient data |

**Root cause:** v2's screening filter successfully enriched the score distribution (more medium/high examples), but with 42% less training data and fewer epochs, the model couldn't learn the harder distribution.

### Solution in v3

Merge v1 and v2 datasets to get:
- Larger dataset (7,827 unique articles, 88 duplicates removed)
- Better balance of distribution coverage and data volume
- Distribution: 88% low, 10% medium, 2% high

## Dimension Framework

### Core Discovery Dimensions

| Dimension | Weight | Question |
|-----------|--------|----------|
| **Discovery Novelty** | 25% | HOW new or unknown is this? (familiar → rarely reported → little known → first discovery → paradigm-changing) |
| **Heritage Significance** | 20% | HOW important culturally? (local interest → regional heritage → national significance → cross-cultural heritage → UNESCO-level) |
| **Cross-Cultural Connection** | 25% | HOW does it connect cultures? (single culture → cultural exchange → multi-culture bridge → civilization dialogue → universal insight) |

### Assessment Dimensions

| Dimension | Weight | Question |
|-----------|--------|----------|
| **Human Resonance** | 15% | HOW does it touch humanity? (abstract → intellectual → emotional → personal transformation → shared human experience) |
| **Evidence Quality** | 15% | HOW documented? (speculation → opinion → journalism → expert analysis → academic/archaeological) **[GATEKEEPER]** |

### Why These Dimensions Work

Cultural discovery is inherently about connections across time and space:

| Scenario | Discovery | Heritage | Cross-Cultural | Resonance | Evidence |
|----------|-----------|----------|----------------|-----------|----------|
| **Silk Road temple syncretism** | 8 (unexpected) | 9 (UNESCO-level) | **10** (Buddhist+Zoroastrian) | 8 | 9 (archaeological) |
| **Local food festival** | 2 (familiar) | 4 (local) | 3 (single culture) | 6 | 7 |
| **Museum repatriation debate** | 5 | 7 | **8** (colonial legacy) | 7 | 8 |
| **Ancient genome study** | **9** (new finding) | 8 | 7 | 5 (abstract) | **10** (peer-reviewed) |
| **Tourism marketing piece** | 1 | 3 | 2 | 4 | **2** (promotional) |

## Gatekeeper Rule

**Evidence Quality < 3 → cap overall signal at 3.0**

Speculation without documentation cannot claim cultural significance.

## Signal Tiers

| Tier | Score Range | Description | Target Use |
|------|-------------|-------------|------------|
| **HIGH** | ≥ 7.0 | Significant discovery or deep cross-cultural insight | Featured content |
| **MEDIUM** | ≥ 4.0 | Meaningful cultural content with some discovery value | Regular feed |
| **LOW** | < 4.0 | Superficial, speculative, or single-culture content | Filter out |

## Files

```
filters/cultural-discovery/v3/
├── README.md                 # This file
├── config.yaml               # Filter configuration
├── prompt-compressed.md      # Oracle prompt (calibrated)
├── prefilter.py              # Keyword-based prefilter
├── base_scorer.py            # Shared scoring logic
├── inference.py              # Production inference module
├── training_metadata.json    # Training configuration
├── training_history.json     # Training curves
└── model/                    # Trained LoRA adapter
    ├── adapter_model.safetensors
    ├── adapter_config.json
    └── tokenizer files...
```

## Training Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Validation MAE** | **0.77** | <0.80 | ✅ PASS |
| Training epochs | 6 | - | - |
| Best epoch | 4 | - | - |
| Trainable params | 18.5M | - | - |
| Training examples | 6,261 | - | - |
| Validation examples | 783 | - | - |
| Test examples | 783 | - | - |
| Model size | ~74MB (adapter) | - | - |

### Per-Dimension Validation MAE (Epoch 4)

| Dimension | MAE | Status |
|-----------|-----|--------|
| discovery_novelty | 0.54 | ✅ Best |
| heritage_significance | 0.58 | ✅ |
| cross_cultural_connection | 0.62 | ✅ |
| human_resonance | 0.77 | ✅ |
| evidence_quality | 1.36 | ⚠️ Hardest dimension |

### Tier-Level Performance (vs v1)

| Tier | v1 MAE | v3 MAE | Improvement |
|------|--------|--------|-------------|
| LOW (0-3.9) | 0.75 | 0.60 | +20% |
| MEDIUM (4-6.9) | 2.85 | 1.73 | **+39%** |
| HIGH (7-10) | 3.49 | 2.69 | **+23%** |

The key improvement is on medium and high-tier articles, which matter most for discovery.

## Development History

### v1 (2026-01)
- First production version
- 5,000 scored articles (random sampling)
- Issue: 94% low-scoring data → regression-to-mean

### v2 (2026-01)
- Applied screening filter before oracle scoring
- 2,919 articles with enriched distribution
- Issue: Not enough data for harder distribution

### v3 (2026-01)
- Merged v1 + v2 datasets (7,827 unique articles)
- Best of both: data volume + distribution coverage
- **Result:** 0.77 MAE, 39% better on medium-tier

## Usage

### Python API

```python
from filters.cultural_discovery.v3.inference import CulturalDiscoveryScorer

scorer = CulturalDiscoveryScorer()

article = {
    "title": "Ancient Silk Road Temple Reveals Buddhist-Zoroastrian Syncretism",
    "content": "Excavations at a 4th-century temple in Uzbekistan..."
}

result = scorer.score_article(article)

print(f"Tier: {result['tier']}")
print(f"Weighted Average: {result['weighted_average']:.2f}")
for dim, score in result['scores'].items():
    print(f"  {dim}: {score:.2f}")
```

### CLI

```bash
# Interactive demo
python filters/cultural-discovery/v3/inference.py

# Batch scoring
python filters/cultural-discovery/v3/inference.py \
    --input articles.jsonl \
    --output results.jsonl
```

## Target Applications

- **ovr.news Wisdom tab**: Featured cultural discovery content
- **Busara**: Cross-cultural understanding and heritage preservation
- **Content curation**: Identifying meaningful cultural journalism

## Changelog

### v3.0 (2026-01-29)
- Merged v1 (4,996) + v2 (2,919) datasets → 7,827 unique articles
- Achieved 0.77 MAE (vs v1's 0.82, v2's 1.47)
- 39% improvement on medium-tier, 23% improvement on high-tier
- Production ready for ovr.news and Busara
