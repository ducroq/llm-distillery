# Cultural Discovery Filter v1

## Concept

Surfaces articles about:
1. **Discoveries** - New findings about art, culture, and history
2. **Connections** - Bridges between different peoples and civilizations

Target publications: **ovr.news** (Wisdom tab), **Busara**

## Dimensions

| Dimension | Weight | Category | Focus |
|-----------|--------|----------|-------|
| discovery_novelty | 0.25 | Discovery | What's new? What didn't we know? |
| heritage_significance | 0.20 | Discovery | Cultural/historical importance |
| cross_cultural_connection | 0.25 | Connection | Bridges between peoples |
| human_resonance | 0.15 | Connection | Lived experience, not dry facts |
| evidence_quality | 0.15 | Assessment | Well-researched? (gatekeeper) |

## High-Scoring Examples

- Archaeological discovery reveals shared ancestry of distant cultures
- Art restoration uncovers hidden layers showing cultural exchange
- Historical research connects modern tradition to ancient roots
- Cross-cultural music collaboration revives endangered heritage

## Filtered Out

- Political/conflict framing of cultural differences
- Tourism listicles ("10 must-see temples")
- Celebrity art auction news
- Cultural appropriation debates (polarizing, not connecting)

## Status

**Phase 6: Training** - Complete

### Completed

1. [x] Draft prompt-compressed.md (oracle instructions)
2. [x] Write prefilter rules (12/12 tests passing)
3. [x] Calibrate on 100 sample articles
4. [x] Collect training data (4,996 articles scored)
5. [x] Prepare training splits (80/10/10)
6. [x] Train student model (Qwen2.5-1.5B)

### Training Data Summary

- **Total scored:** 4,996 articles
- **With signal (≥3.0):** 445 articles (8.9%)
- **High tier (≥7.0):** 33 articles (0.7%)
- **Smooth distribution:** Examples at every 0.5 increment from 0→8

See `docs/adr/cultural-discovery-v1-calibration.md` for full analysis.

### Training Results

**Best validation MAE: 0.82** (target was <1.0)

Trained 9 epochs total (3 initial + 6 extended). Best model at epoch 6.

| Epoch | Train MAE | Val MAE | Val RMSE | Best? |
|-------|-----------|---------|----------|-------|
| 1     | 1.68      | 1.02    | 1.60     |       |
| 2     | 0.90      | 0.90    | 1.42     |       |
| 3     | 0.82      | 0.83    | 1.31     |       |
| 4     | 0.77      | 0.84    | 1.31     |       |
| 5     | 0.77      | 0.83    | 1.30     |       |
| 6     | 0.76      | **0.82**| 1.30     | Yes   |
| 7     | 0.76      | 0.82    | 1.29     |       |
| 8     | 0.76      | 0.82    | 1.29     |       |
| 9     | 0.76      | 0.82    | 1.29     |       |

**Per-dimension validation MAE (best epoch 6):**

| Dimension | MAE | RMSE |
|-----------|-----|------|
| discovery_novelty | 0.59 | 1.09 |
| heritage_significance | 0.72 | 1.28 |
| cross_cultural_connection | 0.68 | 1.11 |
| human_resonance | 0.88 | 1.32 |
| evidence_quality | 1.24 | 1.61 |

**Note:** evidence_quality plateaued at ~1.25 (epochs 5-9). This is likely a data limitation -- only 33 high-tier articles in training. More targeted evidence-quality training data would help for v2.

**Model:** Qwen2.5-1.5B + LoRA (18.5M/1.56B trainable, 1.18%)
**Hardware:** RTX 4080 (16GB), ~15 min/epoch
**Data:** 3,995 train / 500 val / 501 test

### Next Steps

1. [ ] Evaluate on test set (MAE per dimension)
2. [ ] Deploy to HuggingFace Hub
