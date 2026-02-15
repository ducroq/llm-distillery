# ADR 006: Two-Stage Hybrid Inference Pipeline

**Date**: 2026-02-14
**Status**: Accepted
**Context**: Inference bottleneck at scale — single-stage Qwen2.5-1.5B (20-50ms/article) too slow for thousands of articles/day

## Decision

Use a **two-stage hybrid inference pipeline** that combines fast embedding probes (Stage 1) with the existing fine-tuned model (Stage 2):

```
Article -> Prefilter -> Stage 1 (embedding + MLP probe, ~1.3ms)
                            |
                +-----------+-----------+
                |                       |
          weighted_avg < 4.5     weighted_avg >= 4.5
                |                       |
          Return probe scores    Stage 2 (fine-tuned model, ~39ms)
          (fast, approximate)           |
                                  Return precise result
```

Key parameters:
- **Embedding model**: multilingual-e5-small (1.3ms/article, 100 languages, 384d)
- **Probe**: Two-layer MLP (256, 128) trained on 24K production-scored articles
- **Stage 1 threshold**: 4.5 (accepts lower accuracy on borderline MEDIUM 4.0-4.5 range)

## Context

### The Problem

Scoring thousands of articles per day, the single-stage Qwen2.5-1.5B inference (20-50ms/article) is becoming a bottleneck. Need faster inference without sacrificing accuracy on the articles that matter (MEDIUM and HIGH tier).

### Research Foundation

Embedding vs fine-tuning research (`research/embedding_vs_finetuning/`) showed:
- **Fine-tuned Qwen2.5-1.5B**: MAE 0.68 (best accuracy)
- **Embedding MLP probes**: MAE ~0.80 (18% worse, but 20-50x faster)
- **Key insight**: Probes are good enough to identify obvious LOW articles

### Why Not Just Use Probes?

Probes are ~18% worse on MAE overall, and critically worse on boundary cases (MEDIUM/HIGH distinction). For tier assignment accuracy, the fine-tuned model is necessary. But ~68% of articles are LOW — for those, probe accuracy is sufficient.

## Architecture

### Shared infrastructure (`filters/common/`)

- `embedding_stage.py`: EmbeddingStage class with singleton embedding model loading
- `hybrid_scorer.py`: HybridScorer abstract base class

### Per-filter integration

Each filter adds:
- `probe/embedding_probe.pkl`: Trained MLP probe
- `inference_hybrid.py`: Filter-specific HybridScorer subclass
- `config.yaml`: hybrid_inference section

### No changes to existing code

The hybrid scorer wraps the existing scorer (Stage 2) without modifying it. Existing `inference.py` files remain unchanged and can be used standalone.

## Measured Performance (RTX 4080, 5K articles, 79% MEDIUM / 21% LOW)

### Embedding model comparison

| Model | Params | Dim | Speed | Probe MAE | Recall@4.5 |
|-------|--------|-----|-------|-----------|------------|
| e5-small | 118M | 384 | **1.3ms** | 0.451 | 72.3% |
| e5-base | 278M | 768 | 3.0ms | 0.404 | 73.8% |
| e5-large | 560M | 1024 | 9.2ms | 0.391 | 75.0% |

**Selected: e5-small** — 7x faster than e5-large, only 2.7% less recall. All models support 100 languages.

### Benchmark results (threshold=4.5, e5-small)

| Metric | Standard | Hybrid | Change |
|--------|----------|--------|--------|
| Time per article | 37.9ms | **18.1ms** | **2.09x faster** |
| Stage 1 (embed+probe) | — | 1.3ms | all articles |
| Stage 2 (Qwen model) | 37.9ms | 37.9ms | 44% of articles |
| Skip rate | 0% | 56% | — |

Production (~68% LOW) expected speedup: ~2.5-3x.

### Threshold calibration (production data)

Calibrated on 24,304 production-scored articles (19K MEDIUM + 5K LOW).

**Probe v1** (trained on research data): Systematic bias of -0.54, FN rate 9.0% at threshold 3.0. Unusable.

**Probe v2** (retrained on production data): Bias +0.007, val MAE 0.49, weighted avg MAE 0.39.

**Selected threshold: 4.5** — borderline MEDIUM articles (4.0-4.5 range) get approximate probe scores instead of precise model scores. Acceptable trade-off: these articles are already near the LOW/MEDIUM boundary and precise scoring adds limited value.

### Probe score distribution

| Tier | At threshold 4.5 |
|------|------------------|
| LOW articles | 99.0% filtered out |
| MEDIUM articles | 41.7% get probe scores (the 4.0-4.5 borderline range) |

## Consequences

### Positive

- ~2x faster average inference in production
- Scales to higher article volumes without hardware changes
- HIGH tier articles always get precise Stage 2 scoring
- Existing inference code untouched — hybrid is opt-in
- Generalizes to all filters (shared infrastructure)

### Negative

- All articles pay Stage 1 cost (10.5ms) — MEDIUM+ articles are slightly slower
- Borderline MEDIUM articles (4.0-4.5) get less precise probe scores
- Additional dependency: sentence-transformers library
- Probe must be retrained when filter is retrained

### Trade-offs Accepted

- Accept approximate scores on LOW + borderline MEDIUM articles (probe MAE 0.49)
- Accept slightly slower upper-MEDIUM/HIGH articles for much faster LOW articles
- Tier thresholds are somewhat arbitrary; precision on the 4.0-4.5 boundary matters less

## Generalization

To add hybrid inference to a new filter:

1. Generate embeddings for training data (embedding model is shared)
2. Train MLP probe (~2 seconds): `python research/embedding_vs_finetuning/train_probes.py`
3. Copy probe to `filters/{name}/{version}/probe/embedding_probe.pkl`
4. Create `inference_hybrid.py` subclassing `HybridScorer`
5. Calibrate threshold: `python evaluation/calibrate_hybrid_threshold.py`

## References

- Research: `research/embedding_vs_finetuning/`
- Embedding stage: `filters/common/embedding_stage.py`
- Hybrid scorer: `filters/common/hybrid_scorer.py`
- Calibration: `evaluation/calibrate_hybrid_threshold.py`
- First integration: `filters/uplifting/v5/inference_hybrid.py`
