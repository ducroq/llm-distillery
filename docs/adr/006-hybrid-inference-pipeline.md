# ADR 006: Two-Stage Hybrid Inference Pipeline

**Date**: 2026-02-14
**Status**: Accepted
**Context**: Inference bottleneck at scale — single-stage Qwen2.5-1.5B (20-50ms/article) too slow for thousands of articles/day

## Decision

Use a **two-stage hybrid inference pipeline** that combines fast embedding probes (Stage 1) with the existing fine-tuned model (Stage 2):

```
Article -> Prefilter -> Stage 1 (embedding + MLP probe, ~8ms)
                            |
                +-----------+-----------+
                |                       |
          weighted_avg < 3.5     weighted_avg >= 3.5
                |                       |
          Return LOW result      Stage 2 (fine-tuned model, ~25ms)
          (probe scores)                |
                                  Return precise result
```

Key parameters:
- **Embedding model**: multilingual-e5-large (best probe accuracy from research)
- **Probe**: Two-layer MLP (256, 128) trained on frozen embeddings
- **Stage 1 threshold**: 3.5 (calibrated on 24K production articles for <2% FN rate on MEDIUM+)

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

## Expected Performance

| Metric | Standard | Hybrid | Change |
|--------|----------|--------|--------|
| LOW articles | ~25ms | ~8ms | 3x faster |
| MEDIUM+ articles | ~25ms | ~33ms (8+25) | 32% slower |
| Average (68% LOW) | ~25ms | ~14ms | 40% faster |
| MEDIUM+ accuracy | Baseline | Identical | No change |
| LOW accuracy | Baseline | ~18% worse MAE | Acceptable |
| False negative rate | 0% | <2% | Acceptable |

### Threshold calibration (production data)

Calibrated on 24,304 production-scored articles (19K MEDIUM + 5K LOW).

**Probe v1** (trained on research data): Systematic bias of -0.54, FN rate 9.0% at threshold 3.0. Unusable.

**Probe v2** (retrained on production data): Bias +0.007, val MAE 0.49, weighted avg MAE 0.39.

| Threshold | FN Rate | FN Count | Notes |
|-----------|---------|----------|-------|
| 3.00 | 0.1% | 5/3783 | Very safe, minimal speedup |
| 3.25 | 0.6% | 22/3783 | Safe |
| **3.50** | **1.7%** | **63/3783** | **Selected: best <2% FN with meaningful speedup** |
| 3.75 | 4.5% | 171/3783 | Too aggressive |

In production with ~68% LOW articles, threshold 3.5 is expected to yield ~1.5-2x speedup.

## Consequences

### Positive

- ~40% faster average inference time
- Scales to higher article volumes without hardware changes
- No accuracy loss on MEDIUM/HIGH tier articles
- Existing inference code untouched — hybrid is opt-in
- Generalizes to all filters (shared infrastructure)

### Negative

- MEDIUM+ articles slightly slower (double pass through Stage 1 + Stage 2)
- Stage 1 LOW articles have less precise scores (~18% worse MAE)
- Additional dependency: sentence-transformers library
- Probe must be retrained when filter is retrained

### Trade-offs Accepted

- Accept ~18% worse MAE on LOW articles (they're LOW either way)
- Accept <2% false negatives (MEDIUM articles misclassified as LOW by Stage 1)
- Accept slightly slower MEDIUM+ articles for much faster LOW articles

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
