# Commerce Prefilter v2 Design

**Status:** Planning
**Date:** 2026-01-23
**Based on:** [Embedding vs Fine-tuning Research](../../../research/embedding_vs_finetuning/results/commerce_comparison_report.md)

## Summary

Replace fine-tuned DistilBERT with frozen embeddings + MLP classifier, achieving:
- **Better accuracy**: 98.3% F1 vs 97.8% F1
- **Simpler maintenance**: Only retrain small MLP, not full transformer
- **Faster classifier inference**: 0.009ms vs 1.8ms (embedding time separate)

## Architecture Change

### v1 (Current)
```
Article → [Fine-tuned DistilBERT] → commerce_score (0-1)
                  ↓
              516MB model
```

### v2 (Proposed)
```
Article → [Frozen Embedder] → 768-dim vector → [MLP Classifier] → commerce_score (0-1)
               ↓                                      ↓
          ~420MB model                            <1MB weights
          (sentence-transformers)                 (sklearn/torch)
```

## Model Selection

### Embedder Options

| Model | Dims | Languages | Size | Notes |
|-------|------|-----------|------|-------|
| `all-mpnet-base-v2` | 768 | English | 420MB | Best F1 in tests |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 100+ | 970MB | Multilingual version |
| `BAAI/bge-small-en-v1.5` | 384 | English | 130MB | Smaller, still 97.2% F1 |
| `BAAI/bge-m3` | 1024 | 100+ | 2.2GB | Best multilingual, larger |

**Recommendation:** Start with `paraphrase-multilingual-mpnet-base-v2` for multilingual parity with v1.

### Classifier

- **Architecture**: MLP (256 → 128 → 1)
- **Training**: ~1 minute on CPU
- **Inference**: <0.01ms per sample (after embedding)

## Implementation Plan

### Phase 1: Multilingual Validation (1-2 hours)

1. Test `paraphrase-multilingual-mpnet-base-v2` on existing test set
2. Verify F1 >= 97% with multilingual embedder
3. If fails, test `BAAI/bge-m3`

**Deliverable:** Confirmed embedder choice

### Phase 2: Training Pipeline (2-3 hours)

1. Create `train_embedding_classifier.py`:
   - Load training data
   - Generate embeddings (cache to disk)
   - Train MLP with early stopping
   - Save model weights + scaler

2. Create `v2/` directory structure:
   ```
   filters/common/commerce_prefilter/v2/
   ├── config.yaml
   ├── inference.py
   ├── models/
   │   ├── embeddings/           # Cached if needed
   │   ├── mlp_weights.pkl       # MLP state dict
   │   └── scaler.pkl            # StandardScaler
   └── README.md
   ```

**Deliverable:** Trained v2 model

### Phase 3: Inference Implementation (1-2 hours)

1. Create `v2/inference.py`:
   ```python
   class CommercePrefilterV2:
       def __init__(self, threshold=0.95):
           self.embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
           self.classifier = load_mlp('models/mlp_weights.pkl')
           self.scaler = load_scaler('models/scaler.pkl')
           self.threshold = threshold

       def is_commerce(self, article) -> dict:
           text = f"{article['title']} {article['content']}"
           embedding = self.embedder.encode([text])
           scaled = self.scaler.transform(embedding)
           score = self.classifier.predict_proba(scaled)[0, 1]
           return {
               'is_commerce': score >= self.threshold,
               'score': score,
               'version': 'v2'
           }
   ```

2. Ensure API compatibility with v1

**Deliverable:** Working v2 inference

### Phase 4: Validation (2-3 hours)

1. **Test set evaluation**: Confirm F1 >= 97.5%
2. **Edge case testing**:
   - Aldi/retail grocery commercials
   - Electrek green deals
   - Product reviews vs product announcements
3. **Backtest on NexusMind scored articles** (56K articles)

**Deliverable:** Validation report

### Phase 5: Shadow Mode Deployment (1 week)

1. Run v2 alongside v1 in NexusMind
2. Log both predictions, compare disagreements
3. Manual review of disagreements

**Deliverable:** Shadow mode comparison report

### Phase 6: Production Cutover

1. Update `BasePreFilter` to use v2 by default
2. Keep v1 available as fallback
3. Monitor for regressions

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Multilingual embedder underperforms | Test before committing; fallback to v1 |
| Production edge cases missed | Shadow mode before cutover |
| Embedding latency too high | Consider caching or batch processing |
| Model size increase | Use smaller embedder if needed (bge-small) |

## Success Criteria

| Metric | Target | v1 Baseline |
|--------|--------|-------------|
| F1 Score | >= 97.5% | 97.8% |
| Precision | >= 96% | 96.7% |
| Recall | >= 97% | 98.9% |
| Backtest agreement with v1 | >= 95% | - |
| Edge case accuracy | No Aldi-type misses | Has issues |

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1. Multilingual validation | 1-2 hours | - |
| 2. Training pipeline | 2-3 hours | Phase 1 |
| 3. Inference implementation | 1-2 hours | Phase 2 |
| 4. Validation | 2-3 hours | Phase 3 |
| 5. Shadow mode | 1 week | Phase 4 |
| 6. Production cutover | 1 day | Phase 5 approval |

**Total development time:** ~8-12 hours
**Total validation time:** 1 week shadow mode

## Open Questions

1. Should we cache embeddings for articles that pass through multiple filters?
2. Do we need GPU support for embedding generation in production?
3. Should v2 support both English-only and multilingual modes?

---

*Document created as part of embedding vs fine-tuning research. See research/embedding_vs_finetuning/ for experimental details.*
