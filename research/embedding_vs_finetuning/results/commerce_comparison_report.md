# Commerce Detection: Embedding vs Fine-Tuning

Generated: 2026-01-23

## Executive Summary

Unlike semantic dimension scoring (where fine-tuning is essential), **embedding + classifier matches or exceeds fine-tuned DistilBERT for commerce detection**. The best embedding approach achieves 98.3% F1, beating the 97.8% baseline.

This suggests a simpler, more maintainable approach for the commerce prefilter.

## Results

### Overall Comparison

| Approach | F1 | Precision | Recall | Inference |
|----------|-----|-----------|--------|-----------|
| **all-mpnet-base-v2 + MLP** | **98.3%** | **98.9%** | 97.8% | 0.009ms |
| all-mpnet-base-v2 + SVM | 97.8% | 97.8% | 97.8% | 0.244ms |
| Fine-tuned DistilBERT (v1) | 97.8% | 96.7% | 98.9% | 1.8ms GPU |
| bge-small-en-v1.5 + LogReg | 97.2% | 96.7% | 97.8% | 0.002ms |
| bge-small-en-v1.5 + MLP | 97.2% | 96.7% | 97.8% | 0.005ms |

### Best Model: all-mpnet-base-v2 + MLP

**Confusion Matrix (n=190 test samples):**

|  | Pred: Journalism | Pred: Commerce |
|--|------------------|----------------|
| **True: Journalism** | 100 | 1 |
| **True: Commerce** | 2 | 87 |

- **1 false positive**: Journalism article flagged as commerce
- **2 false negatives**: Commerce articles missed

### Why Embedding Works Here (But Not for Uplifting)

| Factor | Commerce Detection | Uplifting Scoring |
|--------|-------------------|-------------------|
| Task type | Binary classification | 6-dim regression |
| Concept clarity | High (promotional patterns) | Medium (abstract dimensions) |
| Regression to mean | N/A | Severe problem |
| Embedding fit | Good (captures commercial language) | Poor (misses nuanced dimensions) |

**Key insight**: General-purpose embeddings already capture "promotional language" patterns well. Fine-tuning adds little value for this specific task.

## Implications for Commerce Prefilter v2

### Advantages of Embedding Approach

| Aspect | Fine-tuned DistilBERT (v1) | Embedding + MLP (proposed v2) |
|--------|---------------------------|------------------------------|
| Accuracy | 97.8% F1 | **98.3% F1** |
| Model size | ~516MB | ~420MB (mpnet) + <1MB (MLP) |
| Training | Requires GPU, hours | CPU, minutes |
| Maintenance | Retrain full model | Retrain only MLP head |
| Multilingual | Built-in | Requires multilingual embedder |

### Concerns

1. **Multilingual support**: Current v1 uses `distilbert-base-multilingual-cased`. The `all-mpnet-base-v2` is English-only. For multilingual, consider:
   - `paraphrase-multilingual-mpnet-base-v2` (768 dims, 100+ languages)
   - `BAAI/bge-m3` (multilingual BGE)

2. **Embedding generation overhead**: Need to generate embeddings for each article. Could be cached if same embeddings used elsewhere.

3. **Production validation**: Test set performance may not reflect production edge cases (e.g., Aldi commercials).

## Recommendation

**Proceed with v2 using embedding + MLP approach**, with:

1. Test multilingual embedding models first
2. Validate on production edge cases before deployment
3. A/B test against v1 in shadow mode
4. Keep v1 as fallback

## Appendix: Full Results

| Model | Classifier | F1 | Precision | Recall | Inference (ms) |
|-------|------------|-----|-----------|--------|----------------|
| all-mpnet-base-v2 | MLP | 98.3% | 98.9% | 97.8% | 0.009 |
| all-mpnet-base-v2 | SVM_RBF | 97.8% | 97.8% | 97.8% | 0.244 |
| bge-small-en-v1.5 | LogisticRegression | 97.2% | 96.7% | 97.8% | 0.002 |
| bge-small-en-v1.5 | SVM_RBF | 97.2% | 96.7% | 97.8% | 0.185 |
| bge-small-en-v1.5 | MLP | 97.2% | 96.7% | 97.8% | 0.005 |
| all-mpnet-base-v2 | LogisticRegression | 95.6% | 93.5% | 97.8% | 0.006 |
| all-MiniLM-L6-v2 | SVM_RBF | 95.0% | 94.4% | 95.5% | 0.190 |
| all-MiniLM-L6-v2 | LogisticRegression | 94.0% | 91.5% | 96.6% | 0.003 |
| all-MiniLM-L6-v2 | MLP | 93.9% | 92.4% | 95.5% | 0.007 |
