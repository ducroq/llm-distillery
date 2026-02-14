# Embedding vs Fine-Tuning Experiment Results

Generated: 2026-01-23 12:18:06

## Experiment Overview

This experiment compares frozen embeddings + linear/MLP probes against
fine-tuned Qwen2.5-1.5B for semantic dimension scoring.

## Summary Results

**Dataset:** uplifting_v5
**Baseline (Fine-tuned Qwen2.5-1.5B) MAE:** 0.6800

| Embedding Model | Probe | MAE | RMSE | Spearman | vs Baseline |
|-----------------|-------|-----|------|----------|-------------|
|  **e5-large-v2** | mlp | 0.8598 | 1.1079 | 0.6989 | +0.1798 (+26.4%) |
| e5-large-v2 | ridge | 0.8933 | 1.1299 | 0.6741 | +0.2133 (+31.4%) |
| bge-large-en-v1.5 | mlp | 0.9014 | 1.1692 | 0.6655 | +0.2214 (+32.6%) |
| e5-large-v2 | lightgbm | 0.9048 | 1.1451 | 0.6701 | +0.2248 (+33.1%) |
| bge-large-en-v1.5 | ridge | 0.9275 | 1.1747 | 0.6420 | +0.2475 (+36.4%) |
| all-mpnet-base-v2 | mlp | 0.9390 | 1.2176 | 0.6230 | +0.2590 (+38.1%) |
| bge-large-en-v1.5 | lightgbm | 0.9391 | 1.1858 | 0.6404 | +0.2591 (+38.1%) |
| all-mpnet-base-v2 | lightgbm | 0.9526 | 1.2093 | 0.6110 | +0.2726 (+40.1%) |
| all-mpnet-base-v2 | ridge | 0.9589 | 1.2151 | 0.6054 | +0.2789 (+41.0%) |
| all-MiniLM-L6-v2 | mlp | 0.9751 | 1.2577 | 0.5891 | +0.2951 (+43.4%) |
| all-MiniLM-L6-v2 | ridge | 0.9994 | 1.2601 | 0.5654 | +0.3194 (+47.0%) |
| all-MiniLM-L6-v2 | lightgbm | 1.0105 | 1.2688 | 0.5617 | +0.3305 (+48.6%) |

## Best Model Analysis

**Best Linear Probe:** e5-large-v2 (Ridge) - MAE: 0.8933
**Best MLP Probe:** e5-large-v2 (MLP) - MAE: 0.8598
**Best Overall:** e5-large-v2 (mlp) - MAE: 0.8598

**Result:** Fine-tuning provides **significant advantage** over embedding approach.
- Gap: 0.1798 (26.4%)
- Recommend continuing with fine-tuning approach.

## Per-Dimension Analysis (Top 3 Models)

### Per-Dimension Results: e5-large-v2 (mlp)

| Dimension | MAE | RMSE | Spearman |
|-----------|-----|------|----------|
| human_wellbeing_impact | 0.9059 | 1.1493 | 0.6906 |
| social_cohesion_impact | 0.8740 | 1.1305 | 0.7188 |
| justice_rights_impact | 0.8027 | 1.0534 | 0.7450 |
| evidence_level | 0.7383 | 0.9727 | 0.7310 |
| benefit_distribution | 1.0047 | 1.2688 | 0.6401 |
| change_durability | 0.8334 | 1.0490 | 0.6681 |

### Per-Dimension Results: e5-large-v2 (ridge)

| Dimension | MAE | RMSE | Spearman |
|-----------|-----|------|----------|
| human_wellbeing_impact | 0.9384 | 1.1788 | 0.6563 |
| social_cohesion_impact | 0.9041 | 1.1556 | 0.7008 |
| justice_rights_impact | 0.8500 | 1.0875 | 0.7165 |
| evidence_level | 0.7476 | 0.9543 | 0.7394 |
| benefit_distribution | 1.0513 | 1.2885 | 0.5963 |
| change_durability | 0.8683 | 1.0869 | 0.6350 |

### Per-Dimension Results: bge-large-en-v1.5 (mlp)

| Dimension | MAE | RMSE | Spearman |
|-----------|-----|------|----------|
| human_wellbeing_impact | 0.9196 | 1.1847 | 0.6698 |
| social_cohesion_impact | 0.9103 | 1.1928 | 0.6858 |
| justice_rights_impact | 0.8489 | 1.1153 | 0.7037 |
| evidence_level | 0.8317 | 1.0720 | 0.6919 |
| benefit_distribution | 1.0324 | 1.3132 | 0.6053 |
| change_durability | 0.8653 | 1.1214 | 0.6367 |

## Recommendations

### Continue with Fine-Tuning

Fine-tuning provides a 26.4% advantage over embedding approaches.

**Potential improvements to try:**
1. Try larger embedding models (e.g., e5-mistral-7b-instruct)
2. Experiment with task-specific embedding fine-tuning
3. Ensemble multiple embedding models
4. Add domain-specific pre-training

### Efficiency Considerations

| Approach | Embedding Time | Probe Training | Inference |
|----------|---------------|----------------|-----------|
| Fine-tuned Qwen | N/A | 2-3 hours | 20-50ms |
| Embedding + Probe | ~1 min/1000 articles | ~1 min | <1ms |