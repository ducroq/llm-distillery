# Sustainability Technology v1 - Training Report

**Date**: 2025-11-27
**Status**: ✅ **PRODUCTION READY**

---

## Executive Summary

The sustainability_technology v1 filter has successfully completed training and evaluation. The distilled model achieves a **test MAE of 0.690** on a 6-dimensional LCSA-based scoring system, with all quality gates passed.

| Phase | Status | Key Outcome |
|-------|--------|-------------|
| Calibration | ✅ Complete | v3 prompt approved (PC1: 63.8%) |
| Training Data | ✅ Complete | 8,989 examples generated |
| Model Training | ✅ Complete | 3 epochs, val MAE: 0.712 |
| Benchmarking | ✅ Complete | test MAE: 0.690 |
| Deployment | 🔜 Ready | Pending setup |

---

## Filter Specification

**Framework**: Life Cycle Sustainability Assessment (LCSA)
**Base Model**: Qwen/Qwen2.5-1.5B
**Training Mode**: Knowledge Distillation (LoRA)

### Dimensions (6)

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Technology Readiness Level | 15% | Deployment stage (TRL 1-9) |
| Technical Performance | 15% | Real-world reliability and efficiency |
| Economic Competitiveness | 20% | Life Cycle Cost (LCC) competitiveness |
| Life Cycle Environmental Impact | 30% | Holistic environmental assessment |
| Social Equity Impact | 10% | Jobs, ethics, equitable access |
| Governance/Systemic Impact | 10% | Systemic disruption potential |

### Tier Thresholds

| Tier | Threshold | Description |
|------|-----------|-------------|
| High Sustainability | ≥7.0 | Mass deployed, proven sustainable, competitive |
| Medium-High | ≥5.0 | Commercial deployment, good sustainability |
| Medium | ≥3.0 | Pilot/early commercial, mixed profile |
| Low | <3.0 | Lab stage or poor sustainability performance |

### Gatekeepers

- **TRL Gatekeeper**: If `technology_readiness_level < 3.0`, overall score capped at 2.9
- **Rationale**: Lab-only technologies cannot achieve high sustainability scores

---

## Calibration Journey

### Prompt Evolution

| Version | PC1 | False Positive Rate | Status |
|---------|-----|---------------------|--------|
| v1 (Original) | 59.0% | 1.7% (high) | ❌ Rejected |
| v2 (Binary) | 81.4% | 0.4% | ❌ Rejected (lost independence) |
| **v3 (Final)** | **63.8%** | **0.3%** | ✅ Approved |

### v3 Key Features

- Strict technology definition with explicit exclusions
- Independent dimension evaluation guidance
- Balanced scoring (not binary 0 or high)
- 16.7% dimensional redundancy (low)

### Manual Validation Results

Tested on 15 known false positives from v1:
- 53% correctly scored low (<3.5)
- 47% scored medium (5-7) - deemed acceptable as edge cases

---

## Training Data

**Source**: Oracle-scored articles using Gemini Flash
**Total Examples**: 8,989

| Split | Examples | Percentage |
|-------|----------|------------|
| Train | 7,990 | 80% |
| Validation | 999 | 10% |
| Test | 1,000 | 10% |

### Score Distribution

| Range | Percentage | Description |
|-------|------------|-------------|
| High (7-10) | 0.3% | True sustainability tech |
| Medium-High (5-7) | 7.7% | Good candidates |
| Medium (3-5) | 19.3% | Mixed profile |
| Low (0-3) | 72.7% | Not relevant |

---

## Model Architecture

**Base Model**: Qwen/Qwen2.5-1.5B
**Adapter**: LoRA (Low-Rank Adaptation)
**Total Parameters**: 1.56B
**Trainable Parameters**: 18.5M (1.2%)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch Size | 8 |
| Learning Rate | 2e-5 |
| Max Length | 512 tokens |
| Warmup Steps | 500 |
| Include Prompt | No (distillation mode) |

---

## Training Results

### Loss Curves

| Epoch | Train Loss | Val Loss | Train MAE | Val MAE |
|-------|------------|----------|-----------|---------|
| 1 | 3.893 | 1.446 | 1.393 | 0.870 |
| 2 | 1.075 | 1.096 | 0.750 | 0.741 |
| 3 | 0.801 | 1.039 | 0.639 | **0.712** |

### Training Observations

- Rapid initial convergence (epoch 1 → 2)
- Continued improvement through epoch 3
- Healthy train/val gap: 10.2% (no overfitting)
- Model generalizes well

---

## Benchmark Results (Test Set)

**Test Examples**: 1,000
**Evaluation Date**: 2025-11-27

### Overall Metrics

| Metric | Validation | Test | Δ |
|--------|------------|------|---|
| **MAE** | 0.712 | **0.690** | -3.1% ✅ |
| **RMSE** | 1.019 | **0.970** | -4.8% ✅ |

**Key Finding**: Test performance is **better** than validation, confirming excellent generalization.

### Per-Dimension Performance

| Dimension | Val MAE | Test MAE | Test RMSE | Status |
|-----------|---------|----------|-----------|--------|
| Life Cycle Env. Impact | 0.557 | **0.562** | 0.759 | ✅ Best |
| Technical Performance | 0.685 | **0.667** | 0.955 | ✅ |
| Economic Competitiveness | 0.656 | **0.667** | 0.994 | ✅ |
| Social Equity Impact | 0.698 | **0.690** | 0.920 | ✅ |
| Technology Readiness | 0.753 | **0.707** | 1.018 | ✅ |
| Governance/Systemic | 0.921 | **0.850** | 1.134 | ✅ Improved |

### Quality Gates

| Gate | Threshold | Actual | Status |
|------|-----------|--------|--------|
| Overall MAE | < 0.80 | 0.690 | ✅ PASS |
| All dimensions MAE | < 1.0 | Max: 0.850 | ✅ PASS |
| Test ≤ Validation | No overfit | -3.1% | ✅ PASS |
| RMSE | < 1.2 | 0.970 | ✅ PASS |

---

## Comparison with Other Filters

| Filter | Val MAE | Train/Val Gap | Dimensions |
|--------|---------|---------------|------------|
| investment-risk v4 | 0.391 | 67.8% (high) | 8 |
| sustainability_tech_innovation v2 | 0.595 | 24.6% | 6 |
| **sustainability_technology v1** | **0.712** | **10.2%** | **6** |
| uplifting v4 | 0.973 | 12.5% | 5 |

**Observation**: This filter achieves comparable performance to existing filters with excellent generalization (lowest train/val gap).

---

## Model Artifacts

### Saved Files

```
filters/sustainability_technology/v1/
├── config.yaml                    # Filter configuration
├── oracle.py                      # Oracle scoring logic
├── prefilter.py                   # Keyword prefilter
├── semantic_prefilter.py          # Semantic prefilter
├── prompt-compressed.md           # Compressed prompt
├── model/
│   ├── adapter_config.json        # LoRA configuration
│   ├── adapter_model.safetensors  # Trained weights (37MB)
│   ├── tokenizer.json             # Tokenizer
│   └── ...                        # Supporting files
├── training_metadata.json         # Training configuration
├── training_history.json          # Epoch-by-epoch metrics
└── benchmarks/
    ├── test_set_results.json      # Summary metrics
    └── test_set_predictions.json  # Full predictions
```

---

## Inference Pipeline

### Recommended Flow

```
Article → Prefilter → Model → Postfilter → Tier Assignment
```

1. **Prefilter**: Fast keyword + semantic filtering to skip irrelevant articles
2. **Model**: LoRA-adapted Qwen2.5-1.5B scores 6 dimensions
3. **Postfilter**: Apply gatekeepers (TRL < 3 caps overall at 2.9)
4. **Tier**: Assign based on weighted average score

### Expected Performance

- **Throughput**: ~50-100 articles/second (batched, GPU)
- **Latency**: ~10-20ms per article (single, GPU)
- **Memory**: ~4GB VRAM (FP16)

---

## Recommendations

### For Production Deployment

1. ✅ Model is production-ready
2. ✅ All quality gates passed
3. ✅ No overfitting detected
4. Deploy with prefilter for efficiency
5. Monitor dimensional distributions over time

### Potential Improvements (Future Versions)

- Consider merging Social/Governance dimensions (r=0.80)
- Add more training data for high-scoring articles
- Fine-tune on domain-specific articles for better calibration

---

## Conclusion

The sustainability_technology v1 filter successfully distills the oracle's 6-dimensional LCSA scoring into a fast, production-ready model. With a test MAE of 0.690 and all dimensions performing below 1.0 MAE, the model meets all quality requirements for deployment.

**Status**: ✅ **APPROVED FOR PRODUCTION**

---

**Report Generated**: 2025-11-27
**Model Version**: 1.0
**Next Step**: Deployment Setup
