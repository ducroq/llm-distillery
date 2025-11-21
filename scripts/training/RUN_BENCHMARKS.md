# Test Set Benchmarking Guide

This guide explains how to benchmark all three trained models on their test sets.

## Overview

Test set benchmarking validates model performance on held-out data that the model has never seen during training. This is CRITICAL for:

1. **Validating train/val gap concerns** - Especially for investment-risk v4 (67.8% gap)
2. **Confirming production readiness** - Test MAE should be close to validation MAE
3. **Comparing models fairly** - All evaluated on same split methodology

## Expected Results

Based on validation performance, we expect:

| Model | Val MAE | Expected Test MAE | Acceptable Range |
|-------|---------|-------------------|------------------|
| investment-risk v4 | 0.3914 | 0.39-0.45 | If >0.60, use epoch 7 |
| sustainability_tech_innovation v2 | 0.5954 | 0.58-0.65 | Should match validation |
| uplifting v4 | 0.9725 | 0.95-1.05 | Should match validation |

## Commands to Run

### On Remote GPU Machine

Run these commands sequentially on your GPU machine:

```bash
# 1. Investment-Risk v4 (CRITICAL - validate 67.8% gap)
python scripts/training/benchmark_test_set.py \
    --filter filters/investment-risk/v4 \
    --data-dir datasets/training/investment_risk_v4 \
    --batch-size 16

# 2. Sustainability Tech Innovation v2
python scripts/training/benchmark_test_set.py \
    --filter filters/sustainability_tech_innovation/v2 \
    --data-dir datasets/training/sustainability_tech_innovation_v2 \
    --batch-size 16

# 3. Uplifting v4
python scripts/training/benchmark_test_set.py \
    --filter filters/uplifting/v4 \
    --data-dir datasets/training/uplifting_v4 \
    --batch-size 16
```

### Expected Runtime

- Each benchmark: ~2-5 minutes
- Total: ~10-15 minutes for all three models

## Output Files

Each benchmark creates two files in `{filter}/benchmarks/`:

1. **test_set_results.json** - Summary metrics (MAE, RMSE per dimension)
2. **test_set_predictions.json** - Detailed predictions for each test article

## Interpreting Results

### Investment-Risk v4 (Most Critical)

**GOOD (Production Ready):**
- Test MAE: 0.39-0.45
- Similar to validation MAE (0.3914)
- Indicates 67.8% train/val gap is acceptable

**CONCERNING (Requires Action):**
- Test MAE: 0.50-0.60
- Slightly worse than validation
- Consider epoch 7-8 models

**BAD (Overfitted):**
- Test MAE: >0.60
- Much worse than validation (0.3914)
- Must revert to epoch 7 (0.3965 MAE, 45.8% gap)

### Sustainability Tech Innovation v2

**GOOD:**
- Test MAE: 0.58-0.65
- Close to validation (0.5954)
- Model generalizes well

**CONCERNING:**
- Test MAE: >0.70
- Model may have overfit

### Uplifting v4

**GOOD:**
- Test MAE: 0.95-1.05
- Close to validation (0.9725)
- Lowest train/val gap (12.5%) should generalize well

## Transfer Results Back to Local

After running all benchmarks:

```bash
# Copy all benchmark results from remote to local
scp -r user@remote:~/llm-distillery/filters/investment-risk/v4/benchmarks \
    C:/local_dev/llm-distillery/filters/investment-risk/v4/

scp -r user@remote:~/llm-distillery/filters/sustainability_tech_innovation/v2/benchmarks \
    C:/local_dev/llm-distillery/filters/sustainability_tech_innovation/v2/

scp -r user@remote:~/llm-distillery/filters/uplifting/v4/benchmarks \
    C:/local_dev/llm-distillery/filters/uplifting/v4/
```

## Next Steps After Benchmarking

1. **Verify results arrived** - Check all three `benchmarks/` directories
2. **Analyze results** - Compare test vs validation MAE
3. **Generate comparative report** - Cross-model analysis
4. **Make deployment decisions** - Accept or revise models
5. **Commit results** - Add benchmarks to repository

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python scripts/training/benchmark_test_set.py \
    --filter filters/investment-risk/v4 \
    --data-dir datasets/training/investment_risk_v4 \
    --batch-size 8  # Reduced from 16
```

### Model Not Found

Ensure model was copied to filter directory:
```bash
ls filters/investment-risk/v4/model/  # Should contain adapter_model.safetensors
```

### Test Set Not Found

Verify dataset location:
```bash
ls datasets/training/investment_risk_v4/test.jsonl
```

## Critical Decision Points

### If investment-risk v4 test MAE > 0.60:

**Option 1: Use Epoch 7 Model**
- Validation MAE: 0.3965 (only 1.3% worse)
- Train/Val Gap: 45.8% (much better than 67.8%)
- Should have better test performance

**Option 2: Retrain with More Regularization**
- Increase LoRA dropout from 0.05 to 0.10
- Add weight decay
- More warmup steps

**Option 3: Early Stopping**
- Implement validation-based early stopping
- Stop when validation improvement < 0.001 for 2 epochs

### If all models pass:

**Production Deployment Ready!**
- All three models generalize well
- Proceed to Phase 8 (Documentation)
- Plan production deployment (Phase 9)
