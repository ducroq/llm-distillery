# Large Ground Truth Workflow

## Overview

Instead of trying to create perfect training data through stratified sampling (which risks keyword bias), we create a **large semantic ground truth** (20k articles) and then **select** the best subset for training.

## Advantages

✅ **No keyword bias** - Selection based on actual LLM semantic scores
✅ **Reusable** - 20k labeled dataset can be mined for different purposes
✅ **Optimal coverage** - Select articles that maximize dimensional spread
✅ **Future-proof** - Need different training data? Select differently from same 20k
✅ **Quality control** - Can exclude outliers, edge cases, poor examples
✅ **Cost-effective** - Gemini labeling 20k ≈ $3.60 (vs $90 for Claude)

## Workflow Steps

### Step 1: Validate Filter/Prompt (Calibration)

**Before** committing to 20k labels, validate your filter works well:

```bash
# Run calibration on 200 random articles
python -m ground_truth.calibrate_oracle \
    --filter filters/uplifting/v1 \
    --source "datasets/raw/master_dataset*.jsonl" \
    --models gemini-flash,gemini-pro \
    --sample-size 200 \
    --seed 42 \
    --output reports/uplifting_calibration_validation.md
```

**Review:**
- Model agreement (correlations, tier distribution)
- Dimensional score distributions
- Choose which model to use (Flash vs Pro vs Claude)
- Refine prompt if needed

---

### Step 2: Create Large Ground Truth (20k Articles)

Once validated, label 20k random articles:

```bash
# Label 20k articles with chosen model
python -m ground_truth.batch_labeler \
    --filter filters/uplifting/v1 \
    --source "datasets/raw/master_dataset*.jsonl" \
    --output-dir datasets/ground_truth_20k \
    --llm gemini-pro \
    --target-count 20000 \
    --random-sample \
    --seed 42
```

**Specs:**
- **Cost:** ~$3.60 for Gemini Pro (20k × $0.00018)
- **Time:** ~6-8 hours (depends on rate limits)
- **Output:** `datasets/ground_truth_20k/uplifting/labeled_articles.jsonl`

**What you get:**
- 20k articles with full dimensional scores
- Complete semantic analysis for each article
- Tier classifications
- Reasoning for each score

---

### Step 3: Analyze Coverage

Understand the distribution of your ground truth:

```bash
python -m ground_truth.analyze_coverage \
    --labeled-file datasets/ground_truth_20k/uplifting/labeled_articles.jsonl \
    --output datasets/ground_truth_20k/coverage_report.json
```

**This tells you:**
- Score distribution for each dimension (1-10)
- Missing score ranges
- IQR and spread statistics
- Coverage gaps

---

### Step 4: Select Optimal Training Set

**NEW TOOL:** `select_training_set.py` (to be implemented)

```bash
python -m ground_truth.select_training_set \
    --labeled-file datasets/ground_truth_20k/uplifting/labeled_articles.jsonl \
    --output-dir datasets/training_2k \
    --target-size 2000 \
    --strategy maximize_coverage \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --dimensions agency,progress,collective_benefit,connection,innovation,justice,resilience,wonder
```

**Selection Strategies:**

1. **`maximize_coverage`** (Recommended)
   - Ensures all score ranges (1-10) represented for each dimension
   - Balances tier distribution
   - Maximizes IQR for better model training

2. **`balanced_tiers`**
   - Equal representation of tiers (not_uplifting, connection, impact)
   - Good for balanced classification

3. **`iqr_optimization`**
   - Explicitly maximizes IQR for each dimension
   - Over-samples extremes slightly
   - Best for teaching dimensional boundaries

4. **`extreme_focus`**
   - Heavy over-sampling of extremes (1-3, 8-10)
   - Ensures model learns edge cases
   - May sacrifice middle-range coverage

**Output:**
```
datasets/training_2k/
├── train.jsonl           # 1,600 articles (80%)
├── val.jsonl            # 200 articles (10%)
├── test.jsonl           # 200 articles (10%)
├── selection_report.md  # How articles were selected
└── coverage_comparison.md # Coverage before/after selection
```

---

### Step 5: Train Model

Use the selected training set:

```bash
# Train Chinese model (Qwen)
python -m training.finetune_qwen \
    --train-file datasets/training_2k/train.jsonl \
    --val-file datasets/training_2k/val.jsonl \
    --test-file datasets/training_2k/test.jsonl \
    --model qwen/Qwen2.5-7B-Instruct \
    --output models/uplifting_qwen_v1
```

---

## Future Reuse of Ground Truth

The 20k ground truth can be reused for:

### Different Training Sets
```bash
# Create training set focused on extremes
python -m ground_truth.select_training_set \
    --labeled-file datasets/ground_truth_20k/uplifting/labeled_articles.jsonl \
    --target-size 2000 \
    --strategy extreme_focus

# Create larger training set
python -m ground_truth.select_training_set \
    --target-size 5000 \
    --strategy maximize_coverage
```

### Different Splits
```bash
# 70-20-10 split instead of 80-10-10
python -m ground_truth.select_training_set \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --test-ratio 0.1
```

### Difficulty-Stratified Sets
```bash
# Hard examples only (articles with low agreement, edge cases)
python -m ground_truth.select_training_set \
    --filter-by-difficulty hard \
    --target-size 1000
```

### Active Learning
```bash
# Select articles where current model performs worst
python -m ground_truth.select_training_set \
    --model-predictions models/uplifting_qwen_v1/predictions.jsonl \
    --strategy active_learning \
    --target-size 500
```

---

## Cost Analysis

### Option A: Direct Training Set Creation
- Label 2k articles directly: $0.36 (Gemini) or $18 (Claude)
- Risk: Poor dimensional coverage, wasted labels
- Cannot reuse or reselect

### Option B: Large Ground Truth (This Workflow)
- Label 20k articles: $3.60 (Gemini) or $180 (Claude)
- Can create multiple training sets from same 20k
- Optimal coverage guaranteed
- Quality control and filtering possible

**Break-even:** After creating ~10 different training sets, Option B is more cost-effective

---

## Implementation Status

### ✅ Existing Tools
- `batch_labeler.py` - Label large datasets
- `analyze_coverage.py` - Analyze dimensional coverage
- `calibrate_oracle.py` - Validate filter/models

### 🔨 To Be Implemented
- `select_training_set.py` - Select optimal subset from ground truth
  - Strategies: maximize_coverage, balanced_tiers, iqr_optimization, extreme_focus
  - Split creation: train/val/test
  - Coverage reporting

---

## Example: Complete Workflow for Uplifting Filter

```bash
# 1. Validate (30 min, $0.04)
python -m ground_truth.calibrate_oracle \
    --filter filters/uplifting/v1 \
    --source "datasets/raw/*.jsonl" \
    --sample-size 200 \
    --models gemini-flash,gemini-pro

# Review reports/uplifting_calibration.md
# Decide: Use gemini-pro

# 2. Create ground truth (6 hours, $3.60)
python -m ground_truth.batch_labeler \
    --filter filters/uplifting/v1 \
    --source "datasets/raw/*.jsonl" \
    --output-dir datasets/ground_truth_20k \
    --llm gemini-pro \
    --target-count 20000 \
    --random-sample \
    --seed 42

# 3. Analyze coverage
python -m ground_truth.analyze_coverage \
    --labeled-file datasets/ground_truth_20k/uplifting/labeled_articles.jsonl

# 4. Select training set (instant, $0)
python -m ground_truth.select_training_set \
    --labeled-file datasets/ground_truth_20k/uplifting/labeled_articles.jsonl \
    --output-dir datasets/training_2k \
    --target-size 2000 \
    --strategy maximize_coverage

# 5. Train model (varies)
python -m training.finetune_qwen \
    --train-file datasets/training_2k/train.jsonl \
    --val-file datasets/training_2k/val.jsonl \
    --output models/uplifting_qwen_v1
```

**Total Cost:** ~$3.64
**Total Time:** ~7 hours
**Reusability:** Infinite (can create different training sets from same 20k)

---

## Next Steps

1. ✅ Run calibration to validate filter/prompt
2. ⏳ Implement `select_training_set.py`
3. ⏳ Create 20k ground truth
4. ⏳ Select optimal 2k training set
5. ⏳ Train Qwen model

---

## References

- [Batch Labeler Documentation](../guides/batch-labeling-guide.md)
- [Oracle Calibration Guide](../guides/oracle-calibration-guide.md)
- [Coverage Analysis Tool](../../ground_truth/analyze_coverage.py)
