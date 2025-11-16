# Uplifting v4 - Training Guide

**Filter Version:** v4.0
**Oracle Data:** 4,723 scored articles
**Status:** Ready for training

---

## Quick Start

### Step 1: Prepare Training Data

```bash
python training/prepare_data.py \
    --filter filters/uplifting/v4 \
    --input "datasets/scored/uplifting_v1/uplifting/scored_batch_*.jsonl" \
    --output-dir datasets/training/uplifting_v4 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42
```

**Output:**
- `datasets/training/uplifting_v4/train.jsonl` (~3,778 examples)
- `datasets/training/uplifting_v4/val.jsonl` (~472 examples)
- `datasets/training/uplifting_v4/test.jsonl` (~473 examples)
- `datasets/training/uplifting_v4/metadata.json` (stats)

**Verify:**
```bash
wc -l datasets/training/uplifting_v4/*.jsonl
head -1 datasets/training/uplifting_v4/train.jsonl | python -m json.tool
```

---

### Step 2: Train the Model (Knowledge Distillation)

**Recommended: Qwen 2.5-1.5B with Knowledge Distillation**

Based on investment-risk results, knowledge distillation significantly outperforms instruction tuning for regression tasks (52.6% better MAE).

```bash
python training/train.py \
    --filter-dir filters/uplifting/v4 \
    --data-dir datasets/training/uplifting_v4 \
    --output-dir filters/uplifting/v4_distillation \
    --model-name Qwen/Qwen2.5-1.5B \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --max-length 512 \
    --training-mode knowledge_distillation \
    --seed 42
```

**Training Configuration:**
```yaml
model: Qwen/Qwen2.5-1.5B
training_mode: knowledge_distillation
include_prompt: false
max_length: 512 tokens
epochs: 3
batch_size: 8
learning_rate: 2e-05
warmup_steps: 500
```

**Expected Training Time:** 2-3 hours on GPU

**Expected Performance (based on investment-risk results):**
- Target: <1.0 MAE
- Expected: 0.6-0.8 MAE
- Dimensions: 8 (same as investment-risk)

---

### Step 3: Generate Training Reports

After training completes:

```bash
# Generate visualizations
python training/plot_learning_curves.py \
    --history filters/uplifting/v4_distillation/training_history.json

# Generate Word report
python training/generate_training_report.py \
    --filter filters/uplifting/v4 \
    --history filters/uplifting/v4_distillation/training_history.json \
    --metadata filters/uplifting/v4_distillation/training_metadata.json
```

Reports will be saved to: `filters/uplifting/v4_distillation/training_reports/`

---

## Filter-Specific Details

### Dimensions (8 total)

From `config.yaml`:

1. **agency** (14% weight) - People/communities taking effective action
2. **progress** (19% weight) - Movement toward human flourishing
3. **collective_benefit** (38% weight) - Helps many people (GATEKEEPER)
4. **connection** (10% weight) - Collaboration, solidarity
5. **innovation** (8% weight) - Novel solutions working
6. **justice** (3% weight) - Addressing inequality/exploitation
7. **resilience** (3% weight) - Building capacity against shocks
8. **wonder** (5% weight) - Awe-inspiring achievement

### Gatekeeper Rules

**collective_benefit** is the primary gatekeeper:
- Must be ≥5.0 for uplifting content
- Exception: If wonder ≥7.0, only need collective_benefit ≥3.0

This prevents false positives on:
- Corporate success stories
- Individual achievements
- Elite-only benefits

---

## Training Modes

### Recommended: Knowledge Distillation

**Why:** Based on investment-risk v2 results, knowledge distillation is superior for regression tasks.

**Results from investment-risk:**
- Distillation: 0.67 MAE ✅
- Instruction: 1.42 MAE ❌
- Improvement: 52.6% better

**Configuration:**
```bash
--training-mode knowledge_distillation
--include-prompt false
--max-length 512
```

**Input:** Article title + content (max 512 tokens)
**Output:** 8 dimension scores (0-10 scale)

### Alternative: Instruction Tuning (Not Recommended)

Only use if you need the model to explain its reasoning.

**Configuration:**
```bash
--training-mode instruction_tuning
--include-prompt true
--max-length 1024
```

**Expected:** Significantly worse accuracy (based on investment-risk results)

---

## Data Statistics

### Oracle Scoring

**Total Scored:** 4,723 articles
**Oracle:** Gemini Flash 1.5
**Batch Files:** 13 files (scored_batch_001 through scored_batch_013)
**Location:** `datasets/scored/uplifting_v1/uplifting/`

### Expected Training Split (80/10/10)

```
Training:   ~3,778 examples (80%)
Validation:   ~472 examples (10%)
Test:         ~473 examples (10%)
```

**Stratification:** Split maintains tier distribution across train/val/test

---

## Expected Results

Based on investment-risk v2 training (same architecture, same approach):

### Performance Targets

| Metric | Target | Expected | Notes |
|--------|--------|----------|-------|
| Validation MAE | <1.0 | 0.6-0.8 | Primary metric |
| Validation RMSE | <1.5 | 1.0-1.3 | Secondary metric |
| Per-dimension MAE | <1.0 | 0.5-1.0 | All 8 dimensions |
| Overfitting gap | <0.1 | 0.03-0.08 | Train/val MAE difference |

### Training Convergence

**Expected behavior (from investment-risk):**
```
Epoch 1: Large improvement (MAE 1.5+ → 1.0)
Epoch 2: Continued improvement (MAE 1.0 → 0.7)
Epoch 3: Convergence (MAE 0.7 → 0.65)
```

If MAE drops below 0.5, you might be overfitting (but check validation performance).

---

## Troubleshooting

### Issue: "No examples found"

**Cause:** Input path doesn't match scored files
**Fix:** Check that scored files exist:
```bash
ls datasets/scored/uplifting_v1/uplifting/scored_batch_*.jsonl
```

### Issue: "Dimension mismatch"

**Cause:** Scored data uses different filter version
**Fix:** Ensure scored data matches filter config (v4 → v4)

### Issue: "Out of memory during training"

**Cause:** Batch size too large for GPU
**Fix:** Reduce batch size:
```bash
--batch-size 4  # or even 2
```

### Issue: "Training not converging"

**Possible causes:**
1. Learning rate too high → Try `--learning-rate 1e-5`
2. Not enough data → Score more articles
3. Data quality issues → Check oracle scores

---

## Next Steps After Training

### 1. Validate Model Performance

```bash
# Compare to investment-risk benchmarks
# Check training_reports/comparison_report.md
```

**Success criteria:**
- ✅ Validation MAE <1.0
- ✅ All dimensions learned (MAE <1.0 per dimension)
- ✅ Minimal overfitting (<10% train/val gap)

### 2. Optional: Train Instruction Tuning for Comparison

```bash
python training/train.py \
    --filter-dir filters/uplifting/v4 \
    --data-dir datasets/training/uplifting_v4 \
    --output-dir filters/uplifting/v4_instruction \
    --training-mode instruction_tuning \
    --max-length 1024
```

Then compare:
```bash
python training/compare_training_modes.py \
    --distillation-history filters/uplifting/v4_distillation/training_history.json \
    --distillation-metadata filters/uplifting/v4_distillation/training_metadata.json \
    --instruction-history filters/uplifting/v4_instruction/training_history.json \
    --instruction-metadata filters/uplifting/v4_instruction/training_metadata.json
```

### 3. Create Training Report

```bash
# Copy from investment-risk template
cp filters/investment-risk/v2/training_report.md \
   filters/uplifting/v4/training_report.md

# Update with uplifting-specific results
```

### 4. Deploy to Production

Once validated:
1. Update `release_report.md` with training results
2. Deploy `v4_distillation/model/` to production pipeline
3. Monitor performance on live traffic

---

## Dataset Quality Notes

### From Package Validation (v4.0)

**Sample validation results:**
- Prefilter block rate: 82.2% (by design - highly selective)
- Articles scored: 17.8% of random sample
- Uplifting identified: 50% of scored articles
- False positive rate: 0% (v3: 87.5% → v4: 0%)

**Key improvement in v4:**
Inline filters pattern eliminated false positives on:
- Professional knowledge/expertise
- Corporate/business news
- Productivity advice
- Individual achievements

---

## Comparison to Investment-Risk

| Aspect | Investment-Risk v2 | Uplifting v4 |
|--------|-------------------|--------------|
| **Dimensions** | 8 | 8 (same!) |
| **Oracle Data** | 5,150 articles | 4,723 articles |
| **Training Size** | 4,118 examples | ~3,778 examples |
| **Model** | Qwen 2.5-1.5B | Qwen 2.5-1.5B |
| **Training Mode** | Knowledge Distillation | Knowledge Distillation (recommended) |
| **Result** | 0.67 MAE ✅ | Expected: 0.6-0.8 MAE |
| **Status** | Production deployed | Ready to train |

**Insight:** Similar dataset size and architecture suggest similar performance is achievable.

---

## Cost Analysis

### Oracle Scoring (Already Complete)
- 4,723 articles × $0.0005 = **$2.36** (already spent)

### Training Cost
- GPU time: 2-3 hours
- Cloud GPU (A100): ~$3-5
- Local GPU: $0

### Production Deployment
- Inference: <50ms per article
- Cost: $0 (local inference)
- vs Oracle API: $0.0005/article

**ROI:** Student model pays for itself after ~10,000 articles scored.

---

## References

- **Filter Specification:** `config.yaml`
- **Oracle Prompt:** `prompt-compressed.md`
- **Validation Report:** `validation_report.md`
- **Release Report:** `release_report.md`
- **Investment-Risk Training:** `../investment-risk/v2/training_report.md` (reference for expected results)

---

**Ready to train?** Run Step 1 (prepare data) and verify the output before proceeding to training!
