# Investment Risk v2 - Training Plan

**Status:** Ready to train both approaches
**Ground truth:** 5,150 articles scored by Gemini Flash
**Goal:** Compare knowledge distillation vs instruction tuning

---

## Two Training Approaches

### Approach A: Knowledge Distillation (Baseline)
**Student input:** Article only (title + content)
**Training mode:** Learn pattern from oracle's outputs

### Approach B: Instruction Tuning (Experimental)
**Student input:** Prompt + Article (prompt-compressed.md + title + content)
**Training mode:** Learn task with explicit context

---

## Step 1: Prepare Training Data

**Run once** (creates train/val/test splits):

```bash
python training/prepare_data.py \
    --filter filters/investment-risk/v2 \
    --input datasets/scored/investment_risk_v2/investment-risk/scored_batch_*.jsonl \
    --output-dir datasets/training/investment_risk_v2 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

**Expected output:**
- `datasets/training/investment_risk_v2/train.jsonl` (4,120 examples)
- `datasets/training/investment_risk_v2/val.jsonl` (515 examples)
- `datasets/training/investment_risk_v2/test.jsonl` (515 examples)

---

## Step 2A: Train Knowledge Distillation Model

**No prompt, faster inference:**

```bash
python training/train.py \
    --filter filters/investment-risk/v2 \
    --data-dir datasets/training/investment_risk_v2 \
    --output-dir filters/investment-risk/v2_distillation \
    --model-name Qwen/Qwen2.5-1.5B \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --max-length 512
```

**Training mode:** `knowledge_distillation`
**Expected time:** ~2-3 hours on GPU
**Expected cost:** $0 (local training)
**Model saved to:** `filters/investment-risk/v2_distillation/model/`

---

## Step 2B: Train Instruction Tuning Model

**With prompt, potentially better understanding:**

```bash
python training/train.py \
    --filter filters/investment-risk/v2 \
    --data-dir datasets/training/investment_risk_v2 \
    --output-dir filters/investment-risk/v2_instruction \
    --model-name Qwen/Qwen2.5-1.5B \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --max-length 1024 \
    --include-prompt
```

**Training mode:** `instruction_tuning`
**Note:** `--max-length 1024` to accommodate prompt (13,601 chars) + article
**Expected time:** ~3-4 hours on GPU (longer sequences)
**Model saved to:** `filters/investment-risk/v2_instruction/model/`

---

## Step 3: Compare Results

### Metrics to Compare

After training, compare `training_metadata.json`:

| Metric | Distillation | Instruction | Winner |
|--------|--------------|-------------|--------|
| **Validation MAE** | ? | ? | Lower is better |
| **Training time** | ~2-3h | ~3-4h | Faster is better |
| **Inference speed** | ~50ms | ~100ms | Faster is better |
| **Memory usage** | Lower | Higher | Lower is better |

### Generalization Test

Test on edge cases:
- Geopolitical events (nuclear testing, wars)
- Economic crises (bank failures, currency collapse)
- Policy errors (unexpected Fed actions)
- Clickbait (should score low)

### Decision Criteria

**Use Knowledge Distillation if:**
- Validation MAE < Instruction Tuning MAE
- Inference speed is critical
- Lower memory footprint needed

**Use Instruction Tuning if:**
- Validation MAE significantly better (>0.2 improvement)
- Better generalization to edge cases
- Inference speed acceptable (~100ms)

---

## Expected Outcomes

### Hypothesis 1: Similar Performance
- Both models achieve MAE ~0.5-0.8
- Distillation wins on speed (50ms vs 100ms)
- **Recommendation:** Use distillation (faster, cheaper)

### Hypothesis 2: Instruction Tuning Superior
- Instruction MAE < Distillation MAE by >0.2
- Better edge case handling
- **Recommendation:** Use instruction (better quality)

### Hypothesis 3: Mixed Results
- Distillation better on some dimensions
- Instruction better on others
- **Recommendation:** Ensemble or dimension-specific models

---

## Troubleshooting

### If validation MAE > 1.5 (poor performance):
1. Check training data quality (dimensional scores in 0-10 range?)
2. Increase epochs (3 → 5)
3. Adjust learning rate (2e-5 → 1e-5)
4. Check for class imbalance (most articles NOISE?)

### If training runs out of memory:
1. Reduce batch size (8 → 4)
2. Enable gradient checkpointing (already on)
3. Use smaller model (1.5B → 0.5B)
4. Reduce max_length (1024 → 768)

### If training too slow:
1. Use fp16 (add `--use-fp16` flag if implemented)
2. Reduce max_length
3. Use smaller validation set

---

## Next Steps After Training

1. **Evaluate both models:**
   ```bash
   python training/evaluate_model.py \
       --filter filters/investment-risk/v2_distillation \
       --test-data datasets/training/investment_risk_v2/test.jsonl
   ```

2. **Generate comparison report:**
   - Compare validation metrics
   - Test on edge cases
   - Measure inference speed
   - Document decision

3. **Deploy winner:**
   - Copy winning model to `filters/investment-risk/v2/model/`
   - Update `training_metadata.json`
   - Create deployment guide

---

## Timeline

**Day 1:**
- ✅ Ground truth generated (5,150 articles)
- ⏳ Prepare training data (~5 minutes)
- ⏳ Train distillation model (~2-3 hours)

**Day 2:**
- ⏳ Train instruction model (~3-4 hours)
- ⏳ Compare results (~1 hour)
- ⏳ Decide winner

**Day 3:**
- ⏳ Deploy winning model
- ⏳ Document results

---

**Created:** 2025-11-15
**Ready to train:** ✅ YES
**Blockers:** None
