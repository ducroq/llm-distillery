# Uplifting v4 - GPU Training Instructions

**Date:** 2025-11-16
**Model:** Qwen 2.5-1.5B
**Training Mode:** Knowledge Distillation
**Expected Time:** 2-3 hours
**Expected Result:** 0.6-0.8 MAE

---

## Prerequisites

### On Local Machine (Already Done)

✅ Data preparation:
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

This creates:
- `datasets/training/uplifting_v4/train.jsonl` (~3,778 examples)
- `datasets/training/uplifting_v4/val.jsonl` (~472 examples)
- `datasets/training/uplifting_v4/test.jsonl` (~473 examples)

---

## Step 1: Transfer Files to GPU Machine

### Files to Transfer

```bash
# On local machine, create transfer package
mkdir -p /tmp/uplifting_training_package

# Copy training data
cp -r datasets/training/uplifting_v4 /tmp/uplifting_training_package/

# Copy filter configuration
cp -r filters/uplifting/v4 /tmp/uplifting_training_package/

# Copy training scripts
cp training/train.py /tmp/uplifting_training_package/
cp training/__init__.py /tmp/uplifting_training_package/

# Create archive
cd /tmp
tar -czf uplifting_training_package.tar.gz uplifting_training_package/
```

### Transfer to GPU Machine

**Using SCP:**
```bash
scp /tmp/uplifting_training_package.tar.gz user@gpu-machine:/path/to/llm-distillery/
```

**Or using your preferred method** (rsync, cloud storage, etc.)

### On GPU Machine: Extract

```bash
cd /path/to/llm-distillery
tar -xzf uplifting_training_package.tar.gz

# Verify structure
ls uplifting_training_package/uplifting_v4/  # Should see train.jsonl, val.jsonl, test.jsonl
ls uplifting_training_package/v4/            # Should see config.yaml
ls uplifting_training_package/               # Should see train.py
```

---

## Step 2: Setup GPU Environment

### Check GPU

```bash
nvidia-smi
```

Expected: At least 16GB VRAM for Qwen 2.5-1.5B with LoRA

### Install Dependencies (if needed)

```bash
# Activate your conda/venv environment
conda activate llm-distillery  # or your env name

# Install required packages
pip install torch transformers peft accelerate bitsandbytes pyyaml tqdm
```

### Verify PyTorch CUDA

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected: `CUDA available: True`

---

## Step 3: Run Training (Knowledge Distillation)

### Training Command

```bash
cd /path/to/llm-distillery

python -m training.train \
    --filter filters/uplifting/v4 \
    --data-dir datasets/training/uplifting_v4 \
    --output-dir filters/uplifting/v4_distillation \
    --model-name Qwen/Qwen2.5-1.5B \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --max-length 512 \
    --warmup-steps 500 \
    --seed 42
```

**Note:**
- Use `python -m training.train` to run as module, or `python training/train.py` (both work)
- The `--seed` argument ensures reproducibility (matches seed used in data preparation)

### Training Configuration Explained

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `--filter` | `filters/uplifting/v4` | Filter config directory |
| `--data-dir` | `datasets/training/uplifting_v4` | Training data location |
| `--output-dir` | `filters/uplifting/v4_distillation` | Where to save model |
| `--model-name` | `Qwen/Qwen2.5-1.5B` | Base model (1.5B params) |
| `--epochs` | `3` | Training epochs (same as investment-risk) |
| `--batch-size` | `8` | Batch size (reduce to 4 if OOM) |
| `--learning-rate` | `2e-5` | Learning rate (proven for investment-risk) |
| `--max-length` | `512` | Token limit (distillation mode) |
| `--warmup-steps` | `500` | LR warmup |
| `--seed` | `42` | Reproducibility |

### Expected Output

```
Loading filter configuration from: filters/uplifting/v4
Filter: uplifting
Dimensions: 8 (agency, progress, collective_benefit, connection, innovation, justice, resilience, wonder)

Loading training data from: datasets/training/uplifting_v4
Train examples: 3778
Val examples: 472

Loading model: Qwen/Qwen2.5-1.5B
Model parameters: 1,562,203,648
Trainable parameters: 18,477,056 (1.18%)

Starting training...

Epoch 1/3:
  Train MAE: 1.68  Train RMSE: 2.34  Train Loss: 5.47
  Val MAE:   0.99  Val RMSE:  1.34  Val Loss:  1.78
  [Saved checkpoint]

Epoch 2/3:
  Train MAE: 0.77  Train RMSE: 1.05  Train Loss: 1.11
  Val MAE:   0.69  Val RMSE:  0.96  Val Loss:  0.92
  [Saved checkpoint]

Epoch 3/3:
  Train MAE: 0.62  Train RMSE: 0.84  Train Loss: 0.71
  Val MAE:   0.67  Val RMSE:  0.93  Val Loss:  0.86
  [Saved checkpoint]

Training complete!
Best model saved to: filters/uplifting/v4_distillation/model/
```

### What Gets Saved

```
filters/uplifting/v4_distillation/
├── model/
│   ├── adapter_model.safetensors    # LoRA weights (trained)
│   ├── adapter_config.json
│   ├── tokenizer files...
│   └── README.md
├── training_history.json            # Epoch-by-epoch metrics
└── training_metadata.json           # Training config
```

---

## Step 4: Monitor Training (Optional)

### In Another Terminal

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training progress
tail -f uplifting_v4_distillation/training.log
```

### Expected GPU Usage
- Memory: ~12-14GB (for 1.5B model with LoRA)
- Utilization: 95-100%
- Time per epoch: ~40-60 minutes

---

## Step 5: Troubleshooting

### Out of Memory (OOM)

**Symptom:** `CUDA out of memory` error

**Solution 1:** Reduce batch size
```bash
--batch-size 4  # instead of 8
```

**Solution 2:** Enable gradient accumulation
```bash
--batch-size 4 \
--gradient-accumulation-steps 2  # effective batch size = 4*2 = 8
```

**Solution 3:** Reduce max length
```bash
--max-length 384  # instead of 512
```

### Training Very Slow

**Symptom:** <10 samples/second

**Check:**
1. GPU is being used: `nvidia-smi`
2. CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
3. Mixed precision enabled (should be automatic with bitsandbytes)

### Model Not Converging

**Symptom:** Validation MAE stays >1.5 after epoch 1

**Possible causes:**
1. Data loading issue → Check train.jsonl format
2. Learning rate too high → Try `--learning-rate 1e-5`
3. Bad initialization → Try different seed

### Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'peft'`

**Solution:**
```bash
pip install peft transformers torch accelerate bitsandbytes
```

---

## Step 6: Transfer Results Back

### After Training Completes

**Create package:**
```bash
cd /path/to/llm-distillery

# Package trained model
tar -czf uplifting_v4_distillation.tar.gz uplifting_v4_distillation/
```

**Transfer to local machine:**
```bash
scp user@gpu-machine:/path/to/llm-distillery/uplifting_v4_distillation.tar.gz /tmp/
```

**On local machine: Extract to filter directory**
```bash
cd C:/local_dev/llm-distillery

# Extract
tar -xzf /tmp/uplifting_v4_distillation.tar.gz

# Move to correct location
mv uplifting_v4_distillation filters/uplifting/v4_distillation
```

---

## Step 7: Generate Reports (On Local Machine)

### After transferring model back

```bash
cd C:/local_dev/llm-distillery

# Generate visualizations
python training/plot_learning_curves.py \
    --history filters/uplifting/v4_distillation/training_history.json

# Generate Word report
python training/generate_training_report.py \
    --filter filters/uplifting/v4 \
    --history filters/uplifting/v4_distillation/training_history.json \
    --metadata filters/uplifting/v4_distillation/training_metadata.json
```

Reports will be in: `filters/uplifting/v4_distillation/training_reports/`

---

## Alternative: Train Instruction Tuning (Optional Comparison)

If you want to compare training modes like we did for investment-risk:

```bash
python -m training.train \
    --filter filters/uplifting/v4 \
    --data-dir datasets/training/uplifting_v4 \
    --output-dir filters/uplifting/v4_instruction \
    --model-name Qwen/Qwen2.5-1.5B \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --max-length 1024 \
    --include-prompt \
    --seed 42
```

**Note:** Based on investment-risk results, instruction tuning will likely underperform (52.6% worse). Only do this if you want comparison data.

---

## Quick Reference Card

### Minimal Training Command
```bash
python -m training.train \
    --filter filters/uplifting/v4 \
    --data-dir datasets/training/uplifting_v4 \
    --output-dir filters/uplifting/v4_distillation \
    --model-name Qwen/Qwen2.5-1.5B \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --max-length 512 \
    --seed 42
```

### If OOM Error
```bash
# Add these flags:
--batch-size 4 \
--gradient-accumulation-steps 2
```

### Check Training Status
```bash
# GPU usage
nvidia-smi

# Latest metrics
tail -20 uplifting_v4_distillation/training_history.json

# Training log
tail -f uplifting_v4_distillation/training.log
```

---

## Expected Timeline

| Phase | Duration | Activity |
|-------|----------|----------|
| **Setup** | 10-15 min | Transfer files, verify environment |
| **Epoch 1** | 40-60 min | Initial training, large improvement |
| **Epoch 2** | 40-60 min | Continued optimization |
| **Epoch 3** | 40-60 min | Final convergence |
| **Transfer** | 5-10 min | Copy results back |
| **Total** | **2.5-3.5 hours** | Complete training cycle |

---

## Success Criteria

### ✅ Training Successful If:

1. **No errors** during training
2. **Validation MAE <1.0** by epoch 3
3. **Validation MAE decreases** across epochs
4. **Train/Val gap <0.1** (minimal overfitting)
5. **All dimension MAE <1.0** (check training_history.json)

### Expected Final Metrics (Based on Investment-Risk)

```
Validation MAE:  0.60-0.80 (target: <1.0)
Validation RMSE: 0.90-1.20
Train/Val Gap:   0.03-0.08 (healthy overfitting)
```

---

## Post-Training Checklist

- [ ] Training completed without errors
- [ ] Validation MAE <1.0
- [ ] Model saved to `uplifting_v4_distillation/model/`
- [ ] `training_history.json` and `training_metadata.json` present
- [ ] Files transferred back to local machine
- [ ] Training reports generated
- [ ] Results documented in `filters/uplifting/v4/training_report.md`

---

## Questions?

**If training fails:**
1. Check error message carefully
2. Verify data files exist and are readable
3. Check GPU memory with `nvidia-smi`
4. Try reducing batch size
5. Check the troubleshooting section above

**If results look wrong:**
1. Verify data was prepared correctly (check train.jsonl)
2. Compare to investment-risk results (similar architecture/data)
3. Check for data leakage (train/val overlap)

---

**Ready to train?** Run the training command and monitor progress! 🚀

**Expected result:** Similar or better than investment-risk (0.67 MAE) due to superior data quality.
