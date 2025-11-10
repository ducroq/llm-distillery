# training/README.md Updates Needed

**Date:** 2025-11-10
**Context:** Model standardization on Qwen 2.5-1.5B with LoRA

---

## Updates Required

### 1. Training Example Command (Line 47)
**Current:**
```bash
--model-name Qwen/Qwen2.5-7B \
```

**Should be:**
```bash
--model-name Qwen/Qwen2.5-1.5B \
```

---

### 2. Model Architecture Section (Line 57-67)
**Current:**
```markdown
The training pipeline uses **Qwen 2.5-7B** adapted for multi-dimensional regression:

```
Input: [title + content] (max 512 tokens)
         ↓
    Qwen 2.5 Base Model
         ↓
    Regression Head (num_dimensions outputs)
         ↓
Output: [dim1_score, dim2_score, ..., dim8_score]
```
```

**Should be:**
```markdown
The training pipeline uses **Qwen 2.5-1.5B** with **LoRA** for memory-efficient training:

```
Input: [title + content] (max 512 tokens)
         ↓
    Qwen 2.5-1.5B Base Model (frozen, 1.5B params)
         ↓
    LoRA Adapters (trainable, ~37M params)
         ↓
    Regression Head (num_dimensions outputs)
         ↓
Output: [dim1_score, dim2_score, ..., dim8_score]
```

**LoRA Configuration:**
- **Rank (r)**: 16
- **Alpha**: 32
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Dropout**: 0.05
- **Trainable Parameters**: ~37M (2.5% of total model)
- **Memory Savings**: Only adapter weights updated during training
```

---

### 3. Training Config Example (Lines 123-138)
**Current:**
```yaml
model:
  name: "Qwen/Qwen2.5-7B"
  max_length: 512
```

**Should be:**
```yaml
model:
  name: "Qwen/Qwen2.5-1.5B"
  max_length: 512
  use_lora: true
  lora_config:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
```

---

### 4. Hardware Requirements Section (Lines 140-154)
**Current:**
```markdown
## Hardware Requirements

### Qwen 2.5-7B (Recommended)

- **GPU Memory**: 16GB+ (RTX 4090, A100)
- **Training Time**: ~2-4 hours for 7,000 samples
- **Expected Accuracy**: 90-95% vs oracle

### Alternative Models

For limited hardware:

- **Qwen 2.5-1.5B**: 8GB GPU, faster training, slightly lower accuracy
- **Qwen 2.5-3B**: 12GB GPU, balanced performance
```

**Should be:**
```markdown
## Hardware Requirements

### Qwen 2.5-1.5B with LoRA (Standard Configuration)

- **GPU Memory**: 16GB (RTX 4080, RTX 4090)
- **System RAM**: 32GB+ recommended (training processes + data loading)
- **Training Time**: ~1.5-2 hours for 7,000 samples
- **Expected MAE**: ~0.78 (proven with uplifting filter)
- **Precision**: FP32 (no quantization needed)
- **Trainable Parameters**: ~37M (2.5% of model)

**Why This Works on 16GB GPU:**
- Base model: ~3GB (1.5B params in FP32)
- LoRA adapters: ~140MB
- Optimizer states: ~280MB (AdamW)
- Gradients: ~140MB
- Activations (batch 8): ~2-3GB
- Total: ~6-7GB (comfortable margin for 16GB GPU)

### Recommended Workflow: tmux + GPU Machine

**Always use tmux for persistent training sessions:**

```bash
# 1. SSH to GPU machine
ssh user@gpu-machine

# 2. Start tmux session
tmux new -s training

# 3. Activate environment (if needed)
source venv/bin/activate

# 4. Run training
python -m training.train \
    --filter filters/sustainability_tech_deployment/v1 \
    --data-dir datasets/training/sustainability_tech_deployment \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5

# 5. Detach: Press Ctrl+B, then D
# Training continues running in background

# 6. Reattach later: tmux attach -t training
# 7. Kill session when done: tmux kill-session -t training
```

**See [GPU Training Guide](../docs/guides/gpu-training-guide.md) for complete tmux workflow, monitoring, and troubleshooting.**

### Alternative: Larger Models (Cloud GPU Required)

**Only consider if 1.5B model MAE > 1.5 consistently:**

| Model | GPU Required | Cloud Cost | Training Time | When to Use |
|-------|-------------|-----------|---------------|-------------|
| Qwen 2.5-7B-Instruct | 24GB+ | ~$0.40-0.60/hr | 4-6 hours | MAE > 1.5 |
| Qwen 2.5-14B | 40GB+ | ~$1.10/hr | 8-12 hours | Research only |

**Cloud GPU Options:**
- **RunPod**: ~$0.40/hr (RTX 4090 24GB)
- **Vast.ai**: ~$0.30/hr (comparable specs)
- **Lambda Labs**: ~$1.10/hr (A100 40GB)

**Cost per training run:** ~$1.50-3.00

See [Model Selection ADR](../docs/decisions/2025-11-10-model-selection-qwen-1.5b.md) for complete decision rationale and migration path.
```

---

### 5. Add LoRA Troubleshooting Section
**Insert new section after "Poor Accuracy" (after line 307):**

```markdown
### LoRA Not Applied

If training output shows full model parameters (~1.5B trainable):

```bash
# Symptom: "trainable parameters: 1,543,504,896"
# Should see: "LoRA applied: 37,748,736 / 1,543,504,896 parameters (2.45% trainable)"

# Solution: Install peft library
pip install peft

# Verify it's imported correctly
python -c "import peft; print(peft.__version__)"
```

### Model Still Too Large for 16GB GPU

If you get CUDA OOM even with:
- Batch size 1
- Qwen/Qwen2.5-1.5B model
- LoRA enabled

**Checklist:**

1. **Verify model name:**
   ```bash
   # In train.py output, look for:
   # "Loading model: Qwen/Qwen2.5-1.5B"
   # If you see "Qwen/Qwen2.5-7B-Instruct" → WRONG MODEL
   ```

2. **Check for zombie processes:**
   ```bash
   nvidia-smi  # Look for old Python processes
   kill -9 <PID>  # Kill any zombie processes holding GPU memory
   ```

3. **Verify GPU has 16GB:**
   ```bash
   nvidia-smi  # Check "Memory-Usage" column
   # Should show ~15.7GB total
   ```

4. **Check system RAM:**
   ```bash
   free -h  # Needs 32GB+ total
   # If <32GB, training may OOM during data loading
   ```

5. **Last resort - use cloud GPU:**
   - See "Alternative: Larger Models" section above
   - RunPod RTX 4090 24GB: ~$0.40/hr
```

---

### 6. Update Output Structure Documentation (Lines 157-182)

**Add note about model/ directory structure:**

```markdown
## Output Structure

Training saves directly to the filter directory:

```
filters/uplifting/v1/
├── config.yaml                    # Filter configuration
├── prefilter.yaml                 # Pre-filter rules
├── README.md                      # Filter documentation
├── model/                         # Trained model checkpoint
│   ├── config.json               # Model configuration
│   ├── adapter_config.json       # LoRA configuration ⭐
│   ├── adapter_model.bin         # LoRA weights (~140MB) ⭐
│   ├── pytorch_model.bin         # Full model (optional, large)
│   └── tokenizer files           # Tokenizer config + vocab
├── training_history.json          # Metrics per epoch
└── training_metadata.json         # Training configuration
```

**Important:** When deploying to HuggingFace:
- Upload entire `model/` directory
- LoRA adapters (`adapter_*.bin`) are sufficient for inference
- Base model (`pytorch_model.bin`) is ~3GB, adapters are ~140MB

Reports and visualizations are saved separately:
```
reports/
├── uplifting_v1_training_report.docx
└── uplifting_v1_plots/
    ├── overall_metrics.png
    ├── per_dimension_mae.png
    ├── loss_curves.png
    └── training_summary.txt
```
```

---

## Summary of Changes

**Model:**
- 7B → 1.5B (standard)
- Added LoRA configuration (~37M trainable params)
- Emphasized tmux workflow for GPU training

**Hardware:**
- 16GB GPU sufficient (not "16GB+")
- Added system RAM requirement (32GB+)
- Detailed memory breakdown

**Training:**
- 1.5-2 hours (not 2-4)
- MAE 0.78 target (proven)
- Cloud GPU only if quality insufficient

**Deployment:**
- LoRA adapters small (~140MB)
- HuggingFace API target
- Migration path documented

---

## Related Files

Also update:
- [x] `docs/README_UPDATES_NEEDED.md` - Main README tracking
- [ ] `docs/guides/qwen-finetuning-guide.md` - Already updated (1,606 lines removed)
- [ ] `docs/guides/gpu-training-guide.md` - Already created (475 lines)
- [ ] `docs/decisions/2025-11-10-model-selection-qwen-1.5b.md` - Already created (ADR)
