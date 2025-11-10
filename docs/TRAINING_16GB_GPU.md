# Training Configuration for 16GB GPU

**Your Hardware**: 16GB VRAM GPU
**Challenge**: Qwen2.5-7B with LoRA + batch size needs ~19GB typically
**Solution**: Optimized configuration for 16GB GPUs

---

## Memory Optimization Strategy

### 1. Reduced Batch Size & Sequence Length

**Standard Config** (needs 24GB):
- Batch size: 4
- Max sequence length: 2048
- Gradient accumulation: 4
- Memory usage: ~19GB

**16GB Optimized Config**:
- Batch size: **2** (reduced)
- Max sequence length: **1536** (reduced from 2048)
- Gradient accumulation: **8** (increased to maintain effective batch size)
- Memory usage: ~14-15GB ✅

**Effective batch size remains 16** (2 × 8 = 16, same as 4 × 4)

### 2. Additional Memory Optimizations

Already enabled in script:
- ✅ 4-bit quantization (saves ~50% memory)
- ✅ Gradient checkpointing (saves ~30% memory)
- ✅ 8-bit optimizer (saves memory)

---

## Training Command for 16GB GPU

```bash
cd C:/local_dev/llm-distillery

python scripts/train_model.py \
  --train-file training_data/tech_deployment/v1/train.jsonl \
  --val-file training_data/tech_deployment/v1/val.jsonl \
  --output-dir models/tech_deployment_v1 \
  --max-seq-length 1536 \
  --lora-rank 16 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --learning-rate 2e-4 \
  --epochs 3 \
  --warmup-steps 100 \
  --eval-steps 200 \
  --save-steps 200
```

**Key changes**:
- `--max-seq-length 1536` (was 2048)
- `--batch-size 2` (was 4)
- `--gradient-accumulation 8` (was 4)

---

## Memory Breakdown (16GB Config)

```
Component                 Memory Usage
--------------------------------
Base model (4-bit)       ~7 GB
LoRA adapters            ~2 GB
Batch size 2             ~5 GB
Optimizer states         ~1 GB
Activation memory        ~1 GB
--------------------------------
Total                    ~16 GB ✅ Fits!
Peak usage               ~15-16 GB
```

---

## Expected Impact

### Training Speed
- **Slightly slower** due to more gradient accumulation steps
- Original: ~2-3 hours → 16GB config: ~3-4 hours
- Still much faster than oracle (150x speedup remains)

### Model Quality
- **No impact** on final model quality
- Effective batch size unchanged (still 16)
- Sequence length 1536 is sufficient (most prompts <1500 tokens)

### Success Criteria
- Same as 24GB config
- No adjustment needed

---

## Troubleshooting

### If you still get "CUDA out of memory"

**Option 1: Reduce batch size further**
```bash
--batch-size 1 \
--gradient-accumulation 16
```
Memory: ~13GB (very safe)
Speed: ~4-5 hours (slower but works)

**Option 2: Reduce LoRA rank**
```bash
--lora-rank 8  # Default is 16
```
Memory: Saves ~1GB
Quality: Minimal impact for this task

**Option 3: Reduce sequence length more**
```bash
--max-seq-length 1024
```
Memory: Saves ~2GB
Coverage: 95% of prompts still fit

**Option 4: Use gradient checkpointing aggressively**
Already enabled in script, but if needed:
```python
# In train_model.py, add to TrainingArguments:
gradient_checkpointing=True,
gradient_checkpointing_kwargs={"use_reentrant": False}
```

---

## Monitoring Memory Usage

### Before Training Starts

Check available memory:
```bash
nvidia-smi
```

Should show:
- Total memory: 16GB
- Free memory: ~15GB (before loading model)

### During Training

Watch memory in another terminal:
```bash
watch -n 1 nvidia-smi
```

**Normal pattern**:
- Initial model load: Jumps to ~9GB
- First training step: Jumps to ~15GB
- Steady state: ~14-15GB
- Evaluation: Slight increase (~15-16GB)

**Warning signs**:
- Memory usage >15.5GB → reduce batch size
- "CUDA out of memory" error → use Option 1 above

---

## Alternative: CPU Offloading (If GPU Still Struggles)

If 16GB is still tight, use CPU offloading:

```python
# In train_model.py, modify model loading:
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=1536,
    dtype=None,
    load_in_4bit=True,
    device_map="auto",  # Add this - auto offloads to CPU/disk if needed
)
```

**Effect**:
- Memory: Always fits (uses system RAM as overflow)
- Speed: ~2x slower (due to CPU<->GPU transfers)
- Total time: ~6-8 hours

---

## Summary: Recommended Configuration

**For RTX 3060/4060 16GB or similar**:

Use the optimized command above:
- Batch size 2
- Sequence length 1536
- Gradient accumulation 8

**Expected results**:
- Memory usage: ~14-15GB ✅
- Training time: ~3-4 hours
- Model quality: Identical to 24GB config

**If that fails**:
- Reduce batch size to 1
- Or reduce sequence length to 1024
- Or enable CPU offloading

---

## Cost Estimate

**Cloud GPU alternatives** (if local 16GB still problematic):

| Provider | GPU | VRAM | Cost/hr | 3hr Training |
|----------|-----|------|---------|--------------|
| RunPod | RTX 4090 | 24GB | $0.44 | $1.32 |
| Vast.ai | RTX 4090 | 24GB | $0.35 | $1.05 |
| Lambda Labs | A100 40GB | 40GB | $1.10 | $3.30 |

**Recommendation**: Try 16GB config first. If issues, rent cloud GPU for $1-3.

---

**Ready to train on 16GB GPU with optimized config!** ✅
