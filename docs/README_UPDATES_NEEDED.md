# README.md Updates Needed

**Date:** 2025-11-10
**Context:** Following model selection decision (ADR 2025-11-10) and sandbox infrastructure setup

---

## Critical Updates Required

### 1. Model Selection (Line 12)
**Current:**
```markdown
2. **Fine-tunes Qwen2.5-7B-Instruct** (see [model decision](docs/decisions/2025-11-08-local-model-selection.md))
```

**Should be:**
```markdown
2. **Fine-tunes Qwen2.5-1.5B** (see [model decision](docs/decisions/2025-11-10-model-selection-qwen-1.5b.md))
```

**Reason:** We've standardized on 1.5B model for all filters (fits on 16GB GPU, proven effective with uplifting filter MAE 0.778)

---

### 2. Deployment Strategy (Line 14)
**Current:**
```markdown
4. **Deploys locally** for fast, cost-effective batch inference (150x faster than oracle)
```

**Should be:**
```markdown
4. **Deploys via HuggingFace Inference API** for serverless, cost-effective inference
```

**Reason:** Deployment target is HuggingFace API (free tier or $9/month pro), not local inference

---

### 3. Status Date (Line 23)
**Current:**
```markdown
## Current Status (October 2025)
```

**Should be:**
```markdown
## Current Status (November 2025)
```

---

### 4. Training Pipeline Status (Line 41)
**Current:**
```markdown
- **Training Pipeline**: Qwen 2.5-7B fine-tuning with multi-dimensional regression
```

**Should be:**
```markdown
- **Training Pipeline**: Qwen 2.5-1.5B fine-tuning with LoRA for memory-efficient training
```

**Reason:** We're using 1.5B model with LoRA (reduces trainable params from 1.5B to ~37M)

---

### 5. Training Command Example (Line 146)
**Current:**
```bash
python -m training.train \
    --filter filters/uplifting/v1 \
    --data-dir datasets/uplifting_ground_truth_v1_splits \
    --output-dir inference/deployed/uplifting_v1 \
    --model-name Qwen/Qwen2.5-7B \
    --epochs 3 \
    --batch-size 8
```

**Should be:**
```bash
python -m training.train \
    --filter filters/uplifting/v1 \
    --data-dir datasets/uplifting_ground_truth_v1_splits \
    --output-dir filters/uplifting/v1/model \
    --model-name Qwen/Qwen2.5-1.5B \
    --epochs 3 \
    --batch-size 8
```

**Reason:** Updated default model name, output directory should be filter package location

---

### 6. GPU Requirements (Line 151)
**Current:**
```markdown
**Requirements**: 16GB+ GPU (RTX 4090, A100), ~2-4 hours training time
```

**Should be:**
```markdown
**Requirements**: 16GB GPU (RTX 4080, RTX 4090), ~1.5-2 hours training time

**Note**: 1.5B model fits comfortably in 16GB VRAM with LoRA. No quantization needed.
**Tmux Recommended**: Use tmux for persistent sessions during training (see [GPU Training Guide](docs/guides/gpu-training-guide.md))
```

**Reason:** 1.5B model works on 16GB GPU, faster training time, emphasis on tmux usage

---

### 7. Architecture Diagram (Lines 184-212)
**Current:** Shows "LLM Oracle (Claude)" and "Small Model (DeBERTa)"

**Should update to:**
```
┌─────────────────────────────────────────────────────────────┐
│                    GROUND TRUTH GENERATION                   │
│                                                              │
│  Raw Articles  →  LLM Oracle  →  Labeled Dataset            │
│  (50K samples)    (Gemini)       (JSONL + 8D scores)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       FINE-TUNING                            │
│                                                              │
│  Ground Truth  →  Qwen 2.5-1.5B  →  Trained Classifier      │
│  (JSONL)          (with LoRA)        (MAE < 1.0)            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      VALIDATION                              │
│                                                              │
│  Test Set  →  Compare  →  Quality Metrics                   │
│               (Model vs Oracle)   (MAE, per-dim accuracy)   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT                                │
│                                                              │
│  Article  →  HuggingFace API  →  Predictions                │
│  (input)     (<500ms)             (8 dimensions)            │
└─────────────────────────────────────────────────────────────┘
```

**Reason:** Reflects actual implementation (Gemini oracle, Qwen 1.5B with LoRA, HF deployment)

---

### 8. Project Structure - Add Sandbox (After line 224)
**Add new section:**
```
├── sandbox/                   # Experimental workspace (gitignored)
│   ├── README.md             # Sandbox usage guide
│   ├── <dated-experiments>/  # Organized by date
│   └── failed_approaches/    # Document what didn't work
│
├── scratch/                   # Very temporary work (gitignored)
├── playground/                # Testing new libraries (gitignored)
└── experiments/local/         # Local experiments (gitignored)
```

**Location:** Insert after `filters/` section in Project Structure

**Reason:** We added sandbox infrastructure for safe experimentation

---

### 9. Performance Benchmarks Section (Lines 335-344)
**Current:** Shows DeBERTa, BERT, Flan-T5 benchmarks

**Should replace with:**
```markdown
## Performance Benchmarks

### Qwen 2.5 Models (Current Approach)

| Model | Size | Training Time | GPU Memory | Validation MAE |
|-------|------|---------------|------------|----------------|
| Qwen 2.5-1.5B | 1.5B params | ~1.5-2 hours | ~8GB | 0.78 (uplifting) |
| Qwen 2.5-7B-Instruct | 7.6B params | ~4-6 hours | ~20GB+ | Not tested |

**Current Standard**: Qwen/Qwen2.5-1.5B with LoRA
- **Trainable Parameters**: ~37M (2.5% of model)
- **Hardware**: RTX 4080 16GB (no quantization needed)
- **Training**: Full precision (FP32)
- **Deployment**: HuggingFace Inference API

### Migration Path to Larger Models
If 1.5B model quality proves insufficient:
- **Cloud GPU Training**: RunPod (~$0.40/hr), Vast.ai (~$0.30/hr)
- **Larger Models**: Qwen 2.5-7B on 24GB+ GPU
- **Cost**: ~$1.50-3.00 per training run
- See [Model Selection ADR](docs/decisions/2025-11-10-model-selection-qwen-1.5b.md) for details
```

**Reason:** Reflects actual models being used, not hypothetical BERT-family benchmarks

---

### 10. Roadmap Phase 2 (Lines 357-361)
**Current:**
```markdown
### Phase 2: Training Pipeline (Next)
- [ ] PyTorch training script
- [ ] Model architectures (BERT, DeBERTa, T5)
- [ ] Training configs
- [ ] Experiment tracking (W&B)
```

**Should be:**
```markdown
### Phase 2: Training Pipeline ✅ (Completed)
- [x] PyTorch training script (training/train.py)
- [x] Qwen 2.5 model architecture with LoRA
- [x] Multi-dimensional regression training
- [x] Train/val/test splitting with stratified sampling
- [x] Model selection decision (1.5B for 16GB GPU)
- [x] GPU training guide with tmux workflow
- [ ] Experiment tracking (W&B integration)
- [ ] Training report generation (post-training analysis)
```

**Reason:** Training pipeline is completed and documented, just need experiment tracking

---

## Additional Improvements

### Add Deployment Section
Insert new section after "Training Pipeline":

```markdown
### Phase 2.5: Deployment Setup (In Progress)
- [ ] HuggingFace Hub upload automation
- [ ] HuggingFace Inference API integration
- [ ] GitHub Actions workflow for model deployment
- [ ] Postfilter.py for production inference
- [ ] API endpoint testing and validation
```

### Update Cost Analysis (Lines 318-333)
Add HuggingFace API pricing:

```markdown
### HuggingFace Inference API Deployment
- **Free Tier**: 30,000 characters/month (~150 articles/month)
- **Pro Tier**: $9/month for unlimited requests
- **Inference Endpoints**: ~$0.60/hour for dedicated hosting
- **Model Size Limit**: Free tier supports up to 2B parameter models

### Updated ROI Calculation
If processing **4,000 articles/day**:
- **Gemini Flash API**: 4,000 × $0.001 × 365 = **$1,460/year**
- **HuggingFace Pro**: $9 × 12 = **$108/year**
- **Training Cost**: ~$0 (local GPU) or ~$3/filter (cloud)
- **Total Savings**: **$1,352/year per filter** (93% cost reduction)
```

---

## Files That Need Review After README Update

1. **docs/README.md** - Main documentation index
2. **training/README.md** - Training documentation
3. **filters/README.md** - Filter development workflow

Ensure these are consistent with updated README.md

---

## Related Changes Already Completed

✅ Created ADR: `docs/decisions/2025-11-10-model-selection-qwen-1.5b.md`
✅ Updated `training/train.py` default to Qwen/Qwen2.5-1.5B
✅ Added LoRA support to training script
✅ Added `peft` to requirements.txt
✅ Created `sandbox/README.md`
✅ Updated `.gitignore` with sandbox directories
✅ Updated `docs/guides/getting-started.md` (Training Quick Start)
✅ Updated `docs/guides/qwen-finetuning-guide.md` (removed 1,606 lines)
✅ Created `docs/guides/gpu-training-guide.md` (475 lines, tmux workflow)
✅ Updated `docs/guides/tmux-usage.md` (training examples)
✅ Added Phase 1.5 to `AI_AUGMENTED_SOLO_DEV_FRAMEWORK.md`

---

## Action Items

1. **Update README.md** with all changes listed above
2. **Review training/README.md** for consistency
3. **Review docs/README.md** for consistency
4. **Test training command** examples to ensure accuracy
5. **Commit and push** updated documentation

---

**Priority**: Medium (should be done before next user onboarding or external documentation reference)
**Estimated Time**: 30-45 minutes for all updates
**Blocking**: No critical functionality blocked, but documentation accuracy important for new users
