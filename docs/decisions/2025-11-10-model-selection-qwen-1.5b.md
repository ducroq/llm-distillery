# ADR: Model Selection - Qwen/Qwen2.5-1.5B for Filter Training

**Date:** 2025-11-10
**Status:** Accepted
**Decision Maker:** Project Team

---

## Context

We are building a knowledge distillation system where local student models learn to replicate expensive cloud oracle (Gemini Flash) behavior for content filtering. We need to select a base model architecture for training across 6+ filters.

### Initial Situation

- First filter (uplifting) was successfully trained using `Qwen/Qwen2.5-7B` model identifier
- Upon inspection, this model actually loaded **494M parameters** (likely Qwen2.5-0.5B or 1.5B variant)
- Achieved **validation MAE of 0.778** on 8-dimensional regression task
- Training completed successfully on RTX 4080 16GB GPU without quantization

### GPU Constraints Discovery

When attempting to train the second filter (tech_deployment), we tried using `Qwen/Qwen2.5-7B-Instruct`:
- This loaded the actual **7.6 billion parameter model** (~15GB in FP32)
- Encountered CUDA Out of Memory errors even with:
  - Batch size reduced to 1
  - LoRA parameter-efficient training (37M trainable parameters)
  - Gradient checkpointing enabled
- Root cause: Base model consumes 15GB of 15.7GB available VRAM, leaving insufficient room for activations and optimizer states

### Deployment Requirements

- Models will be deployed via **HuggingFace Inference API** (not local inference)
- Need to train 6 filters total across different domains
- Consistency across filters is critical for maintenance
- GitHub Actions workflows will upload models to HuggingFace Hub

---

## Decision

**We will standardize on `Qwen/Qwen2.5-1.5B` for all filter training.**

### Model Specifications
- **Model ID:** `Qwen/Qwen2.5-1.5B`
- **Parameters:** ~1.5 billion
- **Memory footprint:** ~3GB in FP32
- **Training precision:** FP32 (full precision, no quantization)
- **Architecture:** Qwen2.5 base (decoder-only transformer)

---

## Rationale

### 1. Proven Performance
- Uplifting filter achieved 0.778 MAE with similar-sized model
- Demonstrates sufficient capacity for 8-dimensional regression task
- Oracle labels already captured from Gemini Flash, student just needs to learn patterns

### 2. Hardware Compatibility
- Fits comfortably in 16GB GPU VRAM (~3GB model + ~5-8GB training overhead)
- Enables batch size 4-8 for efficient training
- No quantization needed, preserving full precision
- Training time: ~2-4 hours on RTX 4080

### 3. Deployment Simplicity
- **HuggingFace Inference API:**
  - Free tier supports models up to 2B parameters
  - Serverless scaling without managing infrastructure
  - HTTPS API endpoints automatically generated
  - Model cards and versioning built-in

- **Cost Estimate (HF Inference API):**
  - Free tier: 30,000 characters/month (~150 articles/month)
  - Pro tier ($9/month): Unlimited requests
  - Inference Endpoints: ~$0.60/hour for dedicated hosting

### 4. Consistency
- Same model architecture across all 6 filters
- Comparable quality expectations
- Simplified maintenance and debugging
- Unified deployment pipeline

### 5. Future Flexibility
**Migration path to larger models if needed:**
- If filter quality proves insufficient with 1.5B model
- When access to larger GPU becomes available (24GB+ VRAM)
- Cloud training options:
  - **RunPod:** ~$0.40-0.60/hr for RTX 4090 (24GB)
  - **Vast.ai:** ~$0.30-0.50/hr for similar specs
  - **Lambda Labs:** ~$1.10/hr for A100 (40GB)
- Migration process:
  1. Prepare training data (same pipeline)
  2. Train on cloud GPU with larger model (e.g., Qwen2.5-7B-Instruct)
  3. Compare validation metrics
  4. Deploy to HuggingFace (may require paid Inference Endpoints for 7B)

---

## Alternatives Considered

### Alternative 1: Qwen/Qwen2.5-7B-Instruct with 8-bit Quantization
**Rejected due to:**
- Adds complexity (quantization during training)
- Inconsistent with proven uplifting approach
- Potential quality degradation from quantization
- More complex deployment (need to handle quantization in inference)
- Still risks OOM on 16GB GPU depending on batch size

**Why we tried this:**
- Initially believed larger model = better quality
- Thought quantization would solve memory issues

**Lessons learned:**
- Model size ≠ task performance (diminishing returns)
- Smaller models with good training data often sufficient
- Simplicity and consistency > maximum capacity

### Alternative 2: Cloud GPU Training (7B+ models)
**Deferred for future:**
- Costs ~$1.50-3.00 per training run (acceptable)
- Requires environment setup overhead
- Makes iteration slower (upload data, monitor remotely)
- No evidence yet that larger model improves results

**When to reconsider:**
- If validation MAE > 1.5 consistently
- If model fails to learn meaningful patterns
- If we need to train 100B+ scale models

### Alternative 3: Other Model Families
**Not explored due to:**
- Qwen2.5 proven effective for our task
- Strong multilingual capabilities (future-proofing)
- Efficient architecture for inference
- Good HuggingFace integration

---

## Implementation Plan

### 1. Update Training Script Default
```bash
# training/train.py
--model-name "Qwen/Qwen2.5-1.5B"  # Change default
```

### 2. Update Documentation
- `docs/guides/qwen-finetuning-guide.md`
- `docs/guides/gpu-training-guide.md`
- `README.md` (if applicable)

### 3. Retrain Consistency Check
**Optional:** Retrain uplifting filter with explicit model name to verify consistency:
```bash
python -m training.train \
    --filter filters/uplifting/v1 \
    --data-dir datasets/training/uplifting \
    --model-name "Qwen/Qwen2.5-1.5B" \
    --epochs 7 \
    --batch-size 4
```

### 4. Deployment Setup
- Document HuggingFace API integration
- Create deployment workflow for GitHub Actions
- Test inference API with trained model

---

## Consequences

### Positive
- ✅ **Consistent training approach** across all filters
- ✅ **No GPU memory issues** on 16GB hardware
- ✅ **Full precision training** (no quantization artifacts)
- ✅ **Simple deployment** (HuggingFace API handles everything)
- ✅ **Fast local iteration** (train multiple filters in parallel if needed)
- ✅ **Cost-effective** (free HF tier for testing, $9/month pro tier for production)

### Negative
- ⚠️ **Limited model capacity** compared to 7B+ models
- ⚠️ **May need retraining** if quality insufficient (but uplifting suggests this is unlikely)
- ⚠️ **HuggingFace vendor lock-in** for inference (but models remain portable)

### Neutral
- Trade-off: Lower capacity but proven sufficient for task
- Trade-off: Faster training but less exploration of model scaling
- Can always upgrade to larger models later with same training pipeline

---

## Validation Criteria

We will validate this decision by monitoring:

1. **Per-filter validation MAE:**
   - Target: < 1.5 per dimension
   - Uplifting achieved: 0.778

2. **Tier classification accuracy:**
   - Target: ≥ 70% per tier
   - Critical: Minority class recall ≥ 60%

3. **Training stability:**
   - No OOM errors
   - Consistent convergence across filters
   - Training time: 2-4 hours per filter

4. **Inference performance:**
   - HuggingFace API latency: < 500ms per article
   - Cost: < $20/month for expected volume

If these criteria are not met, we will revisit the decision and consider Alternative 2 (cloud GPU with larger models).

---

## References

- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- [HuggingFace Inference API Pricing](https://huggingface.co/pricing)
- Uplifting filter training metadata: `filters/uplifting/v1/training_metadata.json`
- Initial OOM troubleshooting discussion: 2025-11-10

---

## Revision History

- **2025-11-10:** Initial decision - standardize on Qwen/Qwen2.5-1.5B
