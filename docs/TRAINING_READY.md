# Training Setup Complete - Ready to Execute

**Date**: 2025-11-09
**Status**: ✅ READY FOR TRAINING
**Model**: Qwen2.5-7B-Instruct
**Filter**: sustainability_tech_deployment

---

## Summary

Training data has been prepared and is ready for model distillation. All mitigation strategies for class imbalance have been applied.

---

## Dataset Summary

**Total Labels**: 2,080 (1,938 base + 142 supplemental)

**Training Set** (after oversampling):
- Total examples: 2,428
- Vaporware: 1,519 (62.6%)
- Pilot: 303 (12.5%)
- Early Commercial: 303 (12.5%)
- Deployed/Proven: 303 (12.5%)

**Validation Set** (natural distribution):
- Total examples: 209
- Vaporware: 169 (80.9%)
- Pilot: 23 (11.0%)
- Early Commercial: 14 (6.7%)
- Deployed/Proven: 3 (1.4%)

---

## Files Ready

### Training Data
```
training_data/tech_deployment/v1/
├── train.jsonl (2,428 examples, 6.7 MB)
└── val.jsonl (209 examples, 612 KB)
```

### Oracle Labels (Merged)
```
ground_truth/labeled/tech_deployment_merged/
└── all_labels.jsonl (2,080 labels)
```

### Scripts
```
scripts/
├── prepare_training_data.py (completed ✅)
└── train_model.py (ready for execution)
```

### Documentation
```
docs/
├── handling_imbalanced_datasets.md (comprehensive mitigation strategies)
├── stratified_sampling_strategy.md (rationale for approach)
└── TRAINING_READY.md (this file)

reports/
├── stratified_sampling_final_assessment.md (decision to accept dataset)
├── tech_deployment_label_validation.md (quality check - PASS)
└── tech_deployment_label_distribution_analysis.md (imbalance analysis)
```

---

## How to Train

### Prerequisites

**Install Unsloth** (if not already installed):
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

**Hardware Requirements**:
- GPU: 24GB VRAM recommended (RTX 3090/4090, A100, etc.)
- RAM: 32GB+
- Disk: 50GB free space for model + checkpoints

**Estimated Training Time**:
- 3 epochs on RTX 4090: ~2-3 hours
- 3 epochs on A100: ~1-2 hours

### Training Command

```bash
cd C:/local_dev/llm-distillery

python scripts/train_model.py \
  --train-file training_data/tech_deployment/v1/train.jsonl \
  --val-file training_data/tech_deployment/v1/val.jsonl \
  --output-dir models/tech_deployment_v1 \
  --max-seq-length 2048 \
  --lora-rank 16 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --learning-rate 2e-4 \
  --epochs 3 \
  --warmup-steps 100 \
  --eval-steps 200 \
  --save-steps 200
```

### Training Configuration

**Model**: Qwen2.5-7B-Instruct (4-bit quantized)
**Method**: LoRA fine-tuning (rank 16)
**Batch Size**: 4 per device (effective 16 with gradient accumulation)
**Learning Rate**: 2e-4 with cosine schedule
**Optimizer**: AdamW 8-bit
**Mixed Precision**: FP16/BF16 (auto-detected)

**Memory Usage**:
- 4-bit model: ~7GB VRAM
- LoRA adapters: ~2GB
- Batch size 4: ~10GB
- Total: ~19GB VRAM (fits on 24GB GPU)

---

## Monitoring Training

### Metrics to Watch

**Loss Curves**:
- Train loss should decrease steadily
- Val loss should track train loss (small gap OK)
- Large gap → overfitting

**Checkpoints**:
- Saved every 200 steps
- Best model auto-selected by val loss
- Up to 3 checkpoints retained

**Expected Loss Range**:
- Initial: ~2.5-3.0
- Final train: ~0.5-1.0
- Final val: ~0.8-1.2

### Warning Signs

❌ **Val loss increases while train decreases**: Overfitting
- Solution: Reduce epochs, increase dropout

❌ **Loss stuck high (>2.0 after 1 epoch)**: Underfitting or learning rate issue
- Solution: Check learning rate, increase LoRA rank

❌ **NaN/Inf loss**: Numerical instability
- Solution: Reduce learning rate, check data format

---

## Success Criteria (Adjusted for Imbalance)

**After Training Completes**:

**Minimum Acceptable** (POC success):
- Vaporware recall: ≥85%
- Pilot recall: ≥55%
- Early commercial recall: ≥45%
- Deployed recall: ≥30%
- MAE per dimension: ≤1.5

**Target** (good results):
- Vaporware recall: ≥90%
- Pilot recall: ≥65%
- Early commercial recall: ≥55%
- Deployed recall: ≥45%
- MAE per dimension: ≤1.3

**Note**: Only 3 deployed examples in validation set, so deployed recall may be noisy.

---

## After Training

### 1. Evaluate on Validation Set

Create evaluation script to:
- Load best checkpoint
- Run inference on 209 val examples
- Calculate per-tier metrics
- Generate confusion matrix
- Calculate MAE per dimension
- Identify failure cases

### 2. Inference on Full Dataset

If evaluation passes:
- Run on all 147K articles
- Compare tier distributions vs oracle subset
- Measure inference speed vs oracle (target: 50-100x faster)
- Calculate cost savings (target: 150x cheaper)

### 3. Document Results

Create evaluation report:
- Per-tier performance breakdown
- Dimension accuracy analysis
- Failure case analysis
- Cost/speed comparison vs oracle
- Recommendations for production iteration

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `--batch-size` to 2
- Reduce `--max-seq-length` to 1536
- Enable gradient checkpointing (already enabled)

### "Unsloth not found"
```bash
pip uninstall unsloth -y
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### "Model taking too long to train"
- Reduce `--epochs` to 2
- Increase `--batch-size` if memory allows
- Use smaller LoRA rank (`--lora-rank 8`)

### "Poor deployed tier recall"
- Expected with only 3 val examples
- Check train set deployed examples are being learned
- Consider increasing oversampling ratio in prepare_training_data.py

---

## Alternative Training Approaches

If initial training fails to meet minimum criteria:

### Option 1: Add Class Weights

Modify `train_model.py` to use weighted loss:
```python
from transformers import Trainer
class WeightedTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Apply class weights to loss
        loss = super().compute_loss(model, inputs, return_outputs)
        # Weight by tier (implement weighting logic)
        return loss
```

### Option 2: Focal Loss

Replace cross-entropy with focal loss:
- Emphasizes hard examples (deployed tier)
- Reduces focus on easy examples (vaporware)
- Requires custom loss implementation

### Option 3: Two-Stage Training

Stage 1: Binary (deployed vs not)
- Heavily oversample deployed
- Use focal loss
- Target: High deployed recall

Stage 2: Regression for non-deployed
- Only predict 1.0-7.9
- Better balanced dataset

---

## Cost & Time Estimates

**Training**:
- Cloud GPU (A100 80GB): ~$2-3 for 2 hours
- Local GPU: Free (electricity cost negligible)

**Total Project Cost So Far**:
- Oracle labeling: $2.08
- Training: $2-3 (if cloud)
- **Total**: ~$5

**Time Investment**:
- Data preparation: 4 hours
- Training: 2-3 hours
- Evaluation: 2 hours
- **Total**: ~8-9 hours

---

## Next Actions

**Ready to train when you are.**

Just run the training command above. The script will:
1. Load training data (2,428 examples)
2. Load validation data (209 examples)
3. Load Qwen2.5-7B with 4-bit quantization
4. Add LoRA adapters
5. Train for 3 epochs (~2-3 hours)
6. Save best model
7. Output final model to `models/tech_deployment_v1/final/`

**Monitor** via console output - loss, eval metrics printed every 50 steps.

**After training**, create evaluation script to assess performance on validation set.

---

**Status**: ✅ ALL SYSTEMS GO
**Date**: 2025-11-09
