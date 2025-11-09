# Handling Imbalanced Datasets in Model Distillation

**Context**: Tech Deployment Filter Training
**Date**: 2025-11-09
**Problem**: Severe class imbalance (62.6% vaporware in training set after oversampling)

---

## Problem Statement

### Dataset Imbalance

**Original Distribution** (2,080 labels):
| Tier | Count | Percentage |
|------|-------|------------|
| Vaporware (<4.0) | 1,688 | 81.2% |
| Pilot (4.0-5.9) | 224 | 10.8% |
| Early Commercial (6.0-7.9) | 139 | 6.7% |
| Deployed/Proven (≥8.0) | 29 | 1.4% |

**After Oversampling** (2,428 train examples):
| Tier | Count | Percentage |
|------|-------|------------|
| Vaporware | 1,519 | 62.6% |
| Pilot | 303 | 12.5% |
| Early Commercial | 303 | 12.5% |
| Deployed/Proven | 303 | 12.5% |

**Validation Set** (209 examples - no oversampling):
| Tier | Count | Percentage |
|------|-------|------------|
| Vaporware | 169 | 80.9% |
| Pilot | 23 | 11.0% |
| Early Commercial | 14 | 6.7% |
| Deployed/Proven | 3 | 1.4% |

### Why This Imbalance Exists

1. **Natural distribution**: Tech news IS heavily skewed toward vaporware
2. **Corpus limitations**: High-scoring articles genuinely rare (~1.4%)
3. **Oracle accuracy**: Gemini Flash correctly identifies most as vaporware
4. **Cannot fix with sampling**: High-score examples don't exist in sufficient quantity

---

## Mitigation Strategies

### 1. Oversampling Minority Classes ✅ APPLIED

**Implementation**:
```python
def oversample_minority_classes(train_set, target_ratio=0.2):
    """Oversample to achieve target_ratio for minority classes."""
    # Calculate multipliers for each tier
    # Duplicate minority examples to reach target
    # Result: deployed_proven goes from 26 to 303 examples
```

**Effect**:
- Deployed tier: 26 → 303 examples (11.7x multiplication)
- Early commercial: 125 → 303 examples (2.4x)
- Pilot: 202 → 303 examples (1.5x)
- Vaporware: 1,519 (unchanged - majority class)

**Benefits**:
- Model sees minority classes more frequently during training
- Reduces tendency to always predict majority class
- Simple and effective for significant imbalance

**Limitations**:
- No new information added (just duplicates)
- Risk of overfitting to duplicated examples
- Validation set remains imbalanced (reflects reality)

### 2. Class Weighting (RECOMMENDED)

**Implementation in Unsloth/HuggingFace**:
```python
from sklearn.utils import class_weight
import numpy as np

# Calculate weights inversely proportional to frequency
tier_counts = {
    'vaporware': 1519,
    'pilot': 303,
    'early_commercial': 303,
    'deployed_proven': 303
}

weights = {
    'vaporware': 1.0,  # Baseline
    'pilot': 1519/303,  # 5.0x
    'early_commercial': 1519/303,  # 5.0x
    'deployed_proven': 1519/303  # 5.0x
}

# Apply during training via loss function
```

**Benefits**:
- Penalizes misclassification of minority classes more heavily
- Doesn't modify dataset (works on loss function)
- Complementary to oversampling

**How to Apply**:
- Use `class_weight` parameter in HuggingFace Trainer
- Or implement custom weighted loss function
- Adjust weights based on validation performance

### 3. Focal Loss (ADVANCED)

**Concept**:
Focal loss reduces focus on easy-to-classify examples and emphasizes hard examples:

```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
```

Where:
- `α_t`: Class weight
- `γ`: Focusing parameter (typically 2.0)
- `p_t`: Model's estimated probability for correct class

**Effect**:
- Easy examples (p_t ≈ 1): Loss near zero (already learned)
- Hard examples (p_t ≈ 0): Full loss (needs attention)
- Deployed examples likely "hard" → get more focus

**Implementation**:
```python
import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t)**self.gamma * ce_loss
        return focal_loss.mean()

# Use as loss function in training
```

**When to Use**:
- If model still biased toward vaporware after oversampling + class weighting
- Particularly useful for continuous scores (regression vs classification)
- May require hyperparameter tuning (γ, α)

### 4. Stratified Validation (✅ APPLIED)

**Implementation**:
- Split each tier separately (90/10)
- Ensures all tiers represented in validation
- Validation maintains natural distribution (reality check)

**Benefits**:
- Can measure per-tier performance
- Validation reflects real-world distribution
- Prevents overfitting to synthetic balance

### 5. Evaluation Metrics (CRITICAL)

**Don't Use**:
- ❌ Overall accuracy (misleading with imbalance - can get 80% by predicting vaporware always)

**Do Use**:
- ✅ **Per-tier accuracy/precision/recall**
- ✅ **Mean Absolute Error (MAE) per dimension**
- ✅ **Confusion matrix** (visualize misclassifications)
- ✅ **F1-score weighted by tier**

**Example Metrics**:
```python
metrics = {
    'overall_accuracy': 0.82,  # Not very informative
    'tier_metrics': {
        'vaporware': {'precision': 0.90, 'recall': 0.95, 'f1': 0.92},
        'pilot': {'precision': 0.70, 'recall': 0.65, 'f1': 0.67},
        'early_commercial': {'precision': 0.65, 'recall': 0.60, 'f1': 0.62},
        'deployed_proven': {'precision': 0.55, 'recall': 0.50, 'f1': 0.52}
    },
    'mae_per_dimension': {
        'deployment_maturity': 1.2,
        'proof_of_impact': 0.9,
        # ...
    }
}
```

### 6. Success Criteria (ADJUSTED FOR IMBALANCE)

**Original POC Goals**:
- Overall accuracy: ≥88%
- MAE per dimension: ≤1.0

**Adjusted for Imbalance**:
- **Vaporware detection**: ≥85% recall (majority class)
- **Pilot detection**: ≥60% recall (moderate)
- **Early commercial detection**: ≥50% recall (low freq)
- **Deployed detection**: ≥40% recall (very rare - acceptable)
- **MAE per dimension**: ≤1.5 (relaxed)
- **No catastrophic failures**: Model must not predict >8.0 for clear vaporware

**Why Relaxed**:
- Only 3 deployed examples in validation set
- Natural distribution heavily skewed
- POC goal is validate distillation works, not perfect accuracy
- Production iteration can address limitations

---

## Training Configuration

### Recommended Unsloth/HuggingFace Setup

```python
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True  # Use 4-bit quantization for efficiency
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/tech_deployment_v1",
    num_train_epochs=3,  # Start with 3, adjust based on validation
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,  # Mixed precision training
    optim="adamw_8bit",  # 8-bit optimizer
    weight_decay=0.01,
    max_grad_norm=1.0,
    # Class weighting via label smoothing (approximation)
    label_smoothing_factor=0.1
)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    max_seq_length=2048,
    dataset_text_field="text",  # Format: f"{prompt}\n\n{completion}"
)

# Train
trainer.train()
```

### Hyperparameters to Tune

**If deployed tier recall < 30%**:
- Increase oversampling ratio (0.2 → 0.3)
- Add class weights
- Reduce learning rate (2e-4 → 1e-4) - slower, more careful learning
- Increase epochs (3 → 5)

**If model overfits (train loss << val loss)**:
- Increase dropout (0.05 → 0.1)
- Increase weight decay (0.01 → 0.05)
- Reduce LoRA rank (16 → 8)
- Use early stopping

**If model underfits (high train loss)**:
- Increase LoRA rank (16 → 32)
- Increase learning rate (2e-4 → 3e-4)
- Increase epochs

---

## Monitoring Training

### Key Metrics to Track

1. **Loss curves**:
   - Train loss should decrease steadily
   - Val loss should track train loss (gap indicates overfitting)

2. **Per-tier metrics** (custom callback):
   - Track precision/recall for each tier separately
   - Watch for deployed tier recall (target ≥40%)

3. **Confusion matrix** (after each epoch):
   - Where are misclassifications happening?
   - Is model conflating pilot with deployed?

4. **MAE per dimension**:
   - Which dimensions are hardest to learn?
   - proof_of_impact likely challenging (only 4 high-score examples in val)

### Warning Signs

❌ **Model always predicts vaporware**:
- Increase oversampling ratio
- Add class weights
- Check if loss function is working

❌ **Deployed tier recall = 0%**:
- Only 3 examples in val - may be too few
- Check train set deployed examples are diverse
- Consider focal loss

❌ **High variance in deployed predictions**:
- Model hasn't learned consistent pattern
- May need more data (production iteration)

❌ **Catastrophic failures** (predicting 9.0 for obvious vaporware):
- Learning rate too high
- Model overfitting to noise
- Reduce LR, add regularization

---

## Post-Training Evaluation

### Comprehensive Evaluation Script

```python
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Load validation predictions
predictions = load_predictions('val_predictions.jsonl')

# Extract tiers
true_tiers = [p['true_tier'] for p in predictions]
pred_tiers = [p['pred_tier'] for p in predictions]

# Overall metrics
print(classification_report(true_tiers, pred_tiers))

# Confusion matrix
cm = confusion_matrix(true_tiers, pred_tiers,
                      labels=['vaporware', 'pilot', 'early_commercial', 'deployed_proven'])
print(cm)

# Per-dimension MAE
for dim in dimensions:
    true_scores = [p[f'true_{dim}'] for p in predictions]
    pred_scores = [p[f'pred_{dim}'] for p in predictions]
    mae = np.mean(np.abs(np.array(true_scores) - np.array(pred_scores)))
    print(f'{dim}: MAE = {mae:.2f}')

# Deployed tier deep dive
deployed_examples = [p for p in predictions if p['true_tier'] == 'deployed_proven']
print(f'\nDeployed Tier Analysis ({len(deployed_examples)} examples):')
for ex in deployed_examples:
    print(f'  True: {ex["true_score"]:.1f}, Pred: {ex["pred_score"]:.1f}, '
          f'Error: {abs(ex["true_score"] - ex["pred_score"]):.1f}')
```

### Decision Points After Evaluation

**If results meet adjusted success criteria**:
- ✅ Proceed to inference on full 147K dataset
- ✅ Analyze tier distributions vs oracle
- ✅ Document learnings for production iteration

**If deployed tier recall < 20%**:
- Consider generating synthetic deployed examples (paraphrasing)
- Manual curation of high-score examples from other sources
- Accept limitation and focus on vaporware/pilot/early-commercial discrimination

**If dimension MAE > 2.0**:
- Review dimension definition clarity
- Check if oracle labels are consistent for that dimension
- Consider dimension-specific fine-tuning

---

## Alternative Approaches (If POC Fails)

### 1. Two-Stage Model

**Stage 1**: Binary classifier (deployed vs not-deployed)
- Heavily oversample deployed examples
- Use focal loss
- Target: ≥70% deployed recall

**Stage 2**: Regression for non-deployed
- Only predict scores 1-7.9
- Better balanced dataset
- More accurate for majority cases

### 2. Ordinal Regression

Instead of treating as continuous regression, use ordinal approach:
- Tiers are ordered (vaporware < pilot < early_comm < deployed)
- Use ordinal loss function (respects order)
- May improve tier boundary discrimination

### 3. Ensemble with Rule-Based

Combine learned model with keyword-based scoring:
- Model predicts score
- Rule-based system checks for deployment signals
- If strong signals present, boost score
- If vaporware language present, cap score

---

## Summary of Applied Mitigations

**Currently Applied** ✅:
1. Stratified train/val split
2. Oversampling minority classes (20% target ratio)
3. Adjusted success criteria

**Recommended for Training** 🎯:
4. Class weighting in loss function
5. Per-tier metric tracking
6. Early stopping on validation loss

**Consider If Needed** ⚠️:
7. Focal loss (if class weighting insufficient)
8. Synthetic data generation (if deployed recall < 20%)
9. Two-stage or ensemble approach (if POC fails)

**Do Not Apply** ❌:
- SMOTE or other synthetic oversampling (duplicates are simpler and equally effective)
- Undersampling majority class (would discard useful vaporware examples)
- Collecting more data from same corpus (high-score examples don't exist)

---

## Expected Outcomes

### Realistic Expectations

**Best Case** (all mitigations work):
- Vaporware recall: ~90%
- Pilot recall: ~65%
- Early commercial recall: ~55%
- Deployed recall: ~45%
- MAE per dimension: ~1.3

**Acceptable Case** (POC success):
- Vaporware recall: ~85%
- Pilot recall: ~55%
- Early commercial recall: ~45%
- Deployed recall: ~30%
- MAE per dimension: ~1.5

**Failure Case** (need alternative approach):
- Deployed recall: <20%
- Model always predicts <6.0
- MAE per dimension: >2.0

### Next Steps by Outcome

**Best/Acceptable**:
- Proceed to full dataset inference
- Compare distilled model vs oracle (cost/speed/quality)
- Plan production iteration improvements

**Failure**:
- Try alternative approaches (two-stage, ensemble)
- Consider different model architecture
- Manual curation of additional high-score examples

---

**Author**: Claude (AI Assistant)
**Date**: 2025-11-09
