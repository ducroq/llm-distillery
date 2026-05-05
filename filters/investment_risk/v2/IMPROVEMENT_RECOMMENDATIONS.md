# Investment-Risk v2 - Model Improvement Recommendations

**Current Model:** Knowledge Distillation, Qwen 2.5-1.5B
**Current Performance:** 0.67 MAE (✅ meets target <1.0)
**Date:** 2025-11-16

---

## Current Situation Analysis

### What You Have
- **Training Data:** 4,118 examples (80% split)
- **Validation Data:** 515 examples (20% split)
- **Total Oracle-Scored:** 5,150 articles
- **Current MAE:** 0.67 (train: 0.62, val: 0.67)
- **Overfitting:** Minimal (+0.046 gap, only 7% degradation)
- **Status:** ✅ Production ready

### Training Curve Analysis
```
Epoch 1: Train 1.68 → Val 0.99  (massive improvement, learning strong signal)
Epoch 2: Train 0.77 → Val 0.69  (continued improvement)
Epoch 3: Train 0.62 → Val 0.67  (convergence, slight overfitting)
```

**Observation:** Model converged well in 3 epochs. Slight overfitting is acceptable and normal.

---

## Recommendation 1: Larger Dataset

### Your Question: "We have more data, so that could be done"

**Current Data:**
- Available corpus: 402,818 articles total
- Oracle-scored: 5,150 articles
- Used for training: 4,118 articles

### ✅ YES - This is Worth Doing (Medium Priority)

**Why:**
1. **You have budget:** More data available (402k corpus)
2. **Oracle is cheap:** Gemini Flash = $0.0005/article
3. **Diminishing returns exist:** But 10k → 20k can still help

**Expected Impact:**
- Current: 4,118 examples → 0.67 MAE
- With 10k: → **0.55-0.62 MAE** (10-20% improvement likely)
- With 20k: → **0.50-0.58 MAE** (15-25% improvement likely)

### Recommendation: **Label 10,000 Total Articles**

**Cost Analysis:**
```
Current:  5,150 articles × $0.0005 = $2.58
Target:   10,000 articles × $0.0005 = $5.00
Additional: 4,850 articles × $0.0005 = $2.42
```

**Time Investment:** ~1-2 hours (batch API)

**When to Do This:**
- ⏭️ **After production deployment** (not before)
- After 1-2 months of monitoring
- If you need better accuracy (e.g., target <0.6 MAE instead of <1.0)

**How to Do It:**
```bash
# Score additional 5,000 articles
python -m ground_truth.batch_scorer \
    --filter filters/investment-risk/v2 \
    --source datasets/raw/historical_dataset.jsonl \
    --output-dir datasets/scored/investment_risk_v2_expanded \
    --llm gemini-flash \
    --batch-size 50 \
    --target-scored 10000 \
    --random-sample \
    --seed 12345

# Combine with existing data
# Retrain with expanded dataset
python training/train.py \
    --config filters/investment-risk/v2/config.yaml \
    --data-dir datasets/training/investment_risk_v2_10k \
    --epochs 4
```

**Verdict:** ✅ **DO THIS** if you want <0.6 MAE, but **NOT URGENT** (current model is production-ready)

---

## Recommendation 2: Regularization (Dropout, Weight Decay)

### Your Question: "Add regularization to reduce overfitting, why not?"

**Current Overfitting Level:**
- Train/Val Gap: +0.046 MAE (7% degradation)
- Train: 0.625 → Val: 0.671

### ⚠️ PROBABLY NOT WORTH IT (Low Priority)

**Analysis:**

#### 1. Overfitting is MINIMAL
```
Good overfitting:  <10% degradation  ← You are here (7%)
Moderate:          10-25% degradation
Problematic:       >25% degradation
```

Your model shows **excellent generalization** already.

#### 2. LoRA Already Provides Regularization

You're using LoRA (Low-Rank Adaptation):
- Only trains 18.5M params (1.2% of model)
- This is **inherent regularization**
- Full fine-tuning would have much worse overfitting

#### 3. Regularization Could HURT Performance

**Risk:**
- Adding dropout → May increase VAL MAE (0.67 → 0.75+)
- Adding weight decay → May slow convergence
- **Trade-off:** Reduced overfitting vs worse validation accuracy

#### 4. Your Target is Already Met

- Target: <1.0 MAE
- Current: 0.67 MAE
- Margin: 33% safety buffer

**When Regularization WOULD Help:**
- If train MAE was 0.50 but val MAE was 1.00+ (50%+ degradation)
- If you had 100k+ training examples (larger models need more regularization)
- If validation loss was increasing while training loss decreased

### Verdict: ⛔ **DON'T DO THIS**

**Reason:** Your overfitting is healthy and expected. Adding regularization would likely hurt validation performance without providing benefit.

**Exception:** If you expand to 20k+ examples and use a larger model (3B/7B params), then reconsider.

---

## Recommendation 3: Ensemble Methods

### Your Question: "Experiment with ensemble methods for better stability?"

### 🤔 MAYBE - Depends on Your Use Case (Medium Priority)

**What is an Ensemble?**
Train 3-5 models with different:
- Random seeds
- Training data splits
- Hyperparameters

Then average their predictions.

### Expected Impact

**Stability Improvement:**
```
Single model: 0.67 MAE ± 0.05 variance (across random seeds)
Ensemble (5 models): 0.64 MAE ± 0.02 variance
```

**Benefits:**
- ✅ More stable predictions (lower variance)
- ✅ Better calibration (confidence estimates)
- ✅ Robustness to edge cases
- ✅ ~5-10% MAE improvement possible

**Costs:**
- ❌ 5x training time
- ❌ 5x inference time (or clever batching)
- ❌ 5x storage (5 model checkpoints)
- ❌ More complex deployment

### When Ensembles Make Sense

✅ **DO THIS IF:**
1. **Critical application** - Wrong predictions have serious consequences
2. **Stability > Speed** - You value consistency over latency
3. **Budget available** - 5x inference cost is acceptable
4. **Edge cases matter** - You need robustness on rare examples

⛔ **DON'T DO THIS IF:**
1. **Latency critical** - Need <50ms inference
2. **Resource constrained** - Limited GPU/CPU budget
3. **Single model works** - 0.67 MAE already sufficient
4. **Rapid iteration** - Prefer faster training cycles

### Practical Ensemble Strategy

If you decide to do this:

**Option 1: Simple Ensemble (Recommended)**
```bash
# Train 3 models with different seeds
for seed in 42 2025 7777; do
    python training/train.py \
        --config filters/investment-risk/v2/config.yaml \
        --data-dir datasets/training/investment_risk_v2 \
        --seed $seed \
        --output-dir filters/investment-risk/v2_distillation_ensemble/model_seed_${seed}
done

# At inference: average predictions from all 3 models
```

**Expected Result:**
- Single model: 0.67 MAE
- 3-model ensemble: 0.62-0.65 MAE (5-10% improvement)

**Option 2: Advanced Ensemble**
- Train with different train/val splits (k-fold)
- Train with different hyperparameters
- More complex, more gain (potentially 0.60 MAE)

### Verdict: 🤔 **MAYBE DO THIS**

**Recommendation:**
- ⏭️ **After production deployment** (not now)
- ⏭️ After 2-3 months of usage
- ⏭️ If you identify specific failure patterns
- ⏭️ If you need <0.65 MAE and stability is critical

**Alternative:** Use the saved training budget for more data instead (better ROI).

---

## Recommendation 4: Larger Model (Not Asked, But Important)

### Should You Try Qwen 2.5-3B or 7B?

**Current:** 1.5B params → 0.67 MAE

**Expected Performance:**
```
1.5B params: 0.67 MAE  ← Current
3B params:   0.60-0.65 MAE (estimated 5-10% improvement)
7B params:   0.55-0.62 MAE (estimated 10-15% improvement)
```

### ✅ YES - Worth Testing (High Priority if you need <0.6 MAE)

**Pros:**
- ✅ Significant accuracy gain likely
- ✅ Better capacity for 8 dimensions
- ✅ Same training data (no extra cost)
- ✅ Still fast enough (<100ms inference)

**Cons:**
- ❌ Higher GPU memory (need 24GB+ for 7B)
- ❌ Longer training time (2-3x)
- ❌ Larger model storage

**Recommendation:**
If you have access to a GPU with 24GB+ VRAM:
1. Try Qwen 2.5-3B first (sweet spot)
2. If that works well, try 7B
3. Compare: 1.5B (0.67) vs 3B (??) vs 7B (??)

**When to Do This:**
- ⏭️ After production deployment
- ⏭️ If you need <0.6 MAE
- ⏭️ If you have GPU budget

---

## Overall Priority Ranking

### 🥇 Highest Priority (Do First)
1. **Deploy current model to production** - It's ready!
2. **Monitor real-world performance** - Collect edge cases
3. **Gather user feedback** - Find actual failure modes

### 🥈 Medium Priority (Do Later, If Needed)
1. **Larger dataset (10k examples)** - If you need <0.6 MAE
   - Cost: $2.42
   - Time: 2 hours
   - Expected: 0.55-0.62 MAE

2. **Larger model (3B params)** - If you need <0.6 MAE AND have GPU
   - Cost: Training time only
   - Expected: 0.60-0.65 MAE

3. **Simple ensemble (3 models)** - If stability critical
   - Cost: 3x training/inference
   - Expected: 0.62-0.65 MAE

### 🥉 Low Priority (Probably Skip)
1. **Regularization (dropout/weight decay)** - Current overfitting is healthy
   - Risk: May hurt validation performance
   - Reason: LoRA already provides regularization

---

## Recommended Action Plan

### Phase 1: Production Deployment (NOW)
```
✅ Deploy v2_distillation model
✅ Monitor for 1-2 months
✅ Collect edge cases and failure modes
✅ Track actual prediction quality
```

### Phase 2: Data-Driven Improvements (2-3 months)
```
After production data:
1. Analyze failure patterns
2. Decide if you need better accuracy
3. Choose improvement strategy based on findings
```

### Phase 3: If Accuracy Needed (3-6 months)
```
Option A: More Data (Best ROI)
- Score 10k total articles ($2.42)
- Retrain 1.5B model
- Expected: 0.55-0.62 MAE

Option B: Larger Model (Good ROI if GPU available)
- Train Qwen 2.5-3B
- Same dataset
- Expected: 0.60-0.65 MAE

Option C: Both (Maximum Accuracy)
- 10k examples + 3B model
- Expected: 0.50-0.58 MAE
```

---

## Cost-Benefit Analysis

### Current Situation: 0.67 MAE
- **Meets target:** ✅ (<1.0)
- **Production ready:** ✅
- **Cost:** $2.58 (already spent)
- **Recommendation:** Deploy now

### Improvement Options (Ranked by ROI)

| Option | Cost | Time | Expected MAE | ROI | Do It? |
|--------|------|------|--------------|-----|--------|
| **Deploy current** | $0 | 1 day | 0.67 | ∞ | ✅ YES (now) |
| **More data (10k)** | $2.42 | 2 hours | 0.55-0.62 | High | 🤔 Maybe (later) |
| **Larger model (3B)** | $0 | 1 day | 0.60-0.65 | Medium | 🤔 Maybe (later) |
| **Both (10k + 3B)** | $2.42 | 2 days | 0.50-0.58 | High | 🤔 Maybe (later) |
| **Ensemble (3x)** | $0 | 3 days | 0.62-0.65 | Low | ⚠️ Only if critical |
| **Regularization** | $0 | 1 day | 0.67-0.75 | Negative | ❌ NO |

---

## Final Recommendations

### 1️⃣ DEPLOY NOW
Your current model (0.67 MAE) is excellent. Don't overthink it.

### 2️⃣ MORE DATA LATER (if needed)
After 2-3 months in production, if you need better accuracy:
- Score 5k more articles ($2.42)
- Retrain with 10k total
- Expected: 0.55-0.62 MAE

### 3️⃣ SKIP REGULARIZATION
Your overfitting is healthy. Don't fix what isn't broken.

### 4️⃣ ENSEMBLES ONLY IF CRITICAL
Only do this if single-model variance causes actual problems in production.

### 5️⃣ LARGER MODEL IF GPU AVAILABLE
Qwen 2.5-3B is a sweet spot if you have the GPU memory.

---

## Questions to Ask Yourself

Before pursuing improvements, answer these:

1. **Is 0.67 MAE actually a problem?**
   - If no → Deploy and monitor
   - If yes → Why? What's the impact?

2. **What is the cost of prediction errors?**
   - High cost → Consider ensembles
   - Medium cost → Current model fine
   - Low cost → Definitely deploy as-is

3. **Do you have GPU budget for 3B model?**
   - Yes + need <0.6 MAE → Try 3B
   - No → Stick with 1.5B

4. **Is $2.42 worth 10-20% accuracy gain?**
   - Yes + need <0.6 MAE → Score more data
   - No → Deploy as-is

5. **What did production monitoring reveal?**
   - Edge cases → Target those in training data
   - Systematic bias → More data won't help, fix oracle
   - Random noise → Ensembles might help

---

## Summary

**Your Questions:**

1. ✅ **More data?** YES - Do this if you need <0.6 MAE (not urgent)
2. ❌ **Regularization?** NO - Current overfitting is healthy
3. 🤔 **Ensembles?** MAYBE - Only if stability critical (not urgent)

**My Advice:**

Deploy now. Monitor for 2-3 months. Then decide based on real production data.

Your model is **excellent**. Don't let perfect be the enemy of good.

---

**Next Action:** Deploy `v2_distillation/model/` to production! 🚀
