# Investment-Risk v4 Next Steps

**Current Status**: 5K training set scoring in progress (~4-5 hours)

---

## After Scoring Completes

### 1. Analyze Training Set Tier Distribution

**Goal**: Verify balanced coverage across risk tiers

**Commands**:
```bash
# Count scored articles
wc -l datasets/scored/investment-risk_v4_training/investment-risk/scored_batch_*.jsonl

# Analyze tier distribution
python filters/investment-risk/v4/analyze_validation.py
# (May need to adapt for training set path)
```

**Expected Distribution**:
- **RED**: 20-30% (~1,000-1,500 articles)
  - 300 synthetic crisis scenarios
  - 400 synthetic moderate risk
  - 300-800 from real data (Oct-Nov 2025 had ~2%)
- **YELLOW**: 10-20% (~500-1,000 articles)
- **GREEN**: 2-5% (~100-250 articles)
  - 300 synthetic opportunity scenarios (some may score YELLOW/BLUE)
- **BLUE**: 15-25% (~750-1,250 articles)
  - 300 synthetic educational content
  - Real framework/analysis articles
- **NOISE**: 35-45% (~1,750-2,250 articles)
  - 200 synthetic noise
  - Real noise from master_dataset

**Action Items**:
- ✅ If distribution looks good → Proceed to training
- ⚠️ If RED too low (<15%) → Consider adding more crisis scenarios
- ⚠️ If NOISE too high (>50%) → Review prefilter settings

---

### 2. Test Complete Pipeline

**Goal**: Validate prefilter → oracle → postfilter integration

**Test Cases**:

#### A. Test Prefilter (Rule-Based Blocking)
```bash
# Test articles that should be blocked
python -c "
from filters.investment_risk.v4.prefilter import prefilter
test_cases = [
    {'title': '🚀 Buy This AI Stock NOW!', 'content': 'Quick gains! Act fast!'},
    {'title': 'Top 10 Crypto Picks', 'content': 'Don't miss out on these altcoins!'},
]
for article in test_cases:
    result = prefilter(article)
    print(f'{article[\"title\"]}: {result}')
"
```

**Expected**: Both should return `False` (blocked)

#### B. Test Oracle Scoring
```bash
# Score a sample crisis article
python -m ground_truth.batch_scorer \
  --filter filters/investment-risk/v4 \
  --source filters/investment-risk/v4/historical_crisis_test_set.jsonl \
  --output-dir test_output \
  --llm gemini-flash \
  --max-batches 1

# Check scores
python -c "
import json
with open('test_output/investment-risk/scored_batch_001.jsonl') as f:
    for line in f:
        article = json.loads(line)
        analysis = article.get('investment-risk_analysis', {})
        print(f'{article[\"id\"]}: macro={analysis.get(\"macro_risk_severity\")}, credit={analysis.get(\"credit_market_stress\")}')
"
```

**Expected**: Crisis articles should score macro/credit >= 8

#### C. Test Postfilter (Tier Classification)
```bash
python -c "
from filters.investment_risk.v4.postfilter import postfilter
import json

# Load a scored article
with open('datasets/scored/investment-risk_v4_training/investment-risk/scored_batch_001.jsonl') as f:
    article = json.loads(f.readline())

# Apply postfilter
result = postfilter(article)
print(f'Tier: {result[\"tier\"]}')
print(f'Scores: macro={result[\"macro_risk_severity\"]}, credit={result[\"credit_market_stress\"]}')
"
```

**Expected**: Tier classification matches postfilter rules in config.yaml

---

### 3. Prepare Dataset for Model Training

**Goal**: Format scored data for knowledge distillation

**Steps**:

#### A. Combine All Scored Batches
```bash
# Merge all batches into single training file
cat datasets/scored/investment-risk_v4_training/investment-risk/scored_batch_*.jsonl \
  > datasets/training/investment-risk_v4_5k_scored.jsonl

# Verify count
wc -l datasets/training/investment-risk_v4_5k_scored.jsonl
# Expected: 5000 lines
```

#### B. Generate Training Statistics
```bash
python -c "
import json
from collections import Counter

tiers = []
with open('datasets/training/investment-risk_v4_5k_scored.jsonl') as f:
    for line in f:
        article = json.loads(line)
        analysis = article.get('investment-risk_analysis', {})

        # Calculate tier (postfilter logic)
        macro = analysis.get('macro_risk_severity', 0)
        credit = analysis.get('credit_market_stress', 0)
        systemic = analysis.get('systemic_risk', 0)
        evidence = analysis.get('evidence_quality', 0)

        if evidence < 4:
            tier = 'NOISE'
        elif macro >= 7 or credit >= 7 or systemic >= 8:
            tier = 'RED'
        else:
            tier = 'OTHER'

        tiers.append(tier)

counter = Counter(tiers)
total = len(tiers)
print('Tier Distribution:')
for tier, count in counter.most_common():
    print(f'{tier}: {count} ({count/total*100:.1f}%)')
"
```

#### C. Split Train/Val/Test Sets
```bash
python -c "
import json
import random

random.seed(42)

# Load all scored articles
articles = []
with open('datasets/training/investment-risk_v4_5k_scored.jsonl') as f:
    for line in f:
        articles.append(json.loads(line))

random.shuffle(articles)

# Split: 80% train, 10% val, 10% test
train_size = int(len(articles) * 0.8)
val_size = int(len(articles) * 0.1)

train_set = articles[:train_size]
val_set = articles[train_size:train_size+val_size]
test_set = articles[train_size+val_size:]

# Save splits
with open('datasets/training/investment-risk_v4_train.jsonl', 'w') as f:
    for article in train_set:
        f.write(json.dumps(article) + '\n')

with open('datasets/training/investment-risk_v4_val.jsonl', 'w') as f:
    for article in val_set:
        f.write(json.dumps(article) + '\n')

with open('datasets/training/investment-risk_v4_test.jsonl', 'w') as f:
    for article in test_set:
        f.write(json.dumps(article) + '\n')

print(f'Train: {len(train_set)} articles')
print(f'Val: {len(val_set)} articles')
print(f'Test: {len(test_set)} articles')
"
```

---

### 4. Train Student Model (Knowledge Distillation)

**Goal**: Distill Gemini Flash oracle into Qwen2.5-7B student model

**Prerequisites**:
- ✅ Scored training data (5K articles)
- ✅ GPU machine with 24GB+ VRAM (for 7B model)
- ✅ Training script configured

**Command**:
```bash
python -m training.knowledge_distillation \
  --filter investment-risk \
  --version v4 \
  --scored-data datasets/training/investment-risk_v4_train.jsonl \
  --val-data datasets/training/investment-risk_v4_val.jsonl \
  --output-dir models/investment-risk_v4 \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 2e-5 \
  --warmup-steps 100
```

**Expected Training Time**:
- 4,000 training examples × 3 epochs = 12,000 steps
- ~8-12 hours on RTX 4090
- ~16-24 hours on RTX 3090

**Monitoring**:
```bash
# Watch training logs
tail -f models/investment-risk_v4/training.log

# Check validation accuracy
grep "val_accuracy" models/investment-risk_v4/training.log
```

**Target Metrics**:
- **Tier classification accuracy**: >92%
- **Dimensional score MAE**: <0.8 (per dimension)
- **RED precision**: >85% (minimize false alarms)
- **RED recall**: >90% (don't miss crises)

---

### 5. Validate Student Model

**Goal**: Ensure student model replicates oracle behavior

**Test Sets**:
- ✅ Held-out test set (10% of 5K = 500 articles)
- ✅ Crisis validation set (11 historical scenarios)
- ✅ Oct-Nov 2025 validation set (100 articles)

**Validation Script**:
```bash
python -m training.validate_student \
  --model models/investment-risk_v4/final \
  --test-data datasets/training/investment-risk_v4_test.jsonl \
  --oracle-scored datasets/training/investment-risk_v4_test.jsonl \
  --output-report models/investment-risk_v4/validation_report.md
```

**Expected Results**:
- Student vs Oracle tier agreement: >92%
- Student scores crisis articles RED: >95%
- Student NOISE rate matches oracle: ±5%

---

### 6. Deploy to Production

**Prerequisites**:
- ✅ Student model validation passed
- ✅ Crisis detection verified
- ✅ Inference speed tested (<50ms per article)

**Deployment Steps**:

1. **Export model for inference**:
```bash
python -m training.export_model \
  --model-dir models/investment-risk_v4/final \
  --output-dir models/investment-risk_v4_production \
  --quantize int8
```

2. **Update production config**:
```yaml
# production/config.yaml
filters:
  investment-risk:
    version: v4
    model_path: models/investment-risk_v4_production
    prefilter: filters/investment-risk/v4/prefilter.py
    postfilter: filters/investment-risk/v4/postfilter.py
```

3. **A/B test** (optional):
```bash
# Run v3 and v4 in parallel, compare outputs
python -m production.ab_test \
  --filter-a investment-risk:v3 \
  --filter-b investment-risk:v4 \
  --sample-size 1000 \
  --duration 7d
```

4. **Full deployment**:
```bash
# Swap v4 into production
python -m production.deploy \
  --filter investment-risk \
  --version v4 \
  --rollout gradual  # 10% → 50% → 100% over 3 days
```

---

## Quick Reference

**Files Created**:
- ✅ Training dataset: `datasets/raw/investment-risk_v4_5k_mixed_30pct_synthetic.jsonl`
- ✅ Validation report: `filters/investment-risk/v4/validation_report.md`
- ✅ Crisis validation: `filters/investment-risk/v4/crisis_validation_report.md`
- ⏳ Scored training data: `datasets/scored/investment-risk_v4_training/` (in progress)
- ⏳ Trained model: `models/investment-risk_v4/` (pending)

**Key Decisions Made**:
1. ✅ Use 30% synthetic + 70% real for balanced training
2. ✅ Accept variable RED rate (2% calm, 50%+ crisis)
3. ✅ Validated oracle calibration on crisis scenarios
4. ✅ Target metrics: 92% tier accuracy, <0.8 MAE per dimension

**Success Criteria**:
- ✅ v4 oracle scores crisis scenarios RED (validated)
- ⏳ Student model replicates oracle behavior (>92% agreement)
- ⏳ Inference speed <50ms per article
- ⏳ Production RED rate correlates with market stress (VIX, spreads)

---

**Last Updated**: 2025-11-18
**Status**: Scoring in progress
**Next Milestone**: Analyze 5K training set distribution
