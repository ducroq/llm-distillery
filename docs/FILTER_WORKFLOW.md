# Filter Development Workflow

**Purpose**: Step-by-step workflow to create a new filter with balanced, high-quality training data

**Lesson Learned**: Don't randomly sample from imbalanced sources and hope for the best. Plan data sourcing FIRST.

---

## Overview

Each filter needs:
1. **Filter definition** (config, prompt, prefilter)
2. **Balanced dataset** (~3,000-5,000 examples covering all tiers/dimensions)
3. **Train/val/test splits** (80/10/10 with stratification)
4. **Final location**: `datasets/{filter_name}/`

---

## Step 1: Filter Definition

**Location**: `filters/{filter_name}/v1/`

**Required files**:
- `config.yaml` - Dimensions, weights, thresholds
- `prompt.md` - Oracle prompt template
- `prefilter.py` - Fast pre-screening (optional but recommended)
- `README.md` - Scoring guide with examples for each tier

**Action**: Create filter package using existing filters as template

**Quality gate**: README must have clear examples for EACH tier (not just high tier)

---

## Step 2: Data Sourcing Strategy (CRITICAL - Don't Skip!)

**Before labeling anything, answer**:

### What are the natural tier distributions for this filter?

**Example: Tech Deployment**
- Tier 1 (8-10 deployed): ~1-2% of tech news (very rare)
- Tier 2 (6-7.9 commercial): ~5-7% of tech news
- Tier 3 (4-5.9 pilot): ~10-15% of tech news
- Tier 4 (<4 vaporware): ~75-85% of tech news

**Implication**: Random sampling from tech news will NOT give balanced data.

### Where can we find examples of each tier?

**Tier 1 (deployed/proven)**:
- Industry reports (IEA, IRENA for energy)
- Case study databases (IEEE, ACM)
- Company annual reports (Tesla, Vestas, etc.)
- Government statistics (EIA, Eurostat)

**Tier 2 (commercial proven)**:
- Trade publications (PV Magazine, Wind Power Monthly)
- Business news (Bloomberg, Reuters - filter for revenue/deployment)
- Company press releases (pilot → commercial announcements)

**Tier 3 (pilot/demonstration)**:
- University tech transfer news
- Startup announcements
- Grant/funding announcements

**Tier 4 (vaporware/lab)**:
- General tech news (TechCrunch, Hacker News)
- Research publication announcements
- Startup concepts

### Sourcing Plan Template

```yaml
target_distribution:
  tier_1: 20%  # 600 examples
  tier_2: 25%  # 750 examples
  tier_3: 30%  # 900 examples
  tier_4: 25%  # 750 examples
  total: 3000

sourcing_strategy:
  tier_1_sources:
    - IEA annual reports (scrape deployment stats)
    - Company case studies (filtered for "operating since")
    - Government energy databases
    - Target: 800 candidates → label → select best 600

  tier_2_sources:
    - Trade publications (filter for "commercial")
    - Business news (filter for revenue/sales mentions)
    - Target: 1000 candidates → label → select best 750

  tier_3_sources:
    - Grant databases (DoE, EU Horizon)
    - Pilot project announcements
    - Target: 1200 candidates → label → select best 900

  tier_4_sources:
    - General tech news (random sample)
    - Research announcements
    - Target: 1000 candidates → label → select best 750
```

**Quality gate**:
- ✅ Have sourcing plan for ALL tiers
- ✅ Know where high-tier examples exist
- ✅ Target total: 3,000-5,000 examples
- ❌ DO NOT proceed with only one source (e.g., just tech news)

---

## Step 3: Data Collection

**Goal**: Gather ~4,000-6,000 candidate articles (before labeling)

### Option A: Multi-Source Collection

**Process**:
1. Identify sources per tier (from sourcing plan)
2. Scrape/download articles from each source
3. Store in `datasets/raw/{filter_name}_tier{X}_candidates.jsonl`
4. Combine into candidate pool

**Example**:
```bash
datasets/raw/
├── tech_deployment_tier1_candidates.jsonl  # 800 articles from IEA, case studies
├── tech_deployment_tier2_candidates.jsonl  # 1000 articles from trade pubs
├── tech_deployment_tier3_candidates.jsonl  # 1200 articles from grants
├── tech_deployment_tier4_candidates.jsonl  # 1000 articles from tech news
└── tech_deployment_all_candidates.jsonl    # 4000 combined
```

### Option B: Smart Filtering from Existing Corpus

If you only have one source (e.g., tech news RSS), use intelligent filtering:

**Process**:
1. Create tier-specific keyword/regex filters
2. Scan corpus for candidates matching each tier
3. Manually review samples to verify quality
4. Gather 2x target for each tier (will shrink during labeling)

**Example filters**:
```python
tier_1_signals = [
    r'\d+\s*gigawatts?.*deployed',
    r'operating since \d{4}',
    r'market share of \d+%',
    r'\d+\s*million units sold'
]

tier_2_signals = [
    r'commercial.*revenue',
    r'first commercial deployment',
    r'sales.*doubled'
]
```

### Option C: Synthetic Augmentation (if needed)

If tier 1/2 examples are scarce:
1. Find 50-100 real examples
2. Use Gemini to paraphrase/augment
3. Label augmented versions
4. Mix with real examples

**Quality gate**:
- ✅ Have 1.5-2x target examples per tier before labeling
- ✅ Verified examples match tier definitions
- ❌ DO NOT label random samples hoping for balance

---

## Step 4: Oracle Labeling

**Process**:
```bash
# Label candidates for each tier separately
python -m ground_truth.batch_labeler \
  --filter filters/{filter_name}/v1 \
  --source datasets/raw/{filter_name}_tier1_candidates.jsonl \
  --llm gemini-flash \
  --target-count 800 \
  --output-dir ground_truth/temp/{filter_name}/tier1

# Repeat for tier 2, 3, 4...
```

**Cost**: ~$0.001 per article × 4,000 = ~$4

**Time**: ~3 hours for 4,000 articles

**Quality gate**:
- ✅ Label all tier candidates
- ✅ Check actual distribution matches expectations
- ✅ If tier severely under-represented, go back to Step 3

---

## Step 5: Dataset Curation

**Goal**: Select balanced subset from labeled candidates

**Process**:
```python
# Load all labeled candidates
tier_1_labeled = load_labels('ground_truth/temp/{filter_name}/tier1/*.jsonl')
tier_2_labeled = load_labels('ground_truth/temp/{filter_name}/tier2/*.jsonl')
# ...

# Filter by actual oracle scores (may not match tier expectations)
actual_tier_1 = [ex for ex in all_labeled if ex['overall_score'] >= 8.0]
actual_tier_2 = [ex for ex in all_labeled if 6.0 <= ex['overall_score'] < 8.0]
# ...

# Sample to target distribution
curated_dataset = (
    sample(actual_tier_1, 600) +
    sample(actual_tier_2, 750) +
    sample(actual_tier_3, 900) +
    sample(actual_tier_4, 750)
)

# Shuffle and save
shuffle(curated_dataset)
save('datasets/{filter_name}/raw_labels.jsonl', curated_dataset)
```

**Quality gate**:
- ✅ Final dataset: 3,000-5,000 examples
- ✅ Distribution: 15-25% per tier (no tier <10% or >40%)
- ✅ All dimensions have ≥100 examples at each score range (low/mid/high)

---

## Step 6: Train/Val/Test Split

**Process**:
```python
from sklearn.model_selection import train_test_split

# First split: 80% train+val, 20% test
train_val, test = train_test_split(
    curated_dataset,
    test_size=0.20,
    stratify=[ex['tier'] for ex in curated_dataset],
    random_state=42
)

# Second split: 80% train, 20% val (of train+val)
train, val = train_test_split(
    train_val,
    test_size=0.20,  # 20% of 80% = 16% of total
    stratify=[ex['tier'] for ex in train_val],
    random_state=42
)

# Save splits
save('datasets/{filter_name}/train.jsonl', train)  # ~2400 examples (80%)
save('datasets/{filter_name}/val.jsonl', val)      # ~600 examples (16%)
save('datasets/{filter_name}/test.jsonl', test)    # ~600 examples (20%)
```

**Note**: Test set is NEVER used during training - only for final evaluation.

**Quality gate**:
- ✅ All splits have all tiers represented
- ✅ Splits are stratified (proportions match across splits)
- ✅ Files saved in correct location: `datasets/{filter_name}/`

---

## Step 7: Validation Analysis

**Before training, verify dataset quality**:

```bash
python scripts/analyze_dataset.py \
  --dataset datasets/{filter_name}/raw_labels.jsonl \
  --output reports/{filter_name}_dataset_analysis.md
```

**Check**:
- Tier distribution (target: 15-25% each)
- Dimension score distributions (all dimensions have full range 1-10)
- Per-dimension coverage (≥100 examples per score range)
- No missing fields

**Quality gate**:
- ✅ Distribution acceptable (no tier <10%)
- ✅ All dimensions have high-score examples (≥8)
- ✅ No catastrophic gaps

---

## Step 8: Training Data Preparation

**Only after dataset is validated**:

```bash
python scripts/prepare_training_data.py \
  --input datasets/{filter_name}/raw_labels.jsonl \
  --output-dir training_data/{filter_name}/v1 \
  --format instruction  # Creates prompt/completion pairs
```

**Output**:
```
training_data/{filter_name}/v1/
├── train.jsonl  # Ready for Unsloth/HuggingFace
└── val.jsonl
```

---

## Step 9: Model Training

**Now safe to train**:

```bash
python scripts/train_model.py \
  --train-file training_data/{filter_name}/v1/train.jsonl \
  --val-file training_data/{filter_name}/v1/val.jsonl \
  --output-dir models/{filter_name}_v1 \
  --epochs 3
```

---

## Step 10: Evaluation

**After training, evaluate on held-out test set**:

```bash
python scripts/evaluate_model.py \
  --model models/{filter_name}_v1/final \
  --test-file datasets/{filter_name}/test.jsonl \
  --output reports/{filter_name}_evaluation.md
```

**Metrics**:
- Per-tier accuracy, precision, recall
- MAE per dimension
- Confusion matrix
- Failure case analysis

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Random sampling from imbalanced source
**Problem**: Tech deployment filter - randomly sampled tech news → 82% vaporware
**Solution**: Plan sourcing strategy FIRST (Step 2)

### ❌ Mistake 2: Accepting imbalanced data "it's reality"
**Problem**: Model will just learn to predict majority class
**Solution**: Training data ≠ natural distribution. Actively curate balance.

### ❌ Mistake 3: Giving up when stratified sampling finds few examples
**Problem**: "Only 3 deployed examples exist" → accepted imbalance
**Solution**: Look in DIFFERENT sources. Deployed tech exists, just not in general tech news.

### ❌ Mistake 4: Creating intermediate folders everywhere
**Problem**: `ground_truth/labeled/`, `training_data/`, `datasets/curated/` scattered
**Solution**: Final home is `datasets/{filter_name}/`. Temp folders OK but clean up.

### ❌ Mistake 5: Skipping validation before training
**Problem**: Train model, then discover dataset has gaps
**Solution**: Always run Step 7 validation analysis first

---

## Success Criteria (Before Training)

**Dataset must have**:
- ✅ 3,000-5,000 total examples
- ✅ 15-25% per tier (no tier <10% or >40%)
- ✅ All dimensions have examples at 1-3, 4-7, 8-10 score ranges
- ✅ ≥100 examples per dimension at high range (8-10)
- ✅ Clean location: `datasets/{filter_name}/train|val|test.jsonl`
- ✅ Validation report confirms quality

**If these aren't met, GO BACK to Step 3 (data collection), don't proceed.**

---

## Template Checklist (Copy for Each Filter)

```markdown
## {Filter Name} Dataset Development

### Step 1: Filter Definition
- [ ] Created filter package in `filters/{filter_name}/v1/`
- [ ] config.yaml complete
- [ ] prompt.md template ready
- [ ] README.md has tier examples

### Step 2: Data Sourcing Strategy
- [ ] Analyzed natural tier distribution for this domain
- [ ] Identified sources for tier 1 (high-score examples)
- [ ] Identified sources for tier 2
- [ ] Identified sources for tier 3
- [ ] Identified sources for tier 4
- [ ] Written sourcing plan with targets

### Step 3: Data Collection
- [ ] Collected tier 1 candidates: ____ / target
- [ ] Collected tier 2 candidates: ____ / target
- [ ] Collected tier 3 candidates: ____ / target
- [ ] Collected tier 4 candidates: ____ / target
- [ ] Total candidates: ____ (target: 4,000-6,000)

### Step 4: Oracle Labeling
- [ ] Labeled tier 1 candidates: ____ examples
- [ ] Labeled tier 2 candidates: ____ examples
- [ ] Labeled tier 3 candidates: ____ examples
- [ ] Labeled tier 4 candidates: ____ examples
- [ ] Total labeled: ____ (target: 4,000+)

### Step 5: Dataset Curation
- [ ] Filtered by actual scores
- [ ] Sampled to balanced distribution
- [ ] Saved to `datasets/{filter_name}/raw_labels.jsonl`
- [ ] Final count: ____

### Step 6: Splits
- [ ] Created train split: ____ examples
- [ ] Created val split: ____ examples
- [ ] Created test split: ____ examples
- [ ] Verified stratification

### Step 7: Validation
- [ ] Ran dataset analysis
- [ ] Tier distribution acceptable: ____
- [ ] Dimension coverage complete: ____
- [ ] Quality report: `reports/{filter_name}_dataset_analysis.md`

### Step 8: Training Prep
- [ ] Converted to training format
- [ ] Located in `training_data/{filter_name}/v1/`

### Step 9: Training
- [ ] Model trained
- [ ] Checkpoints saved

### Step 10: Evaluation
- [ ] Evaluated on test set
- [ ] Evaluation report created
- [ ] Model meets success criteria
```

---

**Author**: Claude (AI Assistant)
**Date**: 2025-11-09
**Lesson**: Plan data sourcing BEFORE labeling. Training data needs balance, not natural distribution.
