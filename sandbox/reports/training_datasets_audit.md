# Training Datasets Audit Report
**Generated:** 2025-11-12
**Purpose:** Comprehensive audit of all training dataset directories

## Executive Summary

**CRITICAL FINDING:** You have **THREE** training dataset directories with significant inconsistencies:

1. **`datasets/training/uplifting/`** - CURRENT (matches ground truth)
2. **`datasets/training/tech_deployment/`** - OUTDATED (32.3% of current data, wrong name)
3. **`datasets/training/sustainability_tech_deployment/`** - OUTDATED (63.2% of current data)

**Recommendation:** Archive outdated directories and regenerate training datasets from current ground truth.

---

## Section 1: Directory Overview

| Directory | Files | Article Counts | Created | Source Ground Truth | Status |
|-----------|-------|----------------|---------|---------------------|--------|
| **datasets/training/uplifting/** | train.jsonl, val.jsonl, test.jsonl, split_metadata.json | Train: 6,172<br>Val: 771<br>Test: 772<br>**Total: 7,715** | Nov 5, 2025 | uplifting/labeled_articles.jsonl (7,715) | **CURRENT** ✅ |
| **datasets/training/tech_deployment/** | train.jsonl, val.jsonl | Train: 2,428<br>Val: 209<br>**Total: 2,637** | Nov 9, 2025 | sustainability_tech_deployment (RENAMED) | **OUTDATED** ❌ |
| **datasets/training/sustainability_tech_deployment/** | train.jsonl, val.jsonl, test.jsonl | Train: 4,328<br>Val: 413<br>Test: 417<br>**Total: 5,158** | Nov 9, 2025 | sustainability_tech_deployment/labeled_articles.jsonl (8,162) | **OUTDATED** ❌ |
| **archive/datasets/old_processing/splits/** | train.jsonl, val.jsonl, test.jsonl, splits_metadata.json | Train: 36,177<br>Val: 7,667<br>Test: 8,025<br>**Total: 51,869** | Oct 26, 2025 | raw/master_dataset.jsonl (OLD) | **ARCHIVED** 📦 |
| **archive/datasets/old_processing/training/** | sample_train.jsonl, sample_val.jsonl | Sample data only | Oct 29, 2025 | Unknown sample | **ARCHIVED** 📦 |

---

## Section 2: Detailed Analysis Per Directory

### 2.1 `datasets/training/uplifting/` - CURRENT ✅

**Location:** `C:\local_dev\llm-distillery\datasets\training\uplifting\`

**Files:**
```
split_metadata.json    (422 bytes,  Nov 5 18:52)
test.jsonl             (1.6 MB,     Nov 5 18:52)
train.jsonl            (11.9 MB,    Nov 5 18:52)
val.jsonl              (1.6 MB,     Nov 5 18:52)
```

**Article Counts:**
- Train: 6,172 (80%)
- Val: 771 (10%)
- Test: 772 (10%)
- **Total: 7,715**

**Ground Truth Comparison:**
- Source: `datasets/labeled/uplifting/labeled_articles.jsonl`
- Ground truth count: **7,715 articles** (modified Nov 3, 2025)
- Training dataset count: **7,715 articles** (created Nov 5, 2025)
- **Match: 100%** ✅

**Split Metadata:**
```json
{
  "filter_name": "uplifting",
  "filter_version": "1.0",
  "dimension_names": ["agency", "progress", "collective_benefit", "connection",
                      "innovation", "justice", "resilience", "wonder"],
  "total_articles": 7715,
  "train_ratio": 0.8,
  "val_ratio": 0.1,
  "test_ratio": 0.1,
  "seed": 42
}
```

**Data Format:**
```json
{
  "id": "...",
  "title": "...",
  "content": "...",
  "url": "...",
  "labels": [8, 7, 6, 8, 5, 7, 6, 8],
  "dimension_names": ["agency", "progress", ...]
}
```

**Status:** **CURRENT** - Matches ground truth perfectly. Ready for training.

**Recommendation:** ✅ **KEEP** - This is your current, up-to-date training dataset.

---

### 2.2 `datasets/training/tech_deployment/` - OUTDATED ❌

**Location:** `C:\local_dev\llm-distillery\datasets\training\tech_deployment\`

**Files:**
```
train.jsonl    (6.7 MB,  Nov 9 10:08)
val.jsonl      (611 KB,  Nov 9 10:08)
```

**Article Counts:**
- Train: 2,428
- Val: 209
- **Total: 2,637** (NO TEST SET!)

**Ground Truth Comparison:**
- Expected source: `datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl`
- Ground truth count: **8,162 articles** (modified Nov 12, 2025)
- Training dataset count: **2,637 articles** (created Nov 9, 2025)
- **Match: 32.3%** (missing 5,525 articles!) ❌

**Data Format:**
```json
{
  "prompt": "Analyze this article about technology for deployment maturity...",
  "...": "..."
}
```
*(Different format than uplifting - uses prompt/completion format, not label arrays)*

**Naming Issue:**
- Directory named `tech_deployment` but ground truth is `sustainability_tech_deployment`
- This appears to be from an EARLIER labeling run before the filter was renamed

**Timeline Context (from SESSION_STATE.md):**
- Nov 8 session: Original ~4,145 labels created
- Nov 11 session: Additional 4,017 labels added
- Total expected: ~8,162 labels
- This directory has only 2,637 (likely a SUBSET of original 4,145)

**Status:** **OUTDATED** - Contains only 32.3% of current ground truth data.

**Recommendation:** 🗑️ **ARCHIVE/DELETE** - Move to `archive/datasets/old_training/tech_deployment/` and regenerate from current ground truth.

---

### 2.3 `datasets/training/sustainability_tech_deployment/` - OUTDATED ❌

**Location:** `C:\local_dev\llm-distillery\datasets\training\sustainability_tech_deployment\`

**Files:**
```
test.jsonl     (672 KB,   Nov 9 19:17)
train.jsonl    (7.2 MB,   Nov 9 19:17)
val.jsonl      (700 KB,   Nov 9 19:17)
```

**Article Counts:**
- Train: 4,328
- Val: 413
- Test: 417
- **Total: 5,158**

**Ground Truth Comparison:**
- Source: `datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl`
- Ground truth count: **8,162 articles** (modified Nov 12, 2025)
- Training dataset count: **5,158 articles** (created Nov 9, 2025)
- **Match: 63.2%** (missing 3,004 articles!) ❌

**Data Format:**
```json
{
  "id": "ai_light_reading_267206844e00",
  "title": "Greenwatch: Is sustainability in telecom still alive...",
  "content": "In a new monthly roundup of sustainability news...",
  "url": "https://...",
  "labels": [5, 6, 4, ...],
  "dimension_names": [...]
}
```

**Timeline:**
- Training dataset created: Nov 9, 2025 19:17
- Ground truth last modified: Nov 12, 2025 11:37 (3 days AFTER training dataset)
- Ground truth had 8,162 articles on Nov 12, but training dataset only has 5,158

**What Happened:**
According to SESSION_STATE.md:
- Nov 9: Trained model with ~4,145 labels (Val MAE: 1.31)
- Nov 11: Added 4,017 NEW labels via batch labeling
- Nov 12: Ground truth consolidated to 8,162 articles
- **This training dataset (5,158) is from an INTERMEDIATE state** (probably after some batches but before consolidation)

**Status:** **OUTDATED** - Missing 3,004 articles (37% of ground truth).

**Recommendation:** 🔄 **REGENERATE** - Archive this version and create new training splits from the complete 8,162-article ground truth.

---

### 2.4 `archive/datasets/old_processing/splits/` - ARCHIVED 📦

**Location:** `C:\local_dev\llm-distillery\archive\datasets\old_processing\splits\`

**Files:**
```
splits_metadata.json   (296 bytes,   Oct 26 08:05)
test.jsonl             (31.5 MB,     Oct 26 08:05)
train.jsonl            (142.2 MB,    Oct 26 08:05)
val.jsonl              (30.2 MB,     Oct 26 08:05)
```

**Article Counts:**
- Train: 36,177
- Val: 7,667
- Test: 8,025
- **Total: 51,869**

**Purpose:**
- These are splits from the OLD raw master dataset (before filter-specific labeling)
- From Oct 26 (before ground truth labeling began)
- Stratified by `source` field, not by filter labels
- Used for general dataset exploration, NOT for filter training

**Status:** **ARCHIVED** - Historical artifact, no longer relevant.

**Recommendation:** ✅ **KEEP ARCHIVED** - Already properly archived, no action needed.

---

### 2.5 `archive/datasets/old_processing/training/` - ARCHIVED 📦

**Location:** `C:\local_dev\llm-distillery\archive\datasets\old_processing\training\`

**Files:**
```
sample_train.jsonl     (7.9 MB,  Oct 29 19:11)
sample_val.jsonl       (919 KB,  Oct 29 19:11)
```

**Article Counts:**
- Sample data only (small subset for testing)

**Purpose:**
- Early testing/development samples
- Not production training data

**Status:** **ARCHIVED** - Historical artifact.

**Recommendation:** ✅ **KEEP ARCHIVED** - Already properly archived, can be deleted if space needed.

---

## Section 3: Ground Truth vs Training Datasets

### 3.1 Uplifting Filter

| Metric | Ground Truth | Training Dataset | Match |
|--------|-------------|------------------|-------|
| **File** | `datasets/labeled/uplifting/labeled_articles.jsonl` | `datasets/training/uplifting/*` | - |
| **Modified** | Nov 3, 2025 15:18 | Nov 5, 2025 18:52 | ✅ |
| **Article Count** | 7,715 | 7,715 | ✅ 100% |
| **Format** | Oracle labels (full analysis) | Simplified (label arrays) | ✅ |
| **Status** | Current ground truth | Current training data | ✅ |

**Verdict:** ✅ **SYNCHRONIZED** - Training dataset is current and matches ground truth.

---

### 3.2 Sustainability Tech Deployment Filter

| Metric | Ground Truth | tech_deployment (OLD) | sustainability_tech_deployment (NEWER) |
|--------|-------------|----------------------|--------------------------------------|
| **File** | `datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl` | `datasets/training/tech_deployment/*` | `datasets/training/sustainability_tech_deployment/*` |
| **Modified** | Nov 12, 2025 11:37 | Nov 9, 2025 10:08 | Nov 9, 2025 19:17 |
| **Article Count** | **8,162** | 2,637 | 5,158 |
| **Coverage** | 100% | 32.3% ❌ | 63.2% ❌ |
| **Missing Articles** | 0 | 5,525 | 3,004 |
| **Status** | Current (complete) | OUTDATED | OUTDATED |

**Verdict:** ❌ **OUT OF SYNC** - Both training datasets are incomplete and outdated.

**Timeline of Events:**
1. **Nov 8-9:** Original labeling run produced ~4,145 labels
2. **Nov 9 10:08:** `tech_deployment/` created with 2,637 articles (subset)
3. **Nov 9 19:17:** `sustainability_tech_deployment/` created with 5,158 articles (larger subset)
4. **Nov 11:** Batch labeling added 4,017 NEW labels
5. **Nov 12 11:37:** Ground truth consolidated to 8,162 articles

**What Went Wrong:**
- Training datasets created BEFORE final ground truth consolidation
- Both are snapshots of intermediate states
- Neither contains the complete 8,162 articles

---

## Section 4: Recommendations

### 4.1 CURRENT - Keep As-Is

**`datasets/training/uplifting/`** ✅
- **Action:** NONE - Already current and correct
- **Rationale:** Matches ground truth (7,715 articles), created Nov 5 after ground truth finalized (Nov 3)
- **Usage:** Ready for model training

---

### 4.2 OUTDATED - Archive and Regenerate

**`datasets/training/tech_deployment/`** ❌
- **Action:** ARCHIVE → `archive/datasets/old_training/tech_deployment_2637_nov9/`
- **Rationale:**
  - Only 2,637 articles (32.3% of current 8,162)
  - Wrong directory name (should be `sustainability_tech_deployment`)
  - Uses different format (prompt/completion vs label arrays)
- **Next Step:** Delete after archiving

**`datasets/training/sustainability_tech_deployment/`** ❌
- **Action:** ARCHIVE → `archive/datasets/old_training/sustainability_tech_deployment_5158_nov9/`
- **Rationale:**
  - Only 5,158 articles (63.2% of current 8,162)
  - Created before final ground truth consolidation
  - Missing 3,004 articles from Nov 11-12 labeling
- **Next Step:** Regenerate from complete ground truth

---

### 4.3 MISSING - Create New Training Datasets

**New `datasets/training/sustainability_tech_deployment/`** 🔄
- **Action:** REGENERATE from complete ground truth (8,162 articles)
- **Source:** `datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl`
- **Expected Output:**
  - train.jsonl: ~6,530 articles (80%)
  - val.jsonl: ~816 articles (10%)
  - test.jsonl: ~816 articles (10%)
  - Total: 8,162 articles
- **Command:** See Section 5 below

---

### 4.4 CLEANUP - Archive Old Datasets

Already Archived (Good):
- ✅ `archive/datasets/old_processing/splits/` (raw master splits)
- ✅ `archive/datasets/old_processing/training/` (sample data)

Need to Archive:
- 🗑️ `datasets/training/tech_deployment/` → `archive/datasets/old_training/tech_deployment_2637_nov9/`
- 🗑️ `datasets/training/sustainability_tech_deployment/` → `archive/datasets/old_training/sustainability_tech_deployment_5158_nov9/`

---

## Section 5: Action Plan

### Step 1: Archive Outdated Training Datasets

```bash
cd C:/local_dev/llm-distillery

# Create archive directory
mkdir -p archive/datasets/old_training

# Archive tech_deployment (2,637 articles)
mv datasets/training/tech_deployment \
   archive/datasets/old_training/tech_deployment_2637_nov9

# Archive sustainability_tech_deployment (5,158 articles)
mv datasets/training/sustainability_tech_deployment \
   archive/datasets/old_training/sustainability_tech_deployment_5158_nov9

# Verify
ls -la datasets/training/  # Should only show uplifting/
ls -la archive/datasets/old_training/  # Should show both archived dirs
```

---

### Step 2: Regenerate Training Dataset (sustainability_tech_deployment)

**Option A: Using existing script** (if format matches uplifting):

```bash
cd C:/local_dev/llm-distillery

python scripts/prepare_training_data_tech_deployment.py \
  --input datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl \
  --output-dir datasets/training/sustainability_tech_deployment \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --oversample-ratio 0.2 \
  --seed 42
```

**Expected Output:**
```
Loading labels from: datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl
Loaded 8,162 labels

Splitting into train/val/test (80%/10%/10%)...
Oversampling minority classes (target ratio: 20%)...
Converting to training format (score arrays only)...

DATASET STATISTICS
==================
Train: 6,530 labels → ~8,500 examples (after oversampling)
Val:   816 labels
Test:  816 labels

Saved training data:
  Train: datasets/training/sustainability_tech_deployment/train.jsonl
  Val:   datasets/training/sustainability_tech_deployment/val.jsonl
  Test:  datasets/training/sustainability_tech_deployment/test.jsonl
```

**Option B: Using ground_truth module** (if you have a generic splitter):

```bash
python -m ground_truth.create_training_splits \
  --input datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl \
  --output-dir datasets/training/sustainability_tech_deployment \
  --filter-name sustainability_tech_deployment \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

---

### Step 3: Verify New Training Dataset

```bash
# Count articles
wc -l datasets/training/sustainability_tech_deployment/*.jsonl

# Should show:
# 6530 (or ~8500 if oversampled) train.jsonl
# 816 val.jsonl
# 816 test.jsonl
# Total: 8162 (before oversampling)

# Check format
head -1 datasets/training/sustainability_tech_deployment/train.jsonl | python -m json.tool

# Should show:
# {
#   "id": "...",
#   "title": "...",
#   "content": "...",
#   "url": "...",
#   "labels": [5, 6, 4, 7, 3, 5, 4, 6],  # 8 dimension scores
#   "dimension_names": ["deployment_maturity", "technology_performance", ...]
# }
```

---

### Step 4: Retrain Model with Complete Dataset

```bash
# Transfer to GPU machine
scp -r datasets/training/sustainability_tech_deployment/ jeroen@llm-distiller:~/llm-distillery/datasets/training/

# SSH to GPU machine
ssh jeroen@llm-distiller

# Train model
cd ~/llm-distillery
python -m filters.sustainability_tech_deployment.v1.train \
  --data datasets/training/sustainability_tech_deployment/train.jsonl \
  --val-data datasets/training/sustainability_tech_deployment/val.jsonl \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 2e-5 \
  --output-dir filters/sustainability_tech_deployment/v1

# Expected improvements with 2x more data (8,162 vs 4,145):
# - Previous Val MAE: 1.31
# - Target Val MAE: <1.20
# - Better coverage of edge cases
# - More stable predictions across dimensions
```

---

### Step 5: Document Changes

Create `datasets/training/sustainability_tech_deployment/README.md`:

```markdown
# Sustainability Tech Deployment Training Data

**Created:** 2025-11-12
**Source:** datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl
**Ground Truth Articles:** 8,162
**Filter Version:** v1

## Split Statistics

- **Train:** 6,530 articles (80%)
- **Val:** 816 articles (10%)
- **Test:** 816 articles (10%)
- **Seed:** 42 (reproducible)

## Data Format

```json
{
  "id": "article_id",
  "title": "Article title",
  "content": "Truncated content (~800 words)",
  "url": "https://...",
  "labels": [5, 6, 4, 7, 3, 5, 4, 6],
  "dimension_names": [
    "deployment_maturity",
    "technology_performance",
    "cost_trajectory",
    "scale_of_deployment",
    "market_penetration",
    "technology_readiness",
    "supply_chain_maturity",
    "proof_of_impact"
  ]
}
```

## Changelog

- **2025-11-12:** Regenerated from complete 8,162-article ground truth
- **2025-11-09 (archived):** Previous version had 5,158 articles (outdated)
- **2025-11-09 (archived):** Earlier version (tech_deployment) had 2,637 articles (outdated)
```

---

## Section 6: Summary Table

| Directory | Status | Action | Articles | Reason |
|-----------|--------|--------|----------|--------|
| `datasets/training/uplifting/` | ✅ CURRENT | KEEP | 7,715 | Matches ground truth |
| `datasets/training/tech_deployment/` | ❌ OUTDATED | ARCHIVE | 2,637 | Only 32% of current data |
| `datasets/training/sustainability_tech_deployment/` | ❌ OUTDATED | REGENERATE | 5,158 | Only 63% of current data |
| `archive/datasets/old_processing/splits/` | 📦 ARCHIVED | KEEP | 51,869 | Historical raw splits |
| `archive/datasets/old_processing/training/` | 📦 ARCHIVED | KEEP | ~1,000 | Historical samples |

---

## Section 7: Key Insights

### 7.1 Why the Confusion?

**Rapid Iteration:** The sustainability tech deployment filter went through multiple labeling runs:
1. **Nov 8-9:** Initial ~4,145 labels → Created training data (2,637 subset)
2. **Nov 9:** Intermediate state with 5,158 labels → Created training data
3. **Nov 11:** Added 4,017 new labels via batch labeling
4. **Nov 12:** Consolidated to 8,162 total labels

**Training datasets created at steps 1-2 are now outdated** because ground truth continued growing.

### 7.2 Naming Issues

- Original filter named `tech_deployment`
- Later renamed to `sustainability_tech_deployment` for clarity
- Old training directory (`tech_deployment/`) wasn't deleted when new one created
- Result: Two directories with confusingly similar names

### 7.3 Best Practice Going Forward

**Golden Rule:** Create training datasets ONLY after ground truth is finalized.

**Workflow:**
1. ✅ Complete all labeling (batch labeling, consolidation, validation)
2. ✅ Finalize ground truth file (`labeled_articles.jsonl`)
3. ✅ Create training splits from complete ground truth
4. ✅ Document what ground truth was used (article count, date)
5. ✅ If ground truth changes, ARCHIVE old training data and regenerate

**DO NOT:**
- ❌ Create training data from intermediate labeling runs
- ❌ Keep multiple versions of training data without clear naming
- ❌ Train models on incomplete ground truth

---

## Appendix A: File Locations Reference

```
llm-distillery/
├── datasets/
│   ├── labeled/                                    # GROUND TRUTH (source of truth)
│   │   ├── uplifting/
│   │   │   └── labeled_articles.jsonl             # 7,715 articles (Nov 3)
│   │   └── sustainability_tech_deployment/
│   │       └── labeled_articles.jsonl             # 8,162 articles (Nov 12)
│   │
│   └── training/                                   # TRAINING DATASETS
│       ├── uplifting/                              # ✅ CURRENT (7,715)
│       │   ├── train.jsonl
│       │   ├── val.jsonl
│       │   ├── test.jsonl
│       │   └── split_metadata.json
│       ├── tech_deployment/                        # ❌ OUTDATED (2,637)
│       │   ├── train.jsonl
│       │   └── val.jsonl
│       └── sustainability_tech_deployment/         # ❌ OUTDATED (5,158)
│           ├── train.jsonl
│           ├── val.jsonl
│           └── test.jsonl
│
└── archive/
    └── datasets/
        └── old_processing/                         # ARCHIVED (historical)
            ├── splits/                             # Raw master splits (51,869)
            └── training/                           # Sample data
```

---

## Appendix B: Commands Quick Reference

**Archive outdated datasets:**
```bash
mkdir -p archive/datasets/old_training
mv datasets/training/tech_deployment archive/datasets/old_training/tech_deployment_2637_nov9
mv datasets/training/sustainability_tech_deployment archive/datasets/old_training/sustainability_tech_deployment_5158_nov9
```

**Regenerate training data:**
```bash
python scripts/prepare_training_data_tech_deployment.py \
  --input datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl \
  --output-dir datasets/training/sustainability_tech_deployment \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 42
```

**Verify counts:**
```bash
wc -l datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl  # Should be 8162
wc -l datasets/training/sustainability_tech_deployment/*.jsonl  # Should total ~8162
```

**Transfer to GPU machine:**
```bash
scp -r datasets/training/sustainability_tech_deployment/ \
  jeroen@llm-distiller:~/llm-distillery/datasets/training/
```

---

## Appendix C: Expected Results After Cleanup

**After completing the action plan, you should have:**

```
datasets/training/
├── uplifting/                          # ✅ CURRENT (7,715 articles)
│   ├── train.jsonl (6,172)
│   ├── val.jsonl (771)
│   ├── test.jsonl (772)
│   └── split_metadata.json
│
└── sustainability_tech_deployment/     # ✅ CURRENT (8,162 articles)
    ├── train.jsonl (6,530)
    ├── val.jsonl (816)
    ├── test.jsonl (816)
    ├── split_metadata.json
    └── README.md

archive/datasets/old_training/
├── tech_deployment_2637_nov9/          # 📦 ARCHIVED (old, 2,637)
└── sustainability_tech_deployment_5158_nov9/  # 📦 ARCHIVED (old, 5,158)
```

**Both active training datasets synchronized with ground truth.**

---

## Questions?

If you encounter issues:

1. **Article counts don't match:** Check if ground truth was modified after this audit (Nov 12, 2025)
2. **Format mismatches:** Verify `prepare_training_data_tech_deployment.py` produces same format as uplifting
3. **Script not found:** Script is at `C:\local_dev\llm-distillery\scripts\prepare_training_data_tech_deployment.py`
4. **Need different split ratios:** Adjust `--train-ratio`, `--val-ratio`, `--test-ratio` (must sum to 1.0)

---

**End of Report**
