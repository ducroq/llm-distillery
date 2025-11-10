# Data Directory Restructuring Plan

**Date**: 2025-11-09
**Status**: READY TO EXECUTE (wait for background labeling to complete)

## Goal

Consolidate all data under `datasets/` with clear structure organized by pipeline stage.

## New Structure

```
datasets/
  ├── raw/                    # Master corpus (99K articles)
  │   └── master_dataset_*.jsonl
  │
  ├── labeled/                # Oracle (Gemini Flash) labels
  │   ├── tech_deployment/
  │   ├── economic_viability/
  │   └── ...
  │
  └── training/               # Training-ready splits
      ├── tech_deployment/
      │   ├── train.jsonl
      │   ├── val.jsonl
      │   └── test.jsonl
      └── ...
```

## Operations

### 1. Move Operations

```bash
# Move labeled data
mv ground_truth/labeled/* datasets/labeled/

# Move training data (if exists)
if [ -d "training_data" ]; then
  mv training_data/* datasets/training/
fi

# Move uplifting labeled data (PRESERVE - DO NOT DELETE!)
mkdir -p datasets/labeled/uplifting
mv datasets/uplifting_ground_truth_v1/* datasets/labeled/uplifting/

# Move uplifting training splits (PRESERVE - DO NOT DELETE!)
mkdir -p datasets/training/uplifting
mv datasets/uplifting_ground_truth_v1_splits/* datasets/training/uplifting/

# Move curated file we're keeping
mkdir -p datasets/working
mv datasets/curated/unlabeled_prefilter_passed.jsonl datasets/working/
```

### 2. Delete Operations

```bash
# Delete old legacy data (SMALL files only)
rm -rf datasets/sustainability_tech_deployment/
rm -rf datasets/ai_augmented_practice/

# Delete now-empty uplifting directories (after moving contents above)
rmdir datasets/uplifting_ground_truth_v1/
rmdir datasets/uplifting_ground_truth_v1_splits/

# Delete old curated files from failed experiments
rm -f datasets/curated/deployment_focused_500.jsonl
rm -f datasets/curated/tier1_all_candidates.jsonl
rm -f datasets/curated/tier1_deployed_candidates.jsonl
rm -f datasets/curated/tier2_early_commercial_candidates.jsonl
rm -f datasets/curated/tier3_pilot_candidates.jsonl
rm -f datasets/curated/high_deployment_200.jsonl

# Remove now-empty directories
rmdir datasets/curated
rmdir ground_truth/labeled
rmdir training_data
```

### 3. Keep Separate

- `ground_truth/*.py` - Scripts (batch_labeler, calibrate_oracle, etc.)
- `calibrations/` - Oracle calibration data and reports

## Execution

**WAIT** for background process 38487b to complete before executing!

Then run:
```bash
cd C:/local_dev/llm-distillery
bash RESTRUCTURING_PLAN.sh
```

## Size Summary

**Deleting:**
- datasets/sustainability_tech_deployment/ (52K)
- datasets/ai_augmented_practice/ (24K)
- datasets/curated/* old files (~10M)
**Total**: ~10MB

**Moving (PRESERVED):**
- ground_truth/labeled/* → datasets/labeled/ (tech_deployment labels)
- training_data/* → datasets/training/ (if exists)
- datasets/uplifting_ground_truth_v1/ (41M) → datasets/labeled/uplifting/
- datasets/uplifting_ground_truth_v1_splits/ (16M) → datasets/training/uplifting/
- datasets/curated/unlabeled_prefilter_passed.jsonl → datasets/working/
