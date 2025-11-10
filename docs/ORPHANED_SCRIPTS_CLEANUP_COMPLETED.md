# Script Cleanup Completed

**Date:** 2025-11-10
**Status:** COMPLETED

---

## Summary

Successfully cleaned up 17 orphaned/outdated Python scripts from the project:
- **Deleted:** 2 clearly superseded scripts
- **Archived:** 15 completed one-time or legacy scripts

All archived scripts are preserved in `archive/scripts/` for historical reference.

---

## Files Deleted (2)

1. **`ground_truth/generate.py`**
   - Superseded by: `batch_labeler.py` (74K, filter package approach)
   - Reason: Old prompt-based ground truth generation

2. **`ground_truth/generate_top_articles_pdf.py`**
   - Superseded by: `generate_top_articles_pdf_v2.py` (uses post-classifier)
   - Reason: v1 deprecated

---

## Files Archived (15)

### Tech Deployment Preparation (9 files → `archive/scripts/tech_deployment_preparation/`)

All one-time scripts created Nov 8-9 for tech deployment dataset preparation:

1. `scripts/consolidate_tech_deployment_labels.py` - Consolidated 4 label batches
2. `scripts/check_batch2_distribution.py` - Batch 2 distribution analysis
3. `scripts/validate_oracle_labels.py` - Oracle label quality validation
4. `scripts/analyze_disagreements.py` - Oracle disagreement analysis
5. `scripts/merge_historical_data.py` - Historical labeled data merge
6. `scripts/augment_deployed_examples.py` - Found more deployed tier examples
7. `scripts/find_deployment_articles.py` - Found deployment-focused articles
8. `scripts/find_high_deployment_articles.py` - Found high deployment score articles
9. `scripts/find_tier_candidates.py` - Found tier 3 (pilot) candidates

### Ground Truth Legacy (3 files → `archive/scripts/ground_truth_legacy/`)

Superseded by current workflow:

10. `ground_truth/create_splits.py` - Superseded by integrated splitting in prepare_training_data_*.py
11. `ground_truth/test_compressed_quality.py` - One-time compression test
12. `ground_truth/prepare_dataset.py` - Superseded by filter-specific preparation scripts

### Training Legacy (3 files → `archive/scripts/training_legacy/`)

Old Unsloth/text generation approach scripts:

13. `scripts/prepare_training_data.py` - Generic Unsloth approach, superseded by regression training
14. `training/prepare_dataset.py` - Generic version, superseded by filter-specific scripts
15. `training/test_pipeline.py` - Dev/test tool, not currently used

---

## Active Scripts Kept

### Core Workflow Scripts
- ✅ `ground_truth/batch_labeler.py` - PRIMARY oracle labeling tool
- ✅ `scripts/prepare_training_data_tech_deployment.py` - Current preparation approach
- ✅ `training/train.py` - PRIMARY training script

### Utility Scripts
- ✅ `ground_truth/calibrate_oracle.py` - Oracle model comparison
- ✅ `ground_truth/calibrate_prefilter.py` - Prefilter testing
- ✅ `training/generate_training_report.py` - Evaluation reporting
- ✅ `training/plot_learning_curves.py` - Training visualization
- ✅ `training/upload_to_huggingface.py` - Model deployment

---

## Cleanup Impact

**Before:** 50+ scripts, unclear which were active
**After:** ~35 active scripts, 17 archived for reference

**Benefits:**
- Removed completed one-time scripts from active codebase
- Removed superseded legacy approaches (Unsloth/text generation)
- Preserved all scripts in archive/ for historical reference
- Clearer project structure focusing on current regression training workflow

---

**Cleanup Execution Date:** 2025-11-10
**Related:** See `ORPHANED_DOCS_CLEANUP.md` for documentation cleanup
