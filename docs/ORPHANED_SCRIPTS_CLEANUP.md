# Orphaned/Redundant Scripts Cleanup Analysis

**Date:** 2025-11-10
**Purpose:** Identify outdated, one-time use, or orphaned Python scripts for cleanup

---

## Scripts Recommended for Archiving/Deletion

### One-Time Tech Deployment Scripts (10 scripts)

All these scripts were created Nov 8-9 for specific one-time tasks during tech deployment dataset preparation. Their work is complete and captured in the final consolidated dataset.

#### `scripts/` directory - Archive Candidates

1. **`scripts/consolidate_tech_deployment_labels.py`** (Nov 9)
   - Purpose: One-time consolidation of 4 label batches
   - Status: COMPLETED - Result is `datasets/labeled/sustainability_tech_deployment/all_labels.jsonl`
   - Action: ARCHIVE

2. **`scripts/check_batch2_distribution.py`** (Nov 9)
   - Purpose: Check distribution of batch 2 labels
   - Status: COMPLETED - Ad-hoc analysis
   - Action: ARCHIVE

3. **`scripts/validate_oracle_labels.py`** (Nov 9)
   - Purpose: Validate oracle label quality
   - Status: COMPLETED - Validation done, documented in dataset README
   - Action: ARCHIVE

4. **`scripts/analyze_disagreements.py`** (Nov 8)
   - Purpose: Analyze oracle disagreements
   - Status: COMPLETED - Analysis in archived reports
   - Action: ARCHIVE

5. **`scripts/merge_historical_data.py`** (Nov 8)
   - Purpose: Merge historical labeled data
   - Status: COMPLETED - One-time merge
   - Action: ARCHIVE

6. **`scripts/augment_deployed_examples.py`** (Nov 9)
   - Purpose: Find more deployed tier examples
   - Status: COMPLETED - Augmentation done
   - Action: ARCHIVE

7. **`scripts/find_deployment_articles.py`** (Nov 9)
   - Purpose: Find deployment-focused articles
   - Status: COMPLETED - Articles found
   - Action: ARCHIVE

8. **`scripts/find_high_deployment_articles.py`** (Nov 9)
   - Purpose: Find high deployment score articles
   - Status: COMPLETED - Articles found
   - Action: ARCHIVE

9. **`scripts/find_tier_candidates.py`** (Nov 9)
   - Purpose: Find tier 3 (pilot) candidates
   - Status: COMPLETED - Candidates found
   - Action: ARCHIVE

10. **`scripts/prepare_training_data.py`** (Nov 9)
    - Purpose: Generic training data preparation
    - Status: SUPERSEDED by `scripts/prepare_training_data_tech_deployment.py`
    - Note: Generic version, not filter-specific
    - Action: REVIEW - Might be template for future filters

---

### Superseded Ground Truth Scripts (6 scripts)

#### `ground_truth/` directory - Archive/Delete Candidates

11. **`ground_truth/generate.py`** (Oct 25, 3.1K)
    - Purpose: Old prompt-based ground truth generation
    - Status: SUPERSEDED by `batch_labeler.py` (74K, filter package approach)
    - Reason: batch_labeler.py uses filter packages, not raw prompts
    - Action: DELETE

12. **`ground_truth/prepare_dataset.py`** (Oct 26, 11K)
    - Purpose: Prepare training datasets
    - Status: SUPERSEDED by `scripts/prepare_training_data_tech_deployment.py`
    - Reason: Filter-specific version is current approach
    - Action: REVIEW - May be generic template

13. **`ground_truth/create_splits.py`** (Oct 26, 8.9K)
    - Purpose: Create train/val/test splits
    - Status: SUPERSEDED - Functionality in `prepare_training_data_tech_deployment.py`
    - Reason: Splitting now integrated into preparation
    - Action: ARCHIVE

14. **`ground_truth/generate_top_articles_pdf.py`** (Nov 2, 7.9K)
    - Purpose: Generate PDF of top uplifting articles
    - Status: SUPERSEDED by `generate_top_articles_pdf_v2.py` (Nov 3, 11K)
    - Reason: v2 uses post-classifier for better scoring
    - Action: DELETE (v1)

15. **`ground_truth/test_compressed_quality.py`** (Oct 30, 15K)
    - Purpose: Test quality of compressed prompts
    - Status: COMPLETED - One-time test
    - Action: ARCHIVE

16. **`ground_truth/analyze_coverage.py`** (Nov 2, 7.8K)
    - Purpose: Analyze oracle coverage of corpus
    - Status: COMPLETED - Analysis for uplifting filter
    - Action: KEEP (may reuse for other filters)

---

### Potentially Redundant Training Scripts (2 scripts)

#### `training/` directory - Need Review

17. **`training/prepare_dataset.py`** (Nov 5, 8.0K)
    - Purpose: Prepare training dataset
    - Status: UNCLEAR - May duplicate `scripts/prepare_training_data_tech_deployment.py`
    - Action: REVIEW - Compare with scripts version

18. **`training/test_pipeline.py`** (Nov 6, 8.1K)
    - Purpose: Test training pipeline
    - Status: UNCLEAR - May be development/test script
    - Action: REVIEW - Check if still used

---

## Scripts to KEEP (Core Workflow)

### Active Ground Truth Tools
- ✅ `ground_truth/batch_labeler.py` - PRIMARY oracle labeling tool
- ✅ `ground_truth/calibrate_oracle.py` - Oracle model comparison
- ✅ `ground_truth/calibrate_prefilter.py` - Prefilter testing
- ✅ `ground_truth/text_cleaning.py` - Text cleaning utilities
- ✅ `ground_truth/llm_evaluators.py` - LLM evaluation utilities
- ✅ `ground_truth/samplers.py` - Sampling strategies
- ✅ `ground_truth/secrets_manager.py` - Credentials management

### Active Training Tools
- ✅ `training/train.py` - PRIMARY training script
- ✅ `training/generate_training_report.py` - Evaluation reporting
- ✅ `training/plot_learning_curves.py` - Training visualization
- ✅ `training/upload_to_huggingface.py` - Model deployment

### Active Scripts
- ✅ `scripts/prepare_training_data_tech_deployment.py` - Current prep approach

### Utility Scripts
- ✅ `ground_truth/compare_top_10.py` - Compare top articles
- ✅ `ground_truth/find_extremes.py` - Find extreme scores
- ✅ `ground_truth/generate_summary_report.py` - Summary reports
- ✅ `ground_truth/generate_top_articles_pdf_v2.py` - PDF reports (v2)
- ✅ `ground_truth/data_profiler.py` - Dataset profiling

---

## Recommended Cleanup Actions

### Immediate Deletions (2 scripts)
```bash
# Delete clearly superseded scripts
rm ground_truth/generate.py
rm ground_truth/generate_top_articles_pdf.py  # Keep v2 only
```

### Archive One-Time Scripts (13 scripts)
```bash
# Move completed one-time scripts to archive
mkdir -p archive/scripts/tech_deployment_preparation/
mv scripts/consolidate_tech_deployment_labels.py archive/scripts/tech_deployment_preparation/
mv scripts/check_batch2_distribution.py archive/scripts/tech_deployment_preparation/
mv scripts/validate_oracle_labels.py archive/scripts/tech_deployment_preparation/
mv scripts/analyze_disagreements.py archive/scripts/tech_deployment_preparation/
mv scripts/merge_historical_data.py archive/scripts/tech_deployment_preparation/
mv scripts/augment_deployed_examples.py archive/scripts/tech_deployment_preparation/
mv scripts/find_deployment_articles.py archive/scripts/tech_deployment_preparation/
mv scripts/find_high_deployment_articles.py archive/scripts/tech_deployment_preparation/
mv scripts/find_tier_candidates.py archive/scripts/tech_deployment_preparation/

mkdir -p archive/scripts/ground_truth_legacy/
mv ground_truth/create_splits.py archive/scripts/ground_truth_legacy/
mv ground_truth/test_compressed_quality.py archive/scripts/ground_truth_legacy/
```

### Review Before Archiving (4 scripts)
Need to check if these have ongoing utility:
- `scripts/prepare_training_data.py` - Generic vs filter-specific
- `ground_truth/prepare_dataset.py` - May be template
- `training/prepare_dataset.py` - Check if used
- `training/test_pipeline.py` - Check if used

---

## Summary

**Total Potentially Orphaned:** 19 scripts
**Immediate Deletions:** 2 scripts
**Archive Candidates:** 13 scripts
**Review Needed:** 4 scripts

**Cleanup Impact:**
- Removes completed one-time scripts
- Removes superseded legacy approaches
- Keeps all active workflow scripts
- Preserves scripts with potential reuse value

**Next Steps:**
1. Delete the 2 clearly superseded scripts
2. Archive the 13 completed one-time scripts
3. Review the 4 questionable scripts to determine utility
4. Update this analysis based on findings
