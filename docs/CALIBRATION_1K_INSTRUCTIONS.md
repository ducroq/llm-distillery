# Oracle Calibration - 1,000 Article Sample

## Overview

This calibration validates the oracle (Gemini Flash + prompt) on 1,000 articles before generating the full 10K training dataset.

**Purpose**: Ensure the oracle is:
- Scoring articles correctly
- Maintaining dimension independence
- Producing consistent results
- Distributing scores appropriately

---

## Step 1: Run Calibration on GPU Machine

### Prerequisites

1. **Master dataset** synced to GPU:
   ```
   datasets/raw/master_dataset_20251009_20251124.jsonl
   ```

2. **Environment variables** configured:
   - `GOOGLE_API_KEY` for Gemini API

3. **Python dependencies** installed:
   ```bash
   pip install -r requirements.txt
   ```

### Run Calibration

```bash
# SSH to GPU machine
ssh jeroen@llm-distiller

# Navigate to project
cd /home/jeroen/llm-distillery

# Run calibration script
python scripts/calibrate_oracle_1k.py
```

### What It Does

1. **Loads** master dataset (178K articles)
2. **Applies** prefilter (keyword-based)
3. **Samples** 1,000 random articles + 20 duplicates
4. **Scores** all 1,020 articles with oracle
5. **Saves** results to `datasets/calibration/calibration_1k_YYYYMMDD_HHMMSS/`

### Expected Runtime

- **~15-20 minutes** (1,020 API calls to Gemini)

### Output Files

```
datasets/calibration/calibration_1k_20251124_140000/
├── articles_sampled.jsonl       # 1,020 articles before scoring
├── articles_scored.jsonl        # 1,020 articles with oracle scores
├── calibration_stats.json       # Basic statistics
└── calibration_log.txt          # Execution log
```

---

## Step 2: Sync Results Back to Local

After calibration completes on GPU:

1. Open **FreeFileSync** (`llm-distillery-sync.ffs_gui`)
2. Click **"Compare"**
3. Verify calibration folder appears
4. Click **"Synchronize"**

Results will be synced to:
```
C:\local_dev\llm-distillery\datasets\calibration\calibration_1k_YYYYMMDD_HHMMSS\
```

---

## Step 3: Run Analysis (Local Machine)

```bash
# On local machine
cd C:\local_dev\llm-distillery

# Run analysis script
python scripts/analyze_calibration_1k.py datasets/calibration/calibration_1k_YYYYMMDD_HHMMSS
```

### What It Does

1. **Loads** scored articles
2. **Calculates** dimension statistics (mean, std, min, max)
3. **Analyzes** score distribution (high/medium/low)
4. **Checks** dimension independence (correlations)
5. **Tests** consistency (duplicate article scoring)
6. **Generates** comprehensive report

### Output Files

```
datasets/calibration/calibration_1k_YYYYMMDD_HHMMSS/
├── calibration_analysis_report.md   # Markdown report (read this!)
├── calibration_analysis.json        # Analysis statistics
└── correlation_matrix.json          # Dimension correlations
```

---

## Step 4: Review Results

### Read the Report

Open `calibration_analysis_report.md` and check:

#### 1. Score Distribution
```
| Range | Count | Percentage |
|-------|-------|------------|
| high (7-10) | 50-150 | 5-15% |       ✅ Good articles
| medium-high (5-7) | 150-250 | 15-25% |  ✅ Decent articles
| medium (3-5) | 300-400 | 30-40% |      ✅ Moderate relevance
| low (1-3) | 200-300 | 20-30% |        ⚠️  False positives
```

**Good**: Low FP rate (20-30%), some high scorers (5-15%)
**Bad**: Very high FP rate (>40%) or no high scorers (<5%)

#### 2. Dimension Statistics

Check that scores are well-distributed:
- **Mean**: Should be 3-5 range (not clustered at extremes)
- **Std Dev**: Should be 1.5-2.5 (good spread)
- **Min/Max**: Should use full 0-10 range

#### 3. Dimension Independence

**Good**: No high correlations (r > 0.85)
**Acceptable**: Some moderate correlations (0.70-0.85) if they make sense (e.g., TRL <-> Economics)
**Bad**: Many high correlations (dimensions not independent)

#### 4. Scoring Consistency

**Good**: Avg difference ≤ 1.0 on duplicate articles
**Bad**: Avg difference > 1.5 (inconsistent scoring)

---

## Step 5: Decision

### If Approved ✅

Report says: **"APPROVED FOR 10K TRAINING"**

**Next steps**:
1. Generate 10K training dataset
2. Train student model
3. Deploy to production

### If Issues Found ⚠️

Report says: **"REVIEW REQUIRED"**

**Possible issues**:
- High FP rate (>40%) → Tighten prefilter
- High correlations → Adjust prompt to emphasize independence
- Poor consistency → Review prompt clarity

**Next steps**:
1. Manual validation (review sample articles)
2. Adjust prompt or prefilter if needed
3. Re-run calibration

---

## Step 6: Manual Validation (Optional)

Spot-check 50 articles across score ranges:

```bash
# Extract sample articles for manual review
python scripts/extract_calibration_samples.py datasets/calibration/calibration_1k_YYYYMMDD_HHMMSS \
    --high 10 \
    --medium 20 \
    --low 20
```

Review:
- Do high-scoring articles deserve 7-10?
- Do low-scoring articles deserve 1-3?
- Are dimension scores accurate and independent?

---

## Troubleshooting

### "Master dataset not found"

Ensure the dataset is synced to GPU machine:
```bash
ls datasets/raw/master_dataset_20251009_20251124.jsonl
```

If missing, run FreeFileSync to sync it.

### "GOOGLE_API_KEY not set"

Configure API key on GPU machine:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or add to `~/.bashrc` for persistence.

### "Too few articles passed prefilter"

Check prefilter pass rate in log:
```bash
cat datasets/calibration/calibration_1k_YYYYMMDD_HHMMSS/calibration_log.txt | grep "Prefilter:"
```

If very low (<30%), check prefilter keywords.

### "Scoring errors"

Check log for API errors:
```bash
cat datasets/calibration/calibration_1k_YYYYMMDD_HHMMSS/calibration_log.txt | grep "ERROR"
```

Common causes:
- API rate limits (wait and retry)
- API key issues
- Network connectivity

---

## Cost Estimate

- **1,020 API calls** to Gemini Flash
- **Estimated cost**: ~$7.50 (at $0.0075/article)
- **Runtime**: ~15-20 minutes

---

## Summary

1. ✅ Run calibration on GPU (`calibrate_oracle_1k.py`)
2. ✅ Sync results back to local (FreeFileSync)
3. ✅ Run analysis on local (`analyze_calibration_1k.py`)
4. ✅ Review report (`calibration_analysis_report.md`)
5. ✅ Decision: Proceed to 10K or adjust and re-calibrate

---

## Questions?

- Script issues: Check `calibration_log.txt`
- Analysis questions: Review `calibration_analysis_report.md`
- Data issues: Check `calibration_stats.json`
