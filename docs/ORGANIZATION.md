# Repository Organization

This document explains the directory structure and file organization conventions for the LLM Distillery project.

## Directory Structure

```
llm-distillery/
├── calibrations/          # Raw calibration data (JSONL)
│   ├── uplifting/        # Uplifting filter calibration data
│   ├── seece/            # SEECE filter calibration data
│   └── sustainability/   # Sustainability filter calibration data
├── config/               # Configuration files
├── datasets/             # Dataset storage
│   └── raw/             # Raw master datasets
├── docs/                 # Documentation
│   ├── analysis/        # Dataset analysis and profiling
│   ├── architecture/    # Architecture documentation
│   └── guides/          # User guides
├── experiments/          # Experiment tracking
├── ground_truth/         # Ground truth pipeline code
├── inference/            # Inference models
├── prompts/              # Prompt templates
├── reports/              # Analysis reports and comparisons
└── tests/                # Test scripts
```

## File Organization Conventions

### 1. Calibration Data (`calibrations/`)

**Purpose**: Store raw calibration data from LLM oracle comparisons (Claude vs Gemini)

**Structure**:
```
calibrations/
└── {filter_name}/
    ├── claude_labels.jsonl    # Claude's analysis results
    └── gemini_labels.jsonl    # Gemini's analysis results
```

**Guidelines**:
- One subdirectory per filter (e.g., `uplifting`, `seece`, `sustainability`)
- JSONL format with full analysis results including scores, reasoning, and metadata
- Typically 100 articles for calibration comparisons
- Do not commit large calibration files (>10MB) to git

### 2. Reports (`reports/`)

**Purpose**: Store analysis reports, calibration summaries, and LLM comparison results

**Structure**:
```
reports/
├── {filter}_calibration.md          # Main calibration report
├── {filter}_calibration_test.md     # Test calibration runs
└── llm-comparison-YYYY-MM-DD.md    # LLM comparison reports
```

**Guidelines**:
- Markdown format for readability
- Include summary statistics, recommendations, and cost analysis
- Date-stamped comparison reports (YYYY-MM-DD format)
- Include methodology, results, and conclusions
- Delete duplicate reports to avoid confusion

### 3. Dataset Analysis (`docs/analysis/`)

**Purpose**: Dataset profiling, statistics, and quality analysis

**Structure**:
```
docs/analysis/
├── dataset_summary.txt              # High-level dataset stats
├── dataset_analysis.png             # Visualization
└── dataset-profile-YYYY-MM-DD.md   # Detailed profile reports
```

**Guidelines**:
- Focus on dataset characteristics (source distribution, quality scores, word counts)
- Include visualizations where helpful
- Date-stamped profiles for tracking changes over time
- Do not store filter comparison reports here (use `reports/` instead)

### 4. Test Files (`tests/`)

**Purpose**: Test scripts for filters, calibrations, and LLM comparisons

**Structure**:
```
tests/
├── test_llm_comparison.py           # Compare LLMs across filters
├── test_seece_prefilter.py         # Test SEECE pre-filter
├── test_sustainability_prefilter.py # Test sustainability pre-filter
├── test_prefilter.py                # Generic pre-filter tests
├── test_batch_labeler.py            # Test batch labeling
├── test_compression.py              # Test compression
└── test_logging.py                  # Test logging
```

**Guidelines**:
- All test scripts in `tests/` directory
- Run tests from project root: `python tests/test_*.py` or `python -m tests.test_*`
- Tests assume they're run from root (relative paths to `datasets/`, `prompts/`, etc.)
- Output test reports to `reports/` directory (not `docs/analysis/`)

## Workflow Conventions

### Calibration Workflow

1. Generate calibration data:
   ```bash
   python -m ground_truth.batch_labeler --prompt prompts/{filter}.md --llm claude -n 100
   python -m ground_truth.batch_labeler --prompt prompts/{filter}.md --llm gemini -n 100
   ```

2. Save outputs to `calibrations/{filter}/`:
   - `claude_labels.jsonl`
   - `gemini_labels.jsonl`

3. Analyze results and create report in `reports/`:
   - `{filter}_calibration.md`

### LLM Comparison Workflow

1. Run comparison test:
   ```bash
   python tests/test_llm_comparison.py
   ```

2. Review report in `reports/`:
   - `llm-comparison-YYYY-MM-DD-HHMMSS.md`

3. Delete duplicate reports (keep latest)

### Pre-filter Testing Workflow

1. Run pre-filter test:
   ```bash
   python tests/test_seece_prefilter.py
   python tests/test_sustainability_prefilter.py
   ```

2. Review console output for pass rates and statistics

3. Adjust filter thresholds if needed

## Recent Reorganization (2025-10-28)

The following changes were made to improve repository organization:

### Changes Made

1. **Deleted duplicate LLM comparison report**
   - Removed: `docs/analysis/llm-comparison-2025-10-27-141218.md`
   - Kept: `llm-comparison-2025-10-27-142326.md` (moved to `reports/`)

2. **Moved LLM comparison reports**
   - From: `docs/analysis/`
   - To: `reports/`
   - Reason: Keep dataset analysis separate from filter comparison reports

3. **Organized test files**
   - From: Project root (7 test files)
   - To: `tests/` directory
   - Updated paths to output reports to `reports/` instead of `docs/analysis/`

4. **Created calibration directory structure**
   - Added: `calibrations/seece/`
   - Added: `calibrations/sustainability/`
   - Existing: `calibrations/uplifting/`
   - Reason: Consistent structure for all filter calibration data

### Migration Notes

- Test files now in `tests/` but still run from project root
- LLM comparison reports now go to `reports/` by default
- All three filters now have dedicated calibration directories
- Dataset analysis remains in `docs/analysis/`

## Best Practices

1. **Keep calibration data and reports separate**
   - Raw data → `calibrations/{filter}/`
   - Analysis → `reports/{filter}_calibration.md`

2. **Use consistent naming**
   - Calibration files: `{llm}_labels.jsonl`
   - Reports: `{filter}_calibration.md` or `llm-comparison-YYYY-MM-DD.md`

3. **Delete duplicates immediately**
   - Don't keep multiple versions of the same report
   - Use git history for previous versions

4. **Document methodology in reports**
   - Sample size and selection criteria
   - LLM settings (model, temperature, timeout)
   - Cost and performance metrics

5. **Run tests from project root**
   - All relative paths assume root as working directory
   - Use `python tests/test_*.py` not `cd tests && python test_*.py`

## Questions?

- Confused about where a file should go? Check this document first
- Need to add a new filter? Create `calibrations/{filter}/` and `reports/{filter}_calibration.md`
- Found inconsistencies? Update this document and clean up the files

---

Last updated: 2025-10-28
