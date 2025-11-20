# Scripts Reorganization Plan

**Created:** 2025-11-20
**Status:** PROPOSED

---

## Problem Statement

Python scripts are scattered across multiple locations:
- `scripts/` (3 files - general utilities)
- `training/` (9 files - mix of core tools and one-off scripts)
- `ground_truth/` (8 files - mix of core modules and utilities)
- `filters/{filter}/v{n}/` (validation scripts mixed with core filter code)
- `sandbox/` (40+ files - ad-hoc analysis and testing)
- `archive/scripts/` (30+ files - old scripts)

**Issues:**
1. Hard to find relevant scripts
2. No clear separation between core tools vs utilities
3. Scripts mixed with core modules
4. Duplicate functionality across locations

---

## Proposed Structure

### Core Modules (NOT scripts - keep where they are)

```
ground_truth/
├── __init__.py
├── batch_scorer.py          # Core: Universal scoring engine
├── llm_evaluators.py        # Core: Oracle API wrappers
├── secrets_manager.py       # Core: API key management
├── text_cleaning.py         # Core: Text sanitization
└── samplers.py              # Core: Sampling strategies

training/
├── __init__.py
├── prepare_data.py          # Core: Train/val/test splitting
├── validate_training_data.py # Core: Quality validation
├── deduplicate_training_data.py # Core: Duplicate removal
└── train.py                 # Core: Model training (future)

filters/
├── base_prefilter.py        # Core: Base class for prefilters
└── {filter}/{version}/
    ├── prefilter.py         # Core: Filter implementation
    ├── postfilter.py        # Core: Tier classification
    └── config.yaml          # Core: Configuration
```

### Script Utilities (reorganize these)

```
scripts/
├── README.md
│
├── validation/              # Phase 3-5: Validation utilities
│   ├── create_validation_sample.py    # Phase 3: Oracle calibration sampling
│   ├── generate_validation_summary.py # Phase 5: Training data reports
│   └── generate_prefilter_report.py   # Phase 4: Prefilter validation (blocked)
│
├── training/                # Phase 6-7: Training utilities
│   ├── compare_training_modes.py      # Compare distillation vs instruction tuning
│   ├── generate_training_report.py    # Training results report
│   └── plot_learning_curves.py        # Visualize training progress
│
├── oracle/                  # Oracle calibration utilities
│   ├── calibrate_oracle.py           # Compare Flash/Pro/Sonnet
│   └── calibrate_prefilter.py        # Prefilter pass rate analysis
│
├── deployment/              # Phase 9: Deployment utilities
│   └── upload_to_huggingface.py      # Model deployment
│
└── dataset/                 # Dataset utilities
    └── create_random_sample.py       # General sampling

archive/
└── scripts/
    ├── obsolete/            # Old, incompatible scripts
    ├── legacy/              # Old implementations (reference)
    ├── one_off/             # One-time analysis scripts
    └── sandbox/             # Experimental/testing scripts
```

---

## Categorization

### Keep in `scripts/validation/` (Phase 3-5)

**From current `scripts/`:**
- ✅ `create_validation_sample.py` → `scripts/validation/`
- ✅ `generate_validation_summary.py` → `scripts/validation/`
- ✅ `generate_prefilter_report.py` → `scripts/validation/`

**From `ground_truth/`:**
- ✅ `calibrate_oracle.py` → `scripts/oracle/` (or keep in ground_truth?)
- ✅ `calibrate_prefilter.py` → `scripts/oracle/` (or keep in ground_truth?)
- ✅ `create_random_sample.py` → `scripts/dataset/`

### Keep in `scripts/training/` (Phase 6-7)

**From current `training/`:**
- ✅ `compare_training_modes.py` → `scripts/training/`
- ✅ `generate_training_report.py` → `scripts/training/`
- ✅ `plot_learning_curves.py` → `scripts/training/`
- ❌ `prepare_data_patch.py` → DELETE (temp patch file)
- ❌ `prepare_data_v2.py` → DELETE (temp patch file)

### Keep in `scripts/deployment/` (Phase 9)

**From current `training/`:**
- ✅ `upload_to_huggingface.py` → `scripts/deployment/`

### Archive (move to `archive/scripts/`)

**From `filters/{filter}/{version}/`:**
- ✅ `filters/investment-risk/v4/analyze_validation.py` → `archive/scripts/filter_specific/`
- ✅ `filters/investment-risk/v4/generate_synthetic_training_data.py` → `archive/scripts/one_off/`
- ✅ `filters/sustainability_tech_innovation/v*/validate_*.py` → `archive/scripts/filter_specific/`

**From `sandbox/`:**
- ✅ ALL `sandbox/*.py` → `archive/scripts/sandbox/` (already experimental)
- ✅ `sandbox/tests/` → `archive/scripts/sandbox/tests/`

**Keep in sandbox temporarily:**
- Sandbox is already separate - can clean up later

---

## Decision: What Goes Where?

### Core Modules (DO NOT MOVE)

**Criteria:**
- Imported by other modules
- Part of core functionality
- Has `__init__.py` in directory

**Examples:**
- `ground_truth/batch_scorer.py` - Core scoring engine
- `training/prepare_data.py` - Core data preparation
- `filters/base_prefilter.py` - Core base class

### Utility Scripts (MOVE TO scripts/)

**Criteria:**
- Standalone scripts (not imported)
- Generate reports or visualizations
- Optional tools for filter development

**Examples:**
- `generate_validation_summary.py` - Generates reports
- `plot_learning_curves.py` - Visualization
- `calibrate_oracle.py` - Utility for Phase 3

### Archive (MOVE TO archive/)

**Criteria:**
- One-off analysis scripts
- Filter-specific scripts
- Obsolete/superseded scripts
- Experimental/testing scripts

**Examples:**
- `analyze_uplifting_dataset.py` - Filter-specific
- `prepare_data_patch.py` - Temporary patch
- `sandbox/*` - Experimental

---

## Implementation Plan

### Phase 1: Create New Structure (5 min)

```bash
mkdir -p scripts/validation
mkdir -p scripts/training
mkdir -p scripts/oracle
mkdir -p scripts/deployment
mkdir -p scripts/dataset
mkdir -p archive/scripts/filter_specific
mkdir -p archive/scripts/one_off
```

### Phase 2: Move Scripts (10 min)

**Validation scripts:**
```bash
# Already in place - just organize subfolders
```

**Training scripts:**
```bash
mv training/compare_training_modes.py scripts/training/
mv training/generate_training_report.py scripts/training/
mv training/plot_learning_curves.py scripts/training/
```

**Deployment scripts:**
```bash
mv training/upload_to_huggingface.py scripts/deployment/
```

**Oracle scripts:**
```bash
mv ground_truth/calibrate_oracle.py scripts/oracle/
mv ground_truth/calibrate_prefilter.py scripts/oracle/
mv ground_truth/create_random_sample.py scripts/dataset/
```

**Archive filter-specific:**
```bash
mv filters/investment-risk/v4/analyze_validation.py archive/scripts/filter_specific/
mv filters/investment-risk/v4/generate_synthetic_training_data.py archive/scripts/one_off/
mv filters/sustainability_tech_innovation/v1/validate_*.py archive/scripts/filter_specific/
mv filters/sustainability_tech_innovation/v2/validate_*.py archive/scripts/filter_specific/
```

**Delete temp files:**
```bash
rm training/prepare_data_patch.py
rm training/prepare_data_v2.py
```

### Phase 3: Update Imports (15 min)

**Check which scripts are imported:**
```bash
grep -r "from ground_truth.calibrate" .
grep -r "from training.compare_training" .
grep -r "from training.plot_learning" .
```

**Update imports if needed** (unlikely - these are standalone scripts)

### Phase 4: Update Documentation (10 min)

**Update:**
- `scripts/README.md` - New structure
- `training/README.md` - Remove moved scripts
- `ground_truth/README.md` - Remove moved scripts
- `docs/agents/filter-development-guide.md` - Reference new paths

---

## Alternative: Minimal Reorganization

**If full reorganization is too much work:**

### Option A: Just Clean Up `scripts/`

1. Create subfolders in `scripts/` only
2. Move 3 current scripts to appropriate subfolders
3. Leave everything else as-is
4. Document what belongs where

**Pros:** Low effort, immediate clarity for `scripts/`
**Cons:** Doesn't solve broader organization issues

### Option B: Archive-Only

1. Move filter-specific scripts to archive
2. Move sandbox entirely to archive
3. Leave core tools where they are
4. Clean up `scripts/` with subfolders

**Pros:** Reduces clutter, medium effort
**Cons:** Still mixed organization

---

## Recommendation

**Start with Option B (Archive-Only):**

1. ✅ **Immediate:** Organize `scripts/` with subfolders (validation, training, oracle, deployment, dataset)
2. ✅ **Immediate:** Move filter-specific scripts from filters/ to archive
3. ✅ **Immediate:** Delete temp patch files (prepare_data_patch.py, prepare_data_v2.py)
4. ⏳ **Later:** Consider moving calibration tools from ground_truth/ to scripts/oracle/
5. ⏳ **Later:** Clean up sandbox/ (already separate, lower priority)

**Total effort:** 30-45 minutes
**Impact:** Clear script organization without breaking imports

---

## Success Criteria

After reorganization:

1. ✅ `scripts/` has clear subfolder structure by phase
2. ✅ No filter-specific scripts in filters/ directories (only core prefilter/postfilter code)
3. ✅ No temp/patch files in training/
4. ✅ Updated README.md documenting new structure
5. ✅ All core modules still importable (no broken imports)

---

**Last Updated:** 2025-11-20
