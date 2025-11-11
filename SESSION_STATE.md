# Session State - 2025-11-11

## Summary
Completed batch labeling of 3,000 new articles for sustainability tech deployment filter. Ready to retrain model with expanded dataset.

## Current Status

### Labeled Data Inventory
- **New labels (this session):** 4,017 articles
  - Location: `datasets/labeled/sustainability_tech_deployment/`
  - Files: `labeled_batch_001.jsonl` through `labeled_batch_085.jsonl`
  - Consolidated: `datasets/labeled/sustainability_tech_deployment/all_labels.jsonl` (4,017 lines)

- **Previous labels (on GPU machine):** ~4,145 articles
  - Used for original training run (Val MAE 1.31)
  - NOT present on this machine
  - Location on GPU machine: TBD (needs to be located)

- **Total tracked in state:** 8,162 article IDs
  - State file: `datasets/labeled/sustainability_tech_deployment/.labeled_ids.json`
  - This tracks ALL processed articles across both sessions

### Training Results (Previous Run)
- **Model:** Qwen 2.5-1.5B with LoRA
- **Training data:** 4,328 examples
- **Validation MAE:** 1.31 (target: <1.5)
- **Best epoch:** 3
- **Location:** `filters/sustainability_tech_deployment/v1/`
- **Plots:** `reports/sustainability_tech_deployment_v1_plots/loss_curves.png`

## Critical Issues to Address

### 1. Unicode Character Warning
**Problem:** VSCode reports "ambiguous unicode characters" in JSONL files
- Affects: Both sustainability_tech_deployment AND uplifting project
- Source: Articles contain special characters, quotes, em-dashes, non-ASCII symbols
- Impact: May cause encoding issues during training or inference

**Recommended Fix (BEFORE TRAINING):**
```python
# Create a preprocessing script to sanitize unicode
import json
import unicodedata

def clean_unicode(text):
    """Remove or normalize problematic unicode characters."""
    # Normalize to NFKC (compatibility decomposition + canonical composition)
    text = unicodedata.normalize('NFKC', text)
    # Remove zero-width characters, control characters
    text = ''.join(c for c in text if unicodedata.category(c) not in ['Cc', 'Cf', 'Cs'])
    # Replace smart quotes with regular quotes
    replacements = {
        '\u2018': "'", '\u2019': "'",  # Single quotes
        '\u201c': '"', '\u201d': '"',  # Double quotes
        '\u2013': '-', '\u2014': '-',  # En/em dashes
        '\u2026': '...',               # Ellipsis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def sanitize_jsonl(input_file, output_file):
    """Clean unicode in all text fields."""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            item = json.loads(line)
            # Clean text fields
            for field in ['title', 'content', 'description']:
                if field in item and item[field]:
                    item[field] = clean_unicode(item[field])
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

# Run on all_labels.jsonl before training
sanitize_jsonl(
    'datasets/labeled/sustainability_tech_deployment/all_labels.jsonl',
    'datasets/labeled/sustainability_tech_deployment/all_labels_clean.jsonl'
)
```

### 2. Missing Original Training Data
**Problem:** 4,145 labels from original training run are not on this machine

**Action Required:**
1. SSH to GPU machine (jeroen@llm-distiller)
2. Locate original label files:
   - Check: `datasets/labeled/sustainability_tech_deployment/`
   - Check: `ground_truth/labeled/tech_deployment/`
   - Check: Training metadata for data path used
3. Copy to this machine
4. Merge with new 4,017 labels
5. Total training dataset: ~8,162 articles (nearly 2x original)

## Next Steps

### Immediate (Before Training)
1. **Fix unicode issues:**
   - Run sanitization script on `all_labels.jsonl`
   - Create `all_labels_clean.jsonl`
   - Verify with: `grep -P '[^\x00-\x7F]' all_labels_clean.jsonl | wc -l` (should be minimal)

2. **Locate and merge original data:**
   - SSH to GPU machine
   - Find original 4,145 labels
   - Copy to this machine
   - Merge: `cat original.jsonl all_labels_clean.jsonl > combined_8162.jsonl`
   - Deduplicate by article ID if needed

### Training on GPU Machine
```bash
# Option A: Train with current 4,017 labels (if original not found)
python -m filters.sustainability_tech_deployment.v1.train \
    --data datasets/labeled/sustainability_tech_deployment/all_labels_clean.jsonl \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --output-dir filters/sustainability_tech_deployment/v1

# Option B: Train with merged 8,162 labels (recommended)
python -m filters.sustainability_tech_deployment.v1.train \
    --data datasets/labeled/sustainability_tech_deployment/combined_8162_clean.jsonl \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --output-dir filters/sustainability_tech_deployment/v1
```

### Expected Improvements
With 2x more training data:
- Better coverage of edge cases
- Reduced overfitting (train-val gap)
- Target Val MAE: <1.20 (improvement from 1.31)
- More stable per-dimension predictions

## Files to Transfer to GPU Machine

**Essential:**
```
datasets/labeled/sustainability_tech_deployment/all_labels_clean.jsonl  (after unicode fix)
filters/sustainability_tech_deployment/v1/  (filter config)
```

**Optional (for analysis):**
```
datasets/labeled/sustainability_tech_deployment/labeled_batch_*.jsonl  (individual batches)
datasets/labeled/sustainability_tech_deployment/.labeled_ids.json  (state tracking)
```

## Batch Labeling Stats
- **LLM:** Gemini Flash (cheap, fast)
- **Target:** 3,000 new labels
- **Actual:** 3,000 new labels (4,017 total in batches from both sessions)
- **Success rate:** 99.9% (only 3 failures in batch 30 - temporary API quota)
- **Time:** ~2 hours 45 minutes
- **Cost:** Minimal (Gemini Flash free tier)

## Domain Name Search (Side Quest)
**Goal:** Find short domain (2-3 letters) for positive/uplifting website

**Findings:**
- All 17,576 three-letter .nl domains are taken
- Created script: `C:/local_dev/NexusMind-Filter/scripts/check_domains.py`
- Supports multiple TLDs (.eu, .nl, .com, .org)
- Usage: `python check_domains.py eu`

**Status:** Paused - script ready to run for .eu domains

## Projects Cross-Reference

### This Project: llm-distillery
- **Focus:** Training small LLM filters via knowledge distillation
- **Current filter:** sustainability_tech_deployment (8 deployment dimensions)
- **Issue:** Unicode in labeled data

### Related Project: NexusMind-Filter (uplifting filter)
- **Location:** `C:/local_dev/NexusMind-Filter`
- **Focus:** LLM-powered semantic filtering for "uplifting" content
- **Issue:** SAME unicode problem in filtered data
- **Action needed:** Apply same unicode sanitization script there

## Technical Debt / Known Issues

1. **Unicode sanitization needed in TWO projects:**
   - llm-distillery (this project)
   - NexusMind-Filter (uplifting filter)

2. **Data location ambiguity:**
   - Original 4,145 labels whereabouts unknown
   - Need to establish single source of truth for training data

3. **Training metadata path:**
   - Training script should log exact data file path used
   - Easier to reconstruct dataset later

## Session Metadata
- **Date:** 2025-11-11
- **Branch:** main
- **Last commit:** (to be created with this file)
- **Machine:** Windows development machine (not GPU machine)
- **GPU machine:** jeroen@llm-distiller (via SSH)

## Recovery Instructions
To resume this work:

1. **Read this file first:** `SESSION_STATE.md`

2. **Check data status:**
   ```bash
   cd C:/local_dev/llm-distillery
   wc -l datasets/labeled/sustainability_tech_deployment/all_labels.jsonl
   # Should show: 4017
   ```

3. **Apply unicode fix:**
   ```bash
   python scripts/sanitize_unicode.py  # Create this script using code above
   ```

4. **Locate original data on GPU machine:**
   ```bash
   ssh jeroen@llm-distiller
   cd /path/to/llm-distillery
   find datasets -name "*.jsonl" | grep label
   ```

5. **Merge and train:**
   - Follow "Next Steps" section above

## Questions for Next Session
- Where are the original 4,145 labels?
- Do we want to fix unicode before or after merging datasets?
- Should we retrain from scratch or fine-tune existing model?
- Do we want to increase epochs (3 → 5) with more data?
