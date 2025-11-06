# Migration Guide: Repository Restructure

The repository structure has been reorganized for clarity. If you have a trained model on your GPU machine, follow these steps to migrate.

## What Changed

**Old structure (confusing):**
```
inference/deployed/uplifting_v1/
├── model/
├── training_history.json
├── training_metadata.json
├── plots/
└── Training_Report.docx
```

**New structure (clear):**
```
filters/uplifting/v1/
├── config.yaml
├── model/                      # ← Model goes here
├── training_history.json       # ← Metadata here
└── training_metadata.json

reports/
├── uplifting_v1_training_report.docx
└── uplifting_v1_plots/
```

## Migration Steps (GPU Machine)

### Step 1: Pull Latest Changes

```bash
git pull
```

This gets the updated scripts and filter structure, but **won't move your trained model**.

### Step 2: Run Migration Script

**On Linux/Mac:**
```bash
bash migrate_structure.sh
```

**On Windows:**
```bash
migrate_structure.bat
```

The script will:
- ✓ Move `inference/deployed/uplifting_v1/model/` → `filters/uplifting/v1/model/`
- ✓ Move training metadata to filter directory
- ✓ Move plots to `reports/uplifting_v1_plots/`
- ✓ Move training report to `reports/`
- ✓ Remove old `inference/deployed/` directory

### Step 3: Verify

Check that your model is in the right place:

```bash
ls filters/uplifting/v1/model/
# Should show: config.json, model.safetensors, tokenizer files

ls filters/uplifting/v1/
# Should show: config.yaml, model/, training_*.json
```

### Step 4: Test Inference (Optional)

Make sure the model still works:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("filters/uplifting/v1/model")
tokenizer = AutoTokenizer.from_pretrained("filters/uplifting/v1/model")

# Test prediction
text = "Example article title\n\nExample content"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model(**inputs)
print(f"Scores: {outputs.logits[0]}")
```

## Future Training

**No migration needed!** New training automatically uses the correct structure:

```bash
# Training now saves directly to filter directory
python -m training.train \
    --filter filters/uplifting/v1 \
    --data-dir datasets/uplifting_ground_truth_v1_splits \
    --model-name Qwen/Qwen2.5-0.5B \
    --epochs 10

# Model saves to: filters/uplifting/v1/model/
# Plots save to: reports/uplifting_v1_plots/
```

## If You Haven't Trained Yet

**No migration needed!** Just pull and start training:

```bash
git pull
python -m training.train --filter filters/uplifting/v1 ...
```

Everything will go to the right place automatically.

## Troubleshooting

### "Filter directory not found"

You need to pull first:
```bash
git pull  # Gets the filter structure
bash migrate_structure.sh  # Then migrate
```

### "Model already exists"

The script backs up the existing model:
```bash
filters/uplifting/v1/model.backup.old/
```

You can compare them or delete the backup if the new one works.

### Manual Migration

If the script doesn't work, manually move files:

```bash
# Move model
mv inference/deployed/uplifting_v1/model filters/uplifting/v1/

# Move metadata
mv inference/deployed/uplifting_v1/training_*.json filters/uplifting/v1/

# Move plots
mkdir -p reports/uplifting_v1_plots
mv inference/deployed/uplifting_v1/plots/* reports/uplifting_v1_plots/

# Move report
mv inference/deployed/uplifting_v1/*.docx reports/uplifting_v1_training_report.docx

# Clean up
rm -rf inference/deployed
```

## Questions?

See `REPOSITORY_STRUCTURE.md` for detailed explanation of the new layout.
