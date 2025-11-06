# Repository Structure

This document explains the LLM Distillery repository organization.

## Core Principle

**Filters are self-contained units** that include their configuration and trained models. **Reports** document how filters were trained.

## Directory Layout

```
llm-distillery/
├── filters/                    # Filter definitions + trained models
│   └── {filter_name}/
│       └── v{version}/
│           ├── config.yaml            # Filter configuration
│           ├── prefilter.yaml         # Pre-filter rules
│           ├── README.md              # Filter documentation
│           ├── model/                 # Trained model (gitignored)
│           │   ├── config.json
│           │   ├── model.safetensors
│           │   └── tokenizer files
│           ├── training_history.json  # Training metrics per epoch
│           └── training_metadata.json # Training configuration
│
├── reports/                    # Training analysis and documentation
│   ├── {filter_name}_v{version}_training_report.docx
│   └── {filter_name}_v{version}_plots/
│       ├── overall_metrics.png
│       ├── per_dimension_mae.png
│       ├── loss_curves.png
│       └── training_summary.txt
│
├── datasets/                   # Training data (gitignored)
│   └── {filter_name}_ground_truth_v{version}_splits/
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── test.jsonl
│       └── split_metadata.json
│
├── training/                   # Training scripts and tools
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── plot_learning_curves.py
│   ├── generate_training_report.py
│   ├── upload_to_huggingface.py
│   ├── README.md
│   ├── GPU_TEST_GUIDE.md
│   └── HUGGINGFACE_SETUP.md
│
├── labeling/                   # Ground truth generation
├── inference/                  # Batch inference scripts
└── evaluation/                 # Model evaluation tools
```

## Key Design Decisions

### 1. Filters Are Self-Contained

Each filter version contains everything needed to use it:
- Configuration (dimensions, weights, thresholds)
- Pre-filter rules
- Trained model weights
- Training metadata for reproducibility

**Benefits:**
- Easy to version (just copy the filter directory)
- Clear what belongs together
- Simple to deploy (just the filter folder)

### 2. Reports Are Separate

Training reports and visualizations go in `reports/` because:
- They document HOW the filter was trained
- They're for analysis, not deployment
- Multiple training runs can be compared

### 3. No "inference/deployed" Directory

The old `inference/deployed/` structure was confusing because:
- "Deployed" implies production, but files were just local
- Separated model from its filter configuration
- Created ambiguity about where things belong

**New approach:**
- Filter directory = ready to use locally
- Hugging Face = deployed to cloud
- No intermediate "deployed" concept

### 4. Model Weights Are Gitignored

Model files (~1GB) are too large for Git. Instead:
- Track training metadata (what model, what hyperparameters)
- Upload models to Hugging Face for sharing
- Recreate models by retraining with same config

## Workflows

### Training a New Filter

```bash
# 1. Prepare dataset
python -m training.prepare_dataset \
    --filter filters/uplifting/v1 \
    --dataset datasets/uplifting_ground_truth_v1/labeled_articles.jsonl \
    --output-dir datasets/uplifting_ground_truth_v1_splits

# 2. Train (saves to filter directory automatically)
python -m training.train \
    --filter filters/uplifting/v1 \
    --data-dir datasets/uplifting_ground_truth_v1_splits \
    --model-name Qwen/Qwen2.5-0.5B \
    --epochs 10

# 3. Generate visualizations (saves to reports/)
python -m training.plot_learning_curves \
    --history filters/uplifting/v1/training_history.json

# 4. Generate report (saves to reports/)
python -m training.generate_training_report \
    --filter filters/uplifting/v1 \
    --history filters/uplifting/v1/training_history.json \
    --metadata filters/uplifting/v1/training_metadata.json

# 5. Upload to Hugging Face (for deployment)
python -m training.upload_to_huggingface \
    --filter filters/uplifting/v1 \
    --repo-name your-username/uplifting-filter-v1 \
    --private
```

### Using a Trained Filter

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load from filter directory
filter_path = "filters/uplifting/v1/model"
tokenizer = AutoTokenizer.from_pretrained(filter_path)
model = AutoModelForSequenceClassification.from_pretrained(filter_path)

# Or load from Hugging Face (if deployed)
model = AutoModelForSequenceClassification.from_pretrained(
    "your-username/uplifting-filter-v1"
)
```

### Creating a New Filter Version

When improving a filter:

```bash
# 1. Copy the filter directory
cp -r filters/uplifting/v1 filters/uplifting/v2

# 2. Update config.yaml with new version and changes
# Edit filters/uplifting/v2/config.yaml

# 3. Train the new version
python -m training.train --filter filters/uplifting/v2 ...

# 4. Compare reports
diff reports/uplifting_v1_training_report.docx \
     reports/uplifting_v2_training_report.docx
```

## What Gets Committed to Git

**Yes (small files):**
- Filter configurations (config.yaml, prefilter.yaml)
- Filter documentation (README.md)
- Training metadata (training_*.json)
- Training scripts
- Documentation

**No (large files - gitignored):**
- Model weights (*.safetensors, *.bin, etc.)
- Datasets (*.jsonl)
- Training reports (*.docx, *.pdf)
- Visualizations (*.png)

## Migration Notes

If you have the old structure (`inference/deployed/`), migrate with:

```bash
# Move model and metadata to filter directory
mv inference/deployed/{filter}_v{version}/model filters/{filter}/v{version}/
mv inference/deployed/{filter}_v{version}/training_*.json filters/{filter}/v{version}/

# Move reports to reports directory
mv inference/deployed/{filter}_v{version}/*.docx reports/
mv inference/deployed/{filter}_v{version}/plots reports/{filter}_v{version}_plots/

# Remove old structure
rm -rf inference/deployed
```

The training scripts now handle the new structure automatically.
