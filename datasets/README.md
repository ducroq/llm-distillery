# Datasets Directory

Organized storage for all datasets in the LLM Distillery project following ML best practices.

## Directory Structure

```
datasets/
├── raw/                    # Raw merged datasets (source of truth)
│   ├── master_dataset.jsonl             # 51,869 articles (204 MB)
│   └── master_dataset_metadata.json     # Merge statistics
│
├── processed/              # Processed/filtered datasets (future use)
│
├── splits/                 # Train/val/test splits for model training
│   ├── train.jsonl                      # 36,177 articles (69.7%)
│   ├── val.jsonl                        # 7,667 articles (14.8%)
│   ├── test.jsonl                       # 8,025 articles (15.5%)
│   └── splits_metadata.json             # Split statistics
│
├── test/                   # Test datasets for development
│   ├── test_1k.jsonl                    # 1,000 sample articles
│   ├── test_1k_metadata.json
│   └── test_articles.jsonl              # Old test file
│
└── uplifting/              # Filter-specific labeled datasets (future)
    ├── uplifting_labeled.jsonl          # Ground truth from Claude/Gemini
    ├── uplifting_train.jsonl
    ├── uplifting_val.jsonl
    └── uplifting_test.jsonl
```

## Dataset Descriptions

### Raw Datasets (`raw/`)

**Purpose**: Source of truth - raw merged data from historical database

**master_dataset.jsonl** (51,869 articles)
- Complete merged dataset from `I:/Mijn Drive/NexusMind/historical-database`
- Deduplicated by article ID (4,914 duplicates removed)
- Quality validated (required fields, non-empty content)
- Created: 2025-10-26
- Sources: 66 JSONL files from historical database

**Article Format**:
```json
{
  "id": "github_1175a04f0e39",
  "title": "Repository: zhouyuan888888/CARETrans",
  "content": "CARE Transformer: Mobile-Friendly...",
  "source": "github",
  "source_type": "api",
  "url": "https://github.com/...",
  "published_date": "2025-10-07T04:54:53",
  "collected_date": "2025-10-09T12:51:30.357274",
  "language": "en",
  "tags": ["github", "repository", "code"],
  "metadata": {...}
}
```

### Train/Val/Test Splits (`splits/`)

**Purpose**: Stratified splits for model training and evaluation

**Creation**:
```bash
python -m ground_truth.create_splits \
    --input datasets/raw/master_dataset.jsonl \
    --output-dir datasets/splits \
    --stratify-by source
```

**Split Ratios**:
- **Train** (70%): 36,177 articles - For model training
- **Val** (15%): 7,667 articles - For hyperparameter tuning
- **Test** (15%): 8,025 articles - For final evaluation

**Stratification**: By `source` field to ensure balanced representation across all 250+ sources

**Top Sources in Dataset**:
- science_arxiv_cs: 10,361 (20.0%)
- newsapi_general: 4,095 (7.9%)
- science_arxiv_math: 2,917 (5.6%)
- arxiv: 2,148 (4.1%)
- global_news_el_pais: 1,576 (3.0%)
- dutch_news_ad_algemeen: 1,416 (2.7%)
- ...and 244 more sources

### Test Datasets (`test/`)

**Purpose**: Small datasets for quick development and testing

**test_1k.jsonl** (1,000 articles)
- First 1,000 articles from master dataset
- Use for quick experiments, debugging, and prototyping

**test_articles.jsonl**
- Legacy test file from previous development

## Usage

### 1. Training a Model

```python
from pathlib import Path
import json

# Load training data
def load_dataset(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

train = load_dataset('datasets/splits/train.jsonl')
val = load_dataset('datasets/splits/val.jsonl')

# Train model
# model.fit(train, validation_data=val)
```

### 2. Evaluating a Model

```python
# Load test set
test = load_dataset('datasets/splits/test.jsonl')

# Evaluate on held-out test set
# metrics = model.evaluate(test)
```

### 3. Quick Prototyping

```bash
# Use test_1k for fast iteration
python experiment.py --data datasets/test/test_1k.jsonl
```

### 4. Creating New Splits

```bash
# Custom split ratios
python -m ground_truth.create_splits \
    --input datasets/raw/master_dataset.jsonl \
    --output-dir datasets/splits_custom \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1

# Different stratification
python -m ground_truth.create_splits \
    --input datasets/raw/master_dataset.jsonl \
    --output-dir datasets/splits \
    --stratify-by language
```

## Metadata Files

Each dataset includes a metadata JSON file documenting:
- Creation timestamp
- Source files/patterns
- Number of articles
- Statistics (duplicates, invalid entries, etc.)
- Split ratios (for train/val/test)

**Example** (`raw/master_dataset_metadata.json`):
```json
{
  "created": "2025-10-26T07:56:11.911492",
  "source_pattern": "I:/Mijn Drive/NexusMind/historical-database/current/*/*.jsonl",
  "total_articles": 51869,
  "unique_articles": 51869,
  "duplicates_removed": 4914,
  "invalid_articles": 0
}
```

## Git Tracking

**Tracked** (committed to repository):
- ✅ Metadata files (`*_metadata.json`)
- ✅ This README
- ✅ Directory structure (.gitkeep files)

**NOT Tracked** (local only, regenerated as needed):
- ❌ Large JSONL files (`*.jsonl`)
- ❌ Subdirectory contents (raw/, processed/, splits/)

**Rationale**: Large datasets (>200MB) are too big for git. Keep metadata in git to document what datasets exist, regenerate locally when needed.

## Regenerating Datasets

If you clone the repository, regenerate datasets locally:

```bash
# 1. Merge historical database
python -m ground_truth.prepare_dataset \
    --source "I:/Mijn Drive/NexusMind/historical-database/current/*/*.jsonl" \
    --output datasets/raw/master_dataset.jsonl

# 2. Create splits
python -m ground_truth.create_splits \
    --input datasets/raw/master_dataset.jsonl \
    --output-dir datasets/splits \
    --stratify-by source
```

## Best Practices

### DO:
✅ Use `raw/` for source of truth datasets
✅ Use `splits/` for train/val/test data
✅ Use `test/` for quick development experiments
✅ Stratify splits by relevant field (source, language, etc.)
✅ Document datasets with metadata files
✅ Keep splits reproducible (use fixed seed)

### DON'T:
❌ Modify files in `raw/` after creation
❌ Mix training and test data
❌ Use test set for hyperparameter tuning (use validation set)
❌ Commit large JSONL files to git
❌ Delete metadata files

## Dataset Statistics

| Dataset | Articles | Size | Purpose |
|---------|----------|------|---------|
| Master (raw) | 51,869 | 204 MB | Source of truth |
| Train | 36,177 | ~143 MB | Model training |
| Val | 7,667 | ~30 MB | Hyperparameter tuning |
| Test | 8,025 | ~31 MB | Final evaluation |
| Test 1K | 1,000 | 2.4 MB | Quick experiments |

**Total Storage**: ~410 MB (excluding processed datasets)

## Future Additions

Planned dataset types:

- **Labeled Datasets**: Ground truth from Claude/Gemini
  `uplifting/uplifting_50k_labeled.jsonl`

- **Filter-Specific Splits**: Train/val/test for each filter
  `uplifting/uplifting_train.jsonl`

- **Processed Datasets**: Cleaned/filtered/augmented data
  `processed/master_filtered.jsonl`

- **Sampled Datasets**: Stratified samples for specific purposes
  `processed/arxiv_only.jsonl`

## See Also

- [Data Preparation Guide](../docs/guides/data-preparation.md) - How to merge and prepare datasets
- [Architecture Overview](../docs/architecture/overview.md) - Overall system design
- [Calibration Guide](../docs/guides/calibration.md) - Comparing LLM oracles
