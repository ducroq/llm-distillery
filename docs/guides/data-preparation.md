# Data Preparation Guide

## Overview

The data preparation pipeline merges historical article data from multiple sources into a unified master dataset ready for ground truth generation. It provides deduplication, quality validation, and incremental update support.

## Quick Start

### 1. Merge Historical Database

```bash
cd C:\local_dev\llm-distillery

# Full merge - creates new master dataset
python -m ground_truth.prepare_dataset \
    --source "I:/Mijn Drive/NexusMind/historical-database/current/*/*.jsonl" \
    --output datasets/master_dataset.jsonl
```

**Expected Output**:
```
Found 66 source files
Output: datasets\master_dataset.jsonl
Mode: Full (overwrite)
Processing: content_items_20251009_130449.jsonl
...
  Written: 10,000 articles
  Written: 20,000 articles
  ...

Dataset Preparation Statistics
Total articles read:     56,783
Valid articles written:  51,869
Duplicates skipped:      4,914
```

### 2. Incremental Updates

When new articles are collected, use incremental mode to append only new content:

```bash
# Incremental update - appends only new articles
python -m ground_truth.prepare_dataset \
    --source "I:/Mijn Drive/NexusMind/historical-database/current/*/*.jsonl" \
    --output datasets/master_dataset.jsonl \
    --incremental
```

**How it works**:
1. Loads all existing article IDs from master dataset (51,869 IDs)
2. Processes source files and skips articles already in master
3. Appends only new articles

### 3. Create Test Dataset

```bash
# Create smaller dataset for testing (1,000 articles)
python -m ground_truth.prepare_dataset \
    --source "I:/Mijn Drive/NexusMind/historical-database/current/*/*.jsonl" \
    --output datasets/test_1k.jsonl \
    --max-articles 1000
```

## Features

### Deduplication

Articles are deduplicated by their `id` field using a Set for O(1) lookup:

```python
# Tracking seen IDs
self.seen_ids: Set[str] = set()

# Skip duplicates
if article_id in self.seen_ids:
    self.stats['duplicates'] += 1
    continue
```

**Statistics**:
- **56,783 total articles** read from source files
- **51,869 unique articles** written to master dataset
- **4,914 duplicates** automatically skipped

### Quality Validation

Each article is validated for required fields and non-empty content:

```python
def is_valid_article(self, article: Dict) -> bool:
    # Required fields
    required = ['id', 'title', 'content']
    if not all(field in article for field in required):
        return False

    # Must have non-empty content
    if not article.get('content', '').strip():
        return False

    # Must have non-empty title
    if not article.get('title', '').strip():
        return False

    return True
```

**Current Statistics**:
- **0 invalid articles** (all articles had proper structure)
- **0 JSON decode errors**

### Incremental Updates

The incremental mode enables efficient updates when new data is collected:

**Process**:
1. Load existing article IDs from master dataset
2. Process source files
3. Skip articles already in the set
4. Append only new articles

**Example**:
```bash
# First run: Creates master dataset (51,869 articles)
python -m ground_truth.prepare_dataset \
    --source "..." \
    --output datasets/master_dataset.jsonl

# Second run: Loads 51,869 existing IDs, skips all duplicates
python -m ground_truth.prepare_dataset \
    --source "..." \
    --output datasets/master_dataset.jsonl \
    --incremental
```

**Output**:
```
Loading existing IDs from datasets\master_dataset.jsonl
Loaded 51,869 existing article IDs
...
Total articles read:     56,783
Valid articles written:  0
Duplicates skipped:      56,783  ← All were duplicates!
```

### Metadata Generation

Each dataset merge generates a metadata file documenting the process:

**File**: `datasets/master_dataset_metadata.json`

```json
{
  "created": "2025-10-26T07:56:11.911492",
  "source_pattern": "I:/Mijn Drive/NexusMind/historical-database/current/*/*.jsonl",
  "total_articles": 51869,
  "unique_articles": 51869,
  "duplicates_removed": 4914,
  "invalid_articles": 0,
  "output_file": "datasets\\master_dataset.jsonl",
  "statistics": {
    "total_read": 56783,
    "written": 51869,
    "duplicates": 4914
  }
}
```

### Progress Logging

The script logs progress every 10,000 articles:

```
Processing: content_items_20251013_065250.jsonl
  Written: 10,000 articles
Processing: content_items_20251015_185048.jsonl
  Written: 20,000 articles
...
```

## Command-Line Options

### Required Arguments

- `--source`: Glob pattern for source JSONL files
  - Example: `"I:/Mijn Drive/NexusMind/historical-database/current/*/*.jsonl"`
  - Supports wildcards for matching multiple files

- `--output`: Path to output master dataset file
  - Example: `datasets/master_dataset.jsonl`

### Optional Arguments

- `--incremental`: Append only new articles to existing master dataset
  - Default: `False` (full merge, overwrite)
  - Use for adding new data without re-processing existing articles

- `--max-articles`: Maximum number of articles to include
  - Default: `None` (unlimited)
  - Use for creating test datasets

- `--output-dir`: Base output directory
  - Default: `datasets/`
  - Where master dataset and metadata files are saved

## Dataset Structure

### Directory Layout

```
llm-distillery/
├── datasets/
│   ├── master_dataset.jsonl           # Full merged dataset (51,869 articles)
│   ├── master_dataset_metadata.json   # Merge statistics
│   ├── test_1k.jsonl                  # Test dataset (1,000 articles)
│   ├── test_1k_metadata.json          # Test metadata
│   ├── raw/                           # Raw source data (optional)
│   ├── processed/                     # Processed data (optional)
│   └── splits/                        # Train/val/test splits (future)
```

### Article Format

Each article in the master dataset follows this JSONL format:

```json
{
  "id": "github_1175a04f0e39",
  "title": "Repository: zhouyuan888888/CARETrans",
  "content": "CARE Transformer: Mobile-Friendly Linear Visual Transformer...",
  "source": "github",
  "source_type": "api",
  "url": "https://github.com/zhouyuan888888/CARETrans",
  "published_date": "2025-10-07T04:54:53",
  "collected_date": "2025-10-09T12:51:30.357274",
  "language": "en",
  "tags": ["github", "repository", "code"],
  "metadata": {
    "repo_id": 1071248919,
    "full_name": "zhouyuan888888/CARETrans",
    "owner": "zhouyuan888888",
    "stars": 1
  }
}
```

## Use Cases

### 1. Initial Dataset Creation

Create master dataset from all historical data:

```bash
python -m ground_truth.prepare_dataset \
    --source "I:/Mijn Drive/NexusMind/historical-database/current/*/*.jsonl" \
    --output datasets/master_dataset.jsonl
```

**Result**: 51,869 unique articles

### 2. Weekly Updates

Add new articles collected during the week:

```bash
# Run content-aggregator to collect new articles
cd C:\local_dev\content-aggregator
python run_aggregator.py

# Copy new files to historical database
# (manual or automated process)

# Update master dataset with only new articles
cd C:\local_dev\llm-distillery
python -m ground_truth.prepare_dataset \
    --source "I:/Mijn Drive/NexusMind/historical-database/current/*/*.jsonl" \
    --output datasets/master_dataset.jsonl \
    --incremental
```

### 3. Creating Stratified Samples

Create smaller datasets for specific purposes:

```bash
# Small test set for quick experiments
python -m ground_truth.prepare_dataset \
    --source "..." \
    --output datasets/test_100.jsonl \
    --max-articles 100

# Medium validation set
python -m ground_truth.prepare_dataset \
    --source "..." \
    --output datasets/val_5k.jsonl \
    --max-articles 5000

# Large training set
python -m ground_truth.prepare_dataset \
    --source "..." \
    --output datasets/train_50k.jsonl \
    --max-articles 50000
```

### 4. Domain-Specific Subsets

Filter by source type (requires custom filtering):

```bash
# GitHub repositories only
python -m ground_truth.prepare_dataset \
    --source "I:/Mijn Drive/NexusMind/historical-database/current/github/*.jsonl" \
    --output datasets/github_only.jsonl

# ArXiv papers only
python -m ground_truth.prepare_dataset \
    --source "I:/Mijn Drive/NexusMind/historical-database/current/arxiv/*.jsonl" \
    --output datasets/arxiv_only.jsonl
```

## Troubleshooting

### Issue: "No files found matching pattern"

**Cause**: Glob pattern doesn't match any files

**Solutions**:
1. Check the path exists: `ls "I:/Mijn Drive/NexusMind/historical-database/current/"`
2. Verify glob pattern: Use `/*/*.jsonl` for subdirectories
3. Try absolute path: `"I:/Mijn Drive/NexusMind/historical-database/current/*/*.jsonl"`

### Issue: High duplicate rate

**Cause**: Running full merge on existing dataset

**Solutions**:
1. Use `--incremental` flag for updates
2. Delete old master dataset to start fresh
3. Check source files aren't duplicated

### Issue: Out of memory

**Cause**: Loading too many article IDs for incremental updates

**Solutions**:
1. Process in smaller batches
2. Use `--max-articles` to limit dataset size
3. Increase system memory or use disk-based deduplication

## Performance

### Speed

- **Processing**: ~3,000 articles/second
- **Deduplication**: O(1) lookup per article
- **Total time**: ~20 seconds for 51,869 articles

### Memory

- **Incremental mode**: ~10MB per 10K article IDs
- **Full merge**: ~50MB peak memory usage
- **Master dataset**: 51,869 articles = ~15-20MB JSONL file

## Next Steps

After preparing the master dataset:

1. **Sample for calibration** (100 articles)
   ```bash
   python -m ground_truth.calibrate_models \
       --prompt prompts/uplifting.md \
       --source datasets/master_dataset.jsonl \
       --sample-size 100
   ```

2. **Generate ground truth** (50K articles)
   ```bash
   python -m ground_truth.generate \
       --prompt prompts/uplifting.md \
       --input datasets/master_dataset.jsonl \
       --output datasets/uplifting_50k_labeled.jsonl \
       --num-samples 50000 \
       --llm claude
   ```

3. **Train local model**
   ```bash
   python -m training.train \
       --config training/configs/uplifting_deberta.yaml \
       --dataset datasets/uplifting_50k_labeled.jsonl \
       --output inference/models/uplifting_v1
   ```

## See Also

- [Calibration Guide](calibration.md) - Compare Claude vs Gemini
- [Architecture Overview](../architecture/overview.md) - System design
- [Getting Started](getting-started.md) - Full workflow
