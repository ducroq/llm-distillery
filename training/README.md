# Training Pipeline

This directory contains the training pipeline for fine-tuning Qwen 2.5 models on filter-specific datasets.

## Overview

The training pipeline consists of two main steps:

1. **Dataset Preparation**: Split labeled ground truth into train/val/test sets
2. **Model Training**: Fine-tune Qwen 2.5 for multi-dimensional regression

## Quick Start

### Prerequisites

Install required dependencies:

```bash
pip install torch transformers datasets pyyaml tqdm
```

### Step 1: Prepare Dataset

Split your labeled ground truth data:

```bash
python -m training.prepare_dataset \
    --filter filters/uplifting/v1 \
    --dataset datasets/uplifting_ground_truth_v1/labeled_articles.jsonl \
    --output-dir datasets/uplifting_ground_truth_v1_splits \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

**Output**: `train.jsonl`, `val.jsonl`, `test.jsonl`, `split_metadata.json`

### Step 2: Train Model

Train Qwen 2.5 on the prepared dataset:

```bash
python -m training.train \
    --filter filters/uplifting/v1 \
    --data-dir datasets/uplifting_ground_truth_v1_splits \
    --output-dir inference/deployed/uplifting_v1 \
    --model-name Qwen/Qwen2.5-7B \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5
```

**Output**: Trained model, training history, metadata

## Model Architecture

The training pipeline uses **Qwen 2.5-7B** adapted for multi-dimensional regression:

```
Input: [title + content] (max 512 tokens)
         ↓
    Qwen 2.5 Base Model
         ↓
    Regression Head (num_dimensions outputs)
         ↓
Output: [dim1_score, dim2_score, ..., dim8_score]
```

### Loss Function

Mean Squared Error (MSE) across all dimensions:

```python
loss = MSE(predictions, ground_truth_labels)
```

### Evaluation Metrics

- **MAE** (Mean Absolute Error): Average error per dimension
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **Per-dimension metrics**: MAE and RMSE for each dimension

## Dataset Format

### Input Format (labeled_articles.jsonl)

Each article includes:

```json
{
  "id": "article_id",
  "title": "Article title",
  "content": "Article text...",
  "uplifting_analysis": {
    "dimensions": {
      "agency": 7,
      "progress": 6,
      "collective_benefit": 8,
      ...
    }
  }
}
```

### Training Format (train.jsonl)

After preparation:

```json
{
  "id": "article_id",
  "title": "Article title",
  "content": "Article text...",
  "labels": [7, 6, 8, 5, 4, 6, 5, 7],
  "dimension_names": ["agency", "progress", "collective_benefit", ...]
}
```

## Training Configuration

Training configs are stored in `training/configs/<filter>_<version>.yaml`:

```yaml
filter:
  path: "filters/uplifting/v1"
  name: "uplifting"
  version: "1.0"

model:
  name: "Qwen/Qwen2.5-7B"
  max_length: 512

training:
  epochs: 3
  batch_size: 8
  learning_rate: 2e-5
  warmup_steps: 500
```

## Hardware Requirements

### Qwen 2.5-7B (Recommended)

- **GPU Memory**: 16GB+ (RTX 4090, A100)
- **Training Time**: ~2-4 hours for 7,000 samples
- **Expected Accuracy**: 90-95% vs oracle

### Alternative Models

For limited hardware:

- **Qwen 2.5-1.5B**: 8GB GPU, faster training, slightly lower accuracy
- **Qwen 2.5-3B**: 12GB GPU, balanced performance

## Output Structure

Training saves directly to the filter directory:

```
filters/uplifting/v1/
├── config.yaml                    # Filter configuration
├── prefilter.yaml                 # Pre-filter rules
├── README.md                      # Filter documentation
├── model/                         # Trained model checkpoint
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
├── training_history.json          # Metrics per epoch
└── training_metadata.json         # Training configuration
```

Reports and visualizations are saved separately:

```
reports/
├── uplifting_v1_training_report.docx
└── uplifting_v1_plots/
    ├── overall_metrics.png
    ├── per_dimension_mae.png
    ├── loss_curves.png
    └── training_summary.txt
```

## Monitoring Training

Training progress is displayed in real-time:

```
Epoch 1/3
========================================
Training: 100%|████████| 772/772 [23:15<00:00]
  Loss: 1.234
  MAE: 0.876
  RMSE: 1.123

Validation:
  Loss: 1.456
  MAE: 0.945
  RMSE: 1.234

✓ New best validation MAE: 0.945
  Model saved to: filters/uplifting/v1/model
```

## Post-Training Workflow

After training completes, follow these steps:

### 1. Generate Visualizations

```bash
python -m training.plot_learning_curves \
    --history filters/uplifting/v1/training_history.json
```

**Output:** `reports/uplifting_v1_plots/`
- `overall_metrics.png` - MAE/RMSE curves
- `per_dimension_mae.png` - Per-dimension learning
- `loss_curves.png` - Training/validation loss
- `training_summary.txt` - Final metrics table

### 2. Generate Training Report (Optional)

```bash
python -m training.generate_training_report \
    --filter filters/uplifting/v1 \
    --history filters/uplifting/v1/training_history.json \
    --metadata filters/uplifting/v1/training_metadata.json
```

**Output:** `reports/uplifting_v1_training_report.docx`
- Professional Word document with embedded plots
- Executive summary, methodology, results, conclusions

### 3. Upload to Hugging Face (Optional)

```bash
python -m training.upload_to_huggingface \
    --filter filters/uplifting/v1 \
    --repo-name your-username/uplifting-filter-v1 \
    --private
```

See `HUGGINGFACE_SETUP.md` for detailed instructions.

### 4. Evaluate on Test Set

```bash
# TODO: Add evaluation script
python -m evaluation.evaluate \
    --filter filters/uplifting/v1 \
    --test-data datasets/uplifting_ground_truth_v1_splits/test.jsonl
```

See `evaluation/README.md` for evaluation workflow.

### Quick Post-Training Commands

```bash
# After training finishes, run these in sequence:

# 1. Generate plots
python -m training.plot_learning_curves \
    --history filters/uplifting/v1/training_history.json

# 2. Generate report (optional)
python -m training.generate_training_report \
    --filter filters/uplifting/v1 \
    --history filters/uplifting/v1/training_history.json \
    --metadata filters/uplifting/v1/training_metadata.json

# 3. Upload to Hugging Face (optional)
python -m training.upload_to_huggingface \
    --filter filters/uplifting/v1 \
    --repo-name username/uplifting-filter-v1 \
    --private
```

## Troubleshooting

### Out of Memory

Reduce batch size or use smaller model:

```bash
python -m training.train \
    --batch-size 4 \
    --model-name Qwen/Qwen2.5-3B \
    ...
```

### Slow Training

Enable gradient accumulation for larger effective batch sizes:

```bash
# Coming soon: --gradient-accumulation-steps 4
```

### Poor Accuracy

- Increase training epochs: `--epochs 5`
- Adjust learning rate: `--learning-rate 1e-5` or `3e-5`
- Check data quality: Verify ground truth labels are consistent

## Advanced Usage

### Custom Training Config

Load from YAML file:

```bash
# Coming soon
python -m training.train \
    --config training/configs/uplifting_v1.yaml
```

### Resume Training

Resume from checkpoint:

```bash
# Coming soon
python -m training.train \
    --resume training/checkpoints/uplifting_v1/epoch_2
```

### Multi-GPU Training

Use distributed training:

```bash
# Coming soon
torchrun --nproc_per_node=4 -m training.train ...
```
