# Training Pipeline

This directory contains the training pipeline for fine-tuning Qwen 2.5 models on filter-specific datasets.

## Overview

The training pipeline consists of four main steps:

1. **⭐ Oracle Prompt Calibration** (CRITICAL - before batch labeling): Validate oracle prompt on small sample
2. **Batch Labeling**: Create ground truth dataset using validated oracle
3. **Dataset Preparation**: Split labeled ground truth into train/val/test sets
4. **Model Training**: Fine-tune Qwen 2.5 for multi-dimensional regression

## Quick Start

### Prerequisites

Install required dependencies:

```bash
pip install torch transformers datasets pyyaml tqdm
```

### Step 0: Calibrate Oracle Prompt ⭐ REQUIRED BEFORE LABELING

**CRITICAL:** Always calibrate your oracle prompt before batch labeling to avoid wasting money on mis-labeled data.

**Why calibrate?** In the sustainability_tech_deployment filter, we discovered after $8 of labeling that the oracle was broken - scoring AWS IAM tutorials and toothbrushes as "mass-deployed climate tech" (10.0 scores). Calibration on 50 articles ($0.05) would have caught this before batch labeling 8,162 articles.

**Process:**

1. **Create calibration sample (50-100 articles):**
   ```bash
   # Stratified sample: 20 on-topic, 20 off-topic, 10 edge cases
   python sandbox/analysis_scripts/create_calibration_sample.py \
       --input articles_corpus.jsonl \
       --output calibration_sample.jsonl \
       --n-positive 20 \
       --n-negative 20 \
       --n-edge 10
   ```

2. **Label calibration sample:**
   ```bash
   python scripts/label_batch.py \
       --filter filters/{filter_name}/v1 \
       --input calibration_sample.jsonl \
       --output calibration_labeled.jsonl
   ```
   **Cost:** ~$0.05 (50 articles × $0.001)

3. **Run Prompt Calibration Agent:**
   ```
   Task: "Calibrate the {filter_name} oracle prompt using the
   Prompt Calibration Agent from docs/agents/templates/prompt-calibration-agent.md.

   Calibration sample: calibration_labeled.jsonl
   Filter path: filters/{filter_name}/v1

   Generate calibration report with PASS/REVIEW/FAIL decision."
   ```

4. **Review calibration report:**
   ```bash
   cat filters/{filter_name}/v1/calibration_report.md
   ```

   **Decision matrix:**
   - ✅ **PASS** → Proceed to batch labeling (Step 1)
   - ⚠️ **REVIEW** → Fix minor issues, re-calibrate ($0.05)
   - ❌ **FAIL** → Major prompt revision needed, re-calibrate

5. **Iterate until PASS:**
   - Update prompt based on calibration findings
   - Re-label same 50 articles (another $0.05)
   - Review again
   - Typical iterations: 2-3 rounds ($0.10-0.15 total)

6. **⭐ Validate on Fresh Sample (CRITICAL - prevents overfitting):**
   ```bash
   # Create validation sample with DIFFERENT random seed
   python sandbox/analysis_scripts/create_calibration_sample.py \
       --input articles_corpus.jsonl \
       --output validation_sample.jsonl \
       --n-positive 20 \
       --n-negative 20 \
       --n-edge 10 \
       --random-seed 2025  # Different seed = different articles

   # Label validation sample
   python -m ground_truth.batch_labeler \
       --filter filters/{filter_name}/v1 \
       --source validation_sample.jsonl \
       --output-dir validation_labeled \
       --llm gemini-flash \
       --batch-size 50 \
       --max-batches 1

   # Run calibration analysis on validation sample
   Task: "Run Prompt Calibration Agent on validation sample to verify
   prompt improvements generalize to new articles."
   ```

   **Cost:** ~$0.05 (50 articles × $0.001)

   **Why validate?**
   - Ensures prompt fixes didn't overfit to calibration sample
   - Validates improvements generalize to unseen data
   - Like train/test split in ML - calibration = train, validation = test

   **Decision matrix:**
   - ✅ **Validation metrics ≈ Calibration metrics** → Proceed to batch labeling
   - ⚠️ **Validation worse than calibration** → Overfitting detected, revise prompt
   - ❌ **Validation fails targets** → Major issues, start over

**ROI:** Spend $0.20 + 2 hours to save $8-16 + days of rework

**See:**
- `docs/decisions/2025-11-13-prompt-calibration-before-batch-labeling.md` for full rationale
- `docs/decisions/2025-11-14-calibration-validation-split.md` for train/test split pattern

---

### Step 1: Batch Label Dataset (after calibration AND validation PASS)

**Only proceed after Step 0 calibration and validation pass!**

```bash
python scripts/label_batch.py \
    --filter filters/{filter_name}/v1 \
    --input articles_to_label.jsonl \
    --output datasets/labeled/{filter_name}/labeled_articles.jsonl
```

**Cost:** ~$8 for 8,000 articles
**Confidence:** High (oracle validated on calibration sample)

---

### Step 2: Prepare Dataset

Split your labeled ground truth data using the generic preparation script:

```bash
# For uplifting filter
python scripts/prepare_training_data.py \
    --filter filters/uplifting/v1 \
    --input datasets/labeled/uplifting/labeled_articles.jsonl \
    --output-dir datasets/training/uplifting \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1

# For tech deployment filter
python scripts/prepare_training_data.py \
    --filter filters/sustainability_tech_deployment/v1 \
    --input datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl \
    --output-dir datasets/training/sustainability_tech_deployment \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

**Key Feature**: The script automatically reads filter configuration (dimensions, tier boundaries) from `config.yaml`, making it work for any filter without code changes.

**Output**: `train.jsonl`, `val.jsonl`, `test.jsonl` with stratified splits maintaining tier proportions

### Step 2: Train Model

Train Qwen 2.5 on the prepared dataset:

```bash
python -m training.train \
    --filter filters/uplifting/v1 \
    --data-dir datasets/training/uplifting \
    --output-dir filters/uplifting/v1 \
    --model-name Qwen/Qwen2.5-7B \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5
```

**Output**: Trained model, training history, metadata saved to filter directory

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

After training completes, follow these steps to evaluate and deploy your model.

### Step 1: Copy Training Results (From Remote GPU)

If training on remote GPU machine:

```bash
# From local machine
scp user@remote:/path/to/llm-distillery/filters/{filter_name}/v1/training_*.json \
    filters/{filter_name}/v1/
```

### Step 2: Review Training Metrics

Quick sanity check before full evaluation:

```bash
# Check final validation MAE
cat filters/{filter_name}/v1/training_metadata.json | grep best_val_mae

# Review training progression
cat filters/{filter_name}/v1/training_history.json
```

**Quick decision:**
- ✅ Val MAE < 1.0 → Proceed to Step 3
- ⚠️ Val MAE 1.0-1.5 → Proceed to Step 3, but may need more training
- ❌ Val MAE > 1.5 → Consider retraining before full evaluation

### Step 3: Run Model Evaluation Agent ⭐ CRITICAL

Systematically evaluate model for production readiness:

```
Task: "Evaluate the trained {filter_name} model using the Model Evaluation Agent
criteria from docs/agents/templates/model-evaluation-agent.md.

Model location: filters/{filter_name}/v1
Test data: datasets/training/{filter_name}/test.jsonl

Run test evaluation and generate production readiness report."
```

**Agent will:**
- Run test set evaluation (`sandbox/analysis_scripts/evaluate_model.py`)
- Analyze training progression, overfitting, convergence
- Check per-dimension performance
- Generate report: `filters/{filter_name}/v1/model_evaluation.md`
- Recommend: ✅ DEPLOY / ⚠️ REVIEW / ❌ FAIL

**Expected duration:** 15-30 minutes (includes test inference + analysis)

### Step 4: Review Production Readiness Report

```bash
cat filters/{filter_name}/v1/model_evaluation.md
```

**Decision matrix:**
- ✅ **DEPLOY** → Model is production ready (test MAE < 1.0, no overfitting)
  - Proceed to Step 5 (deployment)
- ⚠️ **REVIEW** → Model is borderline (test MAE 1.0-1.2)
  - Discuss trade-offs: Deploy for filtering vs retrain for precision
- ❌ **FAIL** → Model needs retraining (test MAE > 1.2 or severe issues)
  - Review recommendations in report (more epochs, larger model, more data)

### Step 5: Deploy to Production (If Report Says DEPLOY)

```bash
# Copy model to production server
scp -r filters/{filter_name}/v1/model/ production-server:/models/

# Or commit to repository for version control
git add filters/{filter_name}/v1/
git commit -m "Add trained {filter_name} model v1 (test MAE: X.XX)"
```

---

## Optional: Additional Analysis & Reporting

### Generate Visualizations (Optional)

```bash
python -m training.plot_learning_curves \
    --history filters/uplifting/v1/training_history.json
```

**Output:** `reports/uplifting_v1_plots/`
- `overall_metrics.png` - MAE/RMSE curves
- `per_dimension_mae.png` - Per-dimension learning
- `loss_curves.png` - Training/validation loss
- `training_summary.txt` - Final metrics table

### Generate Training Report (Optional)

```bash
python -m training.generate_training_report \
    --filter filters/uplifting/v1 \
    --history filters/uplifting/v1/training_history.json \
    --metadata filters/uplifting/v1/training_metadata.json
```

**Output:** `reports/uplifting_v1_training_report.docx`
- Professional Word document with embedded plots
- Executive summary, methodology, results, conclusions

### Upload to Hugging Face (Optional)

```bash
python -m training.upload_to_huggingface \
    --filter filters/uplifting/v1 \
    --repo-name your-username/uplifting-filter-v1 \
    --private
```

See `HUGGINGFACE_SETUP.md` for detailed instructions.

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

# 3. Upload to Hugging Face (optional, use YOUR username)
python -m training.upload_to_huggingface \
    --filter filters/uplifting/v1 \
    --repo-name YOUR_USERNAME/uplifting-filter-v1 \
    --private
```

**Note:** Replace `YOUR_USERNAME` with your Hugging Face username (check with `hf whoami`).

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
