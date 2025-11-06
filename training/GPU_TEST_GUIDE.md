# GPU Training Test Guide

Quick guide to test training on your GPU machine before running the full training.

## Step 1: Setup on GPU Machine

### 1.1 Clone Repository

```bash
git clone https://github.com/ducroq/llm-distillery.git
cd llm-distillery
```

### 1.2 Install Dependencies

```bash
# Install CUDA-enabled PyTorch (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install transformers datasets pyyaml tqdm
```

### 1.3 Verify GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB' if torch.cuda.is_available() else '')"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4090 (or similar)
Memory: 24.0GB
```

## Step 2: Copy Dataset

You need to transfer the prepared dataset to your GPU machine:

```bash
# On your current machine (with the dataset)
# datasets/uplifting_ground_truth_v1_splits/
#   ├── train.jsonl (6,172 articles)
#   ├── val.jsonl (771 articles)
#   ├── test.jsonl (772 articles)
#   └── split_metadata.json

# Transfer to GPU machine (example using scp)
scp -r datasets/uplifting_ground_truth_v1_splits user@gpu-machine:/path/to/llm-distillery/datasets/
```

Or copy the filter and raw data, then prepare splits on GPU machine:
```bash
# Copy raw data and filter
scp -r datasets/uplifting_ground_truth_v1 user@gpu-machine:/path/to/llm-distillery/datasets/
scp -r filters/uplifting user@gpu-machine:/path/to/llm-distillery/filters/

# Then on GPU machine, prepare splits:
python -m training.prepare_dataset \
    --filter filters/uplifting/v1 \
    --dataset datasets/uplifting_ground_truth_v1/labeled_articles.jsonl \
    --output-dir datasets/uplifting_ground_truth_v1_splits
```

## Step 3: Quick Test (100 samples, 1 epoch)

Test with a tiny subset first to verify everything works:

### 3.1 Create Test Subset

```bash
# Create test directory
mkdir -p datasets/test_subset

# Take first 100 train samples
head -n 100 datasets/uplifting_ground_truth_v1_splits/train.jsonl > datasets/test_subset/train.jsonl

# Take first 10 val samples
head -n 10 datasets/uplifting_ground_truth_v1_splits/val.jsonl > datasets/test_subset/val.jsonl
```

### 3.2 Run Test Training

```bash
# For 16GB GPU, use smaller model (0.5B fits comfortably)
# Note: --output-dir is optional, defaults to filter directory
python -m training.train \
    --filter filters/uplifting/v1 \
    --data-dir datasets/test_subset \
    --output-dir test_training_run \
    --model-name Qwen/Qwen2.5-0.5B \
    --epochs 1 \
    --batch-size 2 \
    --max-length 384
```

**Note**: The training script uses gradient checkpointing to reduce memory usage.

**Model sizing for 16GB GPU:**
- Qwen 2.5-0.5B: ~4-5GB (recommended for 16GB GPU) ✓
- Qwen 2.5-1.5B: ~12-14GB (needs 20GB+)
- Qwen 2.5-7B: ~28GB+ (needs 40GB+ GPU)

**Expected behavior:**
- Model downloads (~1GB for 0.5B model)
- Training starts and shows progress bar
- Completes in ~2-5 minutes
- Shows validation metrics at end
- Saves model to `test_training_run/model/`

**What to check:**
- ✓ No CUDA out of memory errors
- ✓ Training loss decreases
- ✓ GPU utilization is high (check with `nvidia-smi`)
- ✓ Model saves successfully

### 3.3 Check Output

```bash
ls test_training_run/model/
# Should show: config.json, model.safetensors, tokenizer files

cat test_training_run/training_metadata.json
# Should show: training config, metrics, model info
```

## Step 4: Full Training (if test passes)

Once the test works, run full training:

```bash
# For 16GB GPU - use 0.5B model (fits comfortably)
# Model will save to filters/uplifting/v1/model/ by default
python -m training.train \
    --filter filters/uplifting/v1 \
    --data-dir datasets/uplifting_ground_truth_v1_splits \
    --model-name Qwen/Qwen2.5-0.5B \
    --epochs 10 \
    --batch-size 4 \
    --max-length 512 \
    --learning-rate 2e-5

# For 24GB GPU - can use 1.5B model
# python -m training.train \
#     --filter filters/uplifting/v1 \
#     --data-dir datasets/uplifting_ground_truth_v1_splits \
#     --model-name Qwen/Qwen2.5-1.5B \
#     --batch-size 4 \
#     --epochs 10

# For 40GB+ GPU - can use 7B model (original config)
# python -m training.train \
#     --filter filters/uplifting/v1 \
#     --data-dir datasets/uplifting_ground_truth_v1_splits \
#     --model-name Qwen/Qwen2.5-7B \
#     --batch-size 4 \
#     --epochs 10
```

**Training time estimate (6,172 samples, 3 epochs):**
- 0.5B model @ batch_size=4
  - RTX 4090 (16GB): ~2-3 hours
- 1.5B model @ batch_size=4
  - RTX 4090 (24GB): ~3-4 hours
- 7B model @ batch_size=4
  - A100 (40GB): ~4-6 hours

**Memory usage with gradient checkpointing (FP32):**
- 0.5B model: ~4-5GB (fits 16GB easily) ✓
- 1.5B model: ~12-14GB (needs 20GB+)
- 7B model: ~28GB+ (needs 40GB+)

**Note**: FP32 training is used for stability (FP16 causes NaN issues)

## Monitoring Training

### In the Terminal
Training shows real-time progress:
```
Epoch 1/3
========================================
Training: 100%|████████| 772/772 [23:15<00:00]
  Loss: 1.234
  MAE: 0.876
  RMSE: 1.123

Validation metrics:
  Loss: 1.456
  MAE: 0.945
  RMSE: 1.234

✓ New best validation MAE: 0.945
  Model saved to: inference/deployed/uplifting_v1/model
```

### With nvidia-smi
```bash
# In another terminal
watch -n 1 nvidia-smi
```

## Troubleshooting

### Out of Memory
```bash
# Try smaller batch size
--batch-size 4

# Or smaller model
--model-name Qwen/Qwen2.5-3B
```

### Model Download Fails
```bash
# Set HuggingFace cache
export HF_HOME=/path/with/space

# Or download manually first
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-7B')"
```

### Slow Training
```bash
# Check GPU is actually being used
python -c "import torch; print(torch.cuda.is_available())"

# Check data loading isn't bottleneck (add more workers if needed)
# Note: Currently not implemented, but you can modify train.py
```

## What Success Looks Like

After 3 epochs, you should see:
- **Validation MAE**: ~0.8-1.2 (target: <1.0)
- **Training time**: 2-4 hours
- **Model size**: ~14GB saved
- **No errors or warnings**

## Next Steps After Training

1. **Evaluate model** (coming soon):
   ```bash
   python -m evaluation.evaluate \
       --filter filters/uplifting/v1 \
       --model inference/deployed/uplifting_v1 \
       --test-set datasets/uplifting_ground_truth_v1_splits/test.jsonl
   ```

2. **Run inference** (coming soon):
   ```bash
   python -m inference.predict \
       --model inference/deployed/uplifting_v1 \
       --input new_articles.jsonl \
       --output predictions.jsonl
   ```

## Files to Transfer Back

After training completes, transfer back to your main machine:

```bash
# From GPU machine
scp -r inference/deployed/uplifting_v1 user@main-machine:/path/to/llm-distillery/inference/deployed/
```

Or just the training metadata if model is too large:
```bash
scp inference/deployed/uplifting_v1/training_metadata.json user@main-machine:/path/
scp inference/deployed/uplifting_v1/training_history.json user@main-machine:/path/
```
