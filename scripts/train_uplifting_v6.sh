#!/bin/bash
# Train uplifting v6 with Gemma-3-1B on gpu-server
# Usage: bash ~/llm-distillery/scripts/train_uplifting_v6.sh

set -e

cd ~/llm-distillery
VENV_PYTHON=~/gpu-server/nexusmind-scorer/venv/bin/python

echo "=== Uplifting v6 Training ==="
echo "Model: google/gemma-3-1b-pt"
echo "Data:  datasets/training/uplifting_v6"
echo ""

# Check GPU
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

# Check data exists
if [ ! -f datasets/training/uplifting_v6/train.jsonl ]; then
    echo "ERROR: Training data not found at datasets/training/uplifting_v6/"
    exit 1
fi

TRAIN_COUNT=$(wc -l < datasets/training/uplifting_v6/train.jsonl)
VAL_COUNT=$(wc -l < datasets/training/uplifting_v6/val.jsonl)
echo "Train: $TRAIN_COUNT examples"
echo "Val:   $VAL_COUNT examples"
echo ""

# Create output dir
mkdir -p filters/uplifting/v6

# Train
echo "=== Starting Training ==="
PYTHONPATH=. $VENV_PYTHON training/train.py \
    --filter filters/uplifting/v6 \
    --data-dir datasets/training/uplifting_v6 \
    --model-name google/gemma-3-1b-pt \
    --use-head-tail \
    --head-tokens 256 \
    --tail-tokens 256 \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --warmup-steps 500 \
    --max-length 512

echo ""
echo "=== Training Complete ==="
echo "Model saved to: filters/uplifting/v6/model/"
echo "Next: scp the model back to local machine"
