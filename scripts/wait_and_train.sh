#!/bin/bash
# Wait for GPU to be free, then train uplifting v6
# Checks every 5 minutes, starts after initial 60-minute wait

set -e

cd ~/llm-distillery
VENV_PYTHON=~/gpu-server/nexusmind-scorer/venv/bin/python
LOG=~/llm-distillery/training_v6.log

echo "$(date): Waiting 60 minutes before first GPU check..." | tee -a $LOG
sleep 3600

MAX_ATTEMPTS=12  # Try for up to 1 more hour after initial wait
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))

    # Check GPU memory usage
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    echo "$(date): Attempt $ATTEMPT/$MAX_ATTEMPTS - GPU mem: ${GPU_MEM}MiB, util: ${GPU_UTIL}%" | tee -a $LOG

    # If GPU memory < 1000 MiB, it's likely free
    if [ "$GPU_MEM" -lt 1000 ]; then
        echo "$(date): GPU is free! Starting training..." | tee -a $LOG

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
            --max-length 512 2>&1 | tee -a $LOG

        echo "$(date): Training complete!" | tee -a $LOG
        echo "$(date): Model saved to filters/uplifting/v6/model/" | tee -a $LOG
        exit 0
    fi

    echo "$(date): GPU busy, waiting 5 minutes..." | tee -a $LOG
    sleep 300
done

echo "$(date): GPU never freed after $MAX_ATTEMPTS attempts. Giving up." | tee -a $LOG
exit 1
