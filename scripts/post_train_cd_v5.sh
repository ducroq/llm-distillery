#!/bin/bash
# Post-training pipeline for cd v5 — runs after Gemma-3-1B+LoRA training completes on gpu-server.
#
# Does (auto):
#   1. scp model/ adapter back from gpu-server
#   2. scp training_metadata.json + training_history.json back
#   3. Verify model files present + sane
#   4. Fit calibration.json (isotonic regression on val set)
#   5. Verify calibration.json fitted
#
# Does NOT do (manual review required first):
#   - Hub upload  → prints command at the end
#   - NexusMind deploy  → prints command at the end
#
# Usage:
#   bash scripts/post_train_cd_v5.sh
#
# Prerequisites:
#   - Training completed on gpu-server (model in ~/llm-distillery/filters/cultural-discovery/v5/model/)
#   - cd v5 filter package locally has config.yaml + prompt-compressed.md + prefilter.py
#   - HF_TOKEN in environment (for Hub upload command at end)

set -e

FILTER_DIR="filters/cultural_discovery/v5"
GPU_FILTER_DIR="~/llm-distillery/filters/cultural-discovery/v5"  # gpu-server uses hyphen path
DATA_DIR="datasets/training/cultural-discovery_v5"

echo "=============================================================="
echo "Post-training pipeline: cd v5"
echo "=============================================================="

# Step 1: scp model adapter back from gpu-server
echo ""
echo "[1/5] Pulling model/ adapter from gpu-server..."
if ssh gpu-server "test -d ${GPU_FILTER_DIR}/model"; then
    scp -r gpu-server:${GPU_FILTER_DIR}/model ${FILTER_DIR}/
    echo "  ✓ Model adapter pulled to ${FILTER_DIR}/model/"
else
    echo "  ✗ FAIL: ${GPU_FILTER_DIR}/model/ does not exist on gpu-server"
    echo "    Training may not have completed yet. Check ssh gpu-server 'ls ${GPU_FILTER_DIR}/'"
    exit 1
fi

# Step 2: scp training metadata + history back
echo ""
echo "[2/5] Pulling training_metadata.json + training_history.json from gpu-server..."
for f in training_metadata.json training_history.json; do
    if ssh gpu-server "test -f ${GPU_FILTER_DIR}/${f}"; then
        scp gpu-server:${GPU_FILTER_DIR}/${f} ${FILTER_DIR}/${f}
        echo "  ✓ ${f} pulled"
    else
        echo "  ⚠ WARN: ${f} missing on gpu-server"
    fi
done

# Step 3: Verify model files
echo ""
echo "[3/5] Verifying model package..."
ls -la ${FILTER_DIR}/model/ 2>&1 | head -10
required_files="adapter_config.json adapter_model.safetensors tokenizer.json"
for f in ${required_files}; do
    if [ -f "${FILTER_DIR}/model/${f}" ]; then
        echo "  ✓ ${f}"
    else
        echo "  ✗ FAIL: ${f} missing"
        exit 1
    fi
done

# Step 4: Fit calibration
echo ""
echo "[4/5] Fitting calibration.json (isotonic regression on val set)..."
PYTHONPATH=. python scripts/calibration/fit_calibration.py \
    --filter ${FILTER_DIR} \
    --data-dir ${DATA_DIR} \
    --test-data ${DATA_DIR}/test.jsonl

# Step 5: Verify calibration
echo ""
echo "[5/5] Verifying calibration.json..."
if [ -f "${FILTER_DIR}/calibration.json" ]; then
    bytes=$(wc -c < "${FILTER_DIR}/calibration.json")
    echo "  ✓ calibration.json created (${bytes} bytes)"
else
    echo "  ✗ FAIL: calibration.json not created"
    exit 1
fi

# Summary
echo ""
echo "=============================================================="
echo "AUTOMATED STEPS COMPLETE"
echo "=============================================================="
echo ""
echo "Filter package contents:"
ls -la ${FILTER_DIR}/ 2>&1
echo ""
echo "=============================================================="
echo "NEXT (MANUAL — review before running):"
echo "=============================================================="
echo ""
echo "1. Verify the filter package looks complete:"
echo "   PYTHONPATH=. python scripts/deployment/verify_filter_package.py --filter ${FILTER_DIR}"
echo ""
echo "2. (Optional) Read training_metadata.json + training_history.json for MAE numbers"
echo ""
echo "3. Hub upload (private):"
echo "   python scripts/deployment/upload_to_huggingface.py \\"
echo "       --filter ${FILTER_DIR} \\"
echo "       --repo-name jeergrvgreg/cultural-discovery-filter-v5 \\"
echo "       --token \$HF_TOKEN --private"
echo ""
echo "4. NexusMind deploy:"
echo "   bash scripts/deploy_to_nexusmind.sh cultural_discovery v5"
echo "   # (script will refuse if NexusMind target is dirty; --force-dirty escape exists)"
echo ""
echo "5. Update memory + run /curate to capture lessons"
