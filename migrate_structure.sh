#!/bin/bash
# One-time migration script to reorganize trained models
# Run this on GPU machine after git pull

echo "=== LLM Distillery Structure Migration ==="
echo ""
echo "This will move trained models to the new structure:"
echo "  inference/deployed/{filter}_v{version}/ -> filters/{filter}/v{version}/"
echo ""

# Check if old structure exists
if [ ! -d "inference/deployed" ]; then
    echo "[INFO] No inference/deployed/ directory found - nothing to migrate"
    exit 0
fi

# Find all model directories in old structure
OLD_MODELS=$(find inference/deployed -type d -name "model" 2>/dev/null)

if [ -z "$OLD_MODELS" ]; then
    echo "[INFO] No trained models found in inference/deployed/"
    echo "[INFO] Old structure can be safely removed"
    read -p "Remove inference/deployed/ directory? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf inference/deployed/
        echo "[OK] Removed inference/deployed/"
    fi
    exit 0
fi

echo "Found trained models to migrate:"
echo ""

# Process each model
for MODEL_PATH in $OLD_MODELS; do
    # Extract filter info from path
    # e.g., inference/deployed/uplifting_v1/model -> uplifting_v1
    OLD_DIR=$(dirname "$MODEL_PATH")
    FILTER_VERSION=$(basename "$OLD_DIR")

    # Parse filter name and version
    # uplifting_v1 -> uplifting, v1
    if [[ $FILTER_VERSION =~ ^(.+)_v([0-9]+)$ ]]; then
        FILTER_NAME="${BASH_REMATCH[1]}"
        VERSION="${BASH_REMATCH[2]}"

        NEW_FILTER_DIR="filters/${FILTER_NAME}/v${VERSION}"

        echo "  $FILTER_VERSION -> $NEW_FILTER_DIR"

        # Check if filter directory exists
        if [ ! -d "$NEW_FILTER_DIR" ]; then
            echo "    [WARNING] Filter directory not found: $NEW_FILTER_DIR"
            echo "    [WARNING] Skipping (run git pull first)"
            continue
        fi

        # Create backup
        if [ -d "$NEW_FILTER_DIR/model" ]; then
            echo "    [INFO] Model already exists in new location, backing up..."
            mv "$NEW_FILTER_DIR/model" "$NEW_FILTER_DIR/model.backup.$(date +%s)"
        fi

        # Move model
        echo "    Moving model..."
        mv "$MODEL_PATH" "$NEW_FILTER_DIR/"

        # Move training metadata if exists
        if [ -f "$OLD_DIR/training_history.json" ]; then
            echo "    Moving training_history.json..."
            mv "$OLD_DIR/training_history.json" "$NEW_FILTER_DIR/"
        fi

        if [ -f "$OLD_DIR/training_metadata.json" ]; then
            echo "    Moving training_metadata.json..."
            mv "$OLD_DIR/training_metadata.json" "$NEW_FILTER_DIR/"
        fi

        # Move plots to reports if they exist
        if [ -d "$OLD_DIR/plots" ]; then
            PLOTS_DIR="reports/${FILTER_VERSION}_plots"
            echo "    Moving plots to $PLOTS_DIR..."
            mkdir -p "$PLOTS_DIR"
            mv "$OLD_DIR/plots"/* "$PLOTS_DIR/" 2>/dev/null || true
        fi

        # Move report if exists
        REPORT_FILE=$(find "$OLD_DIR" -name "*.docx" 2>/dev/null | head -1)
        if [ -n "$REPORT_FILE" ]; then
            echo "    Moving training report..."
            mkdir -p reports
            mv "$REPORT_FILE" "reports/${FILTER_VERSION}_training_report.docx"
        fi

        echo "    [OK] Migration complete"

    else
        echo "    [WARNING] Couldn't parse filter name from: $FILTER_VERSION"
    fi

    echo ""
done

# Clean up old structure
echo ""
echo "Migration complete!"
echo ""

if [ -d "inference/deployed" ]; then
    echo "Checking if old structure can be removed..."
    REMAINING=$(find inference/deployed -type f 2>/dev/null | wc -l)

    if [ "$REMAINING" -eq 0 ]; then
        echo "[INFO] Old structure is empty"
        read -p "Remove inference/deployed/ directory? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf inference/deployed/
            echo "[OK] Removed inference/deployed/"
        fi
    else
        echo "[INFO] Some files remain in inference/deployed/"
        echo "[INFO] Review manually before removing"
    fi
fi

echo ""
echo "=== Next Steps ==="
echo "1. Verify models are in filters/{filter_name}/v{version}/model/"
echo "2. Test inference with new structure"
echo "3. Remove inference/deployed/ if everything works"
echo ""
echo "New training will automatically use the correct structure."
