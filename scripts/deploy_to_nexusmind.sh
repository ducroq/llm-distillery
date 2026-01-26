#!/bin/bash
# Deploy a filter from llm-distillery to NexusMind
#
# Usage: ./scripts/deploy_to_nexusmind.sh <filter_name> <version> [--push]
#
# Examples:
#   ./scripts/deploy_to_nexusmind.sh uplifting v5
#   ./scripts/deploy_to_nexusmind.sh sustainability_technology v2 --push
#
# What it does:
#   1. Copies filter folder to NexusMind
#   2. Copies filters/common/ (shared utilities)
#   3. Commits changes to NexusMind repo
#   4. Optionally pushes and shows pull commands for servers

set -e  # Exit on error

# Configuration
DISTILLERY_ROOT="C:/local_dev/llm-distillery"
NEXUSMIND_ROOT="C:/local_dev/NexusMind"

# Parse arguments
FILTER_NAME="$1"
VERSION="$2"
PUSH_FLAG="$3"

if [ -z "$FILTER_NAME" ] || [ -z "$VERSION" ]; then
    echo "Usage: $0 <filter_name> <version> [--push]"
    echo ""
    echo "Examples:"
    echo "  $0 uplifting v5"
    echo "  $0 sustainability_technology v2 --push"
    exit 1
fi

FILTER_PATH="filters/${FILTER_NAME}/${VERSION}"
SOURCE_DIR="${DISTILLERY_ROOT}/${FILTER_PATH}"
DEST_DIR="${NEXUSMIND_ROOT}/${FILTER_PATH}"
COMMON_SOURCE="${DISTILLERY_ROOT}/filters/common"
COMMON_DEST="${NEXUSMIND_ROOT}/filters/common"

# Validate source exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Filter not found: $SOURCE_DIR"
    exit 1
fi

echo "=== Deploying ${FILTER_NAME} ${VERSION} to NexusMind ==="
echo ""

# Step 1: Copy filter folder
echo "1. Copying filter: ${FILTER_PATH}"
mkdir -p "$DEST_DIR"
cp -r "${SOURCE_DIR}/"* "$DEST_DIR/"
echo "   Copied to: $DEST_DIR"

# Step 2: Copy common utilities
echo ""
echo "2. Copying common utilities: filters/common/"
mkdir -p "$COMMON_DEST"
cp -r "${COMMON_SOURCE}/"* "$COMMON_DEST/"
echo "   Copied to: $COMMON_DEST"

# Step 3: Git status in NexusMind
echo ""
echo "3. Changes in NexusMind:"
cd "$NEXUSMIND_ROOT"
git status --short

# Step 4: Commit
echo ""
echo "4. Committing changes..."
git add -A
COMMIT_MSG="Update ${FILTER_NAME} ${VERSION} from llm-distillery"
git commit -m "$COMMIT_MSG" || echo "   (No changes to commit)"

# Step 5: Push if requested
if [ "$PUSH_FLAG" == "--push" ]; then
    echo ""
    echo "5. Pushing to origin..."
    git push origin main

    echo ""
    echo "=== Deploy commands for servers ==="
    echo ""
    echo "# Sadalsuud:"
    echo "ssh user@sadalsuud \"cd ~/NexusMind && git pull origin main\""
    echo ""
    echo "# llm-distiller:"
    echo "ssh jeroen@llm-distiller \"cd ~/NexusMind && git pull origin main\""
else
    echo ""
    echo "5. Skipping push (use --push flag to push automatically)"
    echo ""
    echo "=== Next steps ==="
    echo ""
    echo "# Push to origin:"
    echo "cd $NEXUSMIND_ROOT && git push origin main"
    echo ""
    echo "# Then pull on servers:"
    echo "ssh user@sadalsuud \"cd ~/NexusMind && git pull origin main\""
    echo "ssh jeroen@llm-distiller \"cd ~/NexusMind && git pull origin main\""
fi

echo ""
echo "=== Done ==="
