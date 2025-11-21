#!/bin/bash
# Quick verification that diagnostic code is present

echo "Checking for diagnostic code in benchmark_test_set.py..."
echo ""

if grep -q "DEBUG: Score layer keys in saved weights" scripts/training/benchmark_test_set.py; then
    echo "✓ Found: DEBUG section for saved weights"
else
    echo "✗ MISSING: DEBUG section for saved weights"
    echo "  You need to git pull to get commit 03f7332"
    exit 1
fi

if grep -q "DEBUG: Score layer keys in model structure" scripts/training/benchmark_test_set.py; then
    echo "✓ Found: DEBUG section for model structure"
else
    echo "✗ MISSING: DEBUG section for model structure"
    echo "  You need to git pull to get commit 03f7332"
    exit 1
fi

if grep -q "DEBUG: All module names in model" scripts/training/benchmark_test_set.py; then
    echo "✓ Found: DEBUG section for all modules"
else
    echo "✗ MISSING: DEBUG section for all modules"
    echo "  You need to git pull to get commit 03f7332"
    exit 1
fi

echo ""
echo "✓ All diagnostic code present!"
echo ""
echo "You can now run the benchmark:"
echo "python scripts/training/benchmark_test_set.py \\"
echo "    --filter filters/investment-risk/v4 \\"
echo "    --data-dir datasets/training/investment_risk_v4 \\"
echo "    --batch-size 16"
