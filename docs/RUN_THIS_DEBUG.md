# Debug Score Layer Loading - Run This

## On Remote GPU Machine

```bash
cd ~/llm-distillery

# Pull latest diagnostic changes
git pull

# Run investment-risk v4 benchmark with DEBUG output
python scripts/training/benchmark_test_set.py \
    --filter filters/investment-risk/v4 \
    --data-dir datasets/training/investment_risk_v4 \
    --batch-size 16
```

## What This Will Show

The debug output will reveal:

1. **Score layer keys in saved weights** - The exact path where `score.weight` is stored in `adapter_model.safetensors`
2. **Score layer keys in model structure** - Where PEFT expects the score layer to be
3. **All modules with 'score'** - Full module hierarchy

This will tell us exactly why `base_model.model.score.weight` shows as "unexpected key" and how to remap it correctly.

## Expected Debug Output

**IMPORTANT**: You MUST see these DEBUG sections in the output. If they don't appear, the diagnostic code isn't running.

Look for sections like:
```
DEBUG: Score layer keys in saved weights:
  base_model.model.score.weight: shape torch.Size([8, 1536])
  base_model.model.score.bias: shape torch.Size([8])

DEBUG: Score layer keys in model structure:
  base_model.model.model.score.weight
  base_model.model.model.score.bias
```

The key mismatch will be obvious once we see both paths side-by-side.

**If you don't see these DEBUG sections**: Stop and verify you pulled the latest changes (commit 03f7332 or later).

## After Running

Copy the full output (especially the DEBUG sections) and paste back here so I can implement the fix.
