# Sandbox Directory

**Purpose:** Safe space for experimentation without polluting the repository.

**Status:** This entire directory is `.gitignore`'d - nothing here will be committed.

---

## What Goes Here

### ✅ Safe to Add

- **Prototypes** - Test new ideas before implementing properly
- **One-off analyses** - Quick data exploration scripts
- **Debugging code** - Isolated test files for troubleshooting
- **Training experiments** - Try different hyperparameters
- **Failed experiments** - Keep notes on what didn't work
- **Messy notebooks** - Jupyter notebooks during exploration phase
- **Temporary data** - Small test datasets
- **Draft scripts** - Work in progress before moving to `scripts/`

### ❌ Don't Put Here (Use Proper Locations Instead)

- **Production scripts** → `scripts/`
- **Filter implementations** → `filters/`
- **Documented experiments** → `experiments/` (tracked)
- **Important analysis results** → `reports/`
- **Working model checkpoints** → `filters/*/model/`
- **Training data** → `datasets/`

---

## Guidelines

1. **No secrets** - Even though it's gitignored, don't put API keys here
2. **Self-contained** - Code here can import from main repo but shouldn't be imported by main code
3. **Document learnings** - If you discover something useful, document it properly in `/docs` or `/reports`
4. **Clean up occasionally** - This isn't permanent storage, delete old experiments
5. **Use subdirectories** - Organize by topic or date for sanity

---

## Recommended Structure

```
sandbox/
├── README.md                    # This file
├── 2025-11-10_model_testing/   # Dated experiments
│   ├── test_quantization.py
│   └── notes.md
├── hyperparameter_search/       # Organized by topic
│   ├── learning_rate_tests.py
│   └── batch_size_experiments.ipynb
├── debugging/                   # Debugging sessions
│   ├── reproduce_oom_error.py
│   └── test_data_loading.py
└── failed_approaches/           # Document what didn't work
    ├── attempted_4bit_quant.md
    └── why_unsloth_failed.md
```

---

## Related Directories

- **`sandbox/`** - You are here (general experimentation)
- **`scratch/`** - Very temporary work, even messier than sandbox
- **`playground/`** - Testing new libraries/tools
- **`experiments/local/`** - Local training experiments
- **`notebooks/exploratory/`** - Messy Jupyter notebooks

All of these are gitignored for your peace of mind.

---

## Example Use Cases

**Quick model test:**
```bash
# sandbox/test_1.5b_model.py
from transformers import AutoModel
model = AutoModel.from_pretrained("Qwen/Qwen2.5-1.5B")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Try different prefilter threshold:**
```bash
# sandbox/prefilter_tuning.py
# Test different thresholds without modifying production code
import sys
sys.path.append('..')
from filters.sustainability_tech_deployment.v1 import prefilter

# Test various thresholds...
```

**Failed experiment documentation:**
```markdown
# sandbox/failed_approaches/2025-11-10-quantization-attempt.md

## What I Tried
8-bit quantization of Qwen2.5-7B-Instruct

## Why It Failed
Still OOM on 16GB GPU even with batch size 1

## What I Learned
Quantization reduces trainable params but base model still loads at 15GB

## Better Approach
Use smaller model (1.5B) without quantization - see ADR
```

---

**Remember:** This is YOUR workspace. Experiment freely!
