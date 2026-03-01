# Way of Working

Operational principles and runbooks for this repository. For project identity, hard constraints, and task-triggered navigation, see `CLAUDE.md` (auto-loaded every session).

---

## Principles

These expand on the hard constraints in `CLAUDE.md` with operational detail.

### Oracle Output Discipline
Oracles output dimensional scores (0-10) only. Tier classification happens in postprocessing. This separates concerns: oracle scores dimensions, postfilter classifies tiers. Changing tier thresholds never requires re-labeling.

### Dimensional Regression, Not Classification
Student models learn continuous 0-10 scores per dimension, not category labels. This preserves nuance and enables flexible tier boundaries downstream.

### Screen + Merge for Needle-in-Haystack Filters (ADR-003)
Filters with low natural pass rates (uplifting, cultural-discovery) need enriched training data. Random articles provide negatives; pre-screened articles enrich positives. Merge both for balanced, representative training sets.

### Commerce Is the Only Universal Prefilter (ADR-004)
The commerce prefilter blocks obvious promotional content across all filters. Filter-specific noise is handled by the trained model, not by adding more rule-based prefilters.

### PEFT Adapters Stay in OLD Format (ADR-007)
Keep adapter files with `.lora_A.weight` keys (OLD format), not `.lora_A.default.weight` (NEW format). Hub loading via `PeftModel.from_pretrained()` requires OLD format. Local `inference.py` remaps as needed. **Never run `resave_adapter.py` before Hub upload.**

### Fit Calibration After Every Training Run (ADR-008)
Per-dimension isotonic regression corrects MSE score compression. Run `scripts/calibration/fit_calibration.py` on the val set. Commit `calibration.json` with the filter package. The base scorer auto-loads it.

### Oracle Prompts Follow the Scope-Check Pattern
Every production prompt uses a two-step structure: **Step 1: Scope Check** (is this article even about our topic? NO → all dimensions 0-2, stop), then **Step 2: Score Dimensions** (with inline critical filters before each scale table). This prevents fast models (Gemini Flash) from skipping top-level rules and scoring noise articles as if they were in-scope. The scope check also includes a noise detection checklist and an anti-hallucination rule requiring exact quotes as evidence.

### Filter Packages Are Self-Contained
Each filter lives in `filters/{name}/v{N}/` with everything needed: config.yaml, prompt, prefilter, scorer, inference modules, calibration, model weights. Independently deployable and auditable.

### Training Data Conventions
- Raw articles: `datasets/raw/master_dataset.jsonl`
- Oracle-scored: `datasets/scored/{filter}_{version}.jsonl`
- Training splits: `datasets/training/{filter}_{version}/` with 80/10/10 train/val/test
- Active learning (ADR-005): use production filter to find high-scoring candidates, oracle score, add to training data

### llm-distillery Creates, NexusMind Deploys
Filters are developed and trained here. Production scoring happens in NexusMind. Deploy via HuggingFace Hub (primary) and direct SCP to gpu-server (fallback).

---

## Runbook: Filter Development

The 9-phase lifecycle. See `docs/agents/filter-development-guide.md` for detailed checklists per phase, or `docs/guides/filter-creation-workflow.md` for quick steps using uplifting v6 as template.

| Phase | Goal | Key Command / Action |
|-------|------|---------------------|
| 1. Planning | Define dimensions, tiers, gatekeepers | Create `filters/{name}/v1/config.yaml` |
| 2. Architecture | Write oracle prompt with scope check + inline critical filters | Create `prompt-compressed.md` |
| 3. Validation | Calibrate oracle, verify scope check, check dimension redundancy (PCA/correlation on MEDIUM+ articles) | `python -m ground_truth.batch_scorer --filter ... --source ... --target-count 100` |
| 4. Prefilter | Rule-based noise filter | Create `prefilter.py` inheriting `base_prefilter.py` |
| 5. Training Data | Score 5K-10K articles | `python -m ground_truth.batch_scorer --filter ... --source datasets/raw/master_dataset.jsonl` |
| 6. Training | Distill to Gemma-3-1B | `PYTHONPATH=. python training/train.py --config ... --data-dir ... --output-dir ...` |
| 7. Calibration | Fit isotonic calibration | `PYTHONPATH=. python scripts/calibration/fit_calibration.py --filter ... --data-dir ... --test-data ...` |
| 8. Testing | Benchmark vs oracle, integration tests | `pytest tests/`, manual review of 30 articles |
| 9. Deployment | Upload to Hub, copy to NexusMind | See deployment runbook below |

### Common Commands

```bash
# Oracle scoring
python -m ground_truth.batch_scorer --filter filters/{name}/v{N} --source datasets/raw/master_dataset.jsonl

# Prepare training splits
python training/prepare_data.py --filter filters/{name}/v{N} --data-source datasets/scored/{name}_v{N}.jsonl

# Validate training data
python training/validate_training_data.py --data-dir datasets/training/{name}_v{N}

# Train on GPU server
ssh gpu-server
cd ~/llm-distillery
PYTHONPATH=. python training/train.py --config filters/{name}/v{N}/config.yaml \
    --data-dir datasets/training/{name}_v{N} --output-dir filters/{name}/v{N}/model

# Fit calibration (after training)
PYTHONPATH=. python scripts/calibration/fit_calibration.py \
    --filter filters/{name}/v{N} \
    --data-dir datasets/training/{name}_v{N} \
    --test-data datasets/training/{name}_v{N}/test.jsonl
```

---

## Runbook: Deployment to NexusMind

### 1. Upload to HuggingFace Hub

```bash
python scripts/deployment/upload_to_huggingface.py \
    --filter filters/{name}/v{N} \
    --repo-name jeergrvgreg/{name}-filter-v{N} \
    --token $HF_TOKEN --private
```

The upload script automatically verifies Hub loading after upload. If verification fails, check adapter format (must be OLD key format — see ADR-007).

### 2. Copy to gpu-server

```bash
scp -r filters/common/ gpu-server:~/NexusMind/filters/common/
scp -r filters/{name}/v{N}/ gpu-server:~/NexusMind/filters/{name}/v{N}/
ssh gpu-server "sudo systemctl restart nexusmind-scorer"
```

Use `scp`, not `rsync` (rsync has dup() errors on gpu-server).

### 3. Copy to NexusMind repo

Copy `filters/common/` and `filters/{name}/v{N}/` to the NexusMind repo. The filter loader auto-discovers from `filters/` directory.

### 4. Verify

```bash
# On gpu-server
ssh gpu-server "journalctl -u nexusmind-scorer -f"

# In NexusMind
python scripts/run_filters.py --filter {name} --hub --max-items 50
```

---

## Runbook: Datasets

**Pipeline:** Raw articles → Oracle scoring → Training splits

- **Raw**: `datasets/raw/master_dataset.jsonl` — consolidated article corpus
- **Scored**: `datasets/scored/{filter}_{version}.jsonl` — oracle-labeled articles
- **Training**: `datasets/training/{filter}_{version}/` — train.jsonl, val.jsonl, test.jsonl (80/10/10)

**Active learning** (ADR-005): Run production filter on new articles → collect high-scoring candidates → oracle score → add to training data → retrain.

**Naming**: Training data dirs use underscores: `sustainability_technology_v3`, `cultural-discovery_v3` (hyphenated filter names keep their hyphens).

---

## Documentation Practices

- **ADRs**: For architectural decisions. Use template at `docs/templates/`. Short version in `docs/adr/`, detailed in `docs/decisions/`.
- **TODO.md**: Update when starting or finishing work. Source of truth for current sprint.
- **gotcha-log.md**: Capture problems and fixes during work (in memory directory). Format: Problem → Root cause → Fix.
- **MEMORY.md**: Curate at end of session. Keep as index pointing to topic files.
- **Stale docs**: Add deprecation header pointing to replacement, don't delete.

---

*Last updated: 2026-03-01*
