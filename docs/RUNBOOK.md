# Runbook

Operational how-to for deployment, training, and scoring. For project identity and hard constraints, see `CLAUDE.md`. For architectural decisions, see `docs/adr/README.md`.

---

## Deployment to NexusMind

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

### 3. Copy to sadalsuud

```bash
scp -r filters/common/ sadalsuud:~/local_dev/NexusMind/filters/common/
scp -r filters/{name}/v{N}/ sadalsuud:~/local_dev/NexusMind/filters/{name}/v{N}/
```

sadalsuud uses Hub inference — do not include `model/` directories in the SCP payload. venv at `~/local_dev/NexusMind/venv/`.

After calibration updates, redeploy `config.yaml` and `calibration.json` to both gpu-server and sadalsuud.

### 4. Verify

```bash
# On gpu-server
ssh gpu-server "journalctl -u nexusmind-scorer -f"

# In NexusMind
python scripts/run_filters.py --filter {name} --hub --max-items 50
```

---

## Oracle Scoring

```bash
# Validation run (~100 articles, Phase 3)
python -m ground_truth.batch_scorer \
    --filter filters/{name}/v{N} \
    --source datasets/raw/master_dataset.jsonl --target-count 100

# Score articles (full run, Phase 5)
python -m ground_truth.batch_scorer \
    --filter filters/{name}/v{N} \
    --source datasets/raw/master_dataset.jsonl

# Multi-run averaging (for prompt-sensitive filters like thriving)
python scripts/oracle/average_oracle_runs.py \
    --runs datasets/scored/{name}_v{N}_run1.jsonl datasets/scored/{name}_v{N}_run2.jsonl datasets/scored/{name}_v{N}_run3.jsonl \
    --output datasets/scored/{name}_v{N}.jsonl
```

---

## Training

### Prepare data

```bash
python training/prepare_data.py \
    --filter filters/{name}/v{N} \
    --data-source datasets/scored/{name}_v{N}.jsonl

# Validate splits
python training/validate_training_data.py \
    --data-dir datasets/training/{name}_v{N}
```

### Train on GPU server

```bash
# 1. Copy training data to gpu-server first
scp -r datasets/training/{name}_v{N}/ gpu-server:~/llm-distillery/datasets/training/

# 2. SSH and train
ssh gpu-server
cd ~/llm-distillery
source ~/gpu-server/nexusmind-scorer/venv/bin/activate
export PYTHONPATH=.
export HF_HUB_OFFLINE=1

python training/train.py \
    --config filters/{name}/v{N}/config.yaml \
    --data-dir datasets/training/{name}_v{N} \
    --output-dir filters/{name}/v{N}/model
```

### Fit calibration (after training)

```bash
PYTHONPATH=. python scripts/calibration/fit_calibration.py \
    --filter filters/{name}/v{N} \
    --data-dir datasets/training/{name}_v{N} \
    --test-data datasets/training/{name}_v{N}/test.jsonl
```

Calibration writes `calibration.json` and `score_scale_factor` to config.yaml. Commit both with the filter package.

---

## Filter Development Lifecycle

9-phase process. See `docs/agents/filter-development-guide.md` for detailed checklists, or `docs/guides/filter-creation-workflow.md` for quick steps.

| Phase | Goal | Key Action |
|-------|------|------------|
| 1. Planning | Define dimensions, tiers, gatekeepers | Create `filters/{name}/v1/config.yaml` |
| 2. Architecture | Write oracle prompt with scope check + inline critical filters | Create `prompt-compressed.md` |
| 3. Validation | Calibrate oracle on ~100 articles | Small batch scoring run |
| 4. Prefilter | Rule-based noise filter | Create `prefilter.py` inheriting `base_prefilter.py` |
| 5. Training Data | Score 5K-10K articles | Full batch scoring run |
| 6. Training | Distill to Gemma-3-1B + LoRA | Train on gpu-server |
| 7. Calibration | Fit isotonic calibration | `fit_calibration.py` on val set |
| 8. Testing | Benchmark vs oracle | `pytest tests/`, manual review of 30 articles |
| 9. Deployment | Upload to Hub, copy to NexusMind | See deployment section above |

---

## Dataset Conventions

- **Raw**: `datasets/raw/master_dataset.jsonl` — consolidated article corpus
- **Scored**: `datasets/scored/{filter}_{version}.jsonl` — oracle-labeled articles
- **Training**: `datasets/training/{filter}_{version}/` — train.jsonl, val.jsonl, test.jsonl (80/10/10)
- **Naming**: Training data dirs use underscores (`sustainability_technology_v3`), but hyphenated filter names keep hyphens (`cultural-discovery_v3`)
- **Active learning** (ADR-005): Run production filter on new articles → collect high-scoring candidates → oracle score → add to training data → retrain
- **Scored JSONL keys**: Use `analysis_field_name()` from `ground_truth/__init__.py` for consistent field naming

---

*Last updated: 2026-03-28*
