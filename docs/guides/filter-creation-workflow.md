# Filter Creation Workflow

Practical step-by-step for creating a new production filter. Uses **uplifting v6** as the reference template (`filters/uplifting/v6/`).

For detailed validation checklists per phase, see `docs/agents/filter-development-guide.md`.

---

## Reference: uplifting v6 file structure

```
filters/uplifting/v6/
  config.yaml              # Dimensions, weights, tiers, preprocessing, training config
  prompt-compressed.md     # Oracle prompt for ground truth scoring
  prefilter.py             # Fast rule-based noise filter
  base_scorer.py           # Shared scoring logic (calibration, preprocessing, tiers)
  inference.py             # Local inference (loads LoRA adapter from model/)
  inference_hub.py         # HuggingFace Hub inference
  inference_hybrid.py      # Two-stage hybrid inference (probe + model)
  calibration.json         # Post-hoc isotonic regression (fitted on val set)
  README.md                # Results, data sculpting, known limitations
  model/                   # LoRA adapter + tokenizer config
  probe/                   # e5-small MLP probe for hybrid Stage 1
  training_history.json    # Loss curves from training
  training_metadata.json   # Hyperparameters, dataset info
```

---

## Workflow

### 1. Define the filter

Create `filters/<name>/v1/config.yaml`:
- 6-8 scoring dimensions (0-10 scale each)
- Weights summing to 1.0
- Tier thresholds (high/medium/low)
- Gatekeepers (hard dimension thresholds that cap overall score)
- Content type caps (optional domain-specific rules)

Reference: `filters/uplifting/v6/config.yaml`

### 2. Write the oracle prompt

Create `filters/<name>/v1/prompt-compressed.md`:
- Defines how Gemini Flash scores each dimension
- Includes scoring rubrics, contrastive examples, content type handling
- Outputs dimensional scores ONLY (no tier classification)

### 3. Validate dimensions

Score ~50-100 articles with the oracle prompt and check:
- Dimension correlations (PCA/redundancy analysis)
- Oracle calibration (are scores distributed as expected?)
- Gatekeeper effectiveness
- See `docs/agents/filter-development-guide.md` Phase 3

### 4. Build the prefilter

Create `filters/<name>/v1/prefilter.py`:
- Inherits from `filters/common/base_prefilter.py`
- Rule-based filtering (keyword matching, source blocking)
- Target: <10% false negative rate on relevant content
- Commerce prefilter included automatically via base class

### 5. Generate training data

Score 5,000-10,000 articles through the oracle:
```bash
python -m ground_truth.batch_scorer \
    --filter filters/<name>/v1 \
    --source datasets/raw/master_dataset.jsonl
```

For needle-in-haystack filters (low pass rate), use screen+merge strategy (ADR-003):
- Random articles provide negatives
- Pre-screened articles enrich positives

### 6. Prepare training splits

```bash
python training/prepare_data.py \
    --filter filters/<name>/v1 \
    --data-source datasets/scored/<name>_v1.jsonl
```

Produces 80/10/10 train/val/test splits in `datasets/training/<name>_v1/`.

### 7. Train the model

Train Gemma-3-1B with LoRA on GPU server:
```bash
PYTHONPATH=. python training/train.py \
    --config filters/<name>/v1/config.yaml \
    --data-dir datasets/training/<name>_v1 \
    --output-dir filters/<name>/v1/model
```

Key settings (see `filters/uplifting/v6/training_metadata.json`):
- Base model: `google/gemma-3-1b-pt`
- LoRA rank 16, alpha 32
- Max 512 tokens with head+tail preprocessing (256+256)
- Learning rate 2e-4, batch size 16, 3 epochs

**Important**: Use `load_base_model_for_seq_cls()` from `filters/common/model_loading.py` instead of `AutoModelForSequenceClassification` directly (Gemma-3-1B compatibility).

### 8. Write inference code

Copy from uplifting v6 and adapt:

- **`base_scorer.py`** — Change class name, dimension names/weights/tiers, filter metadata. The calibration loading, preprocessing, and score processing logic stays the same.
- **`inference.py`** — Change class name, model path. Model loading pattern stays the same.
- **`inference_hub.py`** — Change class name, Hub repo ID.
- **`inference_hybrid.py`** — Change class name, probe path.
- **`prefilter.py`** — Filter-specific rules.

All scorers inherit from the base class which provides:
- Calibration loading and application (`calibration.json`)
- Head+tail text preprocessing
- Score clamping (0-10), weighted average, gatekeeper logic
- Tier assignment
- Batch inference

### 9. Fit score calibration

After training, fit isotonic regression on the validation set:
```bash
PYTHONPATH=. python scripts/calibration/fit_calibration.py \
    --filter filters/<name>/v1 \
    --data-dir datasets/training/<name>_v1

# Verify on held-out test set:
PYTHONPATH=. python scripts/calibration/fit_calibration.py \
    --filter filters/<name>/v1 \
    --data-dir datasets/training/<name>_v1 \
    --test-data datasets/training/<name>_v1/test.jsonl
```

This generates `calibration.json` in the filter directory. The base scorer picks it up automatically at inference time. See ADR-008.

### 10. Train hybrid probe (optional)

Train an e5-small MLP probe for Stage 1 screening:
```bash
PYTHONPATH=. python research/embedding_vs_finetuning/train_probe.py \
    --filter <name> --version v1 \
    --data-dir datasets/training/<name>_v1
```

Calibrate the threshold using `evaluation/calibrate_hybrid_threshold.py`. Store probe in `filters/<name>/v1/probe/`.

### 11. Deploy to HuggingFace Hub

```bash
PYTHONPATH=. python scripts/deployment/upload_to_huggingface.py \
    --filter-dir filters/<name>/v1 \
    --repo-id <org>/<name>-filter-v1
```

**Do NOT run `resave_adapter.py` before upload** — it changes key format and breaks Hub loading (ADR-007).

### 12. Document and verify

- Write `README.md` with results, per-dimension MAE, tier accuracy, known limitations
- Run interactive demo: `PYTHONPATH=. python -m filters.<name>.v1.inference`
- Verify Hub loading: `PYTHONPATH=. python -m filters.<name>.v1.inference_hub`

---

## Shared libraries

These live in `filters/common/` and are used by all filters:

| Module | Purpose |
|--------|---------|
| `model_loading.py` | `load_base_model_for_seq_cls()` — Gemma-3-1B compatibility |
| `score_calibration.py` | `fit_calibration()`, `apply_calibration()` — isotonic regression |
| `embedding_stage.py` | e5-small embedding + MLP probe for hybrid Stage 1 |
| `hybrid_scorer.py` | Two-stage inference orchestration |
| `base_prefilter.py` | Commerce prefilter + threading safety |
| `text_preprocessing.py` | Head+tail token extraction |

---

## Key decisions

- **Oracle outputs scores only** — Tier classification in postfilter (flexible thresholds)
- **Gemma-3-1B** — Default student model (replaced Qwen2.5-1.5B, Feb 2026)
- **LoRA adapters in OLD format** — `.lora_A.weight` (not `.lora_A.default.weight`) for Hub compatibility
- **Calibration before clamping** — Pipeline: raw logits -> calibrate -> clamp 0-10 -> weighted avg -> gatekeeper -> tier
- **Screen+merge for rare-positive filters** — ADR-003
- **Commerce is the only universal prefilter** — ADR-004

---

*Last updated: 2026-02-19*
