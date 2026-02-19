# Uplifting Filter v6

**Date:** 2026-02-19
**Status:** Trained, evaluated, ready for deployment
**Base model:** Gemma-3-1B (was Qwen2.5-1.5B in v5)

## What Changed from v5

Three improvements over v5:

1. **Gemma-3-1B base model** — Smaller (1B vs 1.5B params), faster (18.9ms vs 21.5ms), and more accurate
2. **Label correction for crime articles** — 57 mislabeled articles fixed in training data
3. **Active learning data enrichment** — 495 additional MEDIUM-tier articles from production

## Results

| Metric | v5 (Qwen-1.5B) | v6 (Gemma-3-1B) | Change |
|--------|----------------|-----------------|--------|
| Val MAE | 0.688 | **0.673** | -2.2% |
| Weighted MAE | — | **0.509** | — |
| Tier Accuracy | 85.4% | **85.2%** | ~same |
| Inference Speed | 21.5 ms | **18.9 ms** | 12% faster |
| Parameters | 1.5B | **1.0B** | 33% smaller |
| Trainable (LoRA) | 18.5M | **13.1M** | 29% fewer |

### Per-Dimension MAE (val set)

| Dimension | v6 MAE |
|-----------|--------|
| human_wellbeing_impact | 0.681 |
| social_cohesion_impact | 0.678 |
| justice_rights_impact | 0.632 |
| evidence_level | 0.638 |
| benefit_distribution | 0.765 |
| change_durability | 0.645 |

### Tier Accuracy Breakdown

| Tier | Accuracy | Count |
|------|----------|-------|
| LOW | 91.9% | 615/669 |
| MEDIUM | 73.6% | 279/379 |
| HIGH | 0.0% | 0/1 |

HIGH-tier accuracy unmeasurable — only 1 HIGH article in val set. This is the known data imbalance issue (see "Known Limitations" below).

## Data Sculpting

v6 training data was improved through two techniques applied to the v5 base dataset:

### 1. Active Learning (MEDIUM-tier enrichment)

Used the production v5 model to identify articles the model found interesting, then had the oracle (Gemini Flash) score them:

```
Production output → Filter by model prediction ≥ 5.0 → Oracle score → Merge with v5 data
```

| Step | Count |
|------|-------|
| Production MEDIUM-tier articles | 4,531 |
| Filtered by predicted ≥ 5.0 | 1,355 |
| After manual curation (removed commerce) | 496 |
| After oracle scoring | 495 |

All 495 articles scored in MEDIUM tier (5.52–6.93). This enriched the training data's middle range, where the model needs the most signal to distinguish "somewhat uplifting" from "moderately uplifting."

**Key insight:** The model's own predictions are well-calibrated (predicted 5.5, oracle scored 5.86), making it an effective selector for hard examples.

### 2. Label Correction (crime article caps)

The oracle systematically overscored individual crime/sentencing articles because:
- `justice_rights_impact` scored 7-8 (conviction = "accountability achieved")
- `evidence_level` scored high (court rulings are well-documented)
- But a single criminal getting convicted is not systemic change or solutions journalism

**Process:**
1. Keyword search for crime/sentencing terms in all MEDIUM+ articles (≥ 4.0 weighted avg)
2. Found 134 candidates (3.3% of MEDIUM+ articles)
3. Manually reviewed each: is this an individual crime case, or systemic reform/landmark ruling?
4. Marked 57 articles as individual crime cases
5. Scaled their dimension scores proportionally to cap weighted average at 2.0 (LOW tier)

This is a targeted fix — not re-scoring with the oracle (expensive), but applying the same logic the oracle *should* have applied: "individual crime case → not uplifting." The correction moves these articles from MEDIUM to LOW, where they belong.

**Examples of corrected articles:**
- "Hogere straf opgelegd in hoger beroep zaak" (6.3 → 2.0)
- "Alice Guo gets life term for human trafficking" (6.4 → 2.0)
- "Medisch specialist mag na misbruik niet meer werken als arts" (7.15 → 2.0)

**Examples preserved as legitimate:**
- "France's rehabilitation program turns prisoners into farmers" (kept at 6.3)
- "Meta lawyers tried to block internal research showing teen harm" (kept at 6.0)
- "Blue states launch new legal attack on gun industry immunity" (kept at 5.8)

Full review log: `datasets/training/uplifting_v6/crime_review.tsv`
Applied changes: `datasets/training/uplifting_v6/crime_caps_applied.json`

### Final Training Data

| Dataset | Articles | LOW (<4) | MEDIUM (4-7) | HIGH (≥7) |
|---------|----------|----------|--------------|-----------|
| v5 | 10,000 | 68.4% | 31.5% | 0.1% (7) |
| v6 (before correction) | 10,495 | 65.3% | 34.6% | 0.1% (8) |
| v6 (after correction) | 10,495 | ~65.8% | ~34.1% | 0.1% (8) |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `google/gemma-3-1b-pt` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Max length | 512 |
| Head+tail | 256 + 256 tokens |
| Warmup steps | 500 |
| Training time | ~60 minutes (RTX 4080) |

### Training Progression

| Epoch | Train MAE | Val MAE | Val Loss |
|-------|-----------|---------|----------|
| 1 | 1.496 | 0.784 | 1.036 |
| 2 | 0.727 | 0.698 | 0.812 |
| 3 | 0.640 | **0.673** | 0.767 |

Best model saved at epoch 3.

## Usage

### Standard Inference (single article)

```python
from filters.uplifting.v6.inference import UpliftingScorer

scorer = UpliftingScorer()  # loads from local model/ directory
result = scorer.score_article({"title": "...", "content": "..."})
# result["tier"] -> "high", "medium", or "low"
# result["weighted_average"] -> 0.0-10.0
# result["scores"] -> per-dimension scores
```

### HuggingFace Hub Inference

```python
from filters.uplifting.v6.inference_hub import UpliftingScorerHub

scorer = UpliftingScorerHub(
    repo_id="jeergrvgreg/uplifting-filter-v6",
    token="hf_...",  # required for private repos
)
```

### Hybrid Inference (two-stage pipeline)

The hybrid scorer uses a fast embedding probe (Stage 1, ~1.3ms) to screen out obvious LOW articles before running the full model (Stage 2, ~19ms). Articles with a probe estimate below the threshold skip Stage 2 entirely.

```python
from filters.uplifting.v6.inference_hybrid import UpliftingHybridScorer

scorer = UpliftingHybridScorer()  # uses default threshold from config.yaml
result = scorer.score_article(article)
# result["stage_used"] -> "stage1_low" (skipped model) or "stage2" (full scoring)
```

#### Configuring the Stage 1 threshold

The threshold controls the speed/accuracy tradeoff. Lower thresholds are more conservative (fewer false negatives, less speedup). Higher thresholds are more aggressive (more speedup, more missed MEDIUM+ articles).

The threshold can be set at three levels, each overriding the previous:

1. **`config.yaml`** (default for the filter version):
   ```yaml
   hybrid_inference:
     stage1:
       threshold: 2.25
   ```

2. **Constructor argument** (per instance):
   ```python
   scorer = UpliftingHybridScorer(threshold=1.75)  # more conservative
   scorer = UpliftingHybridScorer(threshold=3.00)  # more aggressive
   ```

3. **CLI flag** (for batch runs):
   ```bash
   python filters/uplifting/v6/inference_hybrid.py --threshold 1.75 --input articles.jsonl
   ```

#### Threshold guidelines

| Threshold | FN Rate | Use case |
|-----------|---------|----------|
| 1.75 | 0.0% | Maximum safety — no MEDIUM+ articles missed. Minimal speedup. |
| **2.25** | **0.5%** | **Default.** Good balance for single-filter deployments. |
| 2.50 | 1.8% | Acceptable for high-volume screening where some loss is tolerable. |
| 3.00+ | >3% | Not recommended — too many false negatives. |

**Multi-filter deployments:** When running multiple filters on the same article (e.g., NexusMind), false negative rates compound. With 4 filters each at 0.5% FN, the probability of losing an article to *any* filter is ~2%. Consider using a lower threshold (1.75) in multi-filter pipelines to keep compound loss acceptable, or accept the speedup tradeoff if throughput is the priority.

#### Recalibrating the threshold

If the production article distribution changes significantly, recalibrate:

```bash
python evaluation/calibrate_hybrid_threshold.py \
    --filter uplifting --version v6 \
    --val-data datasets/training/uplifting_v6/val.jsonl \
    --probe-path filters/uplifting/v6/probe/embedding_probe_e5small.pkl \
    --embedding-model intfloat/multilingual-e5-small \
    --use-ground-truth
```

## Known Limitations

- **HIGH-tier blind spot** — Only 8 HIGH articles (0.08%) in 10,495. Model cannot learn the upper score range. Active learning enriched MEDIUM but found zero HIGH. Targeted collection from positive news sources needed for v7.
- **Crime fix applied to data, not prompt** — The oracle prompt still lacks the individual_crime content-type cap. New oracle scoring would still overscore crime articles. Fix is in `config.yaml` but not in `prompt-compressed.md`. Apply to prompt for v7.

## Files

```
filters/uplifting/v6/
├── README.md                    # This file
├── PLAN.md                      # Original planning document
├── config.yaml                  # Filter config (updated for v6)
├── training_metadata.json       # Training hyperparameters and results
├── training_history.json        # Per-epoch metrics
└── model/
    ├── adapter_config.json      # LoRA configuration
    ├── adapter_model.safetensors # Trained weights (50 MB)
    ├── tokenizer.json           # Gemma tokenizer
    ├── tokenizer_config.json
    └── README.md

datasets/training/uplifting_v6/
├── train.jsonl                  # 8,396 articles (corrected)
├── val.jsonl                    # 1,049 articles (corrected)
├── test.jsonl                   # 1,050 articles (corrected)
├── crime_review.tsv             # Manual review of 134 crime candidates
└── crime_caps_applied.json      # Log of 57 score corrections
```

---

*Created: 2026-02-19*
