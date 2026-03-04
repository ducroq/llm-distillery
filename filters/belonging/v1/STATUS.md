# Belonging Filter - Development Status

**Last Updated:** 2026-03-04
**Status:** Phase 5 Training Complete — Ready for Hub Upload

---

## Completed

- [x] Filter concept developed (Blue Zones inspiration, non-commercial dimensions)
- [x] Philosophical grounding documented (DEEP_ROOTS.md)
- [x] 6 dimensions defined with weights (config.yaml)
- [x] Oracle prompt created with examples (prompt-compressed.md)
- [x] Prefilter implemented with multilingual patterns (prefilter.py)
- [x] Prefilter tested: 10/10 unit tests passing
- [x] Prefilter run on corpus: 31% pass rate on 37,643 articles
- [x] 72 candidate articles extracted for oracle testing
- [x] RSS sources created in FluxusSource (rss_belonging.yaml, 15/24 working)
- [x] **Phase 1-2 reiteration** (2026-03-01):
  - [x] Prompt restructured with Step 1 scope check (matches production pattern)
  - [x] Inline critical filters added per dimension (before scale tables)
  - [x] Anti-hallucination rule: evidence must be EXACT QUOTE from article
  - [x] `community_fabric` added as gatekeeper dimension (threshold 3.0, cap ~3.42)
  - [x] Config updated: model → Gemma-3-1B, head_tail 256+256, target_samples → 10K
  - [x] `hybrid_inference` section added (placeholder, threshold TBD after training)
  - [x] `gatekeepers` section added to config
- [x] **Phase 3 oracle validation** (2026-03-01):
  - [x] 152 articles scored with Gemini Flash (72 candidates + 80 master dataset)
  - [x] Fixed missing article placeholder — root cause of hallucinated evidence
  - [x] Added `out_of_scope` content type for noise articles
  - [x] Added anti-hallucination warning after validation examples
  - [x] **Results:** 3 HIGH, 16 MEDIUM, 133 LOW — correct tier distribution
  - [x] **Hallucinations:** 0 (after fix)
  - [x] **Gatekeeper:** triggered on 88% of articles, correctly blocks noise
  - [x] **Evidence quality:** exact quotes from articles, "No evidence in article" for absent dimensions
  - [x] **Dimension independence confirmed:** slow_presence consistently lowest (max 6.0), purpose_beyond_self often highest
- [x] **Phase 5: Training** (2026-03-04):
  - [x] Training splits prepared: 5,894 train / 738 val / 738 test (stratified 80/10/10)
  - [x] Trained Gemma-3-1B + LoRA on gpu-server (3 epochs, batch 8, lr 2e-5, head+tail 256+256)
  - [x] Val MAE: 0.5343 (raw) → **0.4891** (calibrated, +8.3%)
  - [x] Isotonic calibration fitted on 738 val articles
  - [x] `inference.py` and `base_scorer.py` created
  - [x] Model and calibration synced to local
- [x] **Phase 4 batch labeling** (2026-03-02 — 2026-03-03):
  - [x] Scope candidates scored: 4,999 articles via Gemini Flash
  - [x] Random negatives scored: 2,500 articles via Gemini Flash (seed=42)
  - [x] Deduplicated and merged: 7,370 unique articles (129 cross-source duplicates removed)
  - [x] Data quality checks: schema, scores, gatekeeper, source diversity all verified
  - [x] **Result:** 802 MEDIUM+ articles (72 HIGH, 730 MEDIUM, 6,568 LOW)

---

## Oracle Validation Results (152 articles)

| Metric | Value |
|--------|-------|
| Articles scored | 152 |
| HIGH (≥7.0) | 3 (2%) |
| MEDIUM (4.0-7.0) | 16 (11%) |
| LOW (<4.0) | 133 (88%) |
| Gatekeeper triggered | 133 (88%) |
| Hallucinated evidence | 0 |
| Content type "out_of_scope" | 127 |

**MEDIUM+ dimension stats:**

| Dimension | Min | Max | Mean |
|-----------|-----|-----|------|
| intergenerational_bonds | 2.0 | 8.5 | 5.7 |
| community_fabric | 3.0 | 8.0 | 6.4 |
| reciprocal_care | 2.0 | 8.0 | 5.4 |
| rootedness | 2.0 | 8.0 | 6.3 |
| purpose_beyond_self | 4.0 | 9.0 | 6.7 |
| slow_presence | 3.0 | 6.0 | 4.2 |

**Note:** `slow_presence` caps at 6.0 in news corpus — unhurried rituals need literary/longform content to score higher. Not a prompt issue.

---

## Flagged: Dimension Redundancy (re-evaluate after batch labeling)

PCA on 152 articles: PC1=92.9% — dominated by the 88% noise articles scoring 0-2 everywhere.

**MEDIUM+ only (n=19):**

| Pair | r | Note |
|------|---|------|
| community_fabric ↔ rootedness | **0.845** | May be measuring the same thing — community is almost always placed |
| reciprocal_care ↔ purpose_beyond_self | 0.731 | Care as purpose expression |
| intergenerational_bonds ↔ community_fabric | -0.39 | Genuinely independent |

**Decision: Keep 6 dimensions for now.** n=19 is too small for structural changes. Re-evaluate when batch labeling produces 50+ MEDIUM+ articles. If CF↔Rt stays >0.80, consider merging into a single "rooted_community" dimension.

---

## Scope Probe Screening (2026-03-02)

Belonging is needle-in-haystack: 2% HIGH, 11% MEDIUM in oracle validation. Randomly sampling 10K articles for batch labeling would waste most of the API budget on obvious LOWs. Per ADR-003 (Screen + Merge), we trained a temporary logistic regression probe on E5-small embeddings to rank the master dataset by MEDIUM+ likelihood.

**Script:** `scripts/train_scope_probe.py`

**Approach:** Binary logistic regression (`class_weight='balanced'`) on 384-dim E5-small embeddings. 152 training articles, 19 MEDIUM+ vs 133 LOW.

| Metric | Value |
|--------|-------|
| LOOCV recall (MEDIUM+) | 14/19 (73.7%) |
| LOOCV precision | 14/31 (45.2%) |
| Missed MEDIUM+ articles | 5 (German, Samoan, Indian police, millet germplasm, state health collaboration) |
| Master dataset total | 178,462 |
| Prefilter blocked | 91,992 (51.5%) |
| Prefilter passed + scored | 86,470 |
| Top 5,000 score range | 0.4822 - 0.6556 (median 0.5077) |
| Estimated MEDIUM+ in top 5K | ~2,258 (from LOOCV precision) |

**Known bias:** Top candidates skew heavily toward The Better India (8/10 top articles). The probe learned source/style as a proxy — expected with n=19 positives. Manual inspection of top-10: ~2-3 genuine belonging, rest are "uplifting Indian community stories" without actual belonging dimensions (intergenerational bonds, rootedness, social fabric). Bottom-10 of top-5000: clear noise (medical studies, CinemaSins, link roundups).

**Output:** `datasets/belonging/scope_candidates.jsonl` (5,000 articles with `_probe_score` field)

**Verdict:** Good enough as a screening tool. 73.7% recall means we capture most positives; the oracle will handle precision during batch labeling.

---

## Phase 4: Batch Labeling (2026-03-02 — 2026-03-03)

Two-pass scoring per ADR-003 (Screen + Merge):

1. **Scope candidates** — 4,999 probe-enriched articles scored with Gemini Flash
2. **Random negatives** — 2,500 random articles from master dataset (seed=42)

Scope candidates provide near-miss negatives (articles that looked like belonging but weren't).
Random negatives provide obvious negatives (tech, science, finance) the model needs to learn to reject trivially.

| Source | Articles | HIGH | MEDIUM | LOW |
|--------|----------|------|--------|-----|
| Scope candidates | 4,999 | 72 (1.4%) | 722 (14.4%) | 4,205 (84.1%) |
| Random negatives | 2,500 | 1 (0.0%) | 36 (1.4%) | 2,463 (98.5%) |
| **Merged (deduplicated)** | **7,370** | **72 (1.0%)** | **730 (9.9%)** | **6,568 (89.1%)** |

129 duplicates removed (articles appearing in both scope candidates and random negatives; scope version kept).

**Data quality checks:**
- Schema: all 7,370 articles have all 6 dimensions with score + evidence fields
- Scores: all in 0-10 range, no nulls
- Gatekeeper: 12 articles with community_fabric < 3.0 but weighted avg > 3.42 — acceptable, gatekeeper cap is applied at inference time in `filter_base_scorer.py`, not in training data
- Content type: minor oracle label drift (non-canonical types like "corporate", "networking") — not a training concern since model learns dimension scores, not content types

**Output:** `datasets/belonging/belonging_all_scored.jsonl` (7,370 articles)

---

## Training Results (Phase 5 — 2026-03-04)

| Metric | Value |
|--------|-------|
| Training articles | 5,894 |
| Val articles | 738 |
| Test articles | 738 |
| Val MAE (raw) | 0.5343 |
| Val MAE (calibrated) | **0.4891** |
| Calibration improvement | +8.3% |
| Base model | Gemma-3-1B |
| LoRA trainable params | 13.05M / 1.01B (1.29%) |
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Head+tail | 256+256 tokens |

**Per-dimension calibrated MAE:**

| Dimension | Raw MAE | Calibrated MAE |
|-----------|---------|----------------|
| intergenerational_bonds | 0.522 | 0.468 |
| community_fabric | 0.590 | 0.557 |
| reciprocal_care | 0.508 | 0.458 |
| rootedness | 0.531 | 0.485 |
| purpose_beyond_self | 0.596 | 0.555 |
| slow_presence | 0.454 | 0.412 |

**Model:** `filters/belonging/v1/model/` (adapter_model.safetensors + tokenizer)
**Calibration:** `filters/belonging/v1/calibration.json` (isotonic regression, 738 val articles)

---

## CF↔Rootedness Re-evaluation (2026-03-04)

Phase 3 flagged community_fabric↔rootedness at r=0.845 (n=19). Re-evaluated with full batch data:

| Sample | n | r |
|--------|---|---|
| Phase 3 (small sample) | 19 | 0.845 |
| **Phase 5 MEDIUM+ (full data)** | **801** | **0.470** |
| All data (inflated by noise floor) | 7,370 | 0.916 |

**Verdict:** r=0.845 was a small-sample artifact. At n=801, CF↔rootedness is moderate (0.470) — not redundant. Only pair above 0.5 is community_fabric↔purpose_beyond_self (0.564). PCA shows 4 meaningful components. **Keep all 6 dimensions.**

## Cross-Filter Orthogonality (2026-03-04)

| Filter pair | r | n | Status |
|-------------|---|---|--------|
| belonging ↔ uplifting | 0.508 | 1,866 | Moderate overlap |
| belonging ↔ cultural-discovery | 0.547 | 1,021 | Moderate overlap |
| belonging ↔ sustainability_technology | 0.119 | 498 | Orthogonal |
| uplifting ↔ cultural-discovery | 0.376 | 1,958 | Weak |

498 belonging MEDIUM+ articles are truly exclusive (LOW on all other filters). Belonging captures distinct content: intergenerational bonds, rootedness, mutual care in ordinary life.

**Decision:** Deploy belonging as separate tab in ovr.news (ADR-009). Add filters first, reduce later.

---

## Next Steps

1. **Hub upload** — Upload model to HuggingFace Hub (private)
2. **Deploy to NexusMind** — Copy filter package, restart scorer service
3. **Add belonging tab to ovr.news** — New tab with NexusMind integration

---

## Key Files

| File | Purpose |
|------|---------|
| `config.yaml` | Dimensions, weights, content type caps, gatekeeper |
| `prompt-compressed.md` | Oracle prompt for LLM scoring |
| `prefilter.py` | Rule-based blocker (saves API costs) |
| `DEEP_ROOTS.md` | Philosophical grounding (Weil, Tönnies, etc.) |
| `calibrations/candidates/belonging_candidates.jsonl` | 72 candidate articles |
| `../../scripts/train_scope_probe.py` | Scope probe: train LR + screen master dataset |
| `../../datasets/belonging/scope_candidates.jsonl` | 5,000 top probe-ranked candidates for batch labeling |
| `../../datasets/belonging/belonging_all_scored.jsonl` | 7,370 merged, deduplicated training articles |
| `../../datasets/belonging/belonging/` | Scored scope candidates (102 batches) |
| `../../datasets/belonging/random_negatives/belonging/` | Scored random negatives (50 batches) |

---

## Dimensions (weights)

1. **intergenerational_bonds** (25%) — Multi-generation connection
2. **community_fabric** (25%) — Local social infrastructure **[GATEKEEPER]**
3. **rootedness** (15%) — Place attachment, staying put
4. **purpose_beyond_self** (15%) — Transcendent meaning
5. **slow_presence** (10%) — Unhurried togetherness
6. **reciprocal_care** (10%) — Mutual support networks

---

## Notes

- Prefilter pass rate (31%) is lower than typical filters because corpus is mostly tech/news
- RSS sources in FluxusSource will improve future data quality for this filter
- Gatekeeper on community_fabric prevents wellness/longevity articles from scoring HIGH without actual community evidence
- Phase 1-2 reiteration aligned prompt and config with uplifting v6 / cultural-discovery v4 production standards
- Master dataset prefilter blocks 52% of articles (91,992/178,462) — saves API costs during batch labeling
