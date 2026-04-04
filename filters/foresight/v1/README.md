# Foresight Filter

**Version**: 1.0
**Status**: Ready for deployment — calibrated test MAE 0.75, on par with cultural-discovery and sustainability_tech
**Evolved from**: signs_of_wisdom concept (2025-12-28)
**Philosophy**: "Foresight is a structural evaluation of a decision, not an emotional response to an event"
**Purpose**: Identify decisions that demonstrate long-term thinking — choices made for generations ahead, not for the next quarter or election cycle.

**ovr.news tab:** Foresight
**Bias corrected:** Short-termism — news rewards the immediate, this lens finds decisions made for generations ahead.

## Core Insight

This filter measures **HOW DECISIONS WERE MADE** (process quality), not just what outcomes occurred.

A foresighted decision might not yet have visible results. A good outcome might have come from luck, not foresight.

Education, knowledge transfer, and capacity building are core expressions of foresight — investing in people who will carry knowledge forward.

---

## Key Dimensions (6)

| # | Dimension | Weight | What it measures |
|---|-----------|--------|------------------|
| 1 | **Time Horizon** | 25% | Decisions where payoff is >10 years out; sacrificing short-term for long-term |
| 2 | **Systems Awareness** | 20% | Interconnections, second-order effects, avoiding silver-bullet thinking |
| 3 | **Course Correction** | 20% | Admitting previous path was wrong; changing direction based on evidence |
| 4 | **Intergenerational Investment** | 15% | Future generations considered; education, knowledge transfer, stewardship |
| 5 | **Institutional Durability** | 10% | Structures that outlast current leaders; governance for the long run |
| 6 | **Evidence Foundation** | 10% | GATEKEEPER — decision grounded in evidence, not ideology or popularity |

---

## Two-Stage Scoring Architecture

Foresight is a needle-in-haystack filter — genuine foresighted decisions are rare in general news. Single-pass oracle scoring on a random corpus produced a bimodal distribution (90% at 0-2, dead zone 2-5) that would be untrainable. We solved this with a two-stage approach:

**Stage 1: Embedding pre-screening (ADR-011)**
- 12 hand-crafted seed articles representing canonical foresight examples
- e5-small cosine similarity against full corpus (178K articles)
- Top-2000 candidates selected for oracle scoring
- Seeds at `datasets/foresight/screening/seed_positives.jsonl`

**Stage 2: Oracle scoring with soft content-type caps**
- Prompt focuses on "how much foresight?" (gradient) not "is this foresight?" (binary)
- Content-type caps raised to 4.0-5.0 (catch false positives, don't create dead zones)
- Evidence Foundation gatekeeper (<= 3.0 caps overall at 3.0)

### Calibration Results (2026-04-01)

**Run 1: Random corpus (300 articles, no screening)**

| Bucket | % |
|--------|---|
| Low (0-2) | 90.0% |
| Mid (2-5) | 9.0% |
| High (5+) | 1.0% |

Mean: 1.19, Stdev: 0.60 — untrainable bimodal distribution.

**Run 2: Screened corpus + soft caps (300 articles)**

| Bucket | % |
|--------|---|
| Low (0-2) | 20.0% |
| Mid (2-5) | 60.3% |
| High (5+) | 19.7% |

Mean: 3.24, Stdev: 1.62, Range: 1.0-8.2 — smooth, trainable distribution.

**Run 3: Full scoring run (1,719 articles — 1,480 screened + 500 random background, 3 failed)**

| Bucket | Count | % |
|--------|-------|---|
| [0-1) | 3 | 0.2% |
| [1-2) | 389 | 22.6% |
| [2-3) | 583 | 33.9% |
| [3-4) | 212 | 12.3% |
| [4-5) | 157 | 9.1% |
| [5-6) | 255 | 14.8% |
| [6-7) | 110 | 6.4% |
| [7-8) | 8 | 0.5% |
| [8-10) | 2 | 0.1% |

Mean: 3.24, Stdev: 1.63, Range: 0.0-8.38. Success rate: 99.8% (1,719/1,722).

Distribution matches calibration batch closely (mean 3.24 in both), confirming the two-stage approach is stable.

### Per-Dimension Stats (Full Run)

| Dimension | Mean | Stdev |
|-----------|------|-------|
| time_horizon | 2.73 | 1.90 |
| systems_awareness | 4.70 | 1.97 |
| course_correction | 2.42 | 2.01 |
| intergenerational_investment | 2.35 | 1.84 |
| institutional_durability | 2.34 | 1.83 |
| evidence_foundation | 5.50 | 1.69 |

All dimensions have stdev > 1.6 — good variance for student model training.

### Content Type Distribution (Full Run)

| Type | Count | % |
|------|-------|---|
| no_decision_context | 1,070 | 62.2% |
| policy_decision | 285 | 16.6% |
| institutional_change | 152 | 8.8% |
| corporate_strategy | 74 | 4.3% |
| education_reform | 62 | 3.6% |
| rhetoric_partial | 30 | 1.7% |
| corporate_with_public_benefit | 24 | 1.4% |
| course_correction | 22 | 1.3% |

### Tier Distribution (Full Run)

| Tier | Count | % |
|------|-------|---|
| High (>=7.0) | 10 | 0.6% |
| Medium (4.0-7.0) | 522 | 30.4% |
| Low (<4.0) | 1,187 | 69.1% |

### Dimension Redundancy Analysis (on 300 calibration articles)

PC1 explains 69% variance (< 85% threshold — PASS). Redundancy ratio 13% (< 50% — PASS).

Highest correlations:
- time_horizon ↔ institutional_durability: r=0.857
- time_horizon ↔ intergenerational_investment: r=0.852
- course_correction ↔ institutional_durability: r=0.796

These are conceptually related but not redundant. Cross-dimension exclusion notes in the prompt mitigate oracle conflation. Evidence_foundation is notably independent (r=0.36-0.38 with most dimensions).

### Training Splits

| Split | Count | % |
|-------|-------|---|
| Train | 1,374 | 80% |
| Val | 172 | 10% |
| Test | 173 | 10% |

Stratified by tier (high 0.7%, medium 30.8%, low 68.5%). Data at `datasets/training/foresight_v1/`.

### GPU Server Staging

All files staged on gpu-server (2026-04-02). Verified:
- Training data: `~/llm-distillery/datasets/training/foresight_v1/` (1,374 + 172 + 173)
- Filter config: `~/llm-distillery/filters/foresight/v1/config.yaml`
- Base model: `google/gemma-3-1b-pt` cached in `~/.cache/huggingface/hub/`
- Common modules: `filters/common/model_loading.py` present
- **Blocker**: Ollama using 13GB VRAM. Stop before training: `ssh gpu-server "sudo systemctl stop ollama"`

Training command (run from `~/llm-distillery/` on gpu-server):
```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. ~/gpu-server/nexusmind-scorer/venv/bin/python3 training/train.py \
    --filter filters/foresight/v1 \
    --data-dir datasets/training/foresight_v1 \
    --output-dir filters/foresight/v1/model \
    --use-head-tail --head-tokens 256 --tail-tokens 256 \
    --epochs 3 --batch-size 16 --learning-rate 2e-4
```

### Training Results (9 epochs: 3 initial + 6 resumed)

| Epoch | Train MAE | Val MAE |
|-------|-----------|---------|
| 1 | 2.49 | 1.44 |
| 2 | 1.25 | 1.07 |
| 3 | 0.99 | 0.99 |
| 4 | 0.78 | 0.88 |
| 5 | 0.70 | 0.84 |
| 6 | 0.62 | 0.90 |
| 7 | 0.57 | 0.86 |
| 8 | 0.55 | 0.84 |
| 9 | 0.51 | **0.80** |

Per-dimension val MAE (epoch 9):
| Dimension | Val MAE |
|-----------|---------|
| institutional_durability | 0.71 |
| evidence_foundation | 0.72 |
| intergenerational_investment | 0.76 |
| time_horizon | 0.78 |
| systems_awareness | 0.88 |
| course_correction | 0.95 |

Train MAE (0.51) is now well below val MAE (0.80) — beginning to overfit. 9 epochs is likely near optimal. Systems_awareness and course_correction remain the hardest dimensions (as predicted by oracle calibration review).

Comparison to other filters:
| Filter | Training articles | Val MAE |
|--------|------------------|---------|
| belonging v1 | 5,900 | 0.49 |
| investment-risk v6 | 8,400 | 0.47 |
| nature_recovery v1 | 2,600 | 0.54 |
| **foresight v1** | **1,374** | **0.80** |

MAE 0.80 with only 1,374 training examples is reasonable. Calibration (isotonic regression) typically improves by 3-8%.

**Overfitting:** Mild. Train-val gap widened from 0.0 (epoch 3) to 0.29 (epoch 9), but val MAE was still improving at epoch 9. Near-optimal — more epochs would worsen the gap without improving val.

**Adapter format:** Verified OLD key format (`.lora_A.weight`, `score.weight`). Hub-compatible. LoRA rank 16, alpha 32, 13M trainable params of 1B total.

### Calibration Results (Round 2, final)

| Metric | Val (346) | Test (346) |
|--------|-----------|------------|
| MAE before | 0.74 | 0.75 |
| MAE after | **0.69** | **0.75** |
| Improvement | +7.5% | -0.2% (neutral) |

Per-dimension calibrated test MAE:
| Dimension | Test MAE |
|-----------|----------|
| intergenerational_investment | 0.63 |
| institutional_durability | 0.65 |
| evidence_foundation | 0.78 |
| time_horizon | 0.78 |
| course_correction | 0.79 |
| systems_awareness | 0.86 |

Compared to production filters:
| Filter | Training data | Calibrated test MAE |
|--------|--------------|-------------------|
| belonging v1 | 5,900 | 0.49 |
| investment-risk v6 | 8,400 | 0.47 |
| nature_recovery v1 | 2,600 | 0.51 |
| sustainability_tech v3 | 8,500 | 0.72 |
| cultural-discovery v4 | 6,400 | 0.74 |
| **foresight v1** | **2,761** | **0.75** |

### Dimension Compression Experiment (4-dim)

Merged time_horizon + intergenerational_investment + institutional_durability → long_term_commitment. Retrained on same data.

| Metric | 6-dim | 4-dim |
|--------|-------|-------|
| Val MAE | **0.80** | 0.81 |
| long_term_commitment | 0.82 avg | **0.68** |
| course_correction | **0.95** | 0.98 |
| Train-val gap | **0.29** | 0.57 |

Merged dimension improved (0.68 vs 0.82 avg) but model overfits faster and course_correction worsened. Overall MAE marginally worse. **Verdict: keep 6 dimensions.** The bottleneck is training data volume, not dimensionality.

### Training Round 2 (doubled data: 2,761 examples, 9 epochs)

Scored 1,734 additional articles (~€1.70) from expanded screening (4K candidates from 215K corpus).

| Epoch | Train MAE | Val MAE |
|-------|-----------|---------|
| 1 | 1.79 | 1.07 |
| 4 | 0.64 | 0.81 |
| 8 | 0.25 | **0.74** |
| 9 | 0.18 | 0.75 |

Per-dimension val MAE (epoch 8):
| Dimension | Round 1 (1,374) | Round 2 (2,761) | Change |
|-----------|----------------|----------------|--------|
| evidence_foundation | 0.72 | **0.67** | -7% |
| institutional_durability | 0.71 | **0.68** | -4% |
| intergenerational_investment | 0.76 | **0.69** | -10% |
| time_horizon | 0.78 | **0.77** | -1% |
| systems_awareness | 0.88 | **0.81** | -8% |
| course_correction | 0.95 | **0.85** | -11% |

Every dimension improved. Doubling data confirmed as the bottleneck (not dimensionality — 4-dim experiment showed no gain).

### Total Oracle Cost

| Run | Articles | Cost |
|-----|----------|------|
| Calibration runs 1+2 | 600 | ~€0.60 |
| Scoring round 1 | 1,719 | ~€1.70 |
| Scoring round 2 | 1,734 | ~€1.70 |
| **Total** | **4,053** | **~€4.00** |

After training, scp model back:
```bash
scp -r gpu-server:~/llm-distillery/filters/foresight/v1/model/ filters/foresight/v1/model/
```

---

## Scoring Philosophy

This is a needle-in-haystack filter. In a general news corpus:
- ~5% of articles contain genuine foresight signals
- ~15% are foresight-adjacent (policy, governance, rhetoric)
- ~80% are irrelevant

The embedding prefilter enriches the oracle scoring pool so the student model gets a balanced distribution across the score range.

---

## Prompt Development

The prompt went through 3 review rounds with specialized agents:

**Round 1** — Found 8 issues including dimension overlap (time_horizon/intergenerational_investment), config schema mismatch, validation example math errors, political skew in examples.

**Round 2** — Verified fixes, found remaining math errors in contrastive examples, Step 1/1b flow ambiguity.

**Round 3** — Compared against all production prompts, checked gotcha compatibility, assessed bimodal distribution risk (confirmed by calibration run 1), oracle calibration risk assessment (rated 4/5).

Key prompt features:
- Cross-dimension exclusion notes on 4 dimensions (prevent oracle conflation)
- 13 contrastive examples with 3 splitting examples for problem pairs
- Anti-hallucination rule (exact quotes only) — unique to this filter
- Soft content-type caps (4.0-5.0) instead of hard caps (2.0-3.0)

---

## Relationship to Other Filters

| Filter | Overlap | Distinction |
|--------|---------|-------------|
| `thriving` | Both care about positive outcomes | Foresight = process, Thriving = outcome |
| `sustainability_technology` | Tech decisions could show foresight | Foresight focuses on decision quality, not tech |
| `nature_recovery` | Recovery could result from wise policy | Foresight looks at the policy decision itself |
| `belonging` | Intergenerational overlap | Belonging = relational fabric, Foresight = decision-making |

---

## Next Steps

- [x] Develop harmonized prompt (belonging v1 structure)
- [x] Agent review rounds (3 rounds, 10+ specialized agents)
- [x] Calibration batch 1 — random corpus (300 articles) — bimodal problem confirmed
- [x] Embedding screening with curated seeds (12 seeds, 178K corpus, top-2K)
- [x] Calibration batch 2 — screened corpus (300 articles) — smooth distribution confirmed
- [x] Dimension redundancy analysis — PC1 69%, redundancy ratio 13% — PASS
- [x] Create prefilter.py (rule-based, permissive — blocks noise, passes policy/governance)
- [x] Full oracle scoring — 1,719 articles (1,480 screened + 500 random bg, 3 failed), ~€1.70
- [x] Prepare training splits — 1,374 train / 172 val / 173 test (stratified by tier)
- [x] Train on gpu-server (Gemma-3-1B + LoRA) — 9 epochs (3 + 6 resumed), best val MAE **0.80**
- [x] Write inference code (base_scorer.py, inference.py)
- [x] Score round 2 — 1,734 additional articles (~€1.70), merged to 3,453 total
- [x] Retrain with doubled data (2,761 examples) — val MAE 0.80→0.74 (-7.2%)
- [x] Fit calibration (round 2) — val MAE 0.74→0.69 (+7.5%), **test MAE 0.75** (neutral)
- [x] Write inference_hub.py (HuggingFace Hub inference)
- [x] Model directory cleanup (consolidated model-r2 → model)
- [x] Train hybrid probe — e5-small MLP, probe MAE 0.99 (rough estimator, expected for conceptually hard filter)
- [x] Write inference_hybrid.py (two-stage hybrid inference)
- [ ] Deploy to HuggingFace Hub
- [ ] Fit normalization.json (ADR-014, needs production data)
- [ ] Deploy to NexusMind (gpu-server + sadalsuud)
- [ ] ovr.news frontend integration

---

*Created: 2026-04-01. Renamed from signs_of_wisdom per ADR-012 lens-aligned naming.*
*Last updated: 2026-04-02.*
