# Foresight Filter

**Version**: 1.0
**Status**: Development — oracle scoring complete (1,719 articles), ready for training splits
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
HF_HUB_OFFLINE=1 PYTHONPATH=. python training/train.py \
    --config filters/foresight/v1/config.yaml \
    --data-dir datasets/training/foresight_v1 \
    --output-dir filters/foresight/v1/model \
    --use-head-tail --head-tokens 256 --tail-tokens 256 \
    --epochs 3 --batch-size 16 --lr 2e-4
```

Expected: ~10-15 min on RTX 4080. After training, scp model back:
```bash
scp -r gpu-server:~/llm-distillery/filters/foresight/v1/model/ filters/foresight/v1/model/
```

### Oracle Scoring Cost

- Run 1 (calibration, random): ~€0.30 (300 articles)
- Run 2 (calibration, screened): ~€0.30 (300 articles)
- Run 3 (full): ~€1.70 (1,719 articles)
- **Total oracle cost: ~€2.30**

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
- [ ] Train on gpu-server (Gemma-3-1B + LoRA) — staged, waiting for GPU availability
- [ ] Fit calibration (isotonic regression)
- [ ] Write inference code (base_scorer, inference, inference_hub, inference_hybrid)
- [ ] Train hybrid probe (e5-small MLP)
- [ ] Deploy to HuggingFace Hub
- [ ] ovr.news frontend integration

---

*Created: 2026-04-01. Renamed from signs_of_wisdom per ADR-012 lens-aligned naming.*
*Last updated: 2026-04-02.*
