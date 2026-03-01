# CLAUDE.md - LLM Distillery

## What Is This?

**LLM Distillery** is a knowledge distillation framework. It trains small, cheap, local classifiers (Gemma-3-1B + LoRA) to replicate expensive cloud LLM scoring (Gemini Flash) at 100x lower cost and 50x faster inference.

**Core workflow:** Oracle (Gemini Flash) scores articles on dimensions (0-10) → Train student model (Gemma-3-1B) → Deploy as filter package

**System context:** llm-distillery creates filters. NexusMind deploys them for production scoring. The interface is the filter package: `filters/{name}/v{N}/` directories copied between repos, plus HuggingFace Hub uploads.

## Tech Stack

- **Oracle**: Gemini Flash ($0.001/article, dimensional scoring)
- **Student**: Gemma-3-1B (`google/gemma-3-1b-pt`) with PEFT/LoRA adapters
- **Calibration**: Per-dimension isotonic regression (ADR-008)
- **Hybrid inference**: e5-small embedding probe (Stage 1) + fine-tuned model (Stage 2, ADR-006)
- **Training data**: 5K-10K oracle-scored articles per filter, 80/10/10 splits

## Hard Constraints

- **Oracle outputs scores only.** Dimensional scores (0-10), never tier/stage classifications. Tier assignment is postprocessing. Changing thresholds must never require re-labeling.
- **Use `load_base_model_for_seq_cls()`** from `filters/common/model_loading.py`. Never use `AutoModelForSequenceClassification` directly — Gemma-3-1B's `gemma3_text` config isn't in the Auto mapping.
- **Keep PEFT adapters in OLD key format.** `.lora_A.weight` / `score.weight`, not `.lora_A.default.weight`. Never run `resave_adapter.py` before Hub upload — it breaks `PeftModel.from_pretrained()`.
- **Fit `calibration.json` after every training run.** Isotonic regression on the val set. Commit with the filter package. The base scorer auto-loads it.

## Production Filters

| Filter | Version | MAE | Training Data | Status |
|--------|---------|-----|---------------|--------|
| **uplifting** | v6 | 0.67 | 10.5K articles | Deployed (HF Hub, private) |
| **sustainability_technology** | v3 | 0.72 | 10.6K articles | Deployed (HF Hub, private) |
| **investment-risk** | v6 | 0.47 | 10.4K articles | Deployed (HF Hub, private) |
| **cultural-discovery** | v4 | 0.74 | 8K articles | Deployed (HF Hub, private) |

### In Development

| Filter | Version | Status | Blocker |
|--------|---------|--------|---------|
| **belonging** | v1 | Needs assessment | Current sprint priority |
| **ai-engineering-practice** | v2 | Ready for oracle scoring | Domain classification added |
| **nature_recovery** | v1 | Early dev | Need harmonized prompt |
| **signs_of_wisdom** | v1 | Early dev | Need harmonized prompt |

## Key Decisions

- **Dimensional regression (0-10)** — not classifications (ADR-001)
- **Screen+merge for needle-in-haystack filters** (ADR-003)
- **Commerce is the only universal prefilter** (ADR-004)
- **Active learning for rare tiers** (ADR-005)
- **Fine-tuning beats embedding probes** — research confirmed
- **Gemma-3-1B** — replaced Qwen2.5; better MAE, faster inference

See `docs/adr/README.md` for full ADR index, `docs/decisions/` for detailed records.

## Before You Start

| When you're... | Read... |
|----------------|---------|
| Developing a new filter | `docs/agents/filter-development-guide.md` — full lifecycle, or `docs/guides/filter-creation-workflow.md` — quick steps |
| Deploying to NexusMind or gpu-server | `docs/WAY-OF-WORKING.md` — deployment runbook with verify steps |
| Training on GPU server | `memory/gpu-server.md` — venv, PYTHONPATH, HF_HUB_OFFLINE |
| Debugging model loading or PEFT issues | `memory/gemma3-model.md` — Auto mapping fix, key format details |
| Making architectural decisions | `docs/adr/README.md` — 8 settled ADRs |
| Checking priorities or planning work | `docs/TODO.md` and `docs/ROADMAP.md` |
| Understanding system design | `docs/ARCHITECTURE.md` |
| Stuck on tooling or infra | `memory/gotcha-log.md` — problem/fix archive |

## Getting Started

```bash
pip install -r requirements.txt

# Configure: add HF token to config/credentials/secrets.ini
# Oracle scoring
python -m ground_truth.batch_scorer --filter filters/uplifting/v6 --source datasets/raw/master_dataset.jsonl

# Prepare training splits
python training/prepare_data.py --filter filters/uplifting/v6 --data-source datasets/scored/uplifting_v6.jsonl

# Fit calibration (after training)
PYTHONPATH=. python scripts/calibration/fit_calibration.py \
    --filter filters/uplifting/v6 --data-dir datasets/training/uplifting_v6 \
    --test-data datasets/training/uplifting_v6/test.jsonl

# Upload to Hub
python scripts/deployment/upload_to_huggingface.py \
    --filter filters/uplifting/v6 --repo-name jeergrvgreg/uplifting-filter-v6 \
    --token $HF_TOKEN --private
```

---

*Last updated: 2026-03-01*
