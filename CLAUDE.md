# CLAUDE.md - Project Context for LLM Distillery

## What is this project?

**LLM Distillery** is a framework for distilling knowledge from large foundation models (Gemini Flash) into small, domain-specific classifiers that run locally at 100x lower cost and 50x faster inference.

**Core workflow:** Oracle (Gemini Flash) scores articles on dimensions -> Train student model (Gemma-3-1B) -> Deploy locally

## Project Structure

```
llm-distillery/
├── CLAUDE.md              # This file (AI context)
├── filters/               # Versioned filter packages
│   ├── uplifting/v6/      # Production ready, deployed
│   ├── sustainability_technology/v1-v3/  # v3 production, deployed
│   ├── investment-risk/v5-v6/  # v6 production, deployed
│   ├── cultural-discovery/v1-v4/  # v4 production, deployed
│   ├── signs_of_wisdom/v1/  # Early development
│   ├── nature_recovery/v1/  # Early development
│   ├── ai-engineering-practice/v2/  # Blocked on data
│   ├── belonging/v1/      # Needs assessment
│   └── todo/              # Planned filters
│   └── common/            # Shared infrastructure
│       ├── filter_base_scorer.py  # Base class for all filter scorers
│       ├── model_loading.py       # Gemma-3 model loading + LoRA utilities
│       ├── score_calibration.py   # Isotonic calibration fit/apply
│       ├── embedding_stage.py     # e5-small probe for hybrid inference
│       ├── hybrid_scorer.py       # Two-stage hybrid inference orchestrator
│       ├── base_prefilter.py      # Base prefilter with commerce detection
│       └── commerce_prefilter/    # ML commerce content detector
├── ground_truth/          # Oracle scoring pipeline (batch_scorer, llm_client)
├── training/              # Model training pipeline (prepare_data, validate, train)
├── evaluation/            # Model evaluation tools
├── research/              # Research experiments
│   └── embedding_vs_finetuning/  # Embedding probes vs fine-tuning comparison
├── datasets/              # Raw, scored, and training datasets
├── docs/                  # Documentation
│   ├── TODO.md            # Active task list
│   ├── IDEAS.md           # Future ideas
│   ├── OPEN_QUESTIONS.md  # Unresolved questions
│   ├── ROADMAP.md         # Now/Next/Later roadmap
│   ├── adr/               # ADRs 001-008 (screening, commerce, active learning, hybrid, deployment, calibration)
│   ├── decisions/         # Detailed decision records
│   └── templates/         # ADR template
├── scripts/               # Utility scripts by phase
└── config/                # Configuration and credentials
```

## Current Status (February 2026)

### Production Ready Filters
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
| **ai-engineering-practice** | v2 | Blocked | Needs FluxusSource hardware sources |
| **nature_recovery** | v1 | Early dev | Need harmonized prompt |
| **signs_of_wisdom** | v1 | Early dev | Need harmonized prompt |

### Planned (filters/todo/)
- future-of-education
- seece (corporate excellence)
- sustainability_economic_viability
- sustainability_policy_effectiveness

### Backlog
- **Commerce prefilter v2** - v1 needs rework for multilingual embeddings and context size.

## Key Decisions (see docs/adr/ and docs/decisions/)

- **Oracle outputs scores only** - Tier classification happens in postfilter (flexible thresholds)
- **Dimensional regression** - Student models learn 0-10 scores, not classifications
- **Gemma-3-1B** - Default student model (replaced Qwen2.5-1.5B), better MAE and faster inference
- **Inline filters** - Fast rules embedded in prompts for model compatibility
- **Screen+merge for needle-in-haystack filters** (ADR-003) - Random data provides negatives, screened data enriches positives; merge both for best results
- **Commerce is only universal prefilter** (ADR-004) - Filter-specific noise handled by trained model, not additional prefilters
- **Active learning for rare tiers** (ADR-005) - Use production filter to find high-scoring candidates, oracle score, add to training data
- **Fine-tuning beats embedding probes** - Research confirmed fine-tuned models significantly outperform frozen embedding + probe approaches
- **Post-hoc isotonic calibration** (ADR-008) - Per-dimension isotonic regression corrects MSE score compression; stored as `calibration.json`, applied at inference via `numpy.interp`

## Development Workflow

### Filter Development

See `docs/guides/filter-creation-workflow.md` for full step-by-step using uplifting v6 as template.

Phases: Planning -> Architecture -> Validation -> Prefilter -> Training Data -> Training -> Inference Code -> Calibration -> Hybrid Probe -> Deploy -> Document

### Common Commands

```bash
# Score training data
python -m ground_truth.batch_scorer --filter filters/uplifting/v5 --source datasets/raw/master_dataset.jsonl

# Prepare training splits
python training/prepare_data.py --filter filters/uplifting/v5 --data-source datasets/scored/...

# Validate training data
python training/validate_training_data.py --data-dir datasets/training/uplifting_v5

# Fit score calibration (after training)
PYTHONPATH=. python scripts/calibration/fit_calibration.py \
    --filter filters/uplifting/v6 \
    --data-dir datasets/training/uplifting_v6 \
    --test-data datasets/training/uplifting_v6/test.jsonl
```

## Deployment Gotchas (see ADR-007)

- **Do NOT run `resave_adapter.py` before Hub upload** — it changes key format and breaks `PeftModel.from_pretrained()`. Local `inference.py` handles remapping at load time.
- **Use `load_base_model_for_seq_cls()`** from `filters/common/model_loading.py` instead of `AutoModelForSequenceClassification` directly. Gemma-3-1B (`gemma3_text` config) is not in the Auto mapping.
- **Upload script verifies Hub loading** automatically after upload. If it fails, check adapter format.
- **Fit `calibration.json` after training** (ADR-008) — the base scorer auto-loads it if present. Run `scripts/calibration/fit_calibration.py` on the val set. Commit `calibration.json` with the filter package.

## Conventions

- **Filter packages**: `filters/{name}/v{version}/` with config.yaml, prompt-compressed.md, prefilter.py, base_scorer.py, inference.py, inference_hub.py, inference_hybrid.py, calibration.json
- **FilterBaseScorer**: All filter `base_scorer.py` files inherit from `filters/common/filter_base_scorer.py` for shared model loading, inference, and calibration logic
- **ADRs**: Short architectural decisions in `docs/adr/`, detailed ones in `docs/decisions/`
- **Datasets**: Raw -> Scored -> Training splits (80/10/10)

## Important Files to Read

Before making changes, understand:
- `docs/TODO.md` - Current tasks and filter status
- `docs/ROADMAP.md` - What's now/next/later
- `docs/ARCHITECTURE.md` - System design
- `docs/adr/` and `docs/decisions/` - Past decisions

---

*Last updated: 2026-02-22*
