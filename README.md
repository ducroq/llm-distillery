# LLM Distillery 🥃

**Transform large language model expertise into fast, specialized local models**

LLM Distillery is a framework for distilling knowledge from large foundation models (Gemini Flash) into small, domain-specific classifiers that run locally at 100x lower cost and 50x faster inference.

## Overview

Large language models excel at nuanced judgment tasks but are expensive and slow for production use. This framework:

1. **Generates ground truth datasets** using Gemini Flash as oracle (dimensional scoring only)
2. **Fine-tunes Qwen2.5-7B-Instruct** for multi-dimensional regression
3. **Validates quality** through comprehensive training data validation
4. **Deploys locally** for fast, cost-effective batch inference (150x faster than oracle)

### Use Cases

- **Content filtering**: Uplifting news, sustainability tech deployment, investment risk signals
- **Multi-dimensional scoring**: Rate content on 8 dimensions simultaneously (0-10 scale)
- **Tier classification**: Flexible postfilter tier assignment without model retraining

## Current Status (November 2025)

### 🚀 Production Ready Filters

- **sustainability_technology v1**: ✅ **DEPLOYED**
  - 6 dimensions (LCSA framework): technology_readiness, technical_performance, economic_competitiveness, life_cycle_environmental_impact, social_equity_impact, governance_systemic_impact
  - Model: Qwen2.5-1.5B + LoRA (18.5M trainable params)
  - Test MAE: 0.690 | All dimensions < 1.0
  - Deployed: [HuggingFace Hub](https://huggingface.co/jeergrvgreg/sustainability-technology-v1)

### 🎯 Training Data Ready

- **uplifting v4**: 6,705 examples (validated, ready for training)
  - 8 dimensions: agency, progress, collective_benefit, connection, innovation, justice, resilience, wonder
  - Philosophy: MEANING not TONE

- **investment-risk v4**: 4,880 examples (validated, ready for training)
  - 8 dimensions: macro_risk, credit_stress, sentiment, valuation, policy, systemic, evidence, actionability
  - Philosophy: "You can't predict crashes, but you can prepare for them"

### ✅ Architecture Harmonization Complete (Nov 2025)

All filters now follow consistent oracle output discipline:
- ✅ **Oracle outputs dimensional scores ONLY** (0-10 per dimension + reasoning)
- ✅ **Tier classification in postfilters** (enables flexible thresholds without retraining)
- ✅ **Harmonized prompt structure** (scope → gatekeepers → article → dimensions)
- ✅ **Inline filters for every dimension** (fast model compatibility)

**Result**: Clean separation of concerns - oracle scores, postfilter classifies. Change tier thresholds without re-labeling training data.

### ✅ Training Pipeline Complete

- **Data preparation**: `training/prepare_data.py` with stratified splitting (tier or score-bin based)
- **Data validation**: `training/validate_training_data.py` with comprehensive quality checks
- **Deduplication**: `training/deduplicate_training_data.py` for cross-split duplicate removal
- **Validation reports**: Auto-generated summaries saved to filter directories

### ✅ Development Tools & Agents

- **Filter Development Guide Agent**: End-to-end lifecycle guidance (9 phases: planning → deployment)
- **Filter Harmonizer Agent**: Automated consistency checking and validation
- **Batch Scoring**: Generic `ground_truth.batch_scorer` supporting all filter packages
- **Dataset Profiling**: Master dataset with 402K articles (Oct-Nov 2025)

### 🚧 Next Steps

- **Train remaining filters**: uplifting v4, investment-risk v4
- **Production deployment**: Batch processing pipeline for high-volume scoring

## Quick Start

### 1. Install Dependencies

```bash
cd C:\local_dev\llm-distillery

# Install required packages
pip install anthropic google-generativeai pyyaml
```

### 2. Set Up API Keys

Create `config/credentials/secrets.ini`:

```ini
[api_keys]
anthropic_api_key = sk-ant-your_key_here
gemini_api_key = AIza_your_key_here
gemini_billing_api_key = AIza_billing_key  # Optional: 150 RPM vs 2 RPM
```

**Important**: This file is git-ignored for security.

### 3. Score Training Data

Score 5K+ articles with oracle:

```bash
python -m ground_truth.batch_scorer \
  --filter filters/uplifting/v4 \
  --source datasets/raw/master_dataset.jsonl \
  --output-dir datasets/scored/uplifting_v4_training \
  --llm gemini-flash \
  --target-count 5000 \
  --batch-size 100
```

**Process**: Stream articles → Prefilter → Oracle scores → Save to batches

### 4. Prepare Training Data

Split scored data into train/val/test sets with stratification:

```bash
python training/prepare_data.py \
  --filter filters/uplifting/v4 \
  --data-source datasets/scored/uplifting_v4_training \
  --output-dir datasets/training/uplifting_v4
```

**Output**: train.jsonl (80%), val.jsonl (10%), test.jsonl (10%) with stratified sampling

### 5. Validate Training Data

Run comprehensive quality checks:

```bash
# Full validation with detailed report
python training/validate_training_data.py \
  --data-dir datasets/training/uplifting_v4 \
  --filter filters/uplifting/v4

# If duplicates found, deduplicate
python training/deduplicate_training_data.py datasets/training/uplifting_v4

# Generate summary report for filter documentation
python scripts/validation/generate_validation_summary.py \
  --data-dir datasets/training/uplifting_v4 \
  --filter-name uplifting \
  --version v4 \
  --output filters/uplifting/v4/TRAINING_DATA_VALIDATION.md
```

**Checks performed**:
- Structural integrity (required fields, ID uniqueness, label array length)
- Data distribution (train/val/test splits at 80/10/10)
- Label quality (score range [0-10], no NaN values, sufficient variance)
- Content quality (non-empty titles/content, reasonable lengths)
- Consistency (dimension names match across splits and config)
- Score distributions per dimension

### 6. Train a Model (Coming Soon)

Fine-tune Qwen2.5-7B on prepared dataset:

```bash
python -m training.train \
  --filter filters/uplifting/v4 \
  --data-dir datasets/training/uplifting_v4 \
  --output-dir models/uplifting_v4 \
  --base-model unsloth/Qwen2.5-7B-Instruct \
  --epochs 3 \
  --batch-size 4
```

**Requirements**: 16GB+ GPU (RTX 4090, A100), ~2-4 hours training time

## Documentation

- **[Full Documentation](docs/README.md)** - Complete docs index
- **[Architecture](docs/ARCHITECTURE.md)** - System design and oracle output discipline
- **[System Overview](docs/SYSTEM_OVERVIEW.md)** - Current state and datasets
- **[Filter Development Guide](docs/agents/README.md)** - 9-phase filter development lifecycle
- **[Repository Structure](docs/REPOSITORY_STRUCTURE.md)** - Directory organization
- **[Decisions Log](docs/DECISIONS.md)** - Strategic decisions and rationale

## Architecture

### Oracle → Student Model Knowledge Distillation

```
┌─────────────────────────────────────────────────────────────┐
│                    GROUND TRUTH GENERATION                   │
│                                                              │
│  Raw Articles  →  Oracle (Gemini Flash)  →  Labeled Dataset │
│  (5K+ samples)    8 dimensional scores      (JSONL)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   DATA PREPARATION & VALIDATION              │
│                                                              │
│  Scored Data  →  Stratified Split  →  Quality Validation    │
│  (batches)       (80/10/10)           (dedupe, checks)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       KNOWLEDGE DISTILLATION                 │
│                                                              │
│  Training Data  →  Student Model  →  Trained Classifier     │
│  (validated)       (Qwen2.5-7B)      (90%+ MAE ≤1.5)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT                              │
│                                                              │
│  Article  →  [Prefilter]  →  [Model]  →  [Postfilter]      │
│  (input)     <5ms Python      20-50ms      <10ms Python     │
│              rule-based        Qwen2.5     tier classify    │
└─────────────────────────────────────────────────────────────┘
```

### Runtime Inference Pipeline

Once trained, the deployed system processes articles through **3 stages**:

```
Article → [Prefilter] → [Student Model] → [Postfilter] → Tier + Scores
           <5ms          20-50ms            <10ms
           (Python)      (Qwen2.5-7B)       (Python)
```

**Stage 1: Prefilter** (Fast, Rule-Based)
- Blocks obvious out-of-scope content
- ~5ms per article
- Target: <10% false negative rate

**Stage 2: Student Model** (LLM Inference)
- Scores 8 dimensions (0-10 each)
- Uses fine-tuned Qwen2.5-7B (local)
- 20-50ms per article

**Stage 3: Postfilter** (Tier Classification)
- Maps dimensional scores → tier
- Applies gatekeeper rules from config.yaml
- Flexible thresholds without retraining
- <10ms per article

**Key Benefit**: Change tier definitions by updating config only, no model retraining needed.

## Project Structure

```
llm-distillery/
├── filters/                    # Versioned filter packages
│   ├── uplifting/v4/          # Uplifting content filter
│   ├── sustainability_tech_innovation/v2/  # Sustainability tech
│   ├── investment-risk/v4/    # Investment risk signals
│   └── README.md              # Filter development guide
│
├── ground_truth/              # Oracle scoring pipeline
│   ├── batch_scorer.py        # Universal scoring engine
│   ├── llm_client.py         # Gemini/Claude API clients
│   └── prefilter_runner.py   # Prefilter execution
│
├── training/                  # Model training pipeline
│   ├── prepare_data.py       # Stratified train/val/test splits
│   ├── validate_training_data.py  # Quality validation
│   ├── deduplicate_training_data.py  # Cross-split deduplication
│   └── train.py              # Model fine-tuning (planned)
│
├── datasets/                  # Generated datasets
│   ├── raw/                  # Raw article collections
│   ├── scored/               # Oracle-scored batches
│   └── training/             # Prepared train/val/test splits
│
├── docs/                      # Documentation
│   ├── agents/               # Development guide agents
│   ├── decisions/            # Architecture decision records
│   ├── ARCHITECTURE.md       # System architecture
│   └── SYSTEM_OVERVIEW.md    # Current state & datasets
│
├── scripts/                   # Utility scripts (organized by phase)
│   ├── validation/           # Phase 3-5: Validation utilities
│   ├── training/             # Phase 6-7: Training utilities
│   ├── oracle/               # Phase 3: Oracle calibration
│   ├── deployment/           # Phase 9: Model deployment
│   └── dataset/              # General dataset utilities
│
└── config/                    # Configuration
    └── credentials/          # API keys (git-ignored)
```

## Filter Development Workflow

See [Filter Development Guide](docs/agents/README.md) for the complete 9-phase workflow:

1. **Planning** - Define dimensions, tiers, gatekeepers
2. **Architecture** - Harmonize prompt structure, inline filters
3. **Validation** - Oracle calibration on sample articles
4. **Prefilter** - Test false negative/positive rates
5. **Training Data** - Score 5K+ articles, validate quality ✅ (current)
6. **Training** - Fine-tune student model
7. **Testing** - Benchmark vs oracle, integration tests
8. **Documentation** - Complete all reports and guides
9. **Deployment** - Production release with monitoring

## Cost Analysis

### Ground Truth Generation (One-time per filter)
- **5K articles** × **$0.001/article** (Gemini Flash)
- **Total**: ~$5-10 per filter

### Local Model Inference (Ongoing)
- **Cost**: $0 (runs locally on GPU)
- **Speed**: 20-50ms per article
- **Accuracy**: 90-95% agreement with oracle (MAE ≤1.5)

### ROI Calculation
If processing **4,000 articles/day**:
- **Oracle API**: 4,000 × $0.001 × 365 = **$1,460/year**
- **Local Model**: Training cost $10 + hosting ~$0 = **$10/year**
- **Savings**: **$1,450/year per filter** (99% cost reduction)

## Roadmap

### ✅ Phase 1: Ground Truth Generation (Complete)
- [x] Harmonized filter architecture
- [x] Generic batch scorer
- [x] Oracle calibration (Flash vs Pro)
- [x] Prefilter validation
- [x] Training data collection (5K+ per filter)
- [x] Data validation pipeline

### 🚧 Phase 2: Model Training (Current)
- [x] Data preparation with stratification
- [x] Training data validation
- [ ] Qwen2.5-7B fine-tuning script
- [ ] Training monitoring & checkpointing
- [ ] Model evaluation framework

### 📝 Phase 3: Deployment (Planned)
- [ ] Inference server (prefilter + model + postfilter)
- [ ] Batch processing pipeline
- [ ] Production monitoring
- [ ] Model registry & versioning

## Contributing

This is a personal research project. Contributions and suggestions welcome via issues.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built for the Content Aggregator project
- Oracle scoring powered by Google Gemini Flash
- Student models based on Qwen2.5-7B-Instruct
- Architecture inspired by knowledge distillation research
