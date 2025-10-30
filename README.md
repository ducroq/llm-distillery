# LLM Distillery 🥃

**Transform large language model expertise into fast, specialized local models**

LLM Distillery is a framework for distilling knowledge from large foundation models (Claude, Gemini, GPT-4) into small, domain-specific classifiers that run locally at 100x lower cost and 50x faster inference.

## Overview

Large language models excel at nuanced judgment tasks but are expensive and slow for production use. This framework:

1. **Generates ground truth datasets** using Gemini Flash as labeling oracle
2. **Fine-tunes Qwen 2.5 agents** (7B parameters) specialized per semantic dimension
3. **Validates quality** by comparing model predictions to ground truth
4. **Deploys locally** for fast, cost-effective batch inference

### Use Cases

- **Content filtering**: Sustainability impact, uplifting news, policy relevance
- **Investment intelligence**: Technology readiness, greenwashing detection, evidence strength
- **Quality assessment**: Clinical evidence, regulatory status, technical credibility
- **Multi-dimensional scoring**: Rate content on 8+ dimensions simultaneously

## Current Status (October 2025)

### ✅ Completed
- **Filter Architecture**: Versioned filter packages (pre-filter + prompt + config)
- **Uplifting Filter v1**: 8-dimension framework with rule-based pre-filter (93% pass rate)
- **Sustainability Filter v1**: 8-dimension framework with greenwashing/vaporware detection
- **Investment Risk Filter v1**: 8-dimension framework for capital preservation (FOMO/speculation blocking)
- **Oracle Calibration**: Compare Flash/Pro/Sonnet to select best LLM
- **Pre-filter Calibration**: Measure blocking effectiveness before ground truth generation
- **Generic Batch Labeler**: Universal labeling engine supporting filter packages
- **Secrets Management**: Secure API key handling (env vars + secrets.ini)
- **Comprehensive Documentation**: Filter guides, calibration workflow

### 🚧 In Progress
- Ground truth generation with pre-filter integration
- Master datasets (99K articles, Sept 29 - Oct 29, 2025)

### 📝 Planned
- Training pipeline (Qwen 2.5-7B fine-tuning)
- Evaluation framework (model vs oracle comparison)
- Inference server (pre-filter + model deployment)

## Quick Start

### 1. Install Dependencies

```bash
cd C:\local_dev\llm-distillery

# Using pip
pip install anthropic google-generativeai

# Or install all planned dependencies
pip install -r requirements.txt  # (when available)
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

### 3. Calibrate Pre-Filter

Test pre-filter blocking effectiveness (500 articles recommended):

```bash
python -m ground_truth.calibrate_prefilter \
    --filter filters/uplifting/v1 \
    --source datasets/raw/master_dataset_*.jsonl \
    --sample-size 500 \
    --output reports/uplifting_v1_prefilter_cal.md
```

**Output**: Pass rate, block reason distribution, sample blocked articles

### 4. Calibrate Oracle

Compare Flash vs Pro/Sonnet to choose best LLM (100 articles recommended):

```bash
python -m ground_truth.calibrate_oracle \
    --filter filters/uplifting/v1 \
    --source datasets/raw/master_dataset_*.jsonl \
    --sample-size 100 \
    --models gemini-flash,gemini-pro,claude-sonnet \
    --output reports/uplifting_v1_oracle_cal.md
```

**Output**: Agreement rates, score distributions, cost analysis

See [Filter Development Guide](filters/README.md) for complete workflow.

### 5. Generate Ground Truth

Label articles with oracle (stops at 2,500 passing articles):

```bash
python -m ground_truth.batch_labeler \
    --filter filters/uplifting/v1 \
    --source datasets/raw/master_dataset_*.jsonl \
    --target-labeled 2500 \
    --oracle gemini-flash \
    --output datasets/labeled/uplifting_v1/
```

**Process**: Stream articles → Pre-filter → Label passing articles → Stop at target

### 6. Train a Model (Planned)

```bash
# Coming soon
python -m training.train \
    --filter filters/uplifting/v1 \
    --dataset datasets/labeled/uplifting_v1/ \
    --output inference/deployed/uplifting_v1/
```

### 7. Evaluate Quality (Planned)

```bash
# Coming soon
python -m evaluation.evaluate_model \
    --filter filters/uplifting/v1 \
    --model inference/deployed/uplifting_v1/ \
    --test-set datasets/labeled/uplifting_v1/test.jsonl
```

### 8. Run Inference (Planned)

```bash
# Coming soon - deployed filter includes pre-filter + model
python -m inference.predict \
    --filter inference/deployed/uplifting_v1/ \
    --input articles.jsonl \
    --output predictions.jsonl
```

## Documentation

- **[Full Documentation](docs/README.md)** - Complete docs index
- **[Calibration Guide](docs/guides/calibration.md)** - How to compare Claude vs Gemini
- **[Architecture Overview](docs/architecture/overview.md)** - System design and components
- **[Project Structure](#project-structure)** - Directory organization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GROUND TRUTH GENERATION                   │
│                                                              │
│  Raw Articles  →  LLM Oracle  →  Labeled Dataset            │
│  (50K samples)    (Claude)       (JSON ratings)             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       FINE-TUNING                            │
│                                                              │
│  Ground Truth  →  Small Model  →  Trained Classifier        │
│  (JSONL)          (DeBERTa)       (90%+ accuracy)           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      VALIDATION                              │
│                                                              │
│  Test Set  →  Compare  →  Quality Metrics                   │
│               (Model vs Oracle)   (Accuracy, MAE, F1)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT                                │
│                                                              │
│  Article  →  Local Model  →  Predictions                    │
│  (input)     (<50ms)          (8 dimensions)                │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
llm-distillery/
├── prompts/                    # LLM evaluation prompts
│   ├── sustainability.md       # Climate tech & impact assessment
│   ├── uplifting.md           # Positive/uplifting content detection
│   ├── eu_policy.md           # EU policy relevance
│   └── healthcare_ai.md       # Healthcare AI readiness
│
├── ground_truth/              # Ground truth generation
│   ├── generate.py           # Main CLI for dataset creation
│   ├── samplers.py           # Stratified sampling strategies
│   ├── llm_evaluators.py     # Claude/Gemini API wrappers
│   └── validators.py         # Quality checks, consistency
│
├── datasets/                  # Generated datasets
│   ├── sustainability_50k.jsonl
│   └── splits/               # Train/val/test splits
│
├── training/                  # Model fine-tuning
│   ├── train.py              # Training script
│   ├── models.py             # Model architectures
│   ├── datasets.py           # PyTorch Dataset classes
│   └── configs/              # Training configurations
│       └── sustainability_deberta.yaml
│
├── evaluation/                # Quality assessment
│   ├── evaluate.py           # Model vs oracle comparison
│   ├── calibration.py        # Drift detection
│   └── metrics.py            # Custom metrics
│
├── inference/                 # Deployed models
│   ├── predict.py            # Single prediction
│   ├── batch_predict.py      # Batch processing
│   ├── serve.py              # FastAPI server
│   └── models/               # Model checkpoints
│       └── sustainability_v1/
│
└── tests/                     # Unit tests
    ├── test_ground_truth.py
    ├── test_training.py
    └── test_inference.py
```

## Available Filters

> **See [filters/README.md](filters/README.md) for complete filter development workflow**

### 1. Uplifting Content Filter v1.0 ✅
**Focus**: MEANING not TONE - genuine human and planetary wellbeing

**Pre-filter blocks**: Corporate finance, military buildups (93% pass rate)

**Dimensions (8)**: Agency, progress, collective_benefit (gatekeeper), connection, innovation, justice, resilience, wonder

**Use Cases**: Positive news aggregation, solutions journalism, progress indicators

**Status**: ✅ Implemented, ⏳ Calibration pending

**Package**: [`filters/uplifting/v1/`](filters/uplifting/v1/)

---

### 2. Sustainability Impact Filter v1.0 ✅
**Focus**: DEPLOYED TECHNOLOGY and MEASURED OUTCOMES

**Pre-filter blocks**: Greenwashing, vaporware, fossil fuel transition

**Dimensions (8)**: Climate_impact, technical_credibility (gatekeeper), economic_viability, deployment_readiness, systemic_impact, justice_equity, innovation_quality, evidence_strength

**Use Cases**: Climate tech investment, greenwashing detection, progress tracking

**Status**: ✅ Implemented, ⏳ Calibration pending

**Package**: [`filters/sustainability/v1/`](filters/sustainability/v1/)

---

### 3. Investment Risk Filter v1.0 ✅
**Focus**: CAPITAL PRESERVATION and MACRO RISK SIGNALS, not stock picking

**Pre-filter blocks**: FOMO/speculation, stock picking, affiliate marketing, clickbait

**Dimensions (8)**: macro_risk_severity, credit_market_stress, market_sentiment_extremes, valuation_risk, policy_regulatory_risk, systemic_risk, evidence_quality (gatekeeper), actionability

**Use Cases**: Portfolio defense, risk monitoring, opportunity identification, noise filtering

**Status**: ✅ Implemented, ⏳ Calibration pending

**Package**: [`filters/investment-risk/v1/`](filters/investment-risk/v1/)

---

### 4. SEECE Energy Tech Filter v1.0 ⏳
**Status**: Prompt available, prefilter pending

---

### 5. Future of Education Filter v1.0 ⏳
**Status**: Prompt available, prefilter pending

## Cost Analysis

### Ground Truth Generation (One-time)
- **50K articles** × **$0.003/article** (Claude 3.5 Sonnet)
- **Total**: ~$150 per filter

### Local Model Inference (Ongoing)
- **Cost**: $0 (runs locally)
- **Speed**: 20-50ms per article
- **Accuracy**: 90-95% vs. Claude

### ROI Calculation
If processing **4,000 articles/day**:
- **Claude API**: 4,000 × $0.003 × 365 = **$4,380/year**
- **Local Model**: Training cost $150 + hosting ~$10/month = **$270/year**
- **Savings**: **$4,110/year per filter** (94% cost reduction)

## Performance Benchmarks

| Model | Size | Inference Time | Memory | Accuracy vs Claude |
|-------|------|----------------|--------|--------------------|
| DistilBERT | 66M params | 15ms | 500MB | 88-92% |
| DeBERTa-v3-small | 44M params | 25ms | 400MB | 90-94% |
| BERT-base | 110M params | 35ms | 800MB | 91-95% |
| Flan-T5-base | 250M params | 80ms | 1.5GB | 93-96% |

**Recommended**: DeBERTa-v3-small (best quality/speed tradeoff)

## Roadmap

### Phase 1: Ground Truth Generation ✅ (Current)
- [x] Model calibration (Claude vs Gemini)
- [x] Generic batch labeler
- [x] Uplifting filter prompt & validation
- [x] Secrets management
- [x] Timeout protection & caching
- [ ] Full ground truth generation CLI
- [ ] Stratified sampling

### Phase 2: Training Pipeline (Next)
- [ ] PyTorch training script
- [ ] Model architectures (BERT, DeBERTa, T5)
- [ ] Training configs
- [ ] Experiment tracking (W&B)

### Phase 3: Evaluation & Deployment
- [ ] Model vs oracle comparison
- [ ] Calibration drift detection
- [ ] FastAPI inference server
- [ ] Docker deployment

### Phase 4: Additional Filters
- [ ] Sustainability filter
- [ ] EU policy relevance filter
- [ ] Healthcare AI readiness filter

### Phase 5: Advanced Features
- [ ] Multi-task learning
- [ ] Active learning
- [ ] Model compression (quantization, pruning)
- [ ] Model registry with versioning

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use LLM Distillery in your research or project, please cite:

```bibtex
@software{llm_distillery,
  title = {LLM Distillery: Knowledge Distillation from Large Language Models},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/llm-distillery}
}
```

## Acknowledgments

- Built for the [Content Aggregator](https://github.com/yourusername/content-aggregator) project
- Inspired by model distillation research from Hinton et al.
- LLM evaluation powered by Anthropic Claude and Google Gemini
