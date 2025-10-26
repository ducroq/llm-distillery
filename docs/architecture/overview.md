# LLM Distillery Architecture

## Overview

LLM Distillery is designed for knowledge distillation from large language models into specialized semantic filters. The architecture follows a multi-stage pipeline optimized for both one-time ground truth generation and continuous model training.

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   LLM DISTILLERY PIPELINE                         │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────┐      ┌─────────────────┐      ┌────────────────┐
│   CALIBRATION   │      │  GROUND TRUTH   │      │    TRAINING    │
│                 │      │   GENERATION    │      │                │
│  Compare models │  →   │  Label dataset  │  →   │  Fine-tune     │
│  Claude/Gemini  │      │  with oracle    │      │  local model   │
└─────────────────┘      └─────────────────┘      └────────────────┘
         ↓                        ↓                        ↓
    Select best            50K labeled            90%+ accuracy
    LLM oracle             articles               vs. oracle

                                  ↓
                    ┌──────────────────────────┐
                    │      DEPLOYMENT          │
                    │                          │
                    │   Local model inference  │
                    │   <50ms, $0 cost         │
                    └──────────────────────────┘
```

## Core Components

### 1. Ground Truth Generation (`ground_truth/`)

**Purpose**: Generate labeled datasets using LLM oracles

**Key Modules**:
- `batch_labeler.py`: Generic batch labeling engine for any semantic filter
- `calibrate_models.py`: Compare Claude vs Gemini to select best oracle
- `secrets_manager.py`: Secure API key management
- `llm_evaluators.py`: LLM API wrappers (Claude, Gemini, GPT-4)
- `samplers.py`: Stratified sampling strategies
- `generate.py`: Main CLI for dataset creation (planned)

**Data Flow**:
```
Content Items (JSONL)
  ↓
GenericBatchLabeler
  ├→ Load prompt template
  ├→ Call LLM API (Claude/Gemini/GPT-4)
  ├→ Parse JSON response
  └→ Post-process (calculate tiers, scores)
  ↓
Labeled Dataset (JSONL)
```

### 2. Semantic Filter Prompts (`prompts/`)

**Purpose**: Define semantic dimensions and evaluation criteria

**Structure**:
- `uplifting.md`: 8-dimension framework for uplifting content
- `sustainability.md`: Climate tech impact assessment (planned)
- Custom prompts: Extensible for any domain

**Prompt Format**:
```markdown
# Filter Name

## Dimensions
1. dimension_1 (0-10): Description
2. dimension_2 (0-10): Description
...

## Tiers
- **impact** (7+): High-impact criteria
- **connection** (4-7): Medium-impact criteria
- **not_X** (0-4): Low-impact criteria

## JSON Output
{
  "dimension_1": 8.5,
  "dimension_2": 6.2,
  ...
  "tier": "impact",
  "overall_score": 7.8,
  "reasoning": "..."
}
```

### 3. Model Calibration (`calibrate_models.py`)

**Purpose**: Compare LLM providers to select best oracle for ground truth

**Process**:
1. Sample N articles (e.g., 100)
2. Label with Claude 3.5 Sonnet
3. Label with Gemini 1.5 Pro
4. Compare distributions and agreement
5. Generate comparison report
6. **Recommend**: Use Claude (higher quality) or Gemini (96% cheaper)

**Output**:
- Tier distribution comparison
- Score statistics (average, median, disagreement)
- Cost analysis
- Sample article comparisons
- Calibration report saved to `reports/`

**Caching**:
- Results cached to `calibrations/<filter_name>/`
- Files: `claude_labels.jsonl`, `gemini_labels.jsonl`
- Enables re-analysis without re-labeling

### 4. Training Pipeline (`training/` - Planned)

**Purpose**: Fine-tune small models on ground truth

**Architecture**:
```python
# Planned structure
training/
  ├── train.py              # Main training script
  ├── models.py             # Model architectures (BERT, DeBERTa, T5)
  ├── datasets.py           # PyTorch Dataset classes
  ├── configs/              # Training configurations
  │   └── uplifting_deberta.yaml
  └── utils.py              # Training utilities
```

**Target Models**:
- **DeBERTa-v3-small** (44M params): Best quality/speed tradeoff
- **DistilBERT** (66M params): Faster inference
- **BERT-base** (110M params): Higher accuracy
- **Flan-T5-base** (250M params): Best accuracy

### 5. Evaluation (`evaluation/` - Planned)

**Purpose**: Validate model quality vs. LLM oracle

**Metrics**:
- Accuracy (tier prediction)
- Mean Absolute Error (dimension scores)
- F1 Score (macro)
- Calibration drift detection

### 6. Inference (`inference/` - Planned)

**Purpose**: Deploy trained models for fast local prediction

**Components**:
- `predict.py`: Single article prediction
- `batch_predict.py`: Batch processing
- `serve.py`: FastAPI server
- `models/`: Trained model checkpoints

## Data Models

### Content Item (Input)

```python
{
  "id": str,           # Unique identifier
  "title": str,        # Article title
  "content": str,      # Full text
  "excerpt": str,      # Summary (optional)
  "url": str,          # Source URL
  "published_at": str, # ISO 8601 timestamp
  "source": str        # Origin (e.g., "arxiv", "medium")
}
```

### Labeled Article (Output)

```python
{
  # Original fields
  "id": str,
  "title": str,
  "content": str,
  ...

  # LLM evaluation
  "dimension_1": float,  # 0-10 score
  "dimension_2": float,
  ...
  "tier": str,           # "impact" | "connection" | "not_X"
  "overall_score": float,# Weighted average (0-10)
  "reasoning": str,      # LLM explanation

  # Metadata
  "model": str,          # "claude-3.5-sonnet" | "gemini-1.5-pro"
  "prompt": str,         # Prompt used
  "timestamp": str       # When labeled
}
```

## Configuration Management

### API Keys (`config/credentials/secrets.ini`)

```ini
[api_keys]
anthropic_api_key = sk-ant-...
gemini_api_key = AIza...
openai_api_key = sk-...

[email_credentials]
smtp_server = smtp.gmail.com
smtp_port = 587
...
```

**Priority**:
1. Environment variables (CI/CD, GitHub Actions)
2. `secrets.ini` file (local development)

**Secrets Manager**:
- Loads from `config/credentials/secrets.ini`
- Falls back to environment variables
- Prioritizes Gemini billing key (150 RPM) over free tier (2 RPM)

## Technology Stack

### Core Dependencies
- **Python 3.10+**
- **Anthropic** (`anthropic`): Claude API
- **Google Generative AI** (`google-generativeai`): Gemini API
- **OpenAI** (`openai`): GPT-4 API (optional)

### Training Dependencies (Planned)
- **PyTorch**: Model training
- **Transformers** (Hugging Face): Pre-trained models
- **Weights & Biases**: Experiment tracking

### Serving Dependencies (Planned)
- **FastAPI**: Inference API server
- **ONNX**: Model optimization

## Performance Characteristics

### Calibration
- **Input**: 100 articles
- **Time**: ~5-10 minutes (100 articles × 2 models)
- **Cost**: ~$0.60 (100 × $0.003 Claude + 100 × $0.0003 Gemini)
- **Success rate**: 100% with 60s timeout protection

### Ground Truth Generation (Planned)
- **Input**: 50,000 articles
- **Time**: 12-24 hours (with rate limiting)
- **Cost**: ~$150 (50K × $0.003)
- **Output**: Labeled dataset for training

### Local Model Inference (Planned)
- **Latency**: 20-50ms per article
- **Cost**: $0 (runs locally)
- **Accuracy**: 90-95% vs. Claude
- **Throughput**: 20-50 articles/second

## Error Handling

### Timeout Protection
- **LLM API calls**: 60-second timeout using threading
- **Cross-platform**: Works on Windows/Linux/Mac
- **Graceful degradation**: Returns `None` on timeout, continues processing

### Rate Limiting
- **Claude**: 50 RPM (Tier 1)
- **Gemini Free**: 2 RPM
- **Gemini Billing**: 150 RPM
- **Strategy**: Exponential backoff on rate limit errors

### Retry Logic
- **Transient errors**: Retry up to 3 times
- **API errors**: Log and skip article
- **JSON parsing**: Attempt to extract valid JSON, fallback to None

## Security

### API Key Management
- **NEVER commit** `config/credentials/secrets.ini` to git
- **gitignore**: Excludes all credential files
- **Environment variables**: Preferred for CI/CD
- **Secrets Manager**: Single source of truth for credentials

### Data Privacy
- **No logging**: Article content never logged to console/files
- **Local processing**: All data stays on your machine
- **No telemetry**: No data sent to third parties (except LLM APIs)

## Scalability

### Current Limitations
- **Sequential processing**: One article at a time
- **No caching**: Re-labels same article if run twice
- **No resume**: Starts from scratch on failure

### Future Enhancements
- **Parallel processing**: Async batch labeling
- **Smart caching**: Skip already-labeled articles
- **Resumable jobs**: Save progress, resume from checkpoint
- **Distributed training**: Multi-GPU support

## Directory Structure

```
llm-distillery/
├── ground_truth/              # Ground truth generation
│   ├── __init__.py
│   ├── batch_labeler.py       # Generic batch labeler ✅
│   ├── calibrate_models.py    # Model calibration ✅
│   ├── secrets_manager.py     # API key management ✅
│   ├── llm_evaluators.py      # LLM API wrappers
│   ├── samplers.py            # Sampling strategies
│   └── generate.py            # Main CLI (planned)
│
├── prompts/                   # Semantic filter prompts
│   ├── uplifting.md          # Uplifting content filter ✅
│   └── sustainability.md      # Sustainability filter (planned)
│
├── calibrations/              # Calibration cache (generated)
│   └── <filter_name>/
│       ├── claude_labels.jsonl
│       └── gemini_labels.jsonl
│
├── reports/                   # Calibration reports (generated)
│   └── <filter>_calibration.md
│
├── datasets/                  # Generated datasets (planned)
│   ├── .gitkeep
│   └── splits/
│
├── training/                  # Model training (planned)
│   ├── train.py
│   ├── models.py
│   └── configs/
│
├── evaluation/                # Model evaluation (planned)
│   ├── evaluate.py
│   └── metrics.py
│
├── inference/                 # Deployed models (planned)
│   ├── predict.py
│   ├── batch_predict.py
│   └── models/
│
├── tests/                     # Unit tests
│   └── (to be added)
│
└── docs/                      # Documentation
    ├── architecture/          # Architecture docs
    ├── guides/                # User guides
    ├── api/                   # API reference
    ├── calibration/           # Calibration guides
    └── examples/              # Code examples
```

## Next Steps

See [docs/guides/roadmap.md](../guides/roadmap.md) for implementation plan.
