# LLM Distillery - Architecture

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLM DISTILLERY PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐
│ Raw Corpus   │  (99,763 articles from content-aggregator)
│ master_*.jsonl│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Prefilter    │  (Fast pattern matching - blocks 90%)
│ prefilter.py │  - Sustainability keyword check
└──────┬───────┘  - Vaporware detection
       │          - Lab-only detection
       ▼          → Passes ~10% (9,839 articles)
┌──────────────┐
│ Oracle       │  (Gemini Flash $0.001/article)
│ Labeling     │  - Smart content compression (~800 words)
│ batch_labeler│  - Multi-dimensional scoring (8 dimensions)
└──────┬───────┘  - JSON output validation
       │          → 2,186 labels so far
       ▼
┌──────────────┐
│ Ground Truth │
│ Curation     │  - Balance tier distribution
│ & Splits     │  - Stratified sampling
└──────┬───────┘  - 80/10/10 train/val/test
       │          → Target: 3,000-5,000 balanced examples
       ▼
┌──────────────┐
│ Training     │  (Unsloth + LoRA)
│ Qwen2.5-7B   │  - Instruction fine-tuning
│ Fine-tuning  │  - 4-bit quantization
└──────┬───────┘  - LoRA adapters (r=16, alpha=16)
       │          → Distilled student model
       ▼
┌──────────────┐
│ Local Model  │  (Free, fast inference)
│ Deployment   │  - 7B parameters
│ (Qwen LoRA)  │  - ~100 tok/sec on 16GB GPU
└──────────────┘  - Replicates oracle behavior
```

## Core Components

### 1. Prefilter System
- **Purpose:** Fast, cheap first-pass filtering before expensive LLM analysis
- **Location:** `filters/{filter_name}/v1/prefilter.py`
- **Responsibilities:**
  - Block obviously irrelevant content (90% blocked)
  - Detect vaporware/future-only/lab-only patterns
  - Apply keyword matching for domain relevance
  - Truncate content to first 2000 chars for speed
- **Key Dependencies:** None (regex-based, lightweight)
- **Performance:** ~10ms per article, saves ~$0.0009 per blocked article

### 2. Batch Labeler (Oracle)
- **Purpose:** Generate ground truth labels using cloud LLM
- **Location:** `ground_truth/batch_labeler.py`
- **Responsibilities:**
  - Load filter prompt templates
  - Apply smart content compression (~800 words / 3000 tokens)
  - Call LLM API (Gemini Flash / Gemini Pro)
  - Validate JSON output structure
  - Handle retries and rate limiting
  - Track costs and metrics
- **Key Dependencies:**
  - `google.generativeai` (Gemini API)
  - Filter prompt templates (`filters/{name}/v1/prompt-compressed.md`)
- **Interface:**
  ```python
  batch_labeler.py --filter filters/X/v1 \
                   --source input.jsonl \
                   --llm gemini-flash \
                   --target-count 2000 \
                   --output-dir ground_truth/labeled/X
  ```

### 3. Filter System
- **Purpose:** Define semantic dimensions and scoring logic per filter
- **Location:** `filters/{filter_name}/v1/`
- **Structure:**
  ```
  filters/sustainability_tech_deployment/v1/
  ├── config.yaml              # Dimension weights, thresholds
  ├── prompt-compressed.md     # Oracle prompt template
  ├── prefilter.py            # Fast pre-screening
  └── README.md               # Tier examples, scoring guide
  ```
- **Responsibilities:**
  - Define dimensions (e.g., deployment_stage, impact_magnitude)
  - Set dimension weights (must sum to 1.0)
  - Provide tier thresholds (e.g., deployed ≥8.0)
  - Document scoring rubric with examples

### 4. Dataset Curation
- **Purpose:** Create balanced training datasets from oracle labels
- **Location:** `scripts/` (various dataset manipulation scripts)
- **Responsibilities:**
  - Merge label batches from multiple runs
  - Analyze tier distribution
  - Apply stratified sampling or synthetic augmentation
  - Create train/val/test splits (80/10/10)
  - Validate dataset quality
- **Key Scripts:**
  - `merge_labels.py` - Combine batch files, deduplicate
  - `analyze_distribution.py` - Tier/dimension statistics
  - `create_splits.py` - Stratified train/val/test

### 5. Training Pipeline
- **Purpose:** Fine-tune Qwen2.5-7B on oracle-labeled data
- **Location:** `training/` (to be implemented)
- **Planned Approach:**
  - Framework: Unsloth (optimized for Qwen + LoRA)
  - Quantization: 4-bit (fit on 16GB GPU)
  - LoRA config: r=16, alpha=16, dropout=0.05
  - Training: 3 epochs, gradient checkpointing
  - Validation: MAE per dimension, tier accuracy
- **Key Dependencies:**
  - `unsloth` - Optimized fine-tuning library
  - `transformers` - HuggingFace model loading
  - `peft` - LoRA adapters

## Data Flow

### Input Format (from content-aggregator)
```json
{
  "id": "article-123",
  "title": "Solar Farm Deployed in Arizona",
  "content": "Full article text...",
  "description": "Brief summary",
  "published_date": "2025-01-15T10:30:00Z",
  "source": "energy_utilities_pv_magazine",
  "url": "https://..."
}
```

### Oracle Output Format
```json
{
  "id": "article-123",
  "title": "...",
  "content": "...",
  "sustainability_tech_deployment_analysis": {
    "dimensions": {
      "deployment_stage": 9,
      "impact_magnitude": 7,
      "evidence_quality": 8,
      ...
    },
    "overall_score": 8.2,
    "tier": "deployed",
    "reasoning": "300MW solar farm operational since 2024..."
  }
}
```

### Training Format (instruction tuning)
```json
{
  "instruction": "Analyze this article for tech deployment...",
  "input": "Title: ...\nContent: ...",
  "output": "{\"dimensions\": {...}, \"overall_score\": 8.2, ...}"
}
```

## Key Principles & Constraints

### Oracle-Student Consistency
**Critical:** Student model must see EXACTLY the same input as oracle during training
- ✅ Content truncation: ~800 words applied to both oracle and training data
- ✅ Smart compression: sentence-boundary aware, preserves key paragraphs
- ✅ Same prompt template used for oracle labeling and instruction fine-tuning

### Cost Optimization
- Prefilter blocks 90% → saves ~$0.0009 per article
- Smart compression reduces token count → fits Gemini Flash free tier
- Target: $2-3 per 3,000-label dataset (achievable with current strategy)

### Quality Over Speed
- Prefer real oracle labels over synthetic augmentation
- Accept labeling time (30-60 min for 2,000 articles) for quality
- Only use synthetic when corpus genuinely exhausted

### Tier Balance Requirements
- No tier should be <10% or >40% of final dataset
- Target: 15-25% per tier for 4-tier classification
- Oversampling allowed during training, not in base dataset

## Technical Constraints

### Hardware
- Training: 16GB GPU (RTX 4070 / similar)
- Inference: 8GB VRAM sufficient for 4-bit quantized Qwen

### API Limits
- Gemini Flash free tier: ~250 requests/day
- Rate limiting: 0.5-1 sec between requests to avoid throttling

### Token Limits
- Gemini Flash: 1M tokens input, 8K output
- Smart compression keeps articles under 3K tokens → comfortable margin
- Qwen context: 32K tokens (not a limiting factor)

## Current Gaps / TODO

- [ ] Training pipeline implementation (Unsloth integration)
- [ ] Evaluation framework (MAE per dimension, confusion matrix)
- [ ] Inference optimization (vLLM or similar for production)
- [ ] Multi-filter orchestration (how to combine multiple filters)
- [ ] Continuous learning (how to incorporate new oracle labels)
