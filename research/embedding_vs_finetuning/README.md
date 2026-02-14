# Multilingual Embedding vs Fine-Tuning Research

**Research Question:** Can frozen **multilingual** embeddings + probes match fine-tuned Qwen2.5-1.5B on semantic dimension scoring?

**Critical Note:** The dataset contains multiple languages (English, French, Spanish, Portuguese). All models must support multilingual text.

## Background

The LLM Distillery project uses fine-tuned Qwen2.5-1.5B models (18.5M trainable parameters via LoRA) to score articles on semantic dimensions. This research explores whether simpler embedding-based approaches can achieve comparable results with lower training cost and faster inference.

## Current Baseline

| Metric | Value |
|--------|-------|
| Model | Qwen2.5-1.5B + LoRA |
| Trainable Parameters | 18.5M |
| Performance (MAE) | 0.68 |
| Inference Time | 20-50ms per article |
| Training Time | 2-3 hours on GPU |

## Current Experiment: Multilingual Models

### Embedding Models to Test

| Model | Languages | Dims | Max Tokens | Purpose |
|-------|-----------|------|------------|---------|
| `multilingual-MiniLM-L12-v2` | 50+ | 384 | 128 | Fast baseline (heavy truncation) |
| `multilingual-mpnet-base-v2` | 50+ | 768 | 128 | Quality baseline (heavy truncation) |
| `multilingual-e5-large` | 100+ | 1024 | 512 | Strong MTEB performer |
| `bge-m3` | 100+ | 1024 | 8192 | **Key test**: multilingual + long context |

### Previous (Invalid) English-Only Models

| Model | Dimensions | Notes |
|-------|------------|-------|
| ~~`all-MiniLM-L6-v2`~~ | 384 | **Invalid**: English-only |
| ~~`all-mpnet-base-v2`~~ | 768 | **Invalid**: English-only |
| ~~`bge-large-en-v1.5`~~ | 1024 | **Invalid**: English-only |
| ~~`e5-large-v2`~~ | 1024 | **Invalid**: English-only |

### Probe Methods

1. **Ridge Regression** - Linear probe with L2 regularization (baseline)
2. **2-Layer MLP** - Tests non-linear combinations (256 -> 128 -> output)
3. **LightGBM** - Tree-based alternative

### Datasets

| Filter | Articles | Dimensions | Baseline MAE |
|--------|----------|------------|--------------|
| `uplifting_v5` | 10,000 | 6 | 0.68 |
| `sustainability_technology_v2` | 7,990 | 8 | 0.71 |
| `investment-risk_v5` | 10,000 | 8 | 0.48 |

## Execution Plan (Remote GPU Server)

### Prerequisites

```bash
pip install -r research/embedding_vs_finetuning/requirements.txt
```

### Phase 1: Sync & Generate Embeddings

```bash
# Sync code to remote
rsync -avz --exclude '__pycache__' --exclude '*.pyc' --exclude 'node_modules' \
  research/embedding_vs_finetuning/ llm-distiller:~/llm-distillery/research/embedding_vs_finetuning/

# SSH and run
ssh llm-distiller
cd ~/llm-distillery

# Generate embeddings (run sequentially to avoid OOM)
python research/embedding_vs_finetuning/embed_articles.py --dataset uplifting_v5 --models multilingual-MiniLM-L12-v2
python research/embedding_vs_finetuning/embed_articles.py --dataset uplifting_v5 --models multilingual-mpnet-base-v2
python research/embedding_vs_finetuning/embed_articles.py --dataset uplifting_v5 --models multilingual-e5-large
python research/embedding_vs_finetuning/embed_articles.py --dataset uplifting_v5 --models bge-m3
```

### Phase 2: Train Probes

```bash
python research/embedding_vs_finetuning/train_probes.py \
    --dataset uplifting_v5 \
    --models multilingual-MiniLM-L12-v2 multilingual-mpnet-base-v2 multilingual-e5-large bge-m3
```

### Phase 3: Sync & Evaluate

```bash
# Sync results back
rsync -avz llm-distiller:~/llm-distillery/research/embedding_vs_finetuning/embeddings/ \
  research/embedding_vs_finetuning/embeddings/
rsync -avz llm-distiller:~/llm-distillery/research/embedding_vs_finetuning/results/ \
  research/embedding_vs_finetuning/results/

# Evaluate
python research/embedding_vs_finetuning/evaluate.py --dataset uplifting_v5
```

### Phase 4: Benchmarks (Remote GPU)

```bash
python research/embedding_vs_finetuning/benchmark_speed.py \
    --dataset uplifting_v5 \
    --models multilingual-MiniLM-L12-v2 multilingual-mpnet-base-v2 multilingual-e5-large bge-m3
```

### Phase 5: Error Analysis

```bash
python research/embedding_vs_finetuning/analyze_error_distribution.py \
    --dataset uplifting_v5 \
    --models multilingual-MiniLM-L12-v2 multilingual-mpnet-base-v2 multilingual-e5-large bge-m3 \
    --probe mlp --device cpu
```

### Phase 6: Generate Report

```bash
python research/embedding_vs_finetuning/generate_report.py \
    --dataset uplifting_v5 \
    --output research/embedding_vs_finetuning/results/Multilingual_Embedding_Research_Report.docx
```

## Directory Structure

```
research/embedding_vs_finetuning/
├── README.md                 # This file
├── config.yaml               # Experiment configuration
├── embed_articles.py         # Generate embeddings
├── train_probes.py           # Train linear/MLP/LightGBM probes
├── evaluate.py               # Evaluate on test set
├── compare_results.py        # Generate reports and visualizations
├── embeddings/               # Cached embeddings (gitignored)
│   └── *.npz                 # Numpy compressed format
└── results/                  # Experiment results
    ├── *.pkl                 # Trained probes
    ├── *_evaluation_results.json
    └── *_comparison_report.md
```

## Previous Findings (Invalid - English-Only Models)

Previous results used English-only embedding models on a multilingual dataset. Those findings are invalid and should not be used for decision-making.

### Semantic Dimension Scoring (Uplifting Filter) - INVALID
**Result: Invalid due to language mismatch**

| Approach | MAE | Verdict |
|----------|-----|---------|
| Fine-tuned Qwen2.5-1.5B | 0.68 | Baseline |
| ~~e5-large-v2 + MLP~~ | ~~0.86~~ | **Invalid** (English-only model) |

The poor performance was partially due to using English-only models on multilingual data.

### Binary Classification (Commerce Detection)
**Result: Embedding approach wins**

| Approach | F1 | Verdict |
|----------|-----|---------|
| all-mpnet-base-v2 + MLP | **98.3%** | Best |
| Fine-tuned DistilBERT | 97.8% | Baseline |

**Note:** This result may still be valid if commerce data is primarily English. To be verified.

## Success Criteria (New Multilingual Experiment)

| Outcome | MAE | Interpretation |
|---------|-----|----------------|
| **Closes gap** | ~0.68-0.75 | Multilingual embeddings work well |
| **Partial improvement** | ~0.75-0.85 | Some value, fine-tuning still better |
| **No improvement** | >0.85 | Gap is fundamental to task-specific learning |

## Key Questions to Answer

1. Does BGE-M3 (multilingual + 8K context) close the gap?
2. How much does truncation hurt short-context models (128 tokens)?
3. Is there a quality/speed trade-off worth considering?
4. Which probe type (Ridge, MLP, LightGBM) works best?

## Dependencies

```bash
pip install sentence-transformers scikit-learn lightgbm torch transformers pandas matplotlib pyyaml scipy tqdm
```

## Notes

- Embeddings are cached to disk (~50MB per model per dataset)
- MLP training uses early stopping with patience=10
- LightGBM trains one model per dimension
- All experiments use random seed 42 for reproducibility
