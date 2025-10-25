# Getting Started with LLM Distillery

Welcome to **LLM Distillery**! This guide will help you get started with generating ground truth datasets and training specialized semantic filters.

## 🎯 What You'll Build

Transform expensive LLM evaluations (Claude/Gemini) into fast, cheap local models:
- **Ground truth**: 50K articles rated by Claude ($150 one-time cost)
- **Local model**: 90%+ accuracy, <50ms inference, $0 ongoing cost
- **Savings**: ~$4,000/year per filter

## 📋 Prerequisites

1. **Python 3.10+**
2. **API Keys**:
   - Anthropic Claude API key (for ground truth generation)
   - Optional: Google Gemini or OpenAI GPT-4
3. **Content Aggregator** data (JSONL files from your aggregator)

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
cd C:\local_dev\llm-distillery

# Using pip
pip install -r requirements.txt

# Or using Poetry (recommended)
poetry install
```

### 2. Configure API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API key
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Verify Content Aggregator Data

Make sure you have JSONL files in your content aggregator:

```bash
ls ../content-aggregator/data/collected/*.jsonl
```

## 📊 Creating Your First Filter: Sustainability

### Step 1: Review the Prompt

The sustainability filter prompt is in `prompts/sustainability.md`. It defines:
- 8 dimensions to score (climate impact, technical credibility, etc.)
- Pre-classification filters (greenwashing, vaporware detection)
- Investment signals to extract
- JSON output format

**Review it** to understand what the LLM will evaluate.

### Step 2: Generate Ground Truth (50K articles)

```bash
python -m ground_truth.generate \
    --prompt prompts/sustainability.md \
    --input-dir ../content-aggregator/data/collected \
    --output datasets/sustainability_50k.jsonl \
    --num-samples 50000 \
    --llm claude
```

**Expected**:
- Time: ~12-24 hours (with rate limiting)
- Cost: ~$150 (50K × $0.003/article)
- Output: `datasets/sustainability_50k.jsonl` with rated articles

**Note**: Currently this is a scaffold - full implementation coming soon!

### Step 3: Train Local Model

```bash
python -m training.train \
    --config training/configs/sustainability_deberta.yaml \
    --dataset datasets/sustainability_50k.jsonl \
    --output inference/models/sustainability_v1
```

**Expected**:
- Time: 4-8 hours on single GPU
- Accuracy: 90-94% vs Claude
- Output: Trained model in `inference/models/sustainability_v1/`

### Step 4: Evaluate Quality

```bash
python -m evaluation.evaluate \
    --model inference/models/sustainability_v1 \
    --test-set datasets/splits/sustainability_test.jsonl \
    --oracle claude
```

**Expected Output**:
```
Model Evaluation Results
========================
Accuracy: 92.3%
Mean Absolute Error: 0.08
F1 Score (macro): 0.91

Dimension-wise Performance:
  climate_impact_potential: MAE 0.07
  technical_credibility: MAE 0.06
  economic_viability: MAE 0.09
  ...
```

### Step 5: Deploy to Content Aggregator

```bash
# Copy trained model
cp -r inference/models/sustainability_v1 ../content-aggregator/models/

# Update aggregator config to use local model
# (See content-aggregator docs for integration)
```

## 🎨 Creating Additional Filters

You can create filters for any semantic dimension!

### Template for New Filters

1. **Create prompt** in `prompts/your_filter.md`:
   - Define dimensions to score
   - Add pre-classification filters
   - Specify JSON output format

2. **Generate ground truth**:
   ```bash
   python -m ground_truth.generate --prompt prompts/your_filter.md --output datasets/your_filter_50k.jsonl
   ```

3. **Train model**:
   ```bash
   python -m training.train --dataset datasets/your_filter_50k.jsonl --output inference/models/your_filter_v1
   ```

### Example Filters to Build

From your downstream applications plan:

1. **EU Policy Relevance** (`prompts/eu_policy.md`)
   - Regulatory impact, compliance relevance, timeline urgency

2. **Healthcare AI Readiness** (`prompts/healthcare_ai.md`)
   - Clinical evidence level, regulatory status, adoption readiness

3. **Uplifting Content** (`prompts/uplifting.md`)
   - Agency, progress, collective benefit, connection, resilience

## 📁 Project Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. CREATE PROMPT                                            │
│     prompts/your_filter.md                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. GENERATE GROUND TRUTH                                    │
│     ground_truth/generate.py                                │
│     → datasets/your_filter_50k.jsonl                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. TRAIN MODEL                                              │
│     training/train.py                                       │
│     → inference/models/your_filter_v1/                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. EVALUATE                                                 │
│     evaluation/evaluate.py                                  │
│     → accuracy reports, calibration metrics                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  5. DEPLOY                                                   │
│     Copy to content-aggregator/models/                      │
│     → Fast local inference on new articles                  │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Current Implementation Status

**✅ Completed**:
- Project structure
- Sustainability filter prompt
- Ground truth generation scaffolding
- LLM evaluator interfaces (Claude, Gemini, GPT-4)
- Stratified sampling logic

**🚧 In Progress** (You'll need to complete):
- Full ground truth generation pipeline
- Training scripts
- Evaluation scripts
- Inference server

**📝 Next Steps**:
1. Implement full `ground_truth/generate.py` pipeline
2. Add training configs and scripts
3. Build evaluation framework
4. Create inference server

## 💡 Tips

### Cost Optimization
- **Start small**: Test with 1,000 samples first ($3)
- **Use caching**: LLM evaluators can cache prompts
- **Resume capability**: Generate in batches, resume if interrupted

### Quality Optimization
- **Stratified sampling**: Ensure diverse representation
- **Active learning**: Sample uncertain cases for more training data
- **Continuous calibration**: Re-evaluate quarterly to detect drift

### Speed Optimization
- **Batch processing**: Rate multiple articles per API call
- **Parallel requests**: Use async for faster generation
- **Model quantization**: Reduce model size for faster inference

## 📚 Further Reading

- [Main README](README.md) - Full project documentation
- [Sustainability Filter Prompt](prompts/sustainability.md) - Example filter
- [Content Aggregator Docs](../content-aggregator/README.md) - Integration guide
- [Downstream Applications Plan](../content-aggregator/docs/separate-projects/implementation-plan-downstream-apps.md) - Use cases

## 🆘 Troubleshooting

**Issue**: API rate limits
**Solution**: Add `time.sleep(1)` between requests, or use exponential backoff

**Issue**: Out of memory during training
**Solution**: Reduce batch size in training config, or use gradient accumulation

**Issue**: Model accuracy too low (<85%)
**Solution**: Generate more ground truth samples, or try different base model (DeBERTa vs BERT)

---

**Ready to start?** Begin with the sustainability filter following the steps above!
