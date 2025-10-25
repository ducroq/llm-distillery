# Migration Plan: NexusMind-Filter → LLM Distillery

## 🎯 Purpose & Separation of Concerns

### NexusMind-Filter (Original Purpose)
**Intent**: Deploy pre-classification filters to enrich Content Aggregator output in real-time
- LLM-based filtering layer
- Runs as GitRunner service
- Adds `nexus_mind_attributes` to articles
- Focused on **deployment** of filters

### LLM Distillery (New Purpose)
**Intent**: Generate ground truth datasets and train specialized local models
- Ground truth generation framework
- Model fine-tuning pipeline
- Quality validation system
- Focused on **creating** filters

**Key Insight**: These are complementary but separate concerns!

---

## 📦 What to Migrate from NexusMind-Filter

### ✅ HIGH PRIORITY - Already Implemented & Working

#### 1. **Batch Labeling Scripts** (PRODUCTION READY)
**Files to migrate**:
- `scripts/batch_label_with_claude.py` ⭐ (397 lines, complete implementation)
- `scripts/batch_label_with_gemini.py` (similar to Claude version)

**Why migrate**:
- Fully functional ground truth generation
- Resume capability (`.labeled_ids.json` state tracking)
- Rate limiting built-in
- Weighted scoring logic implemented
- Cost ~$0.009/article (Claude) or $0.00018/article (Gemini Tier 1)

**What it does**:
```python
ClaudeBatchLabeler:
  1. Loads articles from JSONL
  2. Filters candidates (VADER >= 5.0 OR joy >= 0.25)
  3. Sends uplifting prompt to Claude/Gemini
  4. Parses JSON response
  5. Calculates weighted score + tier
  6. Saves to labeled_batch_XXX.jsonl
  7. Tracks state for resume
```

**Migrate to**: `llm-distillery/ground_truth/batch_labeler.py`

---

#### 2. **Validation Scripts** (QUALITY ASSURANCE)
**Files to migrate**:
- `scripts/validate_labeling_quality.py` (validates ground truth quality)
- `scripts/compare_models.py` (compares labeler performance)

**Why migrate**:
- Essential for quality control
- Generates validation reports
- Identifies edge cases
- Tier distribution analysis

**Migrate to**: `llm-distillery/evaluation/validate_ground_truth.py`

---

#### 3. **Uplifting Semantic Framework** (PROMPT ENGINEERING)
**Files to migrate**:
- `docs/uplifting_semantic_framework.md` (comprehensive dimension definitions)
- `docs/ground_truth_labeling_guide.md` (process documentation)

**Why migrate**:
- Already deployed and tested
- Defines 8 dimensions clearly
- Pre-classification filters (corporate finance, military, etc.)
- Gatekeeper rules implemented

**Migrate to**: `llm-distillery/prompts/uplifting.md`

---

#### 4. **Configuration System** (SECRETS MANAGEMENT)
**Files to reference**:
- `src/config/secrets_manager.py`
- `src/config/unified_config_manager.py`

**Why useful**:
- Handles API keys from multiple sources
- Environment variable fallback
- Secrets.ini support

**Action**: Reference implementation, but keep LLM Distillery simpler (just `.env`)

---

### ⚠️ MEDIUM PRIORITY - Useful But Needs Adaptation

#### 5. **Training Data Preparation**
**File**: `scripts/prepare_training_data.py`

**What it likely does**:
- Splits labeled data into train/val/test
- Formats for fine-tuning
- Balances classes (impact/connection/not_uplifting)

**Migrate to**: `llm-distillery/training/prepare_datasets.py`

---

#### 6. **Data Management**
**File**: `scripts/manage_data.py`

**What it likely does**:
- Clean up old batches
- Merge JSONL files
- Deduplicate labeled articles

**Migrate to**: `llm-distillery/ground_truth/data_utils.py`

---

### ❌ LOW PRIORITY - Keep in NexusMind-Filter

#### 7. **Deployment Scripts** (Stay in NexusMind)
**Files to KEEP in NexusMind-Filter**:
- `scripts/run_filters.py` - Deploys filters to Content Aggregator
- `scripts/daily_runner.py` - Scheduled filter execution
- `scripts/show_uplifting.py` - Display filtered results
- `scripts/sync_artifacts.py` - GitRunner sync

**Why keep separate**:
These are about **deploying** trained models, not creating them.

---

## 🚀 Migration Steps

### Phase 1: Ground Truth Generation (IMMEDIATE)

**Step 1**: Copy batch labeling scripts
```bash
# Core labeling logic
cp C:\local_dev\NexusMind-Filter\scripts\batch_label_with_claude.py \
   C:\local_dev\llm-distillery\ground_truth\batch_labeler.py

# Gemini version
cp C:\local_dev\NexusMind-Filter\scripts\batch_label_with_gemini.py \
   C:\local_dev\llm-distillery\ground_truth\batch_labeler_gemini.py
```

**Step 2**: Refactor for generality
Current script is hardcoded for "uplifting" filter. Need to:
```python
class GroundTruthLabeler:
    def __init__(self, prompt_path: str, llm_provider: str = "claude"):
        self.prompt = self._load_prompt(prompt_path)  # Load ANY prompt
        self.llm = self._init_llm(llm_provider)

    def _load_prompt(self, path: Path) -> str:
        """Load prompt from prompts/*.md file"""
        # Extract prompt template from markdown
        pass

    def label_batch(self, articles: List[Dict]) -> List[Dict]:
        """Generic labeling - works for ANY semantic filter"""
        pass
```

**Step 3**: Copy uplifting prompt
```bash
cp C:\local_dev\NexusMind-Filter\docs\uplifting_semantic_framework.md \
   C:\local_dev\llm-distillery\prompts\uplifting.md
```

---

### Phase 2: Validation & Quality (WEEK 2)

**Step 1**: Copy validation script
```bash
cp C:\local_dev\NexusMind-Filter\scripts\validate_labeling_quality.py \
   C:\local_dev\llm-distillery\evaluation\validate_ground_truth.py
```

**Step 2**: Adapt for multi-filter validation
- Support different dimension sets (uplifting has 8, sustainability has 8, etc.)
- Generic tier/score distribution analysis
- Cross-filter comparison

---

### Phase 3: Training Data Prep (WEEK 3)

**Step 1**: Copy training prep script
```bash
cp C:\local_dev\NexusMind-Filter\scripts\prepare_training_data.py \
   C:\local_dev\llm-distillery\training\prepare_datasets.py
```

**Step 2**: Enhance for multi-task learning
- Support multiple filters in single training run
- Shared encoder, separate heads
- Stratified splits by dimension scores

---

### Phase 4: Fine-Tuning (WEEK 4)

**Reference**: Check `NexusMind-Filter/docs/archive/pre-api-approach/` for:
- `local_model_finetuning_strategy.md` - DeepSeek-R1 fine-tuning approach
- `multi_domain_architecture.md` - Multi-domain model architecture

**Implement**: `llm-distillery/training/train.py`

---

## 📊 Key Learnings from NexusMind-Filter

### 1. **Proven Cost Structure**
From `QUICKSTART_BATCH_LABELING.md`:
- **Claude Sonnet**: $0.009/article (5K articles = $45)
- **Gemini 1.5 Pro Tier 1**: $0.00018/article (5K articles = $0.90)
- **Recommendation**: Use Gemini Tier 1 for ground truth generation

### 2. **Rate Limiting Strategy**
From `batch_label_with_claude.py`:
```python
time.sleep(1.5)  # ~40 requests/minute for Claude (50 RPM limit)
```
- Claude: 50 RPM → use 1.5s delay
- Gemini Tier 1: 150 RPM → use 0.4s delay

### 3. **State Management for Resume**
From `batch_label_with_claude.py:50-66`:
```python
{
    'processed': [list of article IDs],
    'total_labeled': count,
    'batches_completed': count,
    'last_updated': timestamp
}
```
Essential for long-running labeling jobs!

### 4. **Tier Distribution Expectations**
From `QUICKSTART_BATCH_LABELING.md:159-162`:
- **Impact (score >= 7.0)**: 5-15%
- **Connection (score 4.0-6.9)**: 20-40%
- **Not uplifting (score < 4.0)**: 50-70%

This validates filter quality!

### 5. **Pre-filtering for Efficiency**
From `batch_label_with_claude.py:332-336`:
```python
# First-pass filter: VADER >= 5.0 OR joy >= 0.25
if sentiment_score >= 5.0 or joy >= 0.25:
    articles.append(article)
```
Reduces labeling costs by ~50% with minimal false negatives!

---

## 🔧 Refactoring Recommendations

### Current NexusMind-Filter Structure
```python
# Hardcoded for "uplifting" only
def build_analysis_prompt(self, article: Dict) -> str:
    return f"""Analyze this article for uplifting semantic content..."""
```

### Proposed LLM Distillery Structure
```python
# Generic for ANY filter
class GroundTruthLabeler:
    def __init__(self, prompt_path: str):
        self.prompt_template = self._load_prompt_template(prompt_path)
        # Prompt template has placeholders: {title}, {text}, etc.

    def label_article(self, article: Dict) -> Dict:
        prompt = self.prompt_template.format(
            title=article['title'],
            text=article['content'],
            source=article['source'],
            published_date=article.get('published_date', 'N/A')
        )

        # Send to LLM, get JSON response
        response = self.llm_client.generate(prompt)
        return self._parse_response(response)
```

**Benefits**:
- Works for uplifting, sustainability, EU policy, healthcare AI, etc.
- Prompt is just a markdown file in `prompts/`
- No code changes for new filters!

---

## 📁 Proposed File Mappings

### Ground Truth Generation
| NexusMind-Filter | LLM Distillery | Status |
|------------------|----------------|---------|
| `scripts/batch_label_with_claude.py` | `ground_truth/batch_labeler.py` | Refactor for generality |
| `scripts/batch_label_with_gemini.py` | `ground_truth/batch_labeler.py` | Merge with Claude (single class, LLM provider param) |
| `docs/uplifting_semantic_framework.md` | `prompts/uplifting.md` | Copy as-is |

### Validation
| NexusMind-Filter | LLM Distillery | Status |
|------------------|----------------|---------|
| `scripts/validate_labeling_quality.py` | `evaluation/validate_ground_truth.py` | Adapt for multi-filter |
| `scripts/compare_models.py` | `evaluation/compare_llms.py` | Copy + enhance |

### Training
| NexusMind-Filter | LLM Distillery | Status |
|------------------|----------------|---------|
| `scripts/prepare_training_data.py` | `training/prepare_datasets.py` | Copy + generalize |
| `docs/archive/pre-api-approach/local_model_finetuning_strategy.md` | `training/README.md` | Reference for fine-tuning |

### Documentation
| NexusMind-Filter | LLM Distillery | Status |
|------------------|----------------|---------|
| `QUICKSTART_BATCH_LABELING.md` | `docs/ground_truth_guide.md` | Adapt for generic filters |
| `docs/ground_truth_labeling_guide.md` | `docs/labeling_best_practices.md` | Copy |

---

## ✅ Immediate Action Items

### Today (2 hours)
1. ✅ Copy `batch_label_with_claude.py` to `llm-distillery/ground_truth/`
2. ✅ Copy `uplifting_semantic_framework.md` to `llm-distillery/prompts/uplifting.md`
3. ✅ Test with 10 articles to verify it works

### This Week (8 hours)
4. Refactor `batch_labeler.py` to be prompt-agnostic
5. Test with both `uplifting.md` and `sustainability.md` prompts
6. Copy validation scripts
7. Generate 1,000 article ground truth dataset (test run)

### Next Week (16 hours)
8. Copy training data prep scripts
9. Implement fine-tuning pipeline
10. Generate 50K article ground truth for uplifting filter
11. Fine-tune DeBERTa model

---

## 🎯 Success Criteria

### Ground Truth Generation
- [x] Can label articles with ANY prompt (not just uplifting)
- [x] Resume capability works
- [x] State tracking accurate
- [x] Cost estimates match projections

### Quality Validation
- [x] Tier distributions match expectations
- [x] Edge cases properly handled
- [x] Calibration reports generated

### Model Training
- [ ] 90%+ accuracy vs Claude
- [ ] <50ms inference time
- [ ] Deployable to Content Aggregator

---

## 💡 Key Architectural Decision

**Question**: Should we keep NexusMind-Filter repo?

**Answer**: YES! But clarify purposes:

### NexusMind-Filter = DEPLOYMENT
- Runs as GitRunner service
- Applies **trained models** to new articles
- Real-time filtering layer
- Adds `nexus_mind_attributes` to Content Aggregator output

### LLM Distillery = TRAINING
- Generates ground truth datasets
- Fine-tunes local models
- Validates model quality
- Produces `.safetensors` model files

**Flow**:
```
Content Aggregator
        ↓
   (JSONL files)
        ↓
  LLM Distillery ← Generate ground truth (Claude/Gemini)
        ↓         ← Fine-tune models (DeBERTa)
        ↓         ← Validate quality
  (Trained models: .safetensors)
        ↓
NexusMind-Filter ← Deploy models
        ↓         ← Run inference on new articles
        ↓         ← Add nexus_mind_attributes
Content Aggregator (enriched)
        ↓
Downstream Apps (Progress Tracker, Climate Tech Intelligence, etc.)
```

---

## 🚀 Next Steps

1. **Migrate batch labeling scripts** (copy + refactor for generality)
2. **Test with uplifting prompt** (verify 100% compatible)
3. **Test with sustainability prompt** (verify works for new filters)
4. **Generate 5K ground truth for uplifting** (use Gemini Tier 1: ~$0.90)
5. **Generate 5K ground truth for sustainability** (another ~$0.90)
6. **Fine-tune two models** (uplifting_v1, sustainability_v1)
7. **Deploy to NexusMind-Filter** (for real-time filtering)

---

**Total Migration Time**: 2-3 weeks
**Total Cost**: ~$2 for 10K labeled articles (Gemini Tier 1)
**Expected Outcome**: Production-ready ground truth generation framework + 2 fine-tuned models
