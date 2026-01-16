---
license: mit
language:
- en
- nl
- de
- fr
- es
- pt
- it
- zh
- sv
- da
- "no"
- fi
- pl
- cs
- ru
- uk
- el
- hu
- ro
- tr
- ar
tags:
- multilingual
- text-classification
- content-filtering
- multi-dimensional-scoring
- knowledge-distillation
- sustainability
- climate-tech
- clean-energy
- LCSA
- news-analysis
- environmental
library_name: peft
base_model: Qwen/Qwen2.5-1.5B
pipeline_tag: text-classification
metrics:
- mae
- rmse
model-index:
- name: sustainability-technology-v2
  results:
  - task:
      type: text-classification
      name: Multi-dimensional Regression
    metrics:
    - type: mae
      value: 0.717
      name: Test MAE
    - type: mae
      value: 0.654
      name: Validation MAE
---

# Sustainability Technology Filter v2

A fine-tuned **Qwen2.5-1.5B** model with LoRA adapters for multi-dimensional sustainability technology assessment. This model evaluates news articles across **6 LCSA-based dimensions** to identify genuinely impactful sustainability technologies - not just greenwashing or speculation.

## Model Description

### Purpose

This model is designed for automated filtering of sustainability and clean technology news. It scores articles on multiple dimensions derived from the **Life Cycle Sustainability Assessment (LCSA)** framework, enabling:

- **Content curation**: Identify high-quality sustainability technology articles
- **Trend analysis**: Track technology readiness and deployment patterns
- **Research filtering**: Separate substantive innovations from hype

### Key Features

- **Multi-dimensional scoring**: 6 independent LCSA dimensions (0-10 scale)
- **Explicit scope boundaries**: Trained to reject AI/ML papers, consumer electronics, programming tutorials
- **Multilingual support**: 21 languages including EN, DE, FR, ES, PT, NL, ZH, and more
- **Evidence-based**: Focuses on documented deployments and metrics, not announcements

### What's New in v2

- **Improved scope filtering**: Explicit exclusions for off-topic content (AI/ML infrastructure, consumer electronics, military tech)
- **Better dimension independence**: Max correlation 0.61 (down from 0.80+ in v1)
- **Multilingual prefilter**: Keywords in 21 languages for global coverage
- **Lower MAE**: 0.654 validation MAE (vs 0.712 in v1)

---

## Dimensions

The model scores articles on 6 dimensions from the LCSA framework:

### Technology Assessment

| Dimension | Weight | Range | Question |
|-----------|--------|-------|----------|
| **Technology Readiness Level** | 15% | 0-9 | Lab concept to Commercial deployment? |
| **Technical Performance** | 15% | 0-10 | Proven efficiency, reliability, scalability? |
| **Economic Competitiveness** | 20% | 0-10 | Cost-competitive with incumbents? |

### Sustainability Impact

| Dimension | Weight | Range | Question |
|-----------|--------|-------|----------|
| **Life Cycle Environmental Impact** | 30% | 0-10 | Full lifecycle benefits (not just use phase)? |
| **Social Equity Impact** | 10% | 0-10 | Job creation, accessibility, community benefit? |
| **Governance & Systemic Impact** | 10% | 0-10 | Policy alignment, infrastructure readiness? |

### Dimension Descriptions

#### Technology Readiness Level (TRL)
Based on NASA/DOE TRL scale:
- 0: Out of scope (not technology)
- 1-3: Lab/proof of concept
- 4-5: Validated in relevant environment
- 6-7: Demonstrated in operational environment
- 8-9: Commercial deployment at scale

**Scoring Guide:**
| Score | Stage | Example |
|-------|-------|---------|
| 0 | Out of scope | Political news, entertainment |
| 1-2 | Basic research | University lab experiment |
| 3-4 | Proof of concept | Working prototype demonstrated |
| 5-6 | Pilot project | Field trial with real users |
| 7-8 | Early commercial | Product available, limited adoption |
| 9 | Mature deployment | Widespread commercial use |

#### Technical Performance
Measures real-world metrics: efficiency improvements, reliability data, scalability evidence, real-world performance.

**What scores high (7-10):**
- Quantified efficiency gains (e.g., "30% more efficient than previous generation")
- Long-term reliability data (e.g., "5-year field performance data")
- Demonstrated scalability (e.g., "successfully scaled to 1GW production")

**What scores low (0-3):**
- No performance data mentioned
- Only theoretical/simulated results
- Vague claims without metrics

#### Economic Competitiveness
Life Cycle Cost (LCC) assessment: CAPEX/OPEX competitiveness, learning curve trajectory, market adoption, subsidy dependence.

**What scores high (7-10):**
- Cost parity or better than incumbents without subsidies
- Strong market adoption signals
- Clear path to profitability

**What scores low (0-3):**
- No cost information
- Heavily subsidy-dependent
- Significantly more expensive than alternatives

#### Life Cycle Environmental Impact
Holistic environmental assessment: cradle-to-grave emissions, resource extraction impacts, manufacturing footprint, end-of-life recyclability.

**What scores high (7-10):**
- Full lifecycle analysis provided
- Net positive environmental impact demonstrated
- Circular economy considerations addressed

**What scores low (0-3):**
- Only use-phase benefits mentioned (ignoring manufacturing/disposal)
- Potential negative impacts ignored
- No environmental data provided

#### Social Equity Impact
Human-centered sustainability: job creation, geographic accessibility, affordability, community acceptance, just transition.

**What scores high (7-10):**
- Documented job creation numbers
- Accessibility for developing regions
- Community benefit sharing mechanisms

**What scores low (0-3):**
- Benefits only wealthy/developed regions
- No social impact mentioned
- Potential negative social consequences ignored

#### Governance & Systemic Impact
System-level readiness: policy alignment, infrastructure compatibility, supply chain maturity, standards and certification.

**What scores high (7-10):**
- Aligned with existing regulations/standards
- Compatible with current infrastructure
- Mature supply chain

**What scores low (0-3):**
- Regulatory barriers not addressed
- Requires significant infrastructure changes
- Immature or fragile supply chain

---

## Performance

### Overall Metrics

| Metric | Validation | Test |
|--------|------------|------|
| **MAE** | 0.654 | 0.717 |
| **RMSE** | 1.14 | 1.22 |

### Per-Dimension Performance (Test Set)

| Dimension | MAE | RMSE |
|-----------|-----|------|
| social_equity_impact | 0.63 | 1.08 |
| economic_competitiveness | 0.67 | 1.15 |
| life_cycle_environmental_impact | 0.69 | 1.10 |
| governance_systemic_impact | 0.77 | 1.28 |
| technical_performance | 0.77 | 1.30 |
| technology_readiness_level | 0.78 | 1.38 |

### Comparison with v1

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| Validation MAE | 0.712 | **0.654** | -8.1% |
| Test MAE | 0.690 | 0.717 | +3.9% |
| Max dimension correlation | 0.80 | **0.61** | Better independence |

### Dimension Correlations (Sustainability Articles Only)

| | TRL | Tech | Econ | Env | Social | Gov |
|---|-----|------|------|-----|--------|-----|
| TRL | 1.00 | 0.29 | 0.51 | 0.08 | 0.16 | 0.24 |
| Tech | 0.29 | 1.00 | 0.36 | 0.15 | 0.10 | 0.23 |
| Econ | 0.51 | 0.36 | 1.00 | 0.35 | 0.39 | 0.50 |
| Env | 0.08 | 0.15 | 0.35 | 1.00 | 0.27 | 0.38 |
| Social | 0.16 | 0.10 | 0.39 | 0.27 | 1.00 | 0.61 |
| Gov | 0.24 | 0.23 | 0.50 | 0.38 | 0.61 | 1.00 |

Maximum correlation: 0.61 (Social-Governance), indicating good dimension independence.

---

## Gatekeeper Rules

### TRL Gatekeeper

**If technology_readiness_level < 3.0 then overall weighted average capped at 2.9**

Rationale: Lab-only technologies cannot achieve high sustainability scores regardless of theoretical potential.

### Tier Classification

| Tier | Weighted Average | Description |
|------|-----------------|-------------|
| **High** | >= 6.0 | Commercial deployment, proven sustainability |
| **Medium** | >= 4.0 | Pilot/early commercial, promising sustainability |
| **Low** | < 4.0 | Lab stage or poor sustainability profile |

---

## Scope Exclusions

The model scores **0 on all dimensions** for off-topic content:

### Excluded Categories

1. **AI/ML Infrastructure** - Model architectures, LLMs, benchmarks (without sustainability application)
2. **Consumer Electronics** - Smartphone reviews, gaming hardware, GPUs
3. **Programming** - Tutorials, frameworks, developer tools
4. **Other** - Military tech, travel, crypto speculation, entertainment

### In-Scope Topics

- Renewable energy (solar, wind, hydro, geothermal, nuclear)
- Electric vehicles and sustainable transport
- Energy storage (batteries, hydrogen, grid storage)
- Carbon capture and emissions reduction
- Circular economy and waste reduction
- Green building and energy efficiency
- Sustainable agriculture and food tech
- AI/ML **applied to** sustainability

### Exclusion Examples

| Article Type | Expected Score | Reason |
|--------------|----------------|--------|
| "New transformer architecture achieves SOTA on ImageNet" | All 0s | AI/ML infrastructure, not sustainability |
| "Samsung Galaxy S26 review: faster charging" | All 0s | Consumer electronics |
| "Building a REST API with FastAPI" | All 0s | Programming tutorial |
| "FlashAttention 3.0 speeds up LLM inference" | All 0s | ML optimization without sustainability application |
| "AI model predicts solar panel degradation" | Scored normally | AI applied to sustainability |
| "New battery recycling process recovers 95% lithium" | Scored normally | Circular economy technology |

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2.5-1.5B |
| Training Mode | Knowledge Distillation |
| Oracle Model | Gemini Flash 2.0 |
| Trainable Parameters | 18.5M (1.18% LoRA) |
| Epochs | 3 |
| Batch Size | 8 |
| Learning Rate | 2e-5 |
| Max Length | 512 tokens |
| Warmup Steps | 500 |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 8 |
| Alpha | 16 |
| Target Modules | q_proj, v_proj |
| Dropout | 0.05 |

### Data Split

| Split | Examples |
|-------|----------|
| Training | 4,358 |
| Validation | 547 |
| Test | 543 |

### Tier Distribution in Training Data

| Tier | Count | Percentage |
|------|-------|------------|
| Low (< 3) | 4,121 | 75.6% |
| Medium (3-6) | 1,303 | 23.9% |
| High (>= 6) | 24 | 0.4% |

The class imbalance reflects real-world distribution from general news sources.

---

## Usage

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import numpy as np

# Load model
base_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    num_labels=6,
    problem_type="regression"
)
model = PeftModel.from_pretrained(base_model, "jeergrvgreg/sustainability-technology-v2")
tokenizer = AutoTokenizer.from_pretrained("jeergrvgreg/sustainability-technology-v2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

model.eval()

# Score an article
article = "Title: Solar Panel Achieves 30% Efficiency\n\nResearchers developed..."
inputs = tokenizer(article, return_tensors="pt", max_length=512, truncation=True, padding=True)

with torch.no_grad():
    scores = model(**inputs).logits[0].numpy()

dimensions = ["technology_readiness_level", "technical_performance",
              "economic_competitiveness", "life_cycle_environmental_impact",
              "social_equity_impact", "governance_systemic_impact"]
weights = [0.15, 0.15, 0.20, 0.30, 0.10, 0.10]

for dim, score in zip(dimensions, scores):
    print(f"{dim}: {score:.1f}")

weighted_avg = np.average(scores, weights=weights)
if scores[0] < 3.0:
    weighted_avg = min(weighted_avg, 2.9)
print(f"Weighted Average: {weighted_avg:.2f}")
```

### Batch Processing

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import numpy as np

# Load model with GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    num_labels=6,
    problem_type="regression"
)
model = PeftModel.from_pretrained(base_model, "jeergrvgreg/sustainability-technology-v2")
model = model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("jeergrvgreg/sustainability-technology-v2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Batch scoring function
def score_articles(articles, batch_size=8):
    """Score multiple articles efficiently."""
    dimensions = ["technology_readiness_level", "technical_performance",
                  "economic_competitiveness", "life_cycle_environmental_impact",
                  "social_equity_impact", "governance_systemic_impact"]
    weights = [0.15, 0.15, 0.20, 0.30, 0.10, 0.10]

    results = []
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_scores = outputs.logits.cpu().numpy()

        for scores in batch_scores:
            weighted_avg = np.average(scores, weights=weights)
            # Apply TRL gatekeeper
            if scores[0] < 3.0:
                weighted_avg = min(weighted_avg, 2.9)

            # Determine tier
            if weighted_avg >= 6.0:
                tier = "high"
            elif weighted_avg >= 4.0:
                tier = "medium"
            else:
                tier = "low"

            results.append({
                "scores": dict(zip(dimensions, scores.tolist())),
                "weighted_average": float(weighted_avg),
                "tier": tier
            })

    return results

# Example usage
articles = [
    "Title: New Solar Cell Efficiency Record\n\nResearchers achieved 47.1% efficiency...",
    "Title: iPhone 16 Review\n\nApple's latest smartphone features..."
]
results = score_articles(articles)
for i, result in enumerate(results):
    print(f"Article {i+1}: {result['tier']} (wavg: {result['weighted_average']:.2f})")
```

### With Prefilter (Recommended)

For production use, apply keyword prefilter before model inference:

```python
import re

SUSTAINABILITY_KEYWORDS = [
    # Energy
    r'\b(solar|photovoltaic|wind turbine|hydropower|geothermal|nuclear)\b',
    r'\b(renewable energy|clean energy|green energy)\b',
    # Transport
    r'\b(electric vehicle|EV|battery electric|hydrogen fuel cell)\b',
    # Storage
    r'\b(battery storage|grid storage|energy storage|lithium-ion)\b',
    # Carbon
    r'\b(carbon capture|CCS|emissions reduction|decarbonization)\b',
    # Circular
    r'\b(recycling|circular economy|waste reduction|upcycling)\b',
    # Building
    r'\b(green building|energy efficiency|heat pump|insulation)\b',
    # Agriculture
    r'\b(sustainable agriculture|vertical farming|precision agriculture)\b',
]

def passes_prefilter(title, content):
    """Check if article is likely sustainability-relevant."""
    text = f"{title} {content[:1000]}".lower()
    for pattern in SUSTAINABILITY_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

# Usage
if passes_prefilter(article_title, article_content):
    result = score_articles([f"Title: {article_title}\n\n{article_content}"])[0]
else:
    result = {"tier": "prefiltered_out", "weighted_average": 0.0}
```

---

## Example Outputs

### High-Tier Article (wavg = 6.65)
**Title:** "German company to sell more refurbished appliances"

| Dimension | Score | Evidence |
|-----------|-------|----------|
| technology_readiness_level | 7.0 | Miele began selling refurbished washing machines in Netherlands |
| technical_performance | 6.0 | Washing machines suitable for re-use, established quality |
| economic_competitiveness | 7.0 | Clients opt for cheaper refurbished variant |
| life_cycle_environmental_impact | 7.0 | 12,000 electronic components refurbished in 2024 |
| social_equity_impact | 6.0 | Expanded customer base to younger, digitally savvy groups |
| governance_systemic_impact | 6.0 | Requires national infrastructure and sufficient returns |

**Weighted Average:** 6.65 | **Tier:** High

### Medium-Tier Article (wavg = 4.82)
**Title:** "Startup develops new solid-state battery prototype"

| Dimension | Score | Evidence |
|-----------|-------|----------|
| technology_readiness_level | 4.0 | Working prototype demonstrated at lab scale |
| technical_performance | 6.0 | 50% higher energy density than Li-ion |
| economic_competitiveness | 4.0 | Targeting cost parity by 2028 |
| life_cycle_environmental_impact | 5.0 | Uses less cobalt, recyclability addressed |
| social_equity_impact | 4.0 | Would enable affordable EVs |
| governance_systemic_impact | 5.0 | Compatible with existing battery manufacturing |

**Weighted Average:** 4.82 | **Tier:** Medium

### Low-Tier Article (wavg = 0.0)
**Title:** "New transformer architecture achieves SOTA on ImageNet"

All dimensions: 0.0 - Out of scope (AI/ML infrastructure without sustainability application)

**Weighted Average:** 0.0 | **Tier:** Low

---

## Limitations

- **Language**: Training predominantly English; prefilter supports 21 languages
- **High-Tier Data**: Only 0.4% high-tier examples in training
- **Precision**: MAE ~0.7 sufficient for tier classification, not precise scoring
- **Context**: 512 token limit may truncate long articles
- **Temporal**: Trained on 2025-2026 news

### Known Failure Modes

1. **AI + Sustainability ambiguity**: Articles about AI for grid optimization may occasionally score low if the sustainability application is not explicit
2. **Marketing language**: Greenwashing content with sustainability buzzwords may receive medium scores
3. **Regional bias**: Training data skewed toward Western sources; Asian sustainability innovations may be underrepresented

---

## Intended Use

### Primary Use Cases
- News aggregation and filtering
- Research monitoring for clean tech
- Content curation for sustainability dashboards
- Trend analysis across sectors

### Out-of-Scope
- Investment decisions (scores content quality, not viability)
- Policy recommendations (requires expert interpretation)
- Academic paper assessment
- Real-time trading

---

## Ethical Considerations

### Potential Biases
- **Geographic**: Training data primarily from English-language and European sources
- **Sector**: Solar and wind energy overrepresented vs. nuclear and hydropower
- **Stage**: More examples of early-stage than deployed technologies

### Misuse Risks
- **Greenwashing amplification**: Model scores based on article claims, not verification
- **Investment signals**: Not suitable for financial decisions
- **Policy influence**: Tier classifications should not drive policy without expert review

### Mitigations
- Clear documentation of limitations
- Gatekeeper rules to prevent speculative content from scoring high
- Explicit scope exclusions to reduce false positives

---

## Technical Specifications

- **Architecture**: Qwen/Qwen2.5-1.5B + LoRA (r=8, alpha=16)
- **GPU VRAM**: 4GB minimum, 8GB recommended
- **Inference**: ~30ms/article on RTX 3060
- **Model Size**: 73MB (LoRA adapters only)
- **Base Model Size**: 3GB (Qwen2.5-1.5B)

### Hardware Requirements

| Mode | VRAM | Speed |
|------|------|-------|
| FP32 (CPU) | 8GB RAM | ~500ms/article |
| FP16 (GPU) | 4GB VRAM | ~30ms/article |
| INT8 (GPU) | 2GB VRAM | ~25ms/article |

---

## Environmental Impact

- **Hardware**: NVIDIA RTX 4080
- **Training Time**: ~1 hour
- **Carbon**: < 0.1 kg CO2eq

---

## Citation

```bibtex
@misc{sustainability_technology_v2,
  title={Sustainability Technology Filter v2},
  author={NexusMind},
  year={2026},
  url={https://huggingface.co/jeergrvgreg/sustainability-technology-v2}
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.0 | 2026-01-14 | Scope exclusions, multilingual prefilter, improved independence |
| v1.0 | 2025-11-27 | Initial LCSA-based model |

---

## Acknowledgments

- **Base Model**: Qwen team for Qwen2.5-1.5B
- **Oracle**: Google for Gemini Flash 2.0 (knowledge distillation source)
- **Framework**: LCSA methodology for sustainability assessment structure
- **Inspiration**: NASA/DOE TRL scale for technology readiness assessment

---

## Contact

For questions, issues, or contributions:
- **Repository**: [NexusMind](https://github.com/ducroq/NexusMind)
- **Model Hub**: [jeergrvgreg/sustainability-technology-v2](https://huggingface.co/jeergrvgreg/sustainability-technology-v2)

---

**Framework Versions**: PEFT 0.17.1, Transformers 4.47+, PyTorch 2.0+
