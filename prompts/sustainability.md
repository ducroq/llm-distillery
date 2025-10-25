# Sustainability Semantic Filter - Ground Truth Generation Prompt

**Purpose**: Rate content for sustainability relevance, impact potential, and credibility for climate tech investment intelligence and progress tracking.

**Version**: 1.0
**Target LLM**: Claude 3.5 Sonnet / Gemini 1.5 Pro
**Use Case**: Generate ground truth labels for fine-tuning local models

---

## PROMPT TEMPLATE

```
Analyze this article for sustainability impact based on CONCRETE ACTIONS and MEASURABLE OUTCOMES, not aspirational statements or commitments.

ARTICLE:
Title: {title}
Source: {source}
Published: {published_date}
Text: {text}

[Rest of the sustainability filter prompt - copied from the original file]
```

(Note: Full prompt content available in content-aggregator repo)
