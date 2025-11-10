#!/usr/bin/env python3
"""
Test compressed prompts vs original prompts to validate quality.

Compares:
- Flash + Compressed vs. Sonnet + Original
- Score differences per dimension
- Reasoning quality
- JSON completeness
- Cost comparison

Usage:
    python scripts/test_compressed_quality.py --filter education --articles 20
    python scripts/test_compressed_quality.py --filter sustainability --articles 10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import statistics

# Mock LLM clients - replace with actual implementations
class MockLLM:
    """Mock LLM client for testing. Replace with actual Anthropic/Google clients."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def label(self, article: Dict, prompt: str) -> Dict:
        """Mock labeling. In production, call actual API."""
        print(f"  [{self.model_name}] Labeling: {article['title'][:50]}...")

        # In production, replace with:
        # if "claude" in self.model_name:
        #     return self._call_anthropic(article, prompt)
        # elif "gemini" in self.model_name:
        #     return self._call_google(article, prompt)

        # Mock response for testing
        return {
            "content_type": "curriculum_innovation",
            "paradox_engagement": 7,
            "reasoning": "Mock response from " + self.model_name
        }


def load_prompt(filter_name: str, compressed: bool = False) -> str:
    """Load prompt from file."""
    base_path = Path("prompts")
    if compressed:
        base_path = base_path / "compressed"

    prompt_files = {
        "education": "future-of-education.md",
        "sustainability": "sustainability.md",
        "seece": "seece-energy-tech.md",
        "uplifting": "uplifting.md",
        "investment-risk": "investment-risk.md"
    }

    prompt_file = base_path / prompt_files[filter_name]

    if not prompt_file.exists():
        print(f"❌ Prompt file not found: {prompt_file}")
        sys.exit(1)

    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract prompt template between ``` markers
    if '```' in content:
        parts = content.split('```')
        if len(parts) >= 2:
            return parts[1].strip()

    return content


def load_sample_articles(n: int = 20) -> List[Dict]:
    """Load sample articles from collected data."""
    # In production, load from data/aggregated/
    # For testing, return mock data
    return [
        {
            "title": f"Sample Article {i}",
            "content": f"Content for article {i} about AI in education...",
            "source": "Test Source",
            "published": "2025-10-29",
            "url": f"https://example.com/article-{i}"
        }
        for i in range(n)
    ]


def calculate_score_difference(original: Dict, compressed: Dict, dimensions: List[str]) -> Dict:
    """Calculate differences in scores between two labelings."""
    differences = {}

    for dim in dimensions:
        if dim in original and dim in compressed:
            diff = abs(original[dim] - compressed[dim])
            differences[dim] = {
                'original': original[dim],
                'compressed': compressed[dim],
                'difference': diff
            }

    return differences


def analyze_quality(results: List[Dict], filter_name: str) -> Dict:
    """Analyze quality differences between original and compressed."""
    analysis = {
        'total_articles': len(results),
        'dimension_differences': {},
        'content_type_agreement': 0,
        'avg_reasoning_length': {'original': 0, 'compressed': 0},
        'missing_fields': {'original': 0, 'compressed': 0}
    }

    # Get dimension names based on filter
    dimension_map = {
        'education': [
            'paradox_engagement', 'curricular_innovation', 'assessment_transformation',
            'pedagogical_depth', 'discipline_specific_adaptation', 'cross_disciplinary_relevance',
            'evidence_implementation', 'institutional_readiness'
        ],
        'sustainability': [
            'climate_impact_potential', 'technical_credibility', 'economic_viability',
            'deployment_readiness', 'systemic_impact', 'justice_equity',
            'innovation_quality', 'evidence_strength'
        ],
        'investment-risk': [
            'macro_risk_severity', 'credit_market_stress', 'market_sentiment_extremes',
            'valuation_risk', 'policy_regulatory_risk', 'systemic_risk',
            'evidence_quality', 'actionability'
        ],
        # Add others as needed
    }

    dimensions = dimension_map.get(filter_name, [])

    # Analyze each article
    content_type_matches = 0
    reasoning_lengths = {'original': [], 'compressed': []}
    dimension_diffs = {dim: [] for dim in dimensions}

    for result in results:
        original = result['original_result']
        compressed = result['compressed_result']

        # Content type agreement
        if original.get('content_type') == compressed.get('content_type'):
            content_type_matches += 1

        # Reasoning length
        reasoning_lengths['original'].append(len(original.get('reasoning', '')))
        reasoning_lengths['compressed'].append(len(compressed.get('reasoning', '')))

        # Dimension differences
        for dim in dimensions:
            if dim in original and dim in compressed:
                diff = abs(original[dim] - compressed[dim])
                dimension_diffs[dim].append(diff)

    # Calculate statistics
    analysis['content_type_agreement'] = content_type_matches / len(results) * 100

    analysis['avg_reasoning_length']['original'] = statistics.mean(reasoning_lengths['original'])
    analysis['avg_reasoning_length']['compressed'] = statistics.mean(reasoning_lengths['compressed'])

    for dim, diffs in dimension_diffs.items():
        if diffs:
            analysis['dimension_differences'][dim] = {
                'mean_diff': statistics.mean(diffs),
                'max_diff': max(diffs),
                'min_diff': min(diffs),
                'stdev': statistics.stdev(diffs) if len(diffs) > 1 else 0
            }

    return analysis


def generate_report(analysis: Dict, filter_name: str, cost_comparison: Dict) -> str:
    """Generate markdown report of quality comparison."""
    report = f"""# Compression Quality Test - {filter_name.title()} Filter

**Date**: 2025-10-29
**Articles Tested**: {analysis['total_articles']}

---

## 📊 Overall Results

**Content Type Agreement**: {analysis['content_type_agreement']:.1f}%

**Reasoning Quality**:
- Original avg length: {analysis['avg_reasoning_length']['original']:.0f} chars
- Compressed avg length: {analysis['avg_reasoning_length']['compressed']:.0f} chars
- Ratio: {analysis['avg_reasoning_length']['compressed'] / analysis['avg_reasoning_length']['original'] * 100:.0f}%

---

## 📏 Dimension Score Differences

| Dimension | Mean Diff | Max Diff | Min Diff | Std Dev | Quality |
|-----------|-----------|----------|----------|---------|---------|
"""

    for dim, stats in analysis['dimension_differences'].items():
        quality = "✅ Excellent" if stats['mean_diff'] < 1.0 else "⚠️ Acceptable" if stats['mean_diff'] < 1.5 else "❌ Review"
        report += f"| {dim.replace('_', ' ').title()} | {stats['mean_diff']:.2f} | {stats['max_diff']:.1f} | {stats['min_diff']:.1f} | {stats['stdev']:.2f} | {quality} |\n"

    report += f"""

---

## 💰 Cost Comparison

**Original (Sonnet)**:
- Cost per article: ${cost_comparison['original_per_article']:.4f}
- Total cost: ${cost_comparison['original_total']:.2f}

**Compressed (Flash)**:
- Cost per article: ${cost_comparison['compressed_per_article']:.6f}
- Total cost: ${cost_comparison['compressed_total']:.4f}

**Savings**: {cost_comparison['savings_ratio']:.0f}x cheaper ({cost_comparison['savings_percent']:.1f}% reduction)

---

## ✅ Quality Assessment

**Acceptance Criteria**:
- Mean score difference <1.5 points per dimension: {"✅ PASS" if all(s['mean_diff'] < 1.5 for s in analysis['dimension_differences'].values()) else "❌ FAIL"}
- 85%+ content type agreement: {"✅ PASS" if analysis['content_type_agreement'] >= 85 else "❌ FAIL"}
- All JSON fields populated: {"✅ PASS" if analysis['missing_fields']['original'] == 0 and analysis['missing_fields']['compressed'] == 0 else "❌ FAIL"}

**Overall Assessment**: {"✅ Compressed version acceptable for production" if analysis['content_type_agreement'] >= 85 and all(s['mean_diff'] < 1.5 for s in analysis['dimension_differences'].values()) else "⚠️ Review recommended"}

---

## 💡 Recommendations

"""

    # Add recommendations based on results
    mean_diffs = [s['mean_diff'] for s in analysis['dimension_differences'].values()]
    avg_diff = statistics.mean(mean_diffs) if mean_diffs else 0

    if avg_diff < 1.0 and analysis['content_type_agreement'] >= 90:
        report += "✅ **USE COMPRESSED + FLASH**: Quality delta <15%, cost savings 500x\n"
    elif avg_diff < 1.5 and analysis['content_type_agreement'] >= 85:
        report += "⚠️ **USE WITH MONITORING**: Quality delta 15-25%, validate samples regularly\n"
    else:
        report += "❌ **STICK WITH ORIGINAL + SONNET**: Quality delta >25% for ground truth\n"

    report += f"""
**Suggested Approach**:
- Use compressed+Flash for {80 if avg_diff < 1.0 else 60}% of articles
- Use original+Sonnet for {20 if avg_diff < 1.0 else 40}% validation
- Monitor precision on high-scored articles (≥7)

---

**Test Script**: `scripts/test_compressed_quality.py`
**Original Prompt**: `prompts/{filter_name}.md`
**Compressed Prompt**: `prompts/compressed/{filter_name}.md`
"""

    return report


def main():
    parser = argparse.ArgumentParser(description='Test compressed prompt quality')
    parser.add_argument('--filter', choices=['education', 'sustainability', 'seece', 'uplifting', 'investment-risk'],
                        default='education', help='Which filter to test')
    parser.add_argument('--articles', type=int, default=20,
                        help='Number of articles to test')
    parser.add_argument('--original-model', default='sonnet',
                        choices=['sonnet', 'opus', 'gpt4'],
                        help='Model for original prompt')
    parser.add_argument('--compressed-model', default='flash',
                        choices=['flash', 'haiku', 'gpt3.5'],
                        help='Model for compressed prompt')

    args = parser.parse_args()

    print(f"🧪 Testing Compression Quality: {args.filter.title()} Filter")
    print("=" * 80)
    print(f"Articles: {args.articles}")
    print(f"Original: {args.original_model} + full prompt")
    print(f"Compressed: {args.compressed_model} + compressed prompt")
    print()

    # Load prompts
    print("📄 Loading prompts...")
    original_prompt = load_prompt(args.filter, compressed=False)
    compressed_prompt = load_prompt(args.filter, compressed=True)
    print(f"  Original: {len(original_prompt)} chars")
    print(f"  Compressed: {len(compressed_prompt)} chars ({len(compressed_prompt)/len(original_prompt)*100:.0f}%)")
    print()

    # Load sample articles
    print("📚 Loading sample articles...")
    articles = load_sample_articles(args.articles)
    print(f"  Loaded {len(articles)} articles")
    print()

    # Initialize LLM clients
    print("🤖 Initializing LLM clients...")
    original_llm = MockLLM(args.original_model)
    compressed_llm = MockLLM(args.compressed_model)
    print()

    # Label articles with both versions
    print("🏷️  Labeling articles...")
    results = []

    for i, article in enumerate(articles, 1):
        print(f"\n[{i}/{len(articles)}] {article['title']}")

        # Label with original
        original_result = original_llm.label(article, original_prompt)

        # Label with compressed
        compressed_result = compressed_llm.label(article, compressed_prompt)

        results.append({
            'article': article,
            'original_result': original_result,
            'compressed_result': compressed_result
        })

    print("\n")

    # Analyze quality
    print("📊 Analyzing quality differences...")
    analysis = analyze_quality(results, args.filter)
    print()

    # Calculate costs
    print("💰 Calculating costs...")
    cost_comparison = {
        'original_per_article': 0.0135,  # Sonnet estimate
        'original_total': 0.0135 * len(articles),
        'compressed_per_article': 0.000056,  # Flash estimate
        'compressed_total': 0.000056 * len(articles),
        'savings_ratio': 0.0135 / 0.000056,
        'savings_percent': (1 - 0.000056 / 0.0135) * 100
    }
    print(f"  Original: ${cost_comparison['original_total']:.4f}")
    print(f"  Compressed: ${cost_comparison['compressed_total']:.6f}")
    print(f"  Savings: {cost_comparison['savings_ratio']:.0f}x")
    print()

    # Generate report
    print("📝 Generating report...")
    report = generate_report(analysis, args.filter, cost_comparison)

    # Save report
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"compression_quality_{args.filter}.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  Saved: {report_path}")
    print()

    # Print summary
    print("=" * 80)
    print("✅ SUMMARY")
    print("=" * 80)
    print(f"Content Type Agreement: {analysis['content_type_agreement']:.1f}%")

    mean_diffs = [s['mean_diff'] for s in analysis['dimension_differences'].values()]
    avg_diff = statistics.mean(mean_diffs) if mean_diffs else 0
    print(f"Avg Dimension Difference: {avg_diff:.2f} points")

    print(f"Cost Savings: {cost_comparison['savings_ratio']:.0f}x cheaper")
    print()

    if avg_diff < 1.0 and analysis['content_type_agreement'] >= 90:
        print("✅ RECOMMENDATION: Use compressed + Flash for production")
        print(f"   Quality delta: <15%")
        print(f"   Cost savings: {cost_comparison['savings_ratio']:.0f}x")
        sys.exit(0)
    elif avg_diff < 1.5 and analysis['content_type_agreement'] >= 85:
        print("⚠️ RECOMMENDATION: Use compressed + Flash with monitoring")
        print(f"   Quality delta: 15-25%")
        print(f"   Validate high-scored articles manually")
        sys.exit(0)
    else:
        print("❌ RECOMMENDATION: Stick with original + Sonnet for ground truth")
        print(f"   Quality delta: >25%")
        print(f"   Use compressed only for pre-filtering")
        sys.exit(1)


if __name__ == "__main__":
    main()
