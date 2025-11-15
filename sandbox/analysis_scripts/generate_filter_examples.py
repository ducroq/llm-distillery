"""
Generate example outputs for filter packages.

Reads labeled dataset, applies post-filter, and generates a markdown document
showing the top N scoring articles as examples.

Usage:
    python scripts/generate_filter_examples.py \
        --filter filters/sustainability_tech_deployment/v1 \
        --dataset datasets/labeled/sustainability_tech_deployment/labeled_articles.jsonl \
        --top-n 10 \
        --output filters/sustainability_tech_deployment/v1/examples.md
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from postfilter import PostFilter


def load_labeled_articles(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load articles from labeled dataset."""
    articles = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def extract_scores_from_article(article: Dict[str, Any], filter_name: str) -> Dict[str, float]:
    """Extract dimensional scores from article's filter analysis."""
    analysis_key = f"{filter_name}_analysis"
    if analysis_key not in article:
        raise ValueError(f"Article missing {analysis_key}")

    return article[analysis_key]["dimensions"]


def is_climate_related(article: Dict[str, Any]) -> bool:
    """Check if article is climate/sustainability related based on keywords."""
    import re

    text = (article.get('title', '') + ' ' + article.get('content', '')).lower()

    # Multi-word phrases (check first)
    phrases = [
        'electric vehicle', 'heat pump', 'carbon capture', 'energy storage',
        'clean energy', 'net zero', 'carbon neutral', 'wave energy',
        'fuel cell'
    ]
    for phrase in phrases:
        if phrase in text:
            return True

    # Single words that need word boundaries (avoid false matches like "ev" in "every")
    word_boundary_keywords = ['evs', 'sustainability', 'decarbonization', 'biomass', 'hydroelectric']
    for keyword in word_boundary_keywords:
        if re.search(r'\b' + keyword + r'\b', text):
            return True

    # Safe substring matches (unlikely to false-match)
    safe_keywords = [
        'solar', 'wind', 'battery', 'batteries', 'renewable', 'hydrogen',
        'emissions', 'climate', 'geothermal', 'charging', 'photovoltaic',
        'turbine', 'tidal'
    ]
    for keyword in safe_keywords:
        if keyword in text:
            return True

    return False


def classify_articles(articles: List[Dict[str, Any]], post_filter: PostFilter, filter_name: str, filter_climate: bool = False) -> List[Dict[str, Any]]:
    """Classify all articles using post-filter."""
    results = []

    for article in articles:
        # Filter for climate-related articles if requested
        if filter_climate:
            # Exclude generic tech sources (programming tutorials, IT infrastructure)
            source = article.get("source", "")
            excluded_sources = ["community_social_dev_to", "dev_to"]
            if any(excluded in source for excluded in excluded_sources):
                continue
        try:
            # Extract dimensional scores from oracle analysis
            scores = extract_scores_from_article(article, filter_name)

            # Classify using post-filter
            classification = post_filter.classify(scores, flag_reasoning_threshold=7.0)

            # Combine article + classification
            result = {
                "id": article["id"],
                "title": article["title"],
                "url": article.get("url", ""),
                "content": article["content"],
                "source": article.get("source", ""),
                "published_date": article.get("published_date", ""),
                "dimensional_scores": classification["dimensional_scores"],
                "overall_score": classification["overall_score"],
                "tier": classification["tier"],
                "tier_description": classification["tier_description"],
                "needs_reasoning": classification["needs_reasoning"],
                "applied_rules": classification["applied_rules"],
                # Include oracle reasoning if available
                "oracle_reasoning": article.get(f"{filter_name}_analysis", {}).get("overall_assessment", "")
            }
            results.append(result)
        except Exception as e:
            print(f"Warning: Skipped article {article.get('id')}: {e}")

    return results


def generate_markdown_examples(
    results: List[Dict[str, Any]],
    filter_name: str,
    filter_config: Dict[str, Any],
    top_n: int = 10,
    climate_filtered: bool = False
) -> str:
    """Generate markdown document with top N examples."""

    # Sort by overall score descending
    sorted_results = sorted(results, key=lambda x: x["overall_score"], reverse=True)
    top_articles = sorted_results[:top_n]

    # Get filter metadata
    filter_display_name = filter_config["filter"].get("name", filter_name)
    filter_description = filter_config["filter"].get("description", "")

    # Build markdown
    md = f"# {filter_display_name} - Example Outputs"
    if climate_filtered:
        md += " (Climate-Filtered)\n\n"
    else:
        md += "\n\n"

    md += f"**Filter:** {filter_display_name}\n"
    md += f"**Description:** {filter_description}\n"
    if climate_filtered:
        md += f"**Note:** Only showing climate/sustainability-related articles (filtered by keywords)\n"
    md += f"**Total articles analyzed:** {len(results):,}\n"
    md += f"**Top articles shown:** {top_n}\n\n"

    md += "---\n\n"
    md += "## Purpose\n\n"
    md += "This document shows real examples of articles scored by this filter. "
    md += "These are the highest-scoring articles from the labeled dataset, demonstrating "
    md += "what content this filter is designed to surface.\n\n"

    md += "**Use this to:**\n"
    md += "- Understand what types of articles score highly\n"
    md += "- Validate the filter is working as intended\n"
    md += "- Communicate filter behavior to stakeholders\n\n"

    md += "---\n\n"

    # Tier distribution
    tier_counts = {}
    for result in results:
        tier = result["tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    md += "## Dataset Tier Distribution\n\n"
    for tier, count in sorted(tier_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(results)) * 100
        md += f"- **{tier}**: {count:,} articles ({pct:.1f}%)\n"
    md += "\n---\n\n"

    # Top articles
    md += f"## Top {top_n} Scoring Articles\n\n"

    for i, article in enumerate(top_articles, 1):
        md += f"### {i}. {article['title']}\n\n"

        md += f"**Overall Score:** {article['overall_score']} | "
        md += f"**Tier:** {article['tier']} | "
        md += f"**Reasoning Flag:** {'Yes' if article['needs_reasoning'] else 'No'}\n\n"

        # Source info
        md += f"**Source:** {article['source']}\n"
        if article['published_date']:
            md += f"**Published:** {article['published_date'][:10]}\n"
        if article['url']:
            md += f"**URL:** {article['url']}\n"
        md += "\n"

        # Dimensional scores
        md += "**Dimensional Scores:**\n\n"
        for dim_name, score in article['dimensional_scores'].items():
            # Format dimension name (replace underscores with spaces, capitalize)
            display_name = dim_name.replace("_", " ").title()
            md += f"- {display_name}: {score}\n"
        md += "\n"

        # Applied rules
        if article['applied_rules'] and article['applied_rules'][0] != "No gatekeeper rules triggered" and article['applied_rules'][0] != "No content type caps triggered":
            md += f"**Applied Rules:** {', '.join(article['applied_rules'])}\n\n"

        # Content preview (first 300 chars)
        content_preview = article['content'][:300]
        if len(article['content']) > 300:
            content_preview += "..."
        md += f"**Content Preview:**\n> {content_preview}\n\n"

        # Oracle reasoning (if available)
        if article['oracle_reasoning']:
            md += f"**Oracle Assessment:**\n> {article['oracle_reasoning']}\n\n"

        md += "---\n\n"

    # Footer
    md += "## Notes\n\n"
    md += "- Scores are on a 0-10 scale for each dimension\n"
    md += "- Overall score is a weighted average based on dimension weights in config.yaml\n"
    md += "- Tier assignment based on overall score thresholds\n"
    md += "- 'Reasoning Flag' indicates articles that may benefit from oracle-generated explanations\n\n"

    md += f"**Generated from:** {len(results):,} labeled articles\n"
    md += "**Filter version:** v1\n"

    return md


def main():
    parser = argparse.ArgumentParser(description="Generate filter example outputs")
    parser.add_argument("--filter", required=True, help="Filter directory path")
    parser.add_argument("--dataset", required=True, help="Labeled dataset JSONL path")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top articles to show")
    parser.add_argument("--output", help="Output markdown file (default: {filter}/examples.md)")
    parser.add_argument("--filter-climate", action="store_true", help="Only include climate/sustainability-related articles")

    args = parser.parse_args()

    filter_path = Path(args.filter)
    dataset_path = Path(args.dataset)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = filter_path / "examples.md"

    # Load filter config
    with open(filter_path / "config.yaml", 'r', encoding='utf-8') as f:
        import yaml
        config = yaml.safe_load(f)

    filter_name = config["filter"]["name"]

    print(f"Loading dataset from {dataset_path}...")
    articles = load_labeled_articles(dataset_path)
    print(f"Loaded {len(articles):,} articles")

    print(f"Initializing post-filter for {filter_name}...")
    post_filter = PostFilter(str(filter_path))

    if args.filter_climate:
        print(f"Classifying articles (climate-filtered)...")
    else:
        print(f"Classifying articles...")
    results = classify_articles(articles, post_filter, filter_name, filter_climate=args.filter_climate)
    print(f"Classified {len(results):,} articles")

    print(f"Generating markdown examples (top {args.top_n})...")
    markdown = generate_markdown_examples(results, filter_name, config, args.top_n, climate_filtered=args.filter_climate)

    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    print(f"[SUCCESS] Examples generated: {output_path}")

    # Print quick summary
    sorted_results = sorted(results, key=lambda x: x["overall_score"], reverse=True)
    print(f"\nTop 3 articles:")
    for i, article in enumerate(sorted_results[:3], 1):
        print(f"  {i}. [{article['overall_score']:.2f}] {article['title'][:60]}...")


if __name__ == "__main__":
    main()
