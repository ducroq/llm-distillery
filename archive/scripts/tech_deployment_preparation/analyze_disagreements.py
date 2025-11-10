#!/usr/bin/env python3
"""Analyze disagreements between Gemini Flash and Pro oracle labels."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

def load_labels(file_path: Path) -> Dict[str, Dict]:
    """Load JSONL labels into a dict keyed by article ID."""
    labels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                labels[item['id']] = item
    return labels

def get_score_and_tier(item: Dict) -> Tuple[float, str]:
    """Extract score and tier from labeled item."""
    analysis = item.get('sustainability_tech_deployment_analysis', {})
    score = analysis.get('overall_score', 0)
    tier = analysis.get('tier', 'unknown')
    return score, tier

def find_disagreements(flash_labels: Dict, pro_labels: Dict, top_n: int = 5) -> List[Dict]:
    """Find top N articles with largest score disagreements."""
    disagreements = []

    for article_id in flash_labels:
        if article_id in pro_labels:
            flash_score, flash_tier = get_score_and_tier(flash_labels[article_id])
            pro_score, pro_tier = get_score_and_tier(pro_labels[article_id])

            diff = abs(flash_score - pro_score)

            disagreements.append({
                'id': article_id,
                'title': flash_labels[article_id].get('title', 'Unknown'),
                'content': flash_labels[article_id].get('content', '')[:500],
                'flash_score': flash_score,
                'flash_tier': flash_tier,
                'flash_analysis': flash_labels[article_id].get('sustainability_tech_deployment_analysis', {}),
                'pro_score': pro_score,
                'pro_tier': pro_tier,
                'pro_analysis': pro_labels[article_id].get('sustainability_tech_deployment_analysis', {}),
                'difference': diff
            })

    disagreements.sort(key=lambda x: x['difference'], reverse=True)
    return disagreements[:top_n]

def format_analysis(analysis: Dict) -> str:
    """Format analysis for readable output."""
    output = []
    output.append(f"Overall Score: {analysis.get('overall_score', 'N/A')}")
    output.append(f"Tier: {analysis.get('tier', 'N/A')}")
    output.append(f"Content Type: {analysis.get('content_type', 'N/A')}")
    output.append("\nDimension Scores:")

    dimensions = analysis.get('dimensions', {})
    for dim, score in sorted(dimensions.items()):
        output.append(f"  {dim}: {score}")

    output.append(f"\nReasoning: {analysis.get('reasoning', 'N/A')}")

    return "\n".join(output)

def main():
    calibrations_dir = Path("calibrations/sustainability_tech_deployment")

    flash_file = calibrations_dir / "gemini_labels.jsonl"
    pro_file = calibrations_dir / "gemini-pro_labels.jsonl"

    print("Loading labels...")
    flash_labels = load_labels(flash_file)
    pro_labels = load_labels(pro_file)

    print(f"Flash labels: {len(flash_labels)}")
    print(f"Pro labels: {len(pro_labels)}")

    print("\nFinding top 5 disagreements...")
    disagreements = find_disagreements(flash_labels, pro_labels, top_n=5)

    output_file = Path("reports/disagreement_analysis.md")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Gemini Flash vs Pro Disagreement Analysis\n\n")
        f.write("**Sustainability Tech Deployment Filter**\n\n")
        f.write("Analyzing the top 5 articles with largest score disagreements to determine which model is more accurate.\n\n")
        f.write("---\n\n")

        for i, case in enumerate(disagreements, 1):
            f.write(f"## Case {i}: {case['title']}\n\n")
            f.write(f"**Score Difference**: {case['difference']:.2f}\n\n")
            f.write(f"**Article Excerpt**: {case['content'][:300]}...\n\n")

            f.write("### Gemini Flash Analysis\n\n")
            f.write(f"```\n{format_analysis(case['flash_analysis'])}\n```\n\n")

            f.write("### Gemini Pro Analysis\n\n")
            f.write(f"```\n{format_analysis(case['pro_analysis'])}\n```\n\n")

            f.write("### Manual Assessment\n\n")
            f.write("**Which model is more accurate?**\n\n")
            f.write("[To be filled in by manual analysis]\n\n")
            f.write("**Reasoning:**\n\n")
            f.write("[Explain why one model's assessment aligns better with the filter's intent]\n\n")
            f.write("---\n\n")

    print(f"\nDisagreement analysis written to: {output_file}")

    # Print summary to console
    print("\n=== TOP 5 DISAGREEMENTS ===\n")
    for i, case in enumerate(disagreements, 1):
        print(f"{i}. {case['title'][:80]}")
        print(f"   Flash: {case['flash_score']:.2f} ({case['flash_tier']})")
        print(f"   Pro: {case['pro_score']:.2f} ({case['pro_tier']})")
        print(f"   Diff: {case['difference']:.2f}\n")

if __name__ == "__main__":
    main()
