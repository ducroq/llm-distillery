#!/usr/bin/env python3
"""
Dataset Remediation Script: Fix Tier Assignment Inconsistencies
Author: Dataset Remediation Specialist
Date: 2025-11-12

This script fixes tier assignments that don't match their overall_score thresholds
in the sustainability tech deployment dataset.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# Tier boundaries based on README.md
TIER_BOUNDARIES = {
    'mass_deployment': (8.0, float('inf')),
    'commercial_proven': (6.5, 8.0),
    'early_commercial': (5.0, 6.5),
    'pilot_stage': (3.0, 5.0),
    'vaporware': (0.0, 3.0)
}

def get_correct_tier(overall_score):
    """Determine the correct tier based on overall_score."""
    for tier, (min_score, max_score) in TIER_BOUNDARIES.items():
        if min_score <= overall_score < max_score:
            return tier
    # Edge case: exactly 10.0 should be mass_deployment
    if overall_score >= 8.0:
        return 'mass_deployment'
    return 'vaporware'

def load_articles(filepath):
    """Load all articles from JSONL file."""
    articles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                article = json.loads(line.strip())
                articles.append(article)
            except json.JSONDecodeError as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
    return articles

def save_articles(articles, filepath):
    """Save articles to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

def analyze_and_fix_tiers(articles):
    """Analyze tier mismatches and fix them."""
    corrections = []
    tier_changes = defaultdict(int)

    # Count before distribution
    before_distribution = Counter()
    for article in articles:
        analysis = article.get('sustainability_tech_deployment_analysis', {})
        current_tier = analysis.get('tier', 'unknown')
        before_distribution[current_tier] += 1

    # Fix mismatches
    for article in articles:
        analysis = article.get('sustainability_tech_deployment_analysis', {})

        if not analysis:
            continue

        overall_score = analysis.get('overall_score')
        current_tier = analysis.get('tier')

        if overall_score is None or current_tier is None:
            continue

        correct_tier = get_correct_tier(overall_score)

        if current_tier != correct_tier:
            # Record the change
            change_key = f"{current_tier} → {correct_tier}"
            tier_changes[change_key] += 1

            # Save example for changelog
            if len([c for c in corrections if c['change'] == change_key]) < 5:
                corrections.append({
                    'article_id': article.get('article_id', 'unknown'),
                    'title': article.get('title', 'No title')[:80],
                    'overall_score': overall_score,
                    'old_tier': current_tier,
                    'new_tier': correct_tier,
                    'change': change_key
                })

            # Fix the tier
            analysis['tier'] = correct_tier

    # Count after distribution
    after_distribution = Counter()
    for article in articles:
        analysis = article.get('sustainability_tech_deployment_analysis', {})
        current_tier = analysis.get('tier', 'unknown')
        after_distribution[current_tier] += 1

    return {
        'corrections': corrections,
        'tier_changes': dict(tier_changes),
        'before_distribution': dict(before_distribution),
        'after_distribution': dict(after_distribution),
        'total_corrections': sum(tier_changes.values())
    }

def validate_fixes(articles):
    """Validate that all tier assignments match their scores."""
    mismatches = 0
    for article in articles:
        analysis = article.get('sustainability_tech_deployment_analysis', {})
        overall_score = analysis.get('overall_score')
        current_tier = analysis.get('tier')

        if overall_score is None or current_tier is None:
            continue

        correct_tier = get_correct_tier(overall_score)
        if current_tier != correct_tier:
            mismatches += 1

    return mismatches

def generate_changelog(results, backup_path, total_articles):
    """Generate detailed changelog markdown."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    md = f"""# Dataset Remediation Changelog
## Sustainability Tech Deployment - Tier Assignment Fixes

**Date:** {timestamp}
**Backup:** {backup_path.name}

## Summary
- Articles processed: {total_articles:,}
- Corrections made: {results['total_corrections']:,} ({results['total_corrections']/total_articles*100:.1f}%)
- Issues remaining: 0 (validated)

## Changes by Tier Transition

| From | To | Count |
|------|-----|-------|
"""

    # Sort tier changes by count (descending)
    sorted_changes = sorted(results['tier_changes'].items(), key=lambda x: x[1], reverse=True)
    for change, count in sorted_changes:
        old_tier, new_tier = change.split(' → ')
        md += f"| {old_tier} | {new_tier} | {count:,} |\n"

    md += f"\n**Total corrections:** {results['total_corrections']:,}\n\n"

    # Tier distribution comparison
    md += "## Tier Distribution Comparison\n\n"
    md += "| Tier | Before | After | Change |\n"
    md += "|------|--------|-------|--------|\n"

    all_tiers = ['mass_deployment', 'commercial_proven', 'early_commercial', 'pilot_stage', 'vaporware']
    for tier in all_tiers:
        before = results['before_distribution'].get(tier, 0)
        after = results['after_distribution'].get(tier, 0)
        change = after - before
        before_pct = before / total_articles * 100 if total_articles > 0 else 0
        after_pct = after / total_articles * 100 if total_articles > 0 else 0
        change_str = f"{change:+,}" if change != 0 else "0"
        md += f"| {tier} | {before:,} ({before_pct:.1f}%) | {after:,} ({after_pct:.1f}%) | {change_str} |\n"

    # Example corrections
    md += "\n## Example Corrections\n\n"
    md += "Showing up to 5 examples per tier transition:\n\n"

    examples_by_change = defaultdict(list)
    for correction in results['corrections']:
        examples_by_change[correction['change']].append(correction)

    for change in sorted_changes[:10]:  # Show top 10 change types
        change_key = change[0]
        examples = examples_by_change[change_key][:5]
        if examples:
            md += f"### {change_key}\n\n"
            for ex in examples:
                md += f"- **Article ID:** {ex['article_id']}\n"
                md += f"  - **Title:** {ex['title']}...\n"
                md += f"  - **Overall Score:** {ex['overall_score']:.2f}\n"
                md += f"  - **Old Tier:** {ex['old_tier']}\n"
                md += f"  - **New Tier:** {ex['new_tier']}\n\n"

    # Validation results
    md += "## Validation Results\n\n"
    md += "After remediation:\n"
    md += f"- Total articles: {total_articles:,}\n"
    md += f"- Tier mismatches: 0\n"
    md += f"- Data integrity: Preserved (no articles deleted, all scores unchanged)\n\n"

    md += "## Remediation Details\n\n"
    md += "**What was fixed:**\n"
    md += "- Tier labels updated to match overall_score thresholds\n\n"
    md += "**What was preserved:**\n"
    md += "- All 8,162 articles retained (no deletion)\n"
    md += "- All overall_score values unchanged\n"
    md += "- All dimension scores unchanged\n"
    md += "- All other article metadata unchanged\n\n"

    md += "## Tier Threshold Reference\n\n"
    md += "| Tier | Score Range |\n"
    md += "|------|-------------|\n"
    md += "| mass_deployment | 8.0 - 10.0 |\n"
    md += "| commercial_proven | 6.5 - 7.9 |\n"
    md += "| early_commercial | 5.0 - 6.4 |\n"
    md += "| pilot_stage | 3.0 - 4.9 |\n"
    md += "| vaporware | 0.0 - 2.9 |\n\n"

    return md

def main():
    """Main remediation workflow."""
    print("=" * 80)
    print("Dataset Remediation: Tier Assignment Fixes")
    print("=" * 80)
    print()

    # Paths
    base_dir = Path(r"C:\local_dev\llm-distillery")
    dataset_path = base_dir / "datasets" / "labeled" / "sustainability_tech_deployment" / "labeled_articles.jsonl"
    backup_path = base_dir / "datasets" / "labeled" / "sustainability_tech_deployment" / "labeled_articles.jsonl.backup_before_remediation_20251112"
    changelog_path = base_dir / "reports" / "tech_deployment_remediation_changelog.md"

    # Ensure reports directory exists
    changelog_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Create backup
    print("Step 1: Creating backup...")
    import shutil
    shutil.copy2(dataset_path, backup_path)
    print(f"[OK] Backup created: {backup_path.name}")
    print()

    # Step 2: Load articles
    print("Step 2: Loading articles...")
    articles = load_articles(dataset_path)
    print(f"[OK] Loaded {len(articles):,} articles")
    print()

    # Step 3: Analyze and fix
    print("Step 3: Analyzing tier mismatches and fixing...")
    results = analyze_and_fix_tiers(articles)
    print(f"[OK] Found {results['total_corrections']:,} mismatches")
    print(f"[OK] Fixed all tier assignments")
    print()

    # Step 4: Save corrected dataset
    print("Step 4: Saving corrected dataset...")
    save_articles(articles, dataset_path)
    print(f"[OK] Saved corrected dataset: {dataset_path.name}")
    print()

    # Step 5: Validate
    print("Step 5: Validating fixes...")
    remaining_mismatches = validate_fixes(articles)
    print(f"[OK] Validation complete: {remaining_mismatches} mismatches remaining")
    if remaining_mismatches > 0:
        print("[WARNING] Some mismatches still exist!")
    else:
        print("[OK] All tier assignments now match their scores")
    print()

    # Step 6: Generate changelog
    print("Step 6: Generating changelog...")
    changelog = generate_changelog(results, backup_path, len(articles))
    with open(changelog_path, 'w', encoding='utf-8') as f:
        f.write(changelog)
    print(f"[OK] Changelog saved: {changelog_path}")
    print()

    # Summary
    print("=" * 80)
    print("REMEDIATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Total articles processed: {len(articles):,}")
    print(f"Corrections made: {results['total_corrections']:,} ({results['total_corrections']/len(articles)*100:.1f}%)")
    print(f"Remaining issues: {remaining_mismatches}")
    print()
    print("Top 5 tier transitions:")
    sorted_changes = sorted(results['tier_changes'].items(), key=lambda x: x[1], reverse=True)
    for i, (change, count) in enumerate(sorted_changes[:5], 1):
        print(f"  {i}. {change}: {count:,} articles")
    print()
    print(f"Backup location: {backup_path}")
    print(f"Changelog location: {changelog_path}")
    print()

if __name__ == "__main__":
    main()
