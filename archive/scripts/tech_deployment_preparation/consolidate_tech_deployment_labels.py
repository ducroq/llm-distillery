"""
Consolidate all tech_deployment label variants into a single deduplicated dataset.

This script:
1. Loads labels from all tech_deployment variant directories
2. Deduplicates by article ID (keeps first occurrence)
3. Generates distribution report
4. Writes consolidated dataset to tech_deployment_consolidated/
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def load_labels_from_directory(directory: Path):
    """Load all labels from a directory's JSONL files."""
    labels = []
    if not directory.exists():
        return labels

    for jsonl_file in sorted(directory.glob("*.jsonl")):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    labels.append(json.loads(line))

    return labels


def consolidate_labels():
    """Consolidate all tech_deployment variants with deduplication."""

    base_dir = Path("datasets/labeled")
    variants = [
        "tech_deployment",
        "tech_deployment_aggressive_labeling",
        "tech_deployment_supplemental",
        "tech_deployment_tier3_pilot"
    ]

    # Track which variant each label came from
    all_labels = []
    source_counts = Counter()

    print("=" * 70)
    print("LOADING LABELS FROM VARIANTS")
    print("=" * 70)

    for variant in variants:
        variant_path = base_dir / variant / "sustainability_tech_deployment"
        labels = load_labels_from_directory(variant_path)

        if labels:
            print(f"{variant:45} {len(labels):6,} labels")
            source_counts[variant] = len(labels)

            # Tag each label with its source
            for label in labels:
                label['_source_variant'] = variant
                all_labels.append(label)
        else:
            print(f"{variant:45}      0 labels (NOT FOUND)")

    print(f"\n{'Total labels loaded':45} {len(all_labels):6,}")

    # Deduplicate by article ID
    print("\n" + "=" * 70)
    print("DEDUPLICATION")
    print("=" * 70)

    seen_ids = set()
    deduplicated = []
    duplicates_by_variant = defaultdict(int)

    for label in all_labels:
        article_id = label.get('id')
        if article_id not in seen_ids:
            seen_ids.add(article_id)
            deduplicated.append(label)
        else:
            duplicates_by_variant[label['_source_variant']] += 1

    print(f"Unique labels:    {len(deduplicated):,}")
    print(f"Duplicates found: {len(all_labels) - len(deduplicated):,}")

    if duplicates_by_variant:
        print("\nDuplicates by source:")
        for variant, count in sorted(duplicates_by_variant.items()):
            print(f"  {variant:43} {count:6,}")

    # Analyze tier distribution
    print("\n" + "=" * 70)
    print("TIER DISTRIBUTION (Consolidated)")
    print("=" * 70)

    tier_counts = Counter()
    score_ranges = {
        'deployed': [],
        'early_comm': [],
        'pilot': [],
        'vaporware': []
    }

    for label in deduplicated:
        analysis = label.get('sustainability_tech_deployment_analysis', {})
        score = analysis.get('overall_score', 0)

        # Categorize by tier thresholds
        if score >= 8.0:
            tier = 'deployed'
            score_ranges['deployed'].append(score)
        elif score >= 6.0:
            tier = 'early_comm'
            score_ranges['early_comm'].append(score)
        elif score >= 4.0:
            tier = 'pilot'
            score_ranges['pilot'].append(score)
        else:
            tier = 'vaporware'
            score_ranges['vaporware'].append(score)

        tier_counts[tier] += 1

    total = len(deduplicated)

    for tier_name, tier_key in [
        ("Deployed (>=8.0)", 'deployed'),
        ("Early Commercial (6.0-7.9)", 'early_comm'),
        ("Pilot (4.0-5.9)", 'pilot'),
        ("Vaporware (<4.0)", 'vaporware')
    ]:
        count = len(score_ranges[tier_key])
        pct = (count / total * 100) if total > 0 else 0
        print(f"{tier_name:35} {count:6,} ({pct:5.1f}%)")

    print(f"\n{'Total':35} {total:6,}")

    # Score statistics
    print("\n" + "=" * 70)
    print("SCORE STATISTICS")
    print("=" * 70)

    all_scores = [
        label.get('sustainability_tech_deployment_analysis', {}).get('overall_score', 0)
        for label in deduplicated
    ]

    if all_scores:
        print(f"Min score:  {min(all_scores):.2f}")
        print(f"Max score:  {max(all_scores):.2f}")
        print(f"Mean score: {sum(all_scores) / len(all_scores):.2f}")
        print(f"Median:     {sorted(all_scores)[len(all_scores)//2]:.2f}")

    # Write consolidated dataset
    print("\n" + "=" * 70)
    print("WRITING CONSOLIDATED DATASET")
    print("=" * 70)

    output_dir = base_dir / "tech_deployment_consolidated" / "sustainability_tech_deployment"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "all_labels.jsonl"

    # Remove source variant tag before writing
    for label in deduplicated:
        label.pop('_source_variant', None)

    with open(output_file, 'w', encoding='utf-8') as f:
        for label in deduplicated:
            f.write(json.dumps(label) + '\n')

    print(f"Written: {output_file}")
    print(f"Labels:  {len(deduplicated):,}")
    print(f"Size:    {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Consolidated {len(all_labels):,} labels from {len(variants)} variants")
    print(f"✅ Removed {len(all_labels) - len(deduplicated):,} duplicates")
    print(f"✅ Final dataset: {len(deduplicated):,} unique labeled articles")
    print(f"✅ Tier balance: {len(score_ranges['deployed'])} deployed, "
          f"{len(score_ranges['early_comm'])} early comm, "
          f"{len(score_ranges['pilot'])} pilot, "
          f"{len(score_ranges['vaporware'])} vaporware")
    print(f"\n📁 Output: datasets/labeled/tech_deployment_consolidated/")


if __name__ == '__main__':
    consolidate_labels()
