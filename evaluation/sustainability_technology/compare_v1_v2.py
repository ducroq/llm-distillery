#!/usr/bin/env python3
"""
Compare sustainability_technology filter v1 vs v2 on known false positives.

This script:
1. Loads 271 manually identified false positives (articles v1 scored as "medium" but should be "low")
2. Finds these articles in the filtered data
3. Runs both v1 and v2 prefilters
4. Compares results and generates a report
"""

import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_false_positives(ground_truth_dir: Path) -> list[dict]:
    """Load all manually identified false positives."""
    fps = []
    for f in ground_truth_dir.glob("*.json"):
        if f.name == "README.md":
            continue
        try:
            data = json.load(open(f, encoding='utf-8'))
            if isinstance(data, list):
                fps.extend(data)
                print(f"Loaded {len(data)} from {f.name}")
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
    return fps


def find_articles_in_filtered_data(article_ids: set, filtered_dir: Path) -> dict:
    """Find articles by ID in the filtered JSONL files."""
    found = {}

    # Search in medium tier (where false positives were classified)
    medium_dir = filtered_dir / "medium"
    if not medium_dir.exists():
        print(f"Warning: {medium_dir} not found")
        return found

    for jsonl_file in medium_dir.glob("*.jsonl"):
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        article = json.loads(line)
                        aid = article.get('id')
                        if aid in article_ids:
                            found[aid] = article
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {jsonl_file}: {e}")

    return found


def load_prefilter(version: str):
    """Load a prefilter by version."""
    if version == "v1":
        filter_path = PROJECT_ROOT / "filters" / "sustainability_technology" / "v1"
    elif version == "v2":
        filter_path = PROJECT_ROOT / "filters" / "sustainability_technology" / "v2"
    else:
        raise ValueError(f"Unknown version: {version}")

    prefilter_path = filter_path / "prefilter.py"
    if not prefilter_path.exists():
        raise FileNotFoundError(f"Prefilter not found: {prefilter_path}")

    import importlib.util
    spec = importlib.util.spec_from_file_location(f"prefilter_{version}", prefilter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the prefilter class
    for name, obj in vars(module).items():
        if isinstance(obj, type) and 'PreFilter' in name and obj.__module__ == module.__name__:
            return obj()

    raise ValueError(f"No PreFilter class found in {prefilter_path}")


def run_comparison(articles: dict, v1_prefilter, v2_prefilter) -> dict:
    """Run both prefilters on articles and compare results."""
    results = {
        'v1_pass': [],
        'v1_block': [],
        'v2_pass': [],
        'v2_block': [],
        'both_pass': [],
        'both_block': [],
        'v1_pass_v2_block': [],  # Improvement: v2 correctly blocks
        'v1_block_v2_pass': [],  # Regression: v2 incorrectly passes
        'details': []
    }

    for aid, article in articles.items():
        # Run v1
        try:
            v1_result, v1_reason = v1_prefilter.apply_filter(article)
        except AttributeError:
            v1_result, v1_reason = v1_prefilter.should_label(article)

        # Run v2
        try:
            v2_result, v2_reason = v2_prefilter.apply_filter(article)
        except AttributeError:
            v2_result, v2_reason = v2_prefilter.should_label(article)

        detail = {
            'article_id': aid,
            'title': article.get('title', '')[:80],
            'v1_pass': v1_result,
            'v1_reason': v1_reason,
            'v2_pass': v2_result,
            'v2_reason': v2_reason
        }
        results['details'].append(detail)

        # Categorize
        if v1_result:
            results['v1_pass'].append(aid)
        else:
            results['v1_block'].append(aid)

        if v2_result:
            results['v2_pass'].append(aid)
        else:
            results['v2_block'].append(aid)

        if v1_result and v2_result:
            results['both_pass'].append(aid)
        elif not v1_result and not v2_result:
            results['both_block'].append(aid)
        elif v1_result and not v2_result:
            results['v1_pass_v2_block'].append(aid)
        else:
            results['v1_block_v2_pass'].append(aid)

    return results


def generate_report(results: dict, total_fps: int, output_path: Path):
    """Generate a markdown report with comparison results."""

    total_found = len(results['details'])
    v1_fp_rate = len(results['v1_pass']) / total_found * 100 if total_found > 0 else 0
    v2_fp_rate = len(results['v2_pass']) / total_found * 100 if total_found > 0 else 0
    improvement = v1_fp_rate - v2_fp_rate

    report = f"""# Sustainability Technology Filter: v1 vs v2 Comparison

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary

Testing both filter versions on **{total_fps} known false positives** (articles that were incorrectly classified as "medium" tier but should be "low/off-topic").

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| **False Positive Rate** | {v1_fp_rate:.1f}% | {v2_fp_rate:.1f}% | **{improvement:+.1f}%** |
| Articles passing prefilter | {len(results['v1_pass'])} | {len(results['v2_pass'])} | {len(results['v2_pass']) - len(results['v1_pass']):+d} |
| Articles blocked | {len(results['v1_block'])} | {len(results['v2_block'])} | {len(results['v2_block']) - len(results['v1_block']):+d} |

## Detailed Breakdown

| Category | Count | % of Total |
|----------|-------|------------|
| Both pass (still FP) | {len(results['both_pass'])} | {len(results['both_pass'])/total_found*100:.1f}% |
| Both block (already caught) | {len(results['both_block'])} | {len(results['both_block'])/total_found*100:.1f}% |
| **v1 pass → v2 block (improvement)** | {len(results['v1_pass_v2_block'])} | {len(results['v1_pass_v2_block'])/total_found*100:.1f}% |
| v1 block → v2 pass (regression) | {len(results['v1_block_v2_pass'])} | {len(results['v1_block_v2_pass'])/total_found*100:.1f}% |

## Interpretation

"""

    if improvement > 0:
        report += f"""**v2 is better:** Reduces false positives by {improvement:.1f} percentage points.

The v2 prefilter correctly blocks {len(results['v1_pass_v2_block'])} articles that v1 incorrectly passed.
"""
    elif improvement < 0:
        report += f"""**v2 is worse:** Increases false positives by {abs(improvement):.1f} percentage points.

The v2 prefilter incorrectly passes {len(results['v1_block_v2_pass'])} articles that v1 correctly blocked.
"""
    else:
        report += "**No change:** Both versions perform identically on this test set.\n"

    # Add sample of improvements
    if results['v1_pass_v2_block']:
        report += "\n## Sample Improvements (v1 pass → v2 block)\n\n"
        report += "Articles that v2 correctly rejects:\n\n"
        for detail in results['details'][:10]:
            if detail['article_id'] in results['v1_pass_v2_block']:
                report += f"- **{detail['title']}**\n"
                report += f"  - v2 reason: {detail['v2_reason']}\n"

    # Add sample of remaining FPs
    if results['both_pass']:
        report += "\n## Remaining False Positives (both pass)\n\n"
        report += "Articles that still slip through v2:\n\n"
        count = 0
        for detail in results['details']:
            if detail['article_id'] in results['both_pass'] and count < 10:
                report += f"- **{detail['title']}**\n"
                count += 1

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport written to: {output_path}")
    return report


def main():
    # Paths
    ground_truth_dir = PROJECT_ROOT / "evaluation" / "sustainability_technology" / "ground_truth"
    filtered_dir = Path("I:/Mijn Drive/NexusMind/filtered/sustainability_technology")
    output_path = PROJECT_ROOT / "evaluation" / "sustainability_technology" / "V1_VS_V2_COMPARISON.md"

    print("=" * 60)
    print("Sustainability Technology Filter: v1 vs v2 Comparison")
    print("=" * 60)

    # Step 1: Load false positives
    print("\n1. Loading known false positives...")
    fps = load_false_positives(ground_truth_dir)
    print(f"   Total: {len(fps)} false positives")

    fp_ids = {fp['article_id'] for fp in fps}

    # Step 2: Find articles in filtered data
    print("\n2. Finding articles in filtered data...")
    articles = find_articles_in_filtered_data(fp_ids, filtered_dir)
    print(f"   Found: {len(articles)} / {len(fp_ids)} articles")

    if len(articles) == 0:
        print("\nERROR: No articles found. Check the filtered data path.")
        return

    # Step 3: Load prefilters
    print("\n3. Loading prefilters...")
    try:
        v1_prefilter = load_prefilter("v1")
        print("   v1 loaded")
    except Exception as e:
        print(f"   v1 ERROR: {e}")
        return

    try:
        v2_prefilter = load_prefilter("v2")
        print("   v2 loaded")
    except Exception as e:
        print(f"   v2 ERROR: {e}")
        return

    # Step 4: Run comparison
    print("\n4. Running comparison...")
    results = run_comparison(articles, v1_prefilter, v2_prefilter)

    # Step 5: Generate report
    print("\n5. Generating report...")
    report = generate_report(results, len(fps), output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"v1 false positive rate: {len(results['v1_pass'])/len(articles)*100:.1f}%")
    print(f"v2 false positive rate: {len(results['v2_pass'])/len(articles)*100:.1f}%")
    print(f"Improvement: {len(results['v1_pass_v2_block'])} articles now correctly blocked")
    print(f"Regression: {len(results['v1_block_v2_pass'])} articles now incorrectly passed")


if __name__ == "__main__":
    main()
