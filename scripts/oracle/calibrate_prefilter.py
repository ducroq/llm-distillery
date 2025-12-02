"""
Pre-filter Calibration Script

Tests pre-filter effectiveness by comparing:
1. Pass rate (% of articles that pass pre-filter)
2. Score distributions (passed vs blocked articles)
3. False negatives (blocked articles that would score high)
4. False positives (passed articles that score low)

Usage:
    python -m ground_truth.calibrate_prefilter \
        --filter filters/uplifting/v1 \
        --source datasets/raw/master_dataset_*.jsonl \
        --sample-size 500 \
        --oracle gemini-flash \
        --output reports/uplifting_v1_prefilter_calibration.md
"""

import argparse
import json
import random
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


def load_filter_package(filter_path: Path) -> Tuple:
    """
    Load filter package components.

    Returns:
        (prefilter_instance, prompt_text, config_dict)
    """
    print(f"Loading filter package: {filter_path}")

    # Load prefilter
    prefilter_module_path = filter_path / "prefilter.py"
    if not prefilter_module_path.exists():
        raise FileNotFoundError(f"prefilter.py not found in {filter_path}")

    spec = importlib.util.spec_from_file_location("prefilter", prefilter_module_path)
    prefilter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prefilter_module)

    # Get the prefilter class (assumes naming convention: <FilterName>PreFilterV1)
    prefilter_classes = [
        obj for name, obj in vars(prefilter_module).items()
        if isinstance(obj, type) and name.endswith('PreFilterV1')
    ]

    if not prefilter_classes:
        raise ValueError(f"No PreFilterV1 class found in {prefilter_module_path}")

    prefilter_class = prefilter_classes[0]
    prefilter = prefilter_class()

    print(f"  Loaded: {prefilter_class.__name__}")
    print(f"  Version: {prefilter.VERSION}")

    # Load prompt (for reference, not used in this script)
    prompt_path = filter_path / "prompt.md"
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
    else:
        prompt_text = None

    # Load config (for reference)
    config_path = filter_path / "config.yaml"
    config_dict = {}
    if config_path.exists():
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

    return prefilter, prompt_text, config_dict


def load_sample_articles(source_pattern: str, sample_size: int, seed: int = 42) -> List[Dict]:
    """Load random sample of articles from source files"""
    from glob import glob

    print(f"\nLoading articles from: {source_pattern}")

    all_articles = []
    for filepath in sorted(glob(source_pattern)):
        print(f"  Reading: {Path(filepath).name}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line.strip())
                    all_articles.append(article)
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue

    print(f"  Total articles available: {len(all_articles):,}")

    if len(all_articles) < sample_size:
        print(f"  WARNING: Only {len(all_articles):,} articles available, using all")
        return all_articles

    random.seed(seed)
    sample = random.sample(all_articles, sample_size)
    print(f"  Sampled: {sample_size:,} articles")

    return sample


def calibrate_prefilter(
    filter_path: Path,
    source_pattern: str,
    sample_size: int = 500,
    seed: int = 42
) -> Dict:
    """
    Calibrate pre-filter by analyzing pass rate and blocked content.

    NOTE: This version only runs the prefilter, does NOT call the oracle.
    Oracle calibration should be done first with calibrate_oracle.py.
    """

    # Load filter package
    prefilter, prompt, config = load_filter_package(filter_path)

    # Load sample articles
    articles = load_sample_articles(source_pattern, sample_size, seed)

    # Run prefilter on all articles
    print(f"\n{'='*70}")
    print("RUNNING PRE-FILTER")
    print(f"{'='*70}")

    passed = []
    blocked = []
    block_reasons = []

    for article in articles:
        should_label, reason = prefilter.should_label(article)

        if should_label:
            passed.append(article)
        else:
            blocked.append(article)
            block_reasons.append(reason)

    pass_rate = len(passed) / len(articles)

    print(f"\nPre-filter Results:")
    print(f"  Total articles: {len(articles):,}")
    print(f"  Passed: {len(passed):,} ({pass_rate*100:.1f}%)")
    print(f"  Blocked: {len(blocked):,} ({(1-pass_rate)*100:.1f}%)")

    # Analyze block reasons
    print(f"\nBlock Reason Distribution:")
    reason_counts = Counter(block_reasons)
    for reason, count in reason_counts.most_common():
        pct = count / len(blocked) * 100 if blocked else 0
        print(f"  {reason}: {count:,} ({pct:.1f}%)")

    # Sample blocked articles
    print(f"\n{'='*70}")
    print("SAMPLE BLOCKED ARTICLES (First 5)")
    print(f"{'='*70}")

    for i, (article, reason) in enumerate(zip(blocked[:5], block_reasons[:5]), 1):
        title = article.get('title', 'No title')
        source = article.get('source', 'unknown')
        print(f"\n{i}. [{reason.upper()}]")
        print(f"   Source: {source}")
        print(f"   Title: {title[:100]}")

    # Return calibration results
    return {
        'filter_path': str(filter_path),
        'sample_size': len(articles),
        'pass_rate': pass_rate,
        'passed_count': len(passed),
        'blocked_count': len(blocked),
        'block_reasons': dict(reason_counts),
        'prefilter_stats': prefilter.get_statistics()
    }


def generate_report(results: Dict, output_path: Path):
    """Generate markdown calibration report"""

    filter_name = Path(results['filter_path']).parent.name

    report = f"""# Pre-filter Calibration Report

**Filter**: {filter_name}
**Version**: {Path(results['filter_path']).name}
**Sample Size**: {results['sample_size']:,} articles

---

## Summary

Pre-filter pass rate determines how many articles are sent to the LLM for labeling vs blocked locally.

### Pass Rate: {results['pass_rate']*100:.1f}%

- **Passed**: {results['passed_count']:,} articles → Will be labeled by oracle
- **Blocked**: {results['blocked_count']:,} articles → Saved LLM cost

### Block Reason Distribution

| Reason | Count | Percentage |
|--------|-------|------------|
"""

    for reason, count in sorted(results['block_reasons'].items(), key=lambda x: x[1], reverse=True):
        pct = count / results['blocked_count'] * 100
        report += f"| {reason} | {count:,} | {pct:.1f}% |\n"

    report += f"""
---

## Pre-filter Statistics

"""

    for key, value in results['prefilter_stats'].items():
        report += f"- **{key}**: {value}\n"

    report += f"""
---

## Interpretation

**Pass Rate Analysis**:
- **High pass rate (>70%)**: Pre-filter is conservative, may let low-value content through
- **Moderate pass rate (40-70%)**: Balanced filtering, good for initial training
- **Low pass rate (<40%)**: Aggressive filtering, may miss some relevant content

**Recommended Next Steps**:
1. ✅ Review sample blocked articles to check for false negatives
2. ⏳ Run oracle calibration: `calibrate_oracle.py`
3. ⏳ If needed, adjust pre-filter patterns and re-calibrate
4. ⏳ Generate ground truth: `batch_labeler.py --filter {results['filter_path']}`

---

**Generated**: {Path(results['filter_path']).parent.parent.parent.name}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calibrate pre-filter effectiveness')

    parser.add_argument('--filter', required=True,
                       help='Path to filter package (e.g., filters/uplifting/v1)')
    parser.add_argument('--source', required=True,
                       help='Source files pattern (e.g., datasets/raw/master_dataset_*.jsonl)')
    parser.add_argument('--sample-size', type=int, default=500,
                       help='Number of articles to sample (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', required=True,
                       help='Output report path (e.g., reports/uplifting_v1_prefilter_cal.md)')

    args = parser.parse_args()

    filter_path = Path(args.filter)
    if not filter_path.exists():
        print(f"ERROR: Filter path not found: {filter_path}")
        sys.exit(1)

    print("="*70)
    print("PRE-FILTER CALIBRATION")
    print("="*70)

    # Run calibration
    results = calibrate_prefilter(
        filter_path=filter_path,
        source_pattern=args.source,
        sample_size=args.sample_size,
        seed=args.seed
    )

    # Generate report
    output_path = Path(args.output)
    generate_report(results, output_path)

    print(f"\n{'='*70}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*70}")
    print(f"Pass rate: {results['pass_rate']*100:.1f}%")
    print(f"Report: {output_path}")


if __name__ == '__main__':
    main()
