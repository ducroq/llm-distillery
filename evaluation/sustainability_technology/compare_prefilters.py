#!/usr/bin/env python3
"""
Compare prefilter v2.1 vs v2.2 on frozen test data.

Test data:
- 271 false positives (should be blocked)
- 300 true positives (should pass)

Outputs confusion matrix and comparison metrics.
"""

import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import both prefilter versions
from filters.sustainability_technology.v2.prefilter_v2_1 import SustainabilityTechnologyPreFilterV2 as PrefilterV21
from filters.sustainability_technology.v2.prefilter_v2_2 import SustainabilityTechnologyPreFilterV2 as PrefilterV22

# Test data paths
GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"
TRUE_POSITIVES_FILE = Path(__file__).parent / "true_positives" / "frozen_true_positives.json"


def load_false_positives():
    """Load 271 false positives from ground truth."""
    fps = []
    for json_file in GROUND_TRUTH_DIR.glob("*.json"):
        if json_file.name == "README.md":
            continue
        with open(json_file, 'r', encoding='utf-8') as f:
            fps.extend(json.load(f))
    return fps


def load_true_positives():
    """Load 300 frozen true positives."""
    with open(TRUE_POSITIVES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_prefilter(prefilter, articles, expected_pass: bool):
    """
    Test prefilter on a set of articles.

    Args:
        prefilter: Prefilter instance
        articles: List of article dicts
        expected_pass: True if articles should pass, False if should be blocked

    Returns:
        dict with results
    """
    passed = []
    blocked = []

    for article in articles:
        # Build article dict for prefilter
        art = {
            'title': article.get('title', ''),
            'content': article.get('content', ''),
        }

        should_pass, reason = prefilter.apply_filter(art)

        result = {
            'title': article.get('title', ''),
            'passed': should_pass,
            'reason': reason,
            'category': article.get('category', 'unknown'),
        }

        if should_pass:
            passed.append(result)
        else:
            blocked.append(result)

    # Compute metrics
    total = len(articles)
    if expected_pass:
        # True positives: should pass
        correct = len(passed)
        errors = blocked  # False negatives
        metric_name = "TP Pass Rate"
    else:
        # False positives: should be blocked
        correct = len(blocked)
        errors = passed  # Still FPs
        metric_name = "FP Block Rate"

    return {
        'total': total,
        'passed': len(passed),
        'blocked': len(blocked),
        'correct': correct,
        'correct_rate': correct / total * 100 if total > 0 else 0,
        'metric_name': metric_name,
        'errors': errors,
        'blocked_by_reason': Counter(r['reason'] for r in blocked),
    }


def compare_prefilters():
    """Run full comparison between v2.1 and v2.2."""

    print("=" * 70)
    print("PREFILTER COMPARISON: v2.1 vs v2.2")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Load test data
    fps = load_false_positives()
    tps = load_true_positives()
    print(f"Test data: {len(fps)} FPs, {len(tps)} TPs")
    print()

    # Initialize prefilters
    v21 = PrefilterV21()
    v22 = PrefilterV22()
    print(f"v2.1: {v21.version}")
    print(f"v2.2: {v22.version}")
    print()

    # Test on False Positives (should be blocked)
    print("-" * 70)
    print("Testing on FALSE POSITIVES (should be blocked)")
    print("-" * 70)

    fp_v21 = test_prefilter(v21, fps, expected_pass=False)
    fp_v22 = test_prefilter(v22, fps, expected_pass=False)

    print(f"v2.1 FP Block Rate: {fp_v21['correct_rate']:.1f}% ({fp_v21['blocked']}/{fp_v21['total']})")
    print(f"v2.2 FP Block Rate: {fp_v22['correct_rate']:.1f}% ({fp_v22['blocked']}/{fp_v22['total']})")
    print(f"Improvement: +{fp_v22['correct_rate'] - fp_v21['correct_rate']:.1f}%")
    print()

    # Test on True Positives (should pass)
    print("-" * 70)
    print("Testing on TRUE POSITIVES (should pass)")
    print("-" * 70)

    tp_v21 = test_prefilter(v21, tps, expected_pass=True)
    tp_v22 = test_prefilter(v22, tps, expected_pass=True)

    print(f"v2.1 TP Pass Rate: {tp_v21['correct_rate']:.1f}% ({tp_v21['passed']}/{tp_v21['total']})")
    print(f"v2.2 TP Pass Rate: {tp_v22['correct_rate']:.1f}% ({tp_v22['passed']}/{tp_v22['total']})")
    print(f"Change: {tp_v22['correct_rate'] - tp_v21['correct_rate']:+.1f}%")
    print()

    # Summary comparison
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'v2.1':>10} {'v2.2':>10} {'Change':>10}")
    print("-" * 55)
    print(f"{'FP Block Rate':<25} {fp_v21['correct_rate']:>9.1f}% {fp_v22['correct_rate']:>9.1f}% {fp_v22['correct_rate'] - fp_v21['correct_rate']:>+9.1f}%")
    print(f"{'TP Pass Rate':<25} {tp_v21['correct_rate']:>9.1f}% {tp_v22['correct_rate']:>9.1f}% {tp_v22['correct_rate'] - tp_v21['correct_rate']:>+9.1f}%")
    print()

    # Confusion matrices
    print("=" * 70)
    print("CONFUSION MATRICES")
    print("=" * 70)
    print()

    print("v2.1:")
    print(f"                    Blocked    |    Passed")
    print(f"                 --------------+--------------")
    print(f"    FP (block)  |    {fp_v21['blocked']:4d}      |     {fp_v21['passed']:4d}      |  n={fp_v21['total']}")
    print(f"    TP (pass)   |    {tp_v21['blocked']:4d}      |     {tp_v21['passed']:4d}      |  n={tp_v21['total']}")
    print()

    print("v2.2:")
    print(f"                    Blocked    |    Passed")
    print(f"                 --------------+--------------")
    print(f"    FP (block)  |    {fp_v22['blocked']:4d}      |     {fp_v22['passed']:4d}      |  n={fp_v22['total']}")
    print(f"    TP (pass)   |    {tp_v22['blocked']:4d}      |     {tp_v22['passed']:4d}      |  n={tp_v22['total']}")
    print()

    # Block reasons for v2.2
    print("=" * 70)
    print("v2.2 BLOCK REASONS (FPs)")
    print("=" * 70)
    for reason, count in fp_v22['blocked_by_reason'].most_common():
        print(f"  {reason:40s}: {count:3d}")
    print()

    # New blocks in v2.2 (improvement)
    v21_blocked_titles = {r['title'] for r in fps if not test_single(v21, r)}
    v22_blocked_titles = {r['title'] for r in fps if not test_single(v22, r)}
    new_blocks = v22_blocked_titles - v21_blocked_titles

    print("=" * 70)
    print(f"NEW BLOCKS IN v2.2 ({len(new_blocks)} articles)")
    print("=" * 70)
    for title in list(new_blocks)[:15]:
        title_clean = title.encode('ascii', 'replace').decode('ascii')[:65]
        print(f"  {title_clean}")
    if len(new_blocks) > 15:
        print(f"  ... and {len(new_blocks) - 15} more")
    print()

    # False negatives in v2.2 (TPs incorrectly blocked)
    if tp_v22['errors']:
        print("=" * 70)
        print(f"FALSE NEGATIVES IN v2.2 ({len(tp_v22['errors'])} TPs incorrectly blocked)")
        print("=" * 70)
        for r in tp_v22['errors'][:15]:
            title_clean = r['title'].encode('ascii', 'replace').decode('ascii')[:55]
            print(f"  [{r['reason']:25s}] {title_clean}")
        if len(tp_v22['errors']) > 15:
            print(f"  ... and {len(tp_v22['errors']) - 15} more")
    print()

    # Recommendation
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    fp_improvement = fp_v22['correct_rate'] - fp_v21['correct_rate']
    tp_regression = tp_v21['correct_rate'] - tp_v22['correct_rate']

    if fp_improvement > 5 and tp_regression < 5:
        print("ACCEPT v2.2: Significant FP improvement with acceptable TP regression")
    elif fp_improvement > 0 and tp_regression <= 0:
        print("ACCEPT v2.2: FP improvement with no TP regression")
    elif tp_regression > 10:
        print("REJECT v2.2: TP regression too high (>10%)")
    else:
        print("REVIEW: Marginal improvement, consider if worth the complexity")

    return {
        'v21': {'fp': fp_v21, 'tp': tp_v21},
        'v22': {'fp': fp_v22, 'tp': tp_v22},
    }


def test_single(prefilter, article):
    """Test single article, return True if passes."""
    art = {'title': article.get('title', ''), 'content': article.get('content', '')}
    should_pass, _ = prefilter.apply_filter(art)
    return should_pass


if __name__ == "__main__":
    results = compare_prefilters()
