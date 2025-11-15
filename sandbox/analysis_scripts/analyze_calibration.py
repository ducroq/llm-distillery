"""
Quick script to analyze calibration results.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_calibration(labeled_file: Path, filter_name: str = 'investment-risk'):
    """Analyze labeled calibration sample."""

    # Load all labeled articles
    articles_by_tier = defaultdict(list)
    analysis_key = f'{filter_name}_analysis'

    with open(labeled_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                article = json.loads(line)
                tier = article[analysis_key]['signal_tier']
                articles_by_tier[tier].append(article)

    # Print summary
    total = sum(len(articles) for articles in articles_by_tier.values())
    print(f"Total articles: {total}\n")

    for tier in ['RED', 'YELLOW', 'GREEN', 'BLUE', 'NOISE']:
        count = len(articles_by_tier[tier])
        pct = (count / total * 100) if total > 0 else 0
        print(f"{tier:8} {count:3} ({pct:5.1f}%)")

    print("\n" + "="*80)
    print("YELLOW ARTICLES (Macro Risk Warnings)")
    print("="*80)

    for i, article in enumerate(articles_by_tier['YELLOW'], 1):
        label = article[analysis_key]
        print(f"\n{i}. {article['title'][:70]}")
        print(f"   Source: {article.get('source', 'unknown')}")
        print(f"   Macro Risk: {label['macro_risk_severity']} | Credit Stress: {label['credit_market_stress']} | Systemic: {label['systemic_risk']}")
        print(f"   Evidence Quality: {label['evidence_quality']} | Actionability: {label['actionability']}")
        print(f"   Reasoning: {label['reasoning'][:150]}...")

        # Check for concerning patterns
        flags = label.get('flags', {})
        if any(flags.values()):
            active_flags = [k for k, v in flags.items() if v]
            print(f"   ⚠️  FLAGS: {', '.join(active_flags)}")

    print("\n" + "="*80)
    print("BLUE ARTICLES (Educational/Context)")
    print("="*80)

    for i, article in enumerate(articles_by_tier['BLUE'], 1):
        label = article[analysis_key]
        print(f"\n{i}. {article['title'][:70]}")
        print(f"   Source: {article.get('source', 'unknown')}")
        print(f"   Reasoning: {label['reasoning'][:150]}...")

        # Check if this should actually be NOISE
        flags = label.get('flags', {})
        if any(flags.values()):
            active_flags = [k for k, v in flags.items() if v]
            print(f"   ⚠️  FLAGS: {', '.join(active_flags)}")

    print("\n" + "="*80)
    print("NOISE SAMPLE (First 5)")
    print("="*80)

    for i, article in enumerate(articles_by_tier['NOISE'][:5], 1):
        label = article[analysis_key]
        print(f"\n{i}. {article['title'][:70]}")
        print(f"   Reasoning: {label['reasoning'][:100]}...")
        flags = label.get('flags', {})
        if any(flags.values()):
            active_flags = [k for k, v in flags.items() if v]
            print(f"   FLAGS: {', '.join(active_flags)}")

if __name__ == "__main__":
    labeled_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets/working/investment_risk_calibration_labeled/investment-risk/labeled_batch_001.jsonl")
    analyze_calibration(labeled_file)
