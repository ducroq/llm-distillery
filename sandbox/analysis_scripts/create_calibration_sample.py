"""
Create stratified calibration sample for oracle prompt validation.

Samples articles from a corpus into three categories:
- Positive examples: Obviously in-scope for the filter
- Negative examples: Obviously out-of-scope for the filter
- Edge cases: Borderline or controversial articles

Usage:
    python scripts/create_calibration_sample.py \
        --input articles_corpus.jsonl \
        --output calibration_sample.jsonl \
        --filter-type sustainability_tech \
        --n-positive 20 \
        --n-negative 20 \
        --n-edge 10
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple


def classify_article_sustainability_tech(article: Dict[str, Any]) -> str:
    """
    Classify article as positive/negative/edge for sustainability tech filter.

    Returns:
        'positive': Clearly in-scope (climate tech)
        'negative': Clearly out-of-scope (generic tech, unrelated)
        'edge': Borderline or controversial
        'unknown': Can't determine from keywords
    """
    text = (article.get('title', '') + ' ' + article.get('content', '')).lower()

    # POSITIVE: Climate/sustainability technology (in-scope)
    positive_keywords = [
        # Renewable energy
        r'\bsolar\b', r'\bwind\b', r'\bgeothermal\b', r'\bhydroelectric\b', r'\btidal\b',
        r'\bwave energy\b', r'\brenewable energy\b', r'\bclean energy\b',
        # Storage & EVs
        r'\bbattery\b', r'\bbatteries\b', r'\benergy storage\b',
        r'\belectric vehicle', r'\bevs\b', r'\bev charging\b',
        # Efficiency & electrification
        r'\bheat pump', r'\binsulation\b', r'\benergy efficiency\b',
        # Carbon management
        r'\bcarbon capture\b', r'\bccus\b', r'\bdirect air capture\b',
        # General climate
        r'\bdecarbonization\b', r'\bnet zero\b', r'\bcarbon neutral\b',
        r'\bemissions reduction\b', r'\bclimate tech\b', r'\bcleantech\b'
    ]

    # NEGATIVE: Out-of-scope (generic tech, unrelated)
    negative_keywords = [
        # Cloud & IT infrastructure
        r'\baws\b', r'\bazure\b', r'\bgcp\b', r'\bcloud computing\b',
        r'\bkubernetes\b', r'\bdocker\b', r'\bmicroservices\b',
        # Programming & software
        r'\bpython\b', r'\bjavascript\b', r'\breact\b', r'\bnode\.js\b',
        r'\bapi development\b', r'\bweb development\b', r'\bsoftware engineering\b',
        # Office & productivity
        r'\bexcel\b', r'\bpowerpoint\b', r'\bword processing\b',
        r'\bproductivity software\b', r'\bproject management\b',
        # Generic IT
        r'\biam\b', r'\bauthentication\b', r'\bdatabase\b', r'\bsql\b',
        # Unrelated topics
        r'\btoothbrush', r'\bpersonal hygiene\b', r'\bbeauty products\b',
        r'\bvideo games\b', r'\bstreaming\b', r'\bsocial media\b'
    ]

    # EDGE: Borderline or controversial
    edge_keywords = [
        # Fossil fuels (context matters)
        r'\boil\b', r'\bgas\b', r'\bfossil fuel', r'\bfracking\b', r'\bpetroleum\b',
        # Controversial tech
        r'\bnuclear\b', r'\bbiofuel', r'\bbiomass\b', r'\bhydrogen\b',
        # Carbon markets (questionable efficacy)
        r'\bcarbon credit', r'\bcarbon offset', r'\bemissions trading\b',
        # Greenwashing risk
        r'\bnet zero pledge\b', r'\bcarbon neutral claim\b', r'\besg\b'
    ]

    # Check each category
    positive_match = any(re.search(keyword, text) for keyword in positive_keywords)
    negative_match = any(re.search(keyword, text) for keyword in negative_keywords)
    edge_match = any(re.search(keyword, text) for keyword in edge_keywords)

    # Classification logic
    if negative_match and not positive_match:
        return 'negative'
    elif positive_match and not negative_match and not edge_match:
        return 'positive'
    elif edge_match:
        return 'edge'
    else:
        return 'unknown'


def classify_article_uplifting(article: Dict[str, Any]) -> str:
    """
    Classify article as positive/negative/edge for uplifting filter.

    Returns:
        'positive': Clearly uplifting (progress, solutions, agency)
        'negative': Clearly not uplifting (doom, powerlessness)
        'edge': Borderline (mixed framing)
        'unknown': Can't determine from keywords
    """
    text = (article.get('title', '') + ' ' + article.get('content', '')).lower()

    # POSITIVE: Uplifting content
    positive_keywords = [
        r'\bbreakthrough\b', r'\bsuccess\b', r'\bachievement\b', r'\bprogress\b',
        r'\bsolution\b', r'\binnovation\b', r'\badvancement\b',
        r'\bcommunity action\b', r'\bpeople power\b', r'\bgrassroots\b',
        r'\bwe can\b', r'\btaking action\b', r'\bmaking a difference\b',
        r'\bhope\b', r'\bpositive change\b', r'\bturning point\b'
    ]

    # NEGATIVE: Doom framing
    negative_keywords = [
        r'\bdoom\b', r'\bapocalypse\b', r'\bcatastrophe\b', r'\bdisaster\b',
        r'\bwe\'re doomed\b', r'\btoo late\b', r'\bpoint of no return\b',
        r'\bhopeless\b', r'\bpowerless\b', r'\bnothing we can do\b',
        r'\binevitable collapse\b', r'\bend of civilization\b'
    ]

    # EDGE: Mixed framing
    edge_keywords = [
        r'\bchallenge\b', r'\bdifficult\b', r'\bcomplex\b',
        r'\buncertain\b', r'\bmixed results\b', r'\bprogress but\b',
        r'\bhope if\b', r'\bcan we\b', r'\btime is running out\b'
    ]

    positive_match = any(re.search(keyword, text) for keyword in positive_keywords)
    negative_match = any(re.search(keyword, text) for keyword in negative_keywords)
    edge_match = any(re.search(keyword, text) for keyword in edge_keywords)

    if negative_match and not positive_match:
        return 'negative'
    elif positive_match and not negative_match and not edge_match:
        return 'positive'
    elif edge_match:
        return 'edge'
    else:
        return 'unknown'


def classify_article_investment_risk(article: Dict[str, Any]) -> str:
    """
    Classify article as positive/negative/edge for investment-risk filter.

    Returns:
        'positive': Macro risk signals (recession, credit crisis, systemic risk)
        'negative': FOMO/speculation, stock picks, crypto pumping
        'edge': Market analysis without clear risk signal
        'unknown': Can't determine from keywords
    """
    text = (article.get('title', '') + ' ' + article.get('content', '')).lower()

    # POSITIVE: Macro risk signals
    positive_keywords = [
        r'\brecession\b', r'\byield curve inver', r'\bcredit crisis\b',
        r'\bfed\b.*\brais', r'\binterest rate', r'\binflation\b',
        r'\bbank.*\bfail', r'\bsystemic risk\b', r'\bmarket crash\b',
        r'\bdebt crisis\b', r'\bliquidity\b', r'\bleverage\b',
        r'\bunemployment\b', r'\bcorporate debt\b', r'\bcredit spread',
        r'\bvaluation', r'\bbubble\b', r'\bvolatility\b',
        r'\bcentral bank\b', r'\bmonetary policy\b', r'\bfiscal\b'
    ]

    # NEGATIVE: FOMO, speculation, stock picks
    negative_keywords = [
        r'\bhot stock', r'\bbuy now\b', r'\bnext big thing\b',
        r'\bmeme stock', r'\bcrypto\b.*\bmoon', r'\bto the moon\b',
        r'\bget rich\b', r'\bdouble your money\b', r'\bguaranteed return',
        r'\bstock pick', r'\bearnings prediction', r'\bprice target',
        r'\bfomo\b', r'\bpump\b', r'\bdump\b',
        r'\baffiliate', r'\bpromo code\b', r'\bsign up now\b',
        r'\bwarren buffett.*secret\b', r'\bthis one stock\b'
    ]

    # EDGE: Market analysis without clear signal
    edge_keywords = [
        r'\bmarket outlook\b', r'\banalyst opinion\b', r'\bmarket forecast\b',
        r'\bportfolio strategy\b', r'\basset allocation\b',
        r'\bmarket neutral\b', r'\bmixed signal', r'\bunclear\b',
        r'\bwait and see\b', r'\bmarket watch\b'
    ]

    positive_match = any(re.search(keyword, text) for keyword in positive_keywords)
    negative_match = any(re.search(keyword, text) for keyword in negative_keywords)
    edge_match = any(re.search(keyword, text) for keyword in edge_keywords)

    # Classification logic
    if negative_match and not positive_match:
        return 'negative'
    elif positive_match and not negative_match and not edge_match:
        return 'positive'
    elif edge_match:
        return 'edge'
    else:
        return 'unknown'


def load_articles(corpus_path: Path) -> List[Dict[str, Any]]:
    """Load articles from JSONL corpus."""
    articles = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def stratified_sample(
    articles: List[Dict[str, Any]],
    filter_type: str,
    n_positive: int,
    n_negative: int,
    n_edge: int,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Create stratified sample of articles.

    Returns:
        (sampled_articles, stats_dict)
    """
    random.seed(random_seed)

    # Classify all articles
    classify_fn = {
        'sustainability_tech': classify_article_sustainability_tech,
        'uplifting': classify_article_uplifting,
        'investment_risk': classify_article_investment_risk
    }.get(filter_type)

    if not classify_fn:
        raise ValueError(f"Unknown filter type: {filter_type}")

    print(f"Classifying {len(articles):,} articles...")
    categorized = {
        'positive': [],
        'negative': [],
        'edge': [],
        'unknown': []
    }

    for article in articles:
        category = classify_fn(article)
        categorized[category].append(article)

    # Print distribution
    print(f"\nCategory distribution:")
    for category, items in categorized.items():
        print(f"  {category}: {len(items):,} articles")

    # Sample from each category
    sampled = {
        'positive': random.sample(categorized['positive'], min(n_positive, len(categorized['positive']))),
        'negative': random.sample(categorized['negative'], min(n_negative, len(categorized['negative']))),
        'edge': random.sample(categorized['edge'], min(n_edge, len(categorized['edge'])))
    }

    # Combine and shuffle
    all_sampled = sampled['positive'] + sampled['negative'] + sampled['edge']
    random.shuffle(all_sampled)

    # Statistics
    stats = {
        'total_articles': len(articles),
        'positive_available': len(categorized['positive']),
        'negative_available': len(categorized['negative']),
        'edge_available': len(categorized['edge']),
        'unknown': len(categorized['unknown']),
        'positive_sampled': len(sampled['positive']),
        'negative_sampled': len(sampled['negative']),
        'edge_sampled': len(sampled['edge']),
        'total_sampled': len(all_sampled)
    }

    return all_sampled, stats


def main():
    parser = argparse.ArgumentParser(description="Create calibration sample for oracle prompt validation")
    parser.add_argument("--input", required=True, help="Input corpus JSONL file")
    parser.add_argument("--output", required=True, help="Output calibration sample JSONL file")
    parser.add_argument("--filter-type", required=True, choices=['sustainability_tech', 'uplifting', 'investment_risk'],
                        help="Filter type to calibrate")
    parser.add_argument("--n-positive", type=int, default=20, help="Number of positive examples")
    parser.add_argument("--n-negative", type=int, default=20, help="Number of negative examples")
    parser.add_argument("--n-edge", type=int, default=10, help="Number of edge cases")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading articles from {input_path}...")
    articles = load_articles(input_path)
    print(f"Loaded {len(articles):,} articles")

    print(f"\nCreating stratified sample for {args.filter_type} filter...")
    print(f"Target: {args.n_positive} positive, {args.n_negative} negative, {args.n_edge} edge cases")

    sampled_articles, stats = stratified_sample(
        articles,
        args.filter_type,
        args.n_positive,
        args.n_negative,
        args.n_edge,
        args.random_seed
    )

    # Write output
    print(f"\nWriting calibration sample to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for article in sampled_articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    print(f"\n[SUCCESS] Calibration sample created!")
    print(f"\nFinal sample composition:")
    print(f"  Positive examples: {stats['positive_sampled']}/{stats['positive_available']}")
    print(f"  Negative examples: {stats['negative_sampled']}/{stats['negative_available']}")
    print(f"  Edge cases: {stats['edge_sampled']}/{stats['edge_available']}")
    print(f"  Total: {stats['total_sampled']} articles")

    if stats['total_sampled'] < (args.n_positive + args.n_negative + args.n_edge):
        print(f"\n[WARNING] Could not sample requested quantity. Available articles limited.")

    print(f"\nNext step:")
    print(f"  python scripts/label_batch.py \\")
    print(f"      --filter filters/{{filter_name}}/v1 \\")
    print(f"      --input {output_path} \\")
    print(f"      --output calibration_labeled.jsonl")


if __name__ == "__main__":
    main()
