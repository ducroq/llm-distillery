#!/usr/bin/env python3
"""
Find Extreme Examples for Dimensional Coverage

Searches raw (unlabeled) articles for potential extreme cases on specific dimensions.
Uses keyword-based heuristics to identify articles likely to score very low (1-2)
or very high (9-10) on target dimensions.

This helps fill gaps in training data coverage identified by analyze_coverage.py.

Usage:
    # Find potential high-agency articles
    python -m ground_truth.find_extremes \
        --source "datasets/raw/*.jsonl" \
        --dimension agency \
        --extreme high \
        --limit 50

    # Find potential low-agency articles
    python -m ground_truth.find_extremes \
        --source "datasets/raw/*.jsonl" \
        --dimension agency \
        --extreme low \
        --limit 50

    # Search all dimensions
    python -m ground_truth.find_extremes \
        --source "datasets/raw/*.jsonl" \
        --all-dimensions \
        --limit 20
"""

import json
import argparse
import glob as glob_module
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re


# Keyword heuristics for identifying extreme scores
# These are rough indicators - articles still need LLM labeling
DIMENSION_KEYWORDS = {
    'agency': {
        'high': [
            # Strong agency indicators (9-10)
            'organized', 'mobilized', 'took action', 'launched initiative',
            'founded', 'created movement', 'led campaign', 'built coalition',
            'community organized', 'grassroots', 'activists', 'protest',
            'demanded change', 'fought for', 'took control', 'self-organized',
            'citizens led', 'people power', 'bottom-up', 'direct action'
        ],
        'low': [
            # Weak agency indicators (1-2)
            'powerless', 'helpless', 'unable to', 'prevented from', 'denied',
            'blocked', 'suppressed', 'silenced', 'no choice', 'forced to',
            'government imposed', 'authorities banned', 'prohibited',
            'no voice', 'excluded', 'marginalized', 'disempowered'
        ]
    },
    'progress': {
        'high': [
            # Strong progress indicators (9-10)
            'breakthrough', 'solved', 'achieved', 'succeeded', 'completed',
            'milestone', 'first time', 'record', 'unprecedented',
            'transformed', 'revolutionized', 'major advance', 'game-changer',
            'eradicated', 'cured', 'eliminated', 'zero', 'target met'
        ],
        'low': [
            # Weak progress indicators (1-2)
            'failed', 'stalled', 'setback', 'regression', 'worse',
            'declined', 'crisis', 'emergency', 'catastrophe', 'collapse',
            'no progress', 'remains unchanged', 'still struggling',
            'years behind', 'deteriorated', 'reversed gains'
        ]
    },
    'collective_benefit': {
        'high': [
            # Strong collective benefit (9-10)
            'universal', 'everyone', 'entire community', 'all citizens',
            'public good', 'common benefit', 'shared prosperity',
            'accessible to all', 'free for everyone', 'community-wide',
            'collective ownership', 'serves all', 'benefits society'
        ],
        'low': [
            # Weak collective benefit (1-2)
            'exclusive', 'elite', 'privileged few', 'limited access',
            'private profit', 'shareholders', 'luxury', 'high-end',
            'for the wealthy', 'gated', 'members only', 'restricted',
            'corporate gain', 'benefits few', 'concentrates wealth'
        ]
    },
    'connection': {
        'high': [
            # Strong connection (9-10)
            'bridged divide', 'united', 'brought together', 'reconciliation',
            'cooperation', 'collaboration', 'cross-cultural', 'dialogue',
            'built trust', 'healed', 'restored relationship', 'solidarity',
            'mutual understanding', 'breaking barriers', 'inclusion'
        ],
        'low': [
            # Weak connection (1-2)
            'isolated', 'divided', 'polarized', 'conflict', 'fragmented',
            'alienated', 'disconnected', 'broken ties', 'severed',
            'hostile', 'antagonistic', 'segregated', 'excluded',
            'widening gap', 'mistrust', 'tensions'
        ]
    },
    'innovation': {
        'high': [
            # Strong innovation (9-10)
            'novel', 'unprecedented', 'first of its kind', 'revolutionary',
            'groundbreaking', 'pioneering', 'cutting-edge', 'breakthrough',
            'invented', 'discovered', 'reimagined', 'paradigm shift',
            'game-changing', 'disruptive', 'transformative technology'
        ],
        'low': [
            # Weak innovation (1-2)
            'traditional', 'conventional', 'standard practice', 'routine',
            'same old', 'nothing new', 'status quo', 'business as usual',
            'established method', 'decades-old', 'unchanged', 'familiar'
        ]
    },
    'justice': {
        'high': [
            # Strong justice (9-10)
            'equity', 'fair', 'equal rights', 'reparations', 'restitution',
            'accountability', 'held responsible', 'justice served',
            'systemic change', 'ended discrimination', 'protected rights',
            'corrected injustice', 'compensated victims', 'redistributed'
        ],
        'low': [
            # Weak justice (1-2)
            'injustice', 'unfair', 'discrimination', 'exploitation',
            'corruption', 'abuse of power', 'got away with', 'impunity',
            'no consequences', 'rigged', 'biased', 'oppression',
            'violates rights', 'inequality', 'systemic bias'
        ]
    },
    'resilience': {
        'high': [
            # Strong resilience (9-10)
            'survived', 'recovered', 'rebuilt', 'bounced back',
            'adapted', 'overcame', 'endured', 'persevered', 'withstood',
            'thrived despite', 'emerged stronger', 'transformed crisis',
            'turned around', 'comeback', 'restored after'
        ],
        'low': [
            # Weak resilience (1-2)
            'collapsed', 'devastated', 'unable to recover', 'permanent damage',
            'irreversible', 'destroyed', 'overwhelmed', 'broke down',
            'gave up', 'abandoned', 'no recovery', 'still struggling',
            'years later still', 'never recovered'
        ]
    },
    'wonder': {
        'high': [
            # Strong wonder (9-10)
            'awe-inspiring', 'extraordinary', 'remarkable', 'stunning',
            'breathtaking', 'mind-blowing', 'incredible', 'spectacular',
            'miraculous', 'phenomenal', 'astonishing', 'magical',
            'defied expectations', 'beyond imagination', 'inspiring'
        ],
        'low': [
            # Weak wonder (1-2)
            'mundane', 'ordinary', 'unremarkable', 'boring', 'routine',
            'predictable', 'expected', 'standard', 'nothing special',
            'underwhelming', 'disappointing', 'mediocre', 'lackluster'
        ]
    }
}


class ExtremeExampleFinder:
    """Find articles likely to have extreme dimensional scores."""

    def __init__(self, source_pattern: str):
        """Initialize with source file pattern."""
        self.source_pattern = source_pattern
        self.articles = []

    def load_articles(self, limit: int = None) -> int:
        """Load articles from source files."""
        source_files = glob_module.glob(self.source_pattern)

        if not source_files:
            print(f"⚠️  No files found matching: {self.source_pattern}")
            return 0

        print(f"Loading articles from {len(source_files)} file(s)...")

        article_count = 0
        for source_file in sorted(source_files):
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            article = json.loads(line)
                            self.articles.append(article)
                            article_count += 1

                            if limit and article_count >= limit * 10:  # Load 10x limit for filtering
                                break
                        except json.JSONDecodeError:
                            continue
                if limit and article_count >= limit * 10:
                    break
            except Exception as e:
                print(f"⚠️  Error loading {source_file}: {e}")

        print(f"✅ Loaded {article_count} articles\n")
        return article_count

    def score_article(
        self,
        article: Dict,
        dimension: str,
        extreme: str
    ) -> Tuple[float, List[str]]:
        """
        Score article for likelihood of extreme value on dimension.

        Returns:
            (score, matched_keywords)
            score: 0.0-1.0, higher means more likely to be extreme
            matched_keywords: List of keywords found
        """
        # Get article text
        title = article.get('title', '').lower()
        text = article.get('text', article.get('content', '')).lower()
        full_text = f"{title} {text}"

        # Get keywords for this dimension and extreme
        keywords = DIMENSION_KEYWORDS.get(dimension, {}).get(extreme, [])

        if not keywords:
            return 0.0, []

        # Count keyword matches (weighted by frequency)
        matched = []
        total_score = 0.0

        for keyword in keywords:
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, full_text))

            if matches > 0:
                matched.append(keyword)
                # Logarithmic scoring (1 match = 1.0, 2 = 1.4, 3 = 1.7, etc.)
                total_score += (matches ** 0.5)

        # Normalize by text length (avoid bias toward long articles)
        text_length = len(full_text.split())
        if text_length > 0:
            normalized_score = total_score / (text_length ** 0.5)
        else:
            normalized_score = 0.0

        return normalized_score, matched

    def find_extremes(
        self,
        dimension: str,
        extreme: str,
        limit: int = 50
    ) -> List[Tuple[Dict, float, List[str]]]:
        """
        Find articles likely to have extreme scores on dimension.

        Args:
            dimension: Dimension name (e.g., 'agency')
            extreme: 'high' or 'low'
            limit: Max number of results

        Returns:
            List of (article, score, matched_keywords) sorted by score
        """
        print(f"Searching for {extreme} {dimension} articles...")

        # Score all articles
        scored_articles = []

        for article in self.articles:
            score, keywords = self.score_article(article, dimension, extreme)

            if score > 0:
                scored_articles.append((article, score, keywords))

        # Sort by score (descending)
        scored_articles.sort(key=lambda x: x[1], reverse=True)

        # Take top N
        results = scored_articles[:limit]

        print(f"✅ Found {len(results)} candidate articles\n")

        return results

    def find_all_extremes(
        self,
        limit_per_extreme: int = 20
    ) -> Dict[str, Dict[str, List]]:
        """
        Find extreme examples for all dimensions.

        Returns:
            {dimension: {extreme: [(article, score, keywords)]}}
        """
        results = defaultdict(dict)

        for dimension in DIMENSION_KEYWORDS.keys():
            for extreme in ['high', 'low']:
                print(f"\n{'='*70}")
                print(f"Finding {extreme} {dimension} articles...")
                print(f"{'='*70}\n")

                extremes = self.find_extremes(dimension, extreme, limit_per_extreme)
                results[dimension][extreme] = extremes

        return results


def print_results(
    results: List[Tuple[Dict, float, List[str]]],
    dimension: str,
    extreme: str,
    show_top_n: int = 10
):
    """Print search results in readable format."""
    print(f"{'='*70}")
    print(f"TOP {show_top_n} CANDIDATES: {extreme.upper()} {dimension.upper()}")
    print(f"{'='*70}\n")

    for i, (article, score, keywords) in enumerate(results[:show_top_n], 1):
        title = article.get('title', 'No title')
        article_id = article.get('id', 'unknown')
        text_preview = article.get('text', article.get('content', ''))[:200]

        print(f"{i}. [{score:.3f}] {title}")
        print(f"   ID: {article_id}")
        print(f"   Matched: {', '.join(keywords[:5])}")
        if len(keywords) > 5:
            print(f"            ...and {len(keywords) - 5} more")
        print(f"   Preview: {text_preview}...")
        print()


def save_results(
    results: List[Tuple[Dict, float, List[str]]],
    output_file: str
):
    """Save candidate articles to JSONL file."""
    print(f"\n💾 Saving {len(results)} articles to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for article, score, keywords in results:
            # Add metadata about why this was selected
            article['_search_metadata'] = {
                'score': score,
                'matched_keywords': keywords
            }
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    print(f"✅ Saved to {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Find extreme examples for dimensional coverage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--source',
        required=True,
        help='Source file pattern (e.g., "datasets/raw/*.jsonl")'
    )
    parser.add_argument(
        '--dimension',
        choices=list(DIMENSION_KEYWORDS.keys()),
        help='Dimension to search for'
    )
    parser.add_argument(
        '--extreme',
        choices=['high', 'low'],
        help='Which extreme to find (high=9-10, low=1-2)'
    )
    parser.add_argument(
        '--all-dimensions',
        action='store_true',
        help='Search all dimensions (overrides --dimension)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=50,
        help='Max results per search (default: 50)'
    )
    parser.add_argument(
        '--output',
        help='Save results to JSONL file'
    )
    parser.add_argument(
        '--show',
        type=int,
        default=10,
        help='Number of top results to display (default: 10)'
    )

    args = parser.parse_args()

    # Validation
    if not args.all_dimensions and (not args.dimension or not args.extreme):
        parser.error("Either --all-dimensions or both --dimension and --extreme required")

    # Initialize finder
    finder = ExtremeExampleFinder(args.source)

    # Load articles
    if not finder.load_articles(limit=args.limit * 20):
        print("❌ No articles loaded")
        return 1

    # Find extremes
    if args.all_dimensions:
        # Search all dimensions
        all_results = finder.find_all_extremes(limit_per_extreme=args.limit)

        # Print summary
        print("\n" + "="*70)
        print("SUMMARY: ALL DIMENSIONS")
        print("="*70 + "\n")

        for dimension in DIMENSION_KEYWORDS.keys():
            for extreme in ['high', 'low']:
                results = all_results[dimension][extreme]
                print(f"{dimension:20s} {extreme:4s}: {len(results):3d} candidates")

        # Save if requested
        if args.output:
            # Save all results to separate files
            output_path = Path(args.output)
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            for dimension in DIMENSION_KEYWORDS.keys():
                for extreme in ['high', 'low']:
                    results = all_results[dimension][extreme]
                    filename = f"{dimension}_{extreme}_candidates.jsonl"
                    filepath = output_dir / filename
                    save_results(results, str(filepath))

    else:
        # Search single dimension
        results = finder.find_extremes(args.dimension, args.extreme, args.limit)

        # Print top results
        print_results(results, args.dimension, args.extreme, args.show)

        # Save if requested
        if args.output:
            save_results(results, args.output)

        # Print usage hint
        print("💡 Tip: Label these candidates with batch_labeler to verify scores")
        print(f"   python -m ground_truth.batch_labeler --filter filters/uplifting/v1 --source {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())
