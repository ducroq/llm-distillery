"""
Data profiling tool for understanding raw datasets before distillation.

Analyzes:
- Time distribution
- Source breakdown
- Sentiment distribution
- Content categories
- Quality metrics
- Pre-filtering recommendations

Usage:
    python -m ground_truth.data_profiler datasets/raw/master_dataset.jsonl
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List
import statistics


class DataProfiler:
    """Profile a JSONL dataset to understand its characteristics."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.articles = []
        self.stats = {
            'total_articles': 0,
            'by_source': Counter(),
            'by_language': Counter(),
            'by_sentiment_category': Counter(),
            'by_date': defaultdict(int),
            'sentiment_scores': [],
            'quality_scores': [],
            'word_counts': [],
            'has_entities': 0,
            'sources_breakdown': defaultdict(lambda: {
                'count': 0,
                'avg_sentiment': [],
                'avg_quality': [],
                'avg_words': []
            })
        }

    def load_sample(self, max_articles: int = None):
        """Load articles from dataset."""
        print(f"Loading dataset: {self.dataset_path}")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_articles and i >= max_articles:
                    break

                try:
                    article = json.loads(line.strip())
                    self.articles.append(article)
                except json.JSONDecodeError:
                    continue

        print(f"Loaded {len(self.articles)} articles\n")

    def analyze(self):
        """Analyze the loaded articles."""
        print("Analyzing dataset...")

        for article in self.articles:
            self.stats['total_articles'] += 1

            # Source distribution
            source = article.get('source', 'unknown')
            self.stats['by_source'][source] += 1

            # Language distribution
            language = article.get('language', 'unknown')
            self.stats['by_language'][language] += 1

            # Sentiment analysis
            sentiment_category = article.get('sentiment_category', 'unknown')
            self.stats['by_sentiment_category'][sentiment_category] += 1

            sentiment_score = article.get('sentiment_score', 0)
            if sentiment_score:
                self.stats['sentiment_scores'].append(sentiment_score)

            # Time distribution
            published_date = article.get('published_date')
            if published_date:
                try:
                    # Extract just the date (YYYY-MM-DD)
                    date = published_date[:10]
                    self.stats['by_date'][date] += 1
                except:
                    pass

            # Quality metrics
            quality_score = article.get('metadata', {}).get('quality_score', 0)
            if quality_score:
                self.stats['quality_scores'].append(quality_score)

            # Word count
            word_count = article.get('metadata', {}).get('word_count', 0)
            if word_count:
                self.stats['word_counts'].append(word_count)

            # Entities
            entities = article.get('metadata', {}).get('entities', {})
            if entities and any(entities.values()):
                self.stats['has_entities'] += 1

            # Source breakdown with metrics
            source_stats = self.stats['sources_breakdown'][source]
            source_stats['count'] += 1
            if sentiment_score:
                source_stats['avg_sentiment'].append(sentiment_score)
            if quality_score:
                source_stats['avg_quality'].append(quality_score)
            if word_count:
                source_stats['avg_words'].append(word_count)

    def print_report(self):
        """Print comprehensive profiling report."""
        print("\n" + "="*80)
        print("DATA PROFILING REPORT")
        print("="*80)

        # Overview
        print(f"\nOVERVIEW")
        print(f"{'-'*80}")
        print(f"Total articles: {self.stats['total_articles']:,}")
        print(f"Articles with entities: {self.stats['has_entities']:,} ({self.stats['has_entities']/self.stats['total_articles']*100:.1f}%)")

        # Source distribution
        print(f"\nSOURCE DISTRIBUTION")
        print(f"{'-'*80}")
        for source, count in self.stats['by_source'].most_common():
            percentage = count / self.stats['total_articles'] * 100
            print(f"  {source:20s}: {count:6,} ({percentage:5.1f}%)")

        # Language distribution
        print(f"\nLANGUAGE DISTRIBUTION")
        print(f"{'-'*80}")
        for lang, count in self.stats['by_language'].most_common():
            percentage = count / self.stats['total_articles'] * 100
            print(f"  {lang:20s}: {count:6,} ({percentage:5.1f}%)")

        # Sentiment distribution
        print(f"\nSENTIMENT DISTRIBUTION")
        print(f"{'-'*80}")
        for sentiment, count in self.stats['by_sentiment_category'].most_common():
            percentage = count / self.stats['total_articles'] * 100
            print(f"  {sentiment:20s}: {count:6,} ({percentage:5.1f}%)")

        if self.stats['sentiment_scores']:
            avg_sentiment = statistics.mean(self.stats['sentiment_scores'])
            median_sentiment = statistics.median(self.stats['sentiment_scores'])
            print(f"\n  Average sentiment score: {avg_sentiment:.2f}")
            print(f"  Median sentiment score: {median_sentiment:.2f}")

        # Time distribution
        if self.stats['by_date']:
            print(f"\nTIME DISTRIBUTION")
            print(f"{'-'*80}")
            sorted_dates = sorted(self.stats['by_date'].items())

            # Group by month
            by_month = defaultdict(int)
            for date, count in sorted_dates:
                month = date[:7]  # YYYY-MM
                by_month[month] += count

            print(f"  Earliest article: {sorted_dates[0][0]}")
            print(f"  Latest article: {sorted_dates[-1][0]}")
            print(f"  Total date range: {len(by_month)} months")

            print(f"\n  Top 10 months by volume:")
            for month, count in sorted(by_month.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"    {month}: {count:,} articles")

        # Quality metrics
        if self.stats['quality_scores']:
            print(f"\nQUALITY METRICS")
            print(f"{'-'*80}")
            avg_quality = statistics.mean(self.stats['quality_scores'])
            median_quality = statistics.median(self.stats['quality_scores'])
            print(f"  Average quality score: {avg_quality:.2f}")
            print(f"  Median quality score: {median_quality:.2f}")

        # Word count distribution
        if self.stats['word_counts']:
            print(f"\nCONTENT LENGTH")
            print(f"{'-'*80}")
            avg_words = statistics.mean(self.stats['word_counts'])
            median_words = statistics.median(self.stats['word_counts'])
            min_words = min(self.stats['word_counts'])
            max_words = max(self.stats['word_counts'])

            print(f"  Average word count: {avg_words:.0f}")
            print(f"  Median word count: {median_words:.0f}")
            print(f"  Min word count: {min_words}")
            print(f"  Max word count: {max_words}")

            # Distribution buckets
            buckets = [0, 50, 100, 200, 500, 1000, 5000]
            bucket_counts = defaultdict(int)
            for wc in self.stats['word_counts']:
                for i in range(len(buckets)-1):
                    if buckets[i] <= wc < buckets[i+1]:
                        bucket_counts[f"{buckets[i]}-{buckets[i+1]}"] += 1
                        break
                else:
                    bucket_counts[f"{buckets[-1]}+"] += 1

            print(f"\n  Word count distribution:")
            for bucket in [f"{buckets[i]}-{buckets[i+1]}" for i in range(len(buckets)-1)] + [f"{buckets[-1]}+"]:
                count = bucket_counts[bucket]
                percentage = count / len(self.stats['word_counts']) * 100
                print(f"    {bucket:15s}: {count:6,} ({percentage:5.1f}%)")

        # Source breakdown with metrics
        print(f"\nDETAILED SOURCE BREAKDOWN")
        print(f"{'-'*80}")
        print(f"{'Source':<20} {'Articles':<10} {'Avg Sentiment':<15} {'Avg Quality':<15} {'Avg Words':<15}")
        print(f"{'-'*80}")

        for source, data in sorted(self.stats['sources_breakdown'].items(),
                                   key=lambda x: x[1]['count'], reverse=True):
            count = data['count']
            avg_sent = statistics.mean(data['avg_sentiment']) if data['avg_sentiment'] else 0
            avg_qual = statistics.mean(data['avg_quality']) if data['avg_quality'] else 0
            avg_words = statistics.mean(data['avg_words']) if data['avg_words'] else 0

            print(f"{source:<20} {count:<10,} {avg_sent:<15.2f} {avg_qual:<15.2f} {avg_words:<15.0f}")

    def generate_recommendations(self):
        """Generate pre-filtering recommendations based on analysis."""
        print(f"\nPRE-FILTERING RECOMMENDATIONS")
        print(f"{'-'*80}")

        recommendations = []

        # Sentiment-based filtering
        if self.stats['sentiment_scores']:
            positive_count = sum(1 for s in self.stats['sentiment_scores'] if s >= 5.0)
            positive_pct = positive_count / len(self.stats['sentiment_scores']) * 100

            if positive_pct < 30:
                recommendations.append({
                    'filter': 'Uplifting content',
                    'suggestion': f'Only {positive_pct:.1f}% have sentiment ≥5.0. Use uplifting_pre_filter to reduce by ~70%',
                    'code': '--pre-filter uplifting'
                })
            else:
                recommendations.append({
                    'filter': 'Uplifting content',
                    'suggestion': f'{positive_pct:.1f}% have sentiment ≥5.0. Pre-filter recommended to reduce by ~{100-positive_pct:.0f}%',
                    'code': '--pre-filter uplifting'
                })

        # Word count filtering
        if self.stats['word_counts']:
            short_count = sum(1 for w in self.stats['word_counts'] if w < 50)
            short_pct = short_count / len(self.stats['word_counts']) * 100

            if short_pct > 10:
                recommendations.append({
                    'filter': 'Minimum word count',
                    'suggestion': f'{short_pct:.1f}% have <50 words. Consider filtering short articles',
                    'code': 'Add word_count >= 50 check in pre_filter'
                })

        # Source-based filtering
        if len(self.stats['by_source']) > 5:
            recommendations.append({
                'filter': 'Source selection',
                'suggestion': f'You have {len(self.stats["by_source"])} sources. Consider focusing on specific high-quality sources',
                'code': 'Filter by source in pre_filter function'
            })

        # Quality-based filtering
        if self.stats['quality_scores']:
            low_quality = sum(1 for q in self.stats['quality_scores'] if q < 0.5)
            if low_quality > 0:
                low_quality_pct = low_quality / len(self.stats['quality_scores']) * 100
                recommendations.append({
                    'filter': 'Quality threshold',
                    'suggestion': f'{low_quality_pct:.1f}% have quality <0.5. Filter out low-quality articles',
                    'code': 'Add quality_score >= 0.5 check in pre_filter'
                })

        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['filter']}")
            print(f"   {rec['suggestion']}")
            print(f"   Code: {rec['code']}")

        # Cost estimation
        print(f"\nCOST ESTIMATION")
        print(f"{'-'*80}")

        total = self.stats['total_articles']

        # Estimate with Gemini (cheapest, fastest)
        avg_time_gemini = 15  # seconds per article
        time_hours = (total * avg_time_gemini) / 3600

        print(f"Full dataset ({total:,} articles):")
        print(f"  Estimated time with Gemini: {time_hours:.1f} hours (~{time_hours/24:.1f} days)")
        print(f"  At $0.00015 per request: ~${total * 0.00015:.2f}")

        # With pre-filtering
        if recommendations:
            filtered = int(total * 0.3)  # Assume 70% reduction
            filtered_time = (filtered * avg_time_gemini) / 3600
            print(f"\nWith pre-filtering ({filtered:,} articles, ~70% reduction):")
            print(f"  Estimated time with Gemini: {filtered_time:.1f} hours (~{filtered_time/24:.1f} days)")
            print(f"  At $0.00015 per request: ~${filtered * 0.00015:.2f}")
            print(f"  SAVINGS: ~${(total - filtered) * 0.00015:.2f} and {time_hours - filtered_time:.1f} hours")

    def save_report(self, output_path: str = None):
        """Save report to JSON file."""
        if not output_path:
            output_path = self.dataset_path.parent / f'{self.dataset_path.stem}_profile.json'

        report = {
            'dataset': str(self.dataset_path),
            'analyzed_at': datetime.utcnow().isoformat(),
            'total_articles': self.stats['total_articles'],
            'sources': dict(self.stats['by_source']),
            'languages': dict(self.stats['by_language']),
            'sentiment_categories': dict(self.stats['by_sentiment_category']),
            'metrics': {
                'avg_sentiment': statistics.mean(self.stats['sentiment_scores']) if self.stats['sentiment_scores'] else 0,
                'avg_quality': statistics.mean(self.stats['quality_scores']) if self.stats['quality_scores'] else 0,
                'avg_words': statistics.mean(self.stats['word_counts']) if self.stats['word_counts'] else 0,
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ground_truth.data_profiler <dataset.jsonl> [--limit N]")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # Check for --limit argument
    max_articles = None
    if '--limit' in sys.argv:
        limit_idx = sys.argv.index('--limit')
        if limit_idx + 1 < len(sys.argv):
            max_articles = int(sys.argv[limit_idx + 1])

    profiler = DataProfiler(dataset_path)
    profiler.load_sample(max_articles=max_articles)
    profiler.analyze()
    profiler.print_report()
    profiler.generate_recommendations()
    profiler.save_report()


if __name__ == '__main__':
    main()
