"""
Comprehensive dataset analysis with visualizations.

Generates histograms and statistics for master_dataset.jsonl:
- Date distributions (published vs collected)
- Source breakdown
- Language distribution
- Quality scores
- Word counts
- Any anomalies
"""

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

def analyze_dataset(dataset_path: str, output_dir: str = 'docs/analysis'):
    """Analyze dataset and generate visualizations."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Data collectors
    published_dates = []
    collected_dates = []
    sources = Counter()
    languages = Counter()
    quality_scores = []
    word_counts = []
    articles_by_month = defaultdict(int)
    articles_by_source_month = defaultdict(lambda: defaultdict(int))

    # Anomaly tracking
    missing_published = 0
    invalid_dates = []

    print("Analyzing dataset...")
    print("=" * 80)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        total = 0
        for i, line in enumerate(f):
            try:
                article = json.loads(line.strip())
                total += 1

                # Published dates
                if 'published_date' in article and article['published_date']:
                    try:
                        dt = datetime.fromisoformat(article['published_date'].replace('Z', '+00:00'))
                        # Only accept realistic dates
                        if 2020 <= dt.year <= 2026:
                            published_dates.append(dt)
                            month_key = dt.strftime('%Y-%m')
                            articles_by_month[month_key] += 1
                            articles_by_source_month[article.get('source', 'unknown')][month_key] += 1
                        else:
                            invalid_dates.append((article.get('id'), dt))
                    except Exception as e:
                        invalid_dates.append((article.get('id'), article['published_date']))
                else:
                    missing_published += 1

                # Collected dates
                if 'collected_date' in article and article['collected_date']:
                    try:
                        dt = datetime.fromisoformat(article['collected_date'].replace('Z', '+00:00'))
                        if 2020 <= dt.year <= 2026:
                            collected_dates.append(dt)
                    except:
                        pass

                # Other metrics
                sources[article.get('source', 'unknown')] += 1
                languages[article.get('language', 'unknown')] += 1

                metadata = article.get('metadata', {})
                if 'quality_score' in metadata:
                    quality_scores.append(metadata['quality_score'])
                if 'word_count' in metadata:
                    word_counts.append(metadata['word_count'])

                if (i + 1) % 10000 == 0:
                    print(f"Processed {i+1:,} articles...")

            except Exception as e:
                print(f"Error processing line {i}: {e}")
                continue

    print(f"\nTotal articles: {total:,}")
    print(f"Articles with valid published dates: {len(published_dates):,}")
    print(f"Articles with collected dates: {len(collected_dates):,}")
    print(f"Missing published dates: {missing_published:,}")
    print(f"Invalid/placeholder dates: {len(invalid_dates):,}")

    # Create visualizations
    fig = plt.figure(figsize=(20, 12))

    # 1. Published dates histogram
    ax1 = plt.subplot(3, 3, 1)
    if published_dates:
        published_dates.sort()
        ax1.hist([d.date() for d in published_dates], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Published Date')
        ax1.set_ylabel('Number of Articles')
        ax1.set_title(f'Publication Date Distribution (n={len(published_dates):,})')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.tick_params(axis='x', rotation=45)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Collected dates histogram
    ax2 = plt.subplot(3, 3, 2)
    if collected_dates:
        collected_dates.sort()
        ax2.hist([d.date() for d in collected_dates], bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Collected Date')
        ax2.set_ylabel('Number of Articles')
        ax2.set_title(f'Collection Date Distribution (n={len(collected_dates):,})')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Top 15 sources
    ax3 = plt.subplot(3, 3, 3)
    top_sources = sources.most_common(15)
    source_names = [s[0][:30] for s in top_sources]  # Truncate long names
    source_counts = [s[1] for s in top_sources]
    ax3.barh(source_names, source_counts, color='coral', edgecolor='black')
    ax3.set_xlabel('Number of Articles')
    ax3.set_title('Top 15 Sources')
    ax3.invert_yaxis()

    # 4. Language distribution
    ax4 = plt.subplot(3, 3, 4)
    top_langs = languages.most_common(10)
    lang_names = [l[0] for l in top_langs]
    lang_counts = [l[1] for l in top_langs]
    ax4.bar(lang_names, lang_counts, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Language')
    ax4.set_ylabel('Number of Articles')
    ax4.set_title('Language Distribution (Top 10)')
    ax4.tick_params(axis='x', rotation=45)

    # 5. Quality score distribution
    ax5 = plt.subplot(3, 3, 5)
    if quality_scores:
        ax5.hist(quality_scores, bins=20, color='gold', edgecolor='black', alpha=0.7)
        ax5.set_xlabel('Quality Score')
        ax5.set_ylabel('Number of Articles')
        ax5.set_title(f'Quality Score Distribution (n={len(quality_scores):,})')
        ax5.axvline(np.median(quality_scores), color='red', linestyle='--',
                    label=f'Median: {np.median(quality_scores):.2f}')
        ax5.legend()

    # 6. Word count distribution
    ax6 = plt.subplot(3, 3, 6)
    if word_counts:
        # Filter outliers for better visualization
        filtered_wc = [wc for wc in word_counts if wc < 2000]
        ax6.hist(filtered_wc, bins=50, color='lightseagreen', edgecolor='black', alpha=0.7)
        ax6.set_xlabel('Word Count')
        ax6.set_ylabel('Number of Articles')
        ax6.set_title(f'Word Count Distribution (<2000 words, n={len(filtered_wc):,})')
        ax6.axvline(np.median(word_counts), color='red', linestyle='--',
                    label=f'Median: {np.median(word_counts):.0f}')
        ax6.legend()

    # 7. Articles per month over time
    ax7 = plt.subplot(3, 3, 7)
    if articles_by_month:
        months = sorted(articles_by_month.keys())
        counts = [articles_by_month[m] for m in months]
        month_dates = [datetime.strptime(m, '%Y-%m') for m in months]
        ax7.plot(month_dates, counts, marker='o', color='steelblue', linewidth=2)
        ax7.fill_between(month_dates, counts, alpha=0.3, color='steelblue')
        ax7.set_xlabel('Month')
        ax7.set_ylabel('Number of Articles')
        ax7.set_title('Articles Published per Month')
        ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax7.grid(True, alpha=0.3)

    # 8. Publication vs Collection date scatter (sample)
    ax8 = plt.subplot(3, 3, 8)
    if published_dates and collected_dates:
        # Match published and collected dates
        with open(dataset_path, 'r', encoding='utf-8') as f:
            pub_col_pairs = []
            for line in f:
                try:
                    article = json.loads(line.strip())
                    if 'published_date' in article and 'collected_date' in article:
                        if article['published_date'] and article['collected_date']:
                            pub = datetime.fromisoformat(article['published_date'].replace('Z', '+00:00'))
                            col = datetime.fromisoformat(article['collected_date'].replace('Z', '+00:00'))
                            if 2020 <= pub.year <= 2026 and 2020 <= col.year <= 2026:
                                pub_col_pairs.append((pub, col))
                                if len(pub_col_pairs) >= 5000:  # Sample for performance
                                    break
                except:
                    pass

        if pub_col_pairs:
            pub_dates = [p[0] for p in pub_col_pairs]
            col_dates = [p[1] for p in pub_col_pairs]
            ax8.scatter(pub_dates, col_dates, alpha=0.3, s=10, color='darkviolet')
            ax8.plot([min(pub_dates), max(pub_dates)],
                     [min(pub_dates), max(pub_dates)],
                     'r--', label='y=x (same day)')
            ax8.set_xlabel('Published Date')
            ax8.set_ylabel('Collected Date')
            ax8.set_title(f'Publication vs Collection Date (sample n={len(pub_col_pairs):,})')
            ax8.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax8.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax8.legend()
            ax8.grid(True, alpha=0.3)

    # 9. Top sources by month (stacked area)
    ax9 = plt.subplot(3, 3, 9)
    if articles_by_source_month:
        top_5_sources = [s[0] for s in sources.most_common(5)]
        months = sorted(set(month for source_data in articles_by_source_month.values()
                           for month in source_data.keys()))
        month_dates = [datetime.strptime(m, '%Y-%m') for m in months]

        # Prepare data for stacking
        data_by_source = []
        labels = []
        for source in top_5_sources:
            counts = [articles_by_source_month[source].get(m, 0) for m in months]
            data_by_source.append(counts)
            labels.append(source[:25])  # Truncate

        ax9.stackplot(month_dates, *data_by_source, labels=labels, alpha=0.7)
        ax9.set_xlabel('Month')
        ax9.set_ylabel('Number of Articles')
        ax9.set_title('Top 5 Sources Over Time (Stacked)')
        ax9.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax9.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax9.legend(loc='upper left', fontsize=8)
        ax9.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / 'dataset_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # Generate text report
    report_path = Path(output_dir) / 'dataset_summary.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Total articles: {total:,}\n\n")

        f.write("DATE INFORMATION\n")
        f.write("-" * 80 + "\n")
        if published_dates:
            f.write(f"Articles with valid published dates: {len(published_dates):,}\n")
            f.write(f"  Earliest: {min(published_dates).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Latest: {max(published_dates).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Duration: {(max(published_dates) - min(published_dates)).days} days\n")
        if collected_dates:
            f.write(f"\nArticles with collection dates: {len(collected_dates):,}\n")
            f.write(f"  Earliest: {min(collected_dates).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Latest: {max(collected_dates).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nMissing published dates: {missing_published:,}\n")
        f.write(f"Invalid/placeholder dates: {len(invalid_dates):,}\n")

        f.write("\n\nSOURCE DISTRIBUTION (Top 20)\n")
        f.write("-" * 80 + "\n")
        for source, count in sources.most_common(20):
            pct = count / total * 100
            f.write(f"  {source:50s}: {count:6,} ({pct:5.2f}%)\n")

        f.write("\n\nLANGUAGE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for lang, count in languages.most_common(10):
            pct = count / total * 100
            f.write(f"  {lang:10s}: {count:6,} ({pct:5.2f}%)\n")

        if quality_scores:
            f.write("\n\nQUALITY SCORES\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Articles with quality scores: {len(quality_scores):,}\n")
            f.write(f"  Mean: {np.mean(quality_scores):.3f}\n")
            f.write(f"  Median: {np.median(quality_scores):.3f}\n")
            f.write(f"  Std Dev: {np.std(quality_scores):.3f}\n")
            f.write(f"  Min: {min(quality_scores):.3f}\n")
            f.write(f"  Max: {max(quality_scores):.3f}\n")

        if word_counts:
            f.write("\n\nWORD COUNTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Articles with word counts: {len(word_counts):,}\n")
            f.write(f"  Mean: {np.mean(word_counts):.1f}\n")
            f.write(f"  Median: {np.median(word_counts):.1f}\n")
            f.write(f"  Std Dev: {np.std(word_counts):.1f}\n")
            f.write(f"  Min: {min(word_counts)}\n")
            f.write(f"  Max: {max(word_counts)}\n")

        if invalid_dates:
            f.write("\n\nINVALID/PLACEHOLDER DATES (Sample)\n")
            f.write("-" * 80 + "\n")
            for article_id, date in invalid_dates[:20]:
                f.write(f"  {article_id}: {date}\n")

    print(f"Text report saved to: {report_path}")
    print("\nAnalysis complete!")
    print(f"\nGenerated files:")
    print(f"  - {output_path}")
    print(f"  - {report_path}")

    # Close plot without showing (for non-interactive environments)
    plt.close()


if __name__ == '__main__':
    analyze_dataset('datasets/raw/master_dataset.jsonl')
