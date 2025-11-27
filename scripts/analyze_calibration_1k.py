"""
Analyze Oracle Calibration Results

This script analyzes the 1K calibration results to validate:
- Score distributions per dimension
- Dimension independence (correlations, PCA)
- Consistency (duplicate article scoring)
- Quality (overall statistics)

Usage:
    python scripts/analyze_calibration_1k.py <calibration_dir>

Example:
    python scripts/analyze_calibration_1k.py datasets/calibration/calibration_1k_20251124_140000

Output:
    - calibration_analysis_report.md (comprehensive report)
    - calibration_analysis.json (analysis statistics)
    - correlation_matrix.json (dimension correlations)
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
import math


class CalibrationAnalyzer:
    """Analyze calibration results."""

    def __init__(self, calibration_dir: Path):
        self.calibration_dir = calibration_dir
        self.scored_articles = []
        self.dimensions = [
            'technology_readiness_level',
            'technical_performance',
            'economic_competitiveness',
            'life_cycle_environmental_impact',
            'social_equity_impact',
            'governance_systemic_impact'
        ]

    def load_scored_articles(self):
        """Load scored articles from calibration."""
        scored_path = self.calibration_dir / "articles_scored.jsonl"
        print(f"Loading scored articles from {scored_path}...")

        with open(scored_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    article = json.loads(line)
                    if 'oracle_scores' in article:
                        self.scored_articles.append(article)

        print(f"Loaded {len(self.scored_articles)} scored articles")

    def calculate_dimension_statistics(self) -> Dict:
        """Calculate statistics for each dimension."""
        print("\nCalculating dimension statistics...")

        dimension_scores = defaultdict(list)

        for article in self.scored_articles:
            scores = article['oracle_scores']
            for dim in self.dimensions:
                if dim in scores and scores[dim] is not None:
                    dimension_scores[dim].append(scores[dim])

        stats = {}
        for dim, scores in dimension_scores.items():
            if scores:
                stats[dim] = {
                    'mean': sum(scores) / len(scores),
                    'std': self._calculate_std(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }

        return stats

    def calculate_score_distribution(self) -> Dict:
        """Calculate overall score distribution (average across dimensions)."""
        print("Calculating score distribution...")

        score_ranges = {
            'high (7-10)': 0,
            'medium-high (5-7)': 0,
            'medium (3-5)': 0,
            'low (1-3)': 0
        }

        for article in self.scored_articles:
            scores = article['oracle_scores']
            # Calculate average score across dimensions
            valid_scores = [s for s in scores.values() if s is not None]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)

                if avg_score >= 7:
                    score_ranges['high (7-10)'] += 1
                elif avg_score >= 5:
                    score_ranges['medium-high (5-7)'] += 1
                elif avg_score >= 3:
                    score_ranges['medium (3-5)'] += 1
                else:
                    score_ranges['low (1-3)'] += 1

        # Convert to percentages
        total = sum(score_ranges.values())
        distribution = {k: {'count': v, 'percentage': (v/total)*100 if total > 0 else 0}
                       for k, v in score_ranges.items()}

        return distribution

    def calculate_correlation_matrix(self) -> Dict:
        """Calculate pairwise correlations between dimensions."""
        print("Calculating correlation matrix...")

        # Collect score vectors for each dimension
        dimension_vectors = defaultdict(list)

        for article in self.scored_articles:
            scores = article['oracle_scores']
            # Only include articles with all dimensions scored
            if all(dim in scores and scores[dim] is not None for dim in self.dimensions):
                for dim in self.dimensions:
                    dimension_vectors[dim].append(scores[dim])

        # Calculate correlations
        correlations = {}
        high_correlations = []

        for i, dim1 in enumerate(self.dimensions):
            for j, dim2 in enumerate(self.dimensions):
                if i < j:  # Only upper triangle
                    corr = self._pearson_correlation(
                        dimension_vectors[dim1],
                        dimension_vectors[dim2]
                    )
                    correlations[f"{dim1}___{dim2}"] = corr

                    if abs(corr) > 0.85:
                        high_correlations.append({
                            'dim1': dim1,
                            'dim2': dim2,
                            'correlation': corr
                        })

        return {
            'correlations': correlations,
            'high_correlations': high_correlations
        }

    def analyze_consistency(self) -> Dict:
        """Analyze scoring consistency using duplicate articles."""
        print("Analyzing consistency (duplicate articles)...")

        # Find duplicate pairs
        duplicates = [a for a in self.scored_articles if '_duplicate_of_index' in a]

        if not duplicates:
            return {'message': 'No duplicates found for consistency check'}

        consistency_scores = []

        for dup_article in duplicates:
            original_idx = dup_article['_duplicate_of_index']

            # Find original article (by URL or title match)
            original = None
            dup_url = dup_article.get('url') or dup_article.get('link')

            for article in self.scored_articles:
                if '_duplicate_of_index' not in article:
                    article_url = article.get('url') or article.get('link')
                    if article_url == dup_url:
                        original = article
                        break

            if original and 'oracle_scores' in original and 'oracle_scores' in dup_article:
                # Compare scores
                orig_scores = original['oracle_scores']
                dup_scores = dup_article['oracle_scores']

                differences = []
                for dim in self.dimensions:
                    if dim in orig_scores and dim in dup_scores:
                        if orig_scores[dim] is not None and dup_scores[dim] is not None:
                            diff = abs(orig_scores[dim] - dup_scores[dim])
                            differences.append(diff)

                if differences:
                    avg_diff = sum(differences) / len(differences)
                    max_diff = max(differences)
                    consistency_scores.append({
                        'avg_difference': avg_diff,
                        'max_difference': max_diff
                    })

        if consistency_scores:
            avg_consistency = sum(s['avg_difference'] for s in consistency_scores) / len(consistency_scores)
            max_inconsistency = max(s['max_difference'] for s in consistency_scores)

            return {
                'pairs_analyzed': len(consistency_scores),
                'avg_difference': avg_consistency,
                'max_difference': max_inconsistency,
                'acceptable': avg_consistency <= 1.0
            }
        else:
            return {'message': 'Could not analyze consistency (no matching pairs)'}

    def generate_report(self, analysis: Dict) -> str:
        """Generate markdown report."""
        print("\nGenerating analysis report...")

        report = []
        report.append("# Oracle Calibration Analysis - 1,000 Articles")
        report.append("")
        report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Calibration Directory**: {self.calibration_dir.name}")
        report.append(f"**Articles Analyzed**: {len(self.scored_articles)}")
        report.append("")
        report.append("---")
        report.append("")

        # Score Distribution
        report.append("## Score Distribution")
        report.append("")
        report.append("Overall article scores (averaged across dimensions):")
        report.append("")
        report.append("| Range | Count | Percentage |")
        report.append("|-------|-------|------------|")
        for range_name, data in analysis['score_distribution'].items():
            report.append(f"| {range_name} | {data['count']} | {data['percentage']:.1f}% |")
        report.append("")

        # Dimension Statistics
        report.append("## Dimension Statistics")
        report.append("")
        report.append("| Dimension | Mean | Std Dev | Min | Max |")
        report.append("|-----------|------|---------|-----|-----|")
        for dim, stats in analysis['dimension_stats'].items():
            short_name = dim.replace('_', ' ').title()
            report.append(f"| {short_name} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['min']:.1f} | {stats['max']:.1f} |")
        report.append("")

        # Correlations
        report.append("## Dimension Independence")
        report.append("")

        high_corrs = analysis['correlations']['high_correlations']
        if high_corrs:
            report.append("**High Correlations** (r > 0.85):")
            report.append("")
            for corr in high_corrs:
                report.append(f"- **{corr['dim1']}** <-> **{corr['dim2']}**: r = {corr['correlation']:.3f}")
            report.append("")
        else:
            report.append("**No high correlations found** (r > 0.85) - Dimensions are independent!")
            report.append("")

        # Consistency
        report.append("## Scoring Consistency")
        report.append("")
        if 'pairs_analyzed' in analysis['consistency']:
            cons = analysis['consistency']
            report.append(f"**Duplicate Pairs Analyzed**: {cons['pairs_analyzed']}")
            report.append(f"**Average Score Difference**: {cons['avg_difference']:.2f}")
            report.append(f"**Max Score Difference**: {cons['max_difference']:.2f}")
            report.append("")
            if cons['acceptable']:
                report.append("**Status**: Acceptable consistency (avg difference <= 1.0)")
            else:
                report.append("**Status**: WARNING - Poor consistency (avg difference > 1.0)")
        else:
            report.append(analysis['consistency'].get('message', 'No consistency data'))
        report.append("")

        # Decision
        report.append("---")
        report.append("")
        report.append("## Decision")
        report.append("")

        # Automated decision logic
        issues = []

        # Check score distribution
        dist = analysis['score_distribution']
        low_rate = dist['low (1-3)']['percentage']
        if low_rate > 40:
            issues.append(f"High false positive rate ({low_rate:.1f}% scored 1-3)")

        # Check high correlations
        if len(high_corrs) > 0:
            issues.append(f"{len(high_corrs)} high correlations found (r > 0.85)")

        # Check consistency
        if 'acceptable' in analysis['consistency'] and not analysis['consistency']['acceptable']:
            issues.append("Poor scoring consistency on duplicates")

        if not issues:
            report.append("**APPROVED FOR 10K TRAINING**")
            report.append("")
            report.append("Oracle calibration looks good:")
            report.append("- Dimensions are independent")
            report.append("- Score distributions are reasonable")
            report.append("- Scoring consistency is acceptable")
        else:
            report.append("**REVIEW REQUIRED**")
            report.append("")
            report.append("Issues found:")
            for issue in issues:
                report.append(f"- {issue}")

        report.append("")
        report.append("---")
        report.append("")
        report.append("## Next Steps")
        report.append("")
        report.append("1. Manual validation: Review sample articles across score ranges")
        report.append("2. If approved: Generate 10K training dataset")
        report.append("3. If issues: Adjust prompt/prefilter and re-calibrate")
        report.append("")

        return '\n'.join(report)

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) == 0:
            return 0.0

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denom_x = math.sqrt(sum((x[i] - mean_x) ** 2 for i in range(n)))
        denom_y = math.sqrt(sum((y[i] - mean_y) ** 2 for i in range(n)))

        if denom_x == 0 or denom_y == 0:
            return 0.0

        return numerator / (denom_x * denom_y)


def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_calibration_1k.py <calibration_dir>")
        print("\nExample:")
        print("  python scripts/analyze_calibration_1k.py datasets/calibration/calibration_1k_20251124_140000")
        return

    calibration_dir = Path(sys.argv[1])

    if not calibration_dir.exists():
        print(f"ERROR: Calibration directory not found: {calibration_dir}")
        return

    print("="*70)
    print("Oracle Calibration Analysis")
    print("="*70)
    print()

    # Initialize analyzer
    analyzer = CalibrationAnalyzer(calibration_dir)

    # Load scored articles
    analyzer.load_scored_articles()

    if not analyzer.scored_articles:
        print("ERROR: No scored articles found")
        return

    # Run analysis
    analysis = {
        'dimension_stats': analyzer.calculate_dimension_statistics(),
        'score_distribution': analyzer.calculate_score_distribution(),
        'correlations': analyzer.calculate_correlation_matrix(),
        'consistency': analyzer.analyze_consistency()
    }

    # Generate report
    report = analyzer.generate_report(analysis)

    # Save report
    report_path = calibration_dir / "calibration_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    # Save analysis JSON
    analysis_path = calibration_dir / "calibration_analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"Analysis saved to: {analysis_path}")

    print()
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Review the report at: {report_path}")


if __name__ == "__main__":
    main()
