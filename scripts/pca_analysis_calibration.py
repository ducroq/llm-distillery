"""
PCA Analysis for Oracle Calibration

Analyzes the principal components of the oracle scores to validate:
- Dimension independence (PC1 < 70%)
- Intrinsic dimensionality (how many PCs needed for 90%/95% variance)
- Correlation structure

Usage:
    python scripts/pca_analysis_calibration.py <calibration_dir>

Example:
    python scripts/pca_analysis_calibration.py datasets/calibration/calibration_1k_20251124_180913
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List
import math


class PCAAnalyzer:
    """Perform PCA analysis on calibration data."""

    def __init__(self, calibration_dir: Path):
        self.calibration_dir = calibration_dir
        self.dimensions = [
            'technology_readiness_level',
            'technical_performance',
            'economic_competitiveness',
            'life_cycle_environmental_impact',
            'social_equity_impact',
            'governance_systemic_impact'
        ]
        self.data_matrix = None
        self.mean_vector = None
        self.std_vector = None

    def load_scores(self) -> np.ndarray:
        """Load score matrix from calibration."""
        scored_path = self.calibration_dir / "articles_scored.jsonl"
        print(f"Loading scores from {scored_path}...")

        scores_list = []
        with open(scored_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                article = json.loads(line)
                if 'oracle_scores' not in article:
                    continue

                # Extract scores in dimension order
                scores = article['oracle_scores']
                score_vector = []
                valid = True
                for dim in self.dimensions:
                    if dim in scores and scores[dim] is not None:
                        score_vector.append(float(scores[dim]))
                    else:
                        valid = False
                        break

                if valid:
                    scores_list.append(score_vector)

        data = np.array(scores_list)
        print(f"Loaded {len(data)} complete score vectors")
        return data

    def standardize_data(self, data: np.ndarray) -> np.ndarray:
        """Standardize data (mean=0, std=1)."""
        self.mean_vector = np.mean(data, axis=0)
        self.std_vector = np.std(data, axis=0)

        # Avoid division by zero
        self.std_vector[self.std_vector == 0] = 1.0

        standardized = (data - self.mean_vector) / self.std_vector
        return standardized

    def compute_pca(self, data: np.ndarray) -> Dict:
        """Compute PCA manually."""
        # Standardize
        standardized = self.standardize_data(data)

        # Compute covariance matrix
        n_samples = standardized.shape[0]
        cov_matrix = (standardized.T @ standardized) / (n_samples - 1)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Compute variance explained
        total_variance = eigenvalues.sum()
        variance_explained = eigenvalues / total_variance * 100
        cumulative_variance = np.cumsum(variance_explained)

        return {
            'eigenvalues': eigenvalues.tolist(),
            'variance_explained': variance_explained.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'eigenvectors': eigenvectors.tolist()
        }

    def compute_correlations(self, data: np.ndarray) -> np.ndarray:
        """Compute correlation matrix."""
        # Standardize
        standardized = self.standardize_data(data)

        # Compute correlation matrix
        n_samples = standardized.shape[0]
        corr_matrix = (standardized.T @ standardized) / (n_samples - 1)

        return corr_matrix

    def generate_report(self, pca_results: Dict, corr_matrix: np.ndarray) -> str:
        """Generate PCA analysis report."""
        report = []
        report.append("# PCA Analysis - Oracle Calibration")
        report.append("")
        report.append(f"**Calibration**: {self.calibration_dir.name}")
        report.append(f"**Samples**: {len(self.data_matrix)}")
        report.append("")
        report.append("---")
        report.append("")

        # Variance Explained
        report.append("## Variance Explained by Principal Components")
        report.append("")
        report.append("| PC | Eigenvalue | Variance % | Cumulative % |")
        report.append("|----|------------|------------|--------------|")

        for i, (eigenval, var_pct, cum_pct) in enumerate(zip(
            pca_results['eigenvalues'],
            pca_results['variance_explained'],
            pca_results['cumulative_variance']
        ), 1):
            report.append(f"| PC{i} | {eigenval:.3f} | {var_pct:.1f}% | {cum_pct:.1f}% |")

        report.append("")

        # PC1 Analysis
        pc1_variance = pca_results['variance_explained'][0]
        report.append("## PC1 Analysis")
        report.append("")
        report.append(f"**PC1 Variance**: {pc1_variance:.1f}%")
        report.append("")

        if pc1_variance < 70:
            report.append("✅ **GOOD**: PC1 < 70% indicates a **multi-dimensional problem**")
        else:
            report.append("⚠️ **WARNING**: PC1 >= 70% indicates dimensions may be too correlated")

        report.append("")

        # Intrinsic Dimensionality
        report.append("## Intrinsic Dimensionality")
        report.append("")

        cum_var = pca_results['cumulative_variance']
        dims_90 = next((i+1 for i, v in enumerate(cum_var) if v >= 90), 6)
        dims_95 = next((i+1 for i, v in enumerate(cum_var) if v >= 95), 6)

        report.append(f"- **90% variance**: {dims_90} / 6 dimensions")
        report.append(f"- **95% variance**: {dims_95} / 6 dimensions")
        report.append("")

        redundancy = 100 - (dims_95 / 6 * 100)
        report.append(f"**Redundancy score**: {redundancy:.1f}%")
        report.append("")

        if redundancy < 30:
            report.append("✅ Low redundancy - dimensions capture distinct aspects")
        elif redundancy < 50:
            report.append("⚠️ Moderate redundancy - some overlap between dimensions")
        else:
            report.append("❌ High redundancy - dimensions may be too correlated")

        report.append("")

        # Correlation Matrix
        report.append("## Correlation Matrix")
        report.append("")
        report.append("| | TRL | Tech | Econ | Env | Social | Gov |")
        report.append("|---|-----|------|------|-----|--------|-----|")

        dim_short = ['TRL', 'Tech', 'Econ', 'Env', 'Social', 'Gov']
        for i, dim in enumerate(dim_short):
            row = [dim]
            for j in range(6):
                corr = corr_matrix[i, j]
                row.append(f"{corr:.2f}")
            report.append("| " + " | ".join(row) + " |")

        report.append("")

        # High Correlations
        report.append("## High Correlations (r > 0.70)")
        report.append("")

        high_corrs = []
        for i in range(6):
            for j in range(i+1, 6):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.70:
                    high_corrs.append({
                        'dim1': self.dimensions[i],
                        'dim2': self.dimensions[j],
                        'corr': corr
                    })

        if high_corrs:
            for item in high_corrs:
                report.append(f"- **{item['dim1']}** <-> **{item['dim2']}**: r = {item['corr']:.3f}")
        else:
            report.append("✅ **No high correlations found** - Dimensions are independent")

        report.append("")

        # Decision
        report.append("---")
        report.append("")
        report.append("## Decision")
        report.append("")

        issues = []
        if pc1_variance >= 70:
            issues.append(f"PC1 = {pc1_variance:.1f}% (should be < 70%)")
        if redundancy > 50:
            issues.append(f"Redundancy = {redundancy:.1f}% (high)")
        if len(high_corrs) > 3:
            issues.append(f"{len(high_corrs)} high correlations found")

        if not issues:
            report.append("✅ **APPROVED FOR TRAINING**")
            report.append("")
            report.append("PCA analysis confirms:")
            report.append(f"- Multi-dimensional problem (PC1 = {pc1_variance:.1f}%)")
            report.append("- Low dimension redundancy")
            report.append("- Independent scoring dimensions")
        else:
            report.append("⚠️ **REVIEW REQUIRED**")
            report.append("")
            report.append("Issues found:")
            for issue in issues:
                report.append(f"- {issue}")

        report.append("")

        return '\n'.join(report)


def main():
    """Main execution."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/pca_analysis_calibration.py <calibration_dir>")
        print("\nExample:")
        print("  python scripts/pca_analysis_calibration.py datasets/calibration/calibration_1k_20251124_180913")
        return

    calibration_dir = Path(sys.argv[1])
    if not calibration_dir.exists():
        print(f"ERROR: Calibration directory not found: {calibration_dir}")
        return

    print("="*70)
    print("PCA Analysis - Oracle Calibration")
    print("="*70)
    print()

    # Initialize analyzer
    analyzer = PCAAnalyzer(calibration_dir)

    # Load data
    data = analyzer.load_scores()
    if len(data) == 0:
        print("ERROR: No valid score vectors found")
        return

    analyzer.data_matrix = data

    # Compute PCA
    print("\nComputing PCA...")
    pca_results = analyzer.compute_pca(data)

    # Compute correlations
    print("Computing correlations...")
    corr_matrix = analyzer.compute_correlations(data)

    # Generate report
    print("Generating report...")
    report = analyzer.generate_report(pca_results, corr_matrix)

    # Save report
    report_path = calibration_dir / "pca_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved: {report_path}")

    # Save PCA results as JSON
    results_path = calibration_dir / "pca_analysis.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'pca_results': pca_results,
            'correlation_matrix': corr_matrix.tolist()
        }, f, indent=2)
    print(f"Results saved: {results_path}")

    print()
    print("="*70)
    print("PCA ANALYSIS COMPLETE")
    print("="*70)
    print(f"Review the report at: {report_path}")


if __name__ == "__main__":
    main()
