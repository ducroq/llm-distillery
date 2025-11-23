"""
Analyze prediction error characteristics from test set benchmarks.

This script examines:
1. Prediction bias (systematic over/under-prediction)
2. Error distribution (symmetric vs skewed)
3. Per-dimension error patterns
4. Score range dependencies (do errors vary by true score value?)
5. Correlation structure (which dimensions have correlated errors?)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_predictions(predictions_file: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load predictions and labels from benchmark results."""
    with open(predictions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = np.array(data["predictions"])
    labels = np.array(data["labels"])
    dimension_names = data["dimension_names"]
    article_ids = data["article_ids"]

    return predictions, labels, dimension_names, article_ids


def compute_bias_metrics(predictions: np.ndarray, labels: np.ndarray, dimension_names: List[str]) -> Dict:
    """Compute bias metrics (mean error, not absolute)."""
    results = {}

    # Overall bias (mean error)
    errors = predictions - labels
    overall_bias = np.mean(errors)

    results["overall"] = {
        "mean_error": float(overall_bias),
        "median_error": float(np.median(errors)),
        "std_error": float(np.std(errors)),
        "interpretation": "positive = over-prediction, negative = under-prediction"
    }

    # Per-dimension bias
    results["dimensions"] = {}
    for i, dim_name in enumerate(dimension_names):
        dim_errors = errors[:, i]
        results["dimensions"][dim_name] = {
            "mean_error": float(np.mean(dim_errors)),
            "median_error": float(np.median(dim_errors)),
            "std_error": float(np.std(dim_errors)),
        }

    return results


def compute_error_distribution(predictions: np.ndarray, labels: np.ndarray, dimension_names: List[str]) -> Dict:
    """Analyze error distribution characteristics."""
    results = {}
    errors = predictions - labels

    # Overall distribution
    results["overall"] = {
        "skewness": float(stats.skew(errors.flatten())),
        "kurtosis": float(stats.kurtosis(errors.flatten())),
        "shapiro_wilk_p": float(stats.shapiro(errors.flatten()[:5000])[1]),  # Subsample for speed
    }

    # Per-dimension distribution
    results["dimensions"] = {}
    for i, dim_name in enumerate(dimension_names):
        dim_errors = errors[:, i]
        results["dimensions"][dim_name] = {
            "skewness": float(stats.skew(dim_errors)),
            "kurtosis": float(stats.kurtosis(dim_errors)),
        }

    return results


def analyze_score_range_dependency(predictions: np.ndarray, labels: np.ndarray, dimension_names: List[str]) -> Dict:
    """Check if errors depend on true score value."""
    results = {}
    errors = predictions - labels

    # Bin true scores into ranges
    bins = [0, 2, 4, 6, 8, 10]
    bin_labels = ["0-2", "2-4", "4-6", "6-8", "8-10"]

    results["score_ranges"] = {}

    for i, dim_name in enumerate(dimension_names):
        dim_labels = labels[:, i]
        dim_errors = errors[:, i]
        dim_predictions = predictions[:, i]

        range_analysis = {}
        for j in range(len(bins) - 1):
            mask = (dim_labels >= bins[j]) & (dim_labels < bins[j+1])
            if mask.sum() > 0:
                range_analysis[bin_labels[j]] = {
                    "count": int(mask.sum()),
                    "mean_error": float(np.mean(dim_errors[mask])),
                    "mae": float(np.mean(np.abs(dim_errors[mask]))),
                    "mean_true": float(np.mean(dim_labels[mask])),
                    "mean_pred": float(np.mean(dim_predictions[mask])),
                }

        results["score_ranges"][dim_name] = range_analysis

    return results


def compute_error_correlations(predictions: np.ndarray, labels: np.ndarray, dimension_names: List[str]) -> Dict:
    """Analyze correlation structure of errors across dimensions."""
    errors = predictions - labels

    # Compute error correlation matrix
    error_corr = np.corrcoef(errors.T)

    results = {
        "correlation_matrix": error_corr.tolist(),
        "dimension_names": dimension_names,
    }

    # Find highly correlated dimension pairs
    high_corr_pairs = []
    for i in range(len(dimension_names)):
        for j in range(i + 1, len(dimension_names)):
            corr = error_corr[i, j]
            if abs(corr) > 0.3:  # Threshold for "notable" correlation
                high_corr_pairs.append({
                    "dim1": dimension_names[i],
                    "dim2": dimension_names[j],
                    "correlation": float(corr),
                })

    results["high_correlation_pairs"] = sorted(high_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)

    return results


def identify_outliers(predictions: np.ndarray, labels: np.ndarray, dimension_names: List[str], article_ids: List[str]) -> Dict:
    """Identify articles with largest prediction errors."""
    errors = predictions - labels
    abs_errors = np.abs(errors)

    # Overall worst predictions
    overall_errors = np.mean(abs_errors, axis=1)
    worst_indices = np.argsort(overall_errors)[-20:][::-1]

    results = {
        "worst_overall": []
    }

    for idx in worst_indices:
        results["worst_overall"].append({
            "article_id": article_ids[idx],
            "mae": float(overall_errors[idx]),
            "true_scores": labels[idx].tolist(),
            "predicted_scores": predictions[idx].tolist(),
            "errors": errors[idx].tolist(),
        })

    # Worst per dimension
    results["worst_per_dimension"] = {}
    for i, dim_name in enumerate(dimension_names):
        dim_abs_errors = abs_errors[:, i]
        worst_dim_indices = np.argsort(dim_abs_errors)[-10:][::-1]

        results["worst_per_dimension"][dim_name] = []
        for idx in worst_dim_indices:
            results["worst_per_dimension"][dim_name].append({
                "article_id": article_ids[idx],
                "error": float(errors[idx, i]),
                "true_score": float(labels[idx, i]),
                "predicted_score": float(predictions[idx, i]),
            })

    return results


def create_visualizations(predictions: np.ndarray, labels: np.ndarray, dimension_names: List[str], output_dir: Path, filter_name: str):
    """Create visualization plots for error analysis."""
    errors = predictions - labels

    # 1. Error distribution histogram
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, dim_name in enumerate(dimension_names):
        if i < len(axes):
            ax = axes[i]
            dim_errors = errors[:, i]
            ax.hist(dim_errors, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
            ax.axvline(np.mean(dim_errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dim_errors):.3f}')
            ax.set_xlabel('Error (Predicted - True)')
            ax.set_ylabel('Count')
            ax.set_title(f'{dim_name}\nBias: {np.mean(dim_errors):.3f}, Std: {np.std(dim_errors):.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(dimension_names), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / f"{filter_name}_error_distributions.png", dpi=150)
    plt.close()

    # 2. Scatter plots: Predicted vs True
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, dim_name in enumerate(dimension_names):
        if i < len(axes):
            ax = axes[i]
            ax.scatter(labels[:, i], predictions[:, i], alpha=0.3, s=10)
            ax.plot([0, 10], [0, 10], 'r--', linewidth=2, label='Perfect prediction')
            ax.set_xlabel('True Score')
            ax.set_ylabel('Predicted Score')
            ax.set_title(f'{dim_name}')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.legend()
            ax.grid(True, alpha=0.3)

    for i in range(len(dimension_names), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / f"{filter_name}_pred_vs_true.png", dpi=150)
    plt.close()

    # 3. Error correlation heatmap
    error_corr = np.corrcoef(errors.T)

    plt.figure(figsize=(10, 8))
    sns.heatmap(error_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=dimension_names, yticklabels=dimension_names,
                vmin=-1, vmax=1, square=True)
    plt.title(f'{filter_name}: Error Correlation Matrix\n(How errors across dimensions relate)')
    plt.tight_layout()
    plt.savefig(output_dir / f"{filter_name}_error_correlations.png", dpi=150)
    plt.close()

    # 4. Bias by score range
    bins = [0, 2, 4, 6, 8, 10]
    bin_labels = ["0-2", "2-4", "4-6", "6-8", "8-10"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, dim_name in enumerate(dimension_names):
        if i < len(axes):
            ax = axes[i]
            dim_labels = labels[:, i]
            dim_errors = errors[:, i]

            mean_errors = []
            for j in range(len(bins) - 1):
                mask = (dim_labels >= bins[j]) & (dim_labels < bins[j+1])
                if mask.sum() > 0:
                    mean_errors.append(np.mean(dim_errors[mask]))
                else:
                    mean_errors.append(0)

            ax.bar(bin_labels, mean_errors, alpha=0.7, edgecolor='black')
            ax.axhline(0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('True Score Range')
            ax.set_ylabel('Mean Error')
            ax.set_title(f'{dim_name}')
            ax.grid(True, alpha=0.3, axis='y')

    for i in range(len(dimension_names), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / f"{filter_name}_bias_by_range.png", dpi=150)
    plt.close()

    print(f"Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze prediction error characteristics")
    parser.add_argument(
        "--filter",
        type=Path,
        required=True,
        help="Path to filter directory (e.g., filters/investment-risk/v4)",
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=None,
        help="Path to predictions file (default: {filter}/benchmarks/test_set_predictions.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for analysis (default: {filter}/benchmarks/error_analysis)",
    )

    args = parser.parse_args()

    # Set default paths
    if args.predictions_file is None:
        args.predictions_file = args.filter / "benchmarks" / "test_set_predictions.json"
    if args.output_dir is None:
        args.output_dir = args.filter / "benchmarks" / "error_analysis"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Prediction Error Analysis")
    print("=" * 80)
    print(f"Filter: {args.filter}")
    print(f"Predictions: {args.predictions_file}")
    print(f"Output: {args.output_dir}")
    print()

    # Load data
    print("Loading predictions...")
    predictions, labels, dimension_names, article_ids = load_predictions(args.predictions_file)
    print(f"Loaded {len(predictions)} predictions with {len(dimension_names)} dimensions")
    print()

    # 1. Bias Analysis
    print("Computing bias metrics...")
    bias_results = compute_bias_metrics(predictions, labels, dimension_names)

    print("\n" + "=" * 80)
    print("BIAS ANALYSIS (Mean Error = Predicted - True)")
    print("=" * 80)
    print(f"\nOverall Bias: {bias_results['overall']['mean_error']:.4f}")
    print(f"Overall Median Error: {bias_results['overall']['median_error']:.4f}")
    print(f"Overall Std Error: {bias_results['overall']['std_error']:.4f}")
    print()
    print("Per-Dimension Bias:")
    for dim_name, metrics in bias_results["dimensions"].items():
        bias = metrics["mean_error"]
        direction = "OVER-predicting" if bias > 0 else "UNDER-predicting" if bias < 0 else "NEUTRAL"
        print(f"  {dim_name:30s}: {bias:+.4f}  ({direction})")

    # 2. Error Distribution
    print("\n" + "=" * 80)
    print("ERROR DISTRIBUTION")
    print("=" * 80)
    dist_results = compute_error_distribution(predictions, labels, dimension_names)

    print(f"\nOverall Error Distribution:")
    print(f"  Skewness: {dist_results['overall']['skewness']:.4f}  (0=symmetric, +ve=right tail, -ve=left tail)")
    print(f"  Kurtosis: {dist_results['overall']['kurtosis']:.4f}  (0=normal, +ve=heavy tails, -ve=light tails)")
    print(f"  Shapiro-Wilk p-value: {dist_results['overall']['shapiro_wilk_p']:.4f}  (p<0.05 = not normal)")

    # 3. Score Range Dependency
    print("\n" + "=" * 80)
    print("SCORE RANGE DEPENDENCY")
    print("=" * 80)
    range_results = analyze_score_range_dependency(predictions, labels, dimension_names)

    print("\nChecking if errors vary by true score value...")
    print("(Model might be better/worse at low vs high scores)")
    print()

    for dim_name, ranges in range_results["score_ranges"].items():
        print(f"{dim_name}:")
        for range_label, metrics in ranges.items():
            print(f"  {range_label}: n={metrics['count']:3d}, bias={metrics['mean_error']:+.3f}, mae={metrics['mae']:.3f}")
        print()

    # 4. Error Correlations
    print("=" * 80)
    print("ERROR CORRELATIONS")
    print("=" * 80)
    corr_results = compute_error_correlations(predictions, labels, dimension_names)

    print("\nDimension pairs with correlated errors (|r| > 0.3):")
    if corr_results["high_correlation_pairs"]:
        for pair in corr_results["high_correlation_pairs"]:
            print(f"  {pair['dim1']} <-> {pair['dim2']}: r={pair['correlation']:.3f}")
    else:
        print("  None - errors are independent across dimensions")

    # 5. Outliers
    print("\n" + "=" * 80)
    print("OUTLIER ANALYSIS")
    print("=" * 80)
    outlier_results = identify_outliers(predictions, labels, dimension_names, article_ids)

    print("\nTop 10 articles with highest overall MAE:")
    for i, article in enumerate(outlier_results["worst_overall"][:10], 1):
        print(f"  {i}. {article['article_id']}: MAE={article['mae']:.3f}")

    # Save all results
    results_file = args.output_dir / "error_analysis.json"
    print(f"\nSaving detailed results to: {results_file}")

    full_results = {
        "filter": str(args.filter),
        "num_examples": len(predictions),
        "dimension_names": dimension_names,
        "bias_analysis": bias_results,
        "distribution_analysis": dist_results,
        "score_range_analysis": range_results,
        "correlation_analysis": corr_results,
        "outlier_analysis": outlier_results,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    # Create visualizations
    print("\nGenerating visualizations...")
    filter_name = args.filter.name
    create_visualizations(predictions, labels, dimension_names, args.output_dir, filter_name)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - error_analysis.json (detailed metrics)")
    print(f"  - {filter_name}_error_distributions.png (histograms)")
    print(f"  - {filter_name}_pred_vs_true.png (scatter plots)")
    print(f"  - {filter_name}_error_correlations.png (heatmap)")
    print(f"  - {filter_name}_bias_by_range.png (bias by score range)")


if __name__ == "__main__":
    main()
