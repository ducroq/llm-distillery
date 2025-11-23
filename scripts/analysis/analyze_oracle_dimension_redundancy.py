"""
Analyze dimension redundancy in oracle labels BEFORE student training.

This script examines the oracle-labeled training data to identify:
1. Highly correlated dimensions (candidates for merging)
2. Intrinsic dimensionality (how many independent factors?)
3. Variance explained by each dimension
4. Optimal dimension count via PCA

Use this BEFORE training to decide if dimension reduction would be beneficial.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage


def load_training_data(data_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load oracle labels from training data."""
    labels_list = []
    dimension_names = None

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            labels_list.append(example["labels"])
            if dimension_names is None:
                dimension_names = example["dimension_names"]

    labels = np.array(labels_list)
    return labels, dimension_names


def compute_correlation_analysis(labels: np.ndarray, dimension_names: List[str]) -> Dict:
    """Compute correlation matrix and identify redundant dimensions."""
    # Correlation matrix
    corr_matrix = np.corrcoef(labels.T)

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(dimension_names)):
        for j in range(i + 1, len(dimension_names)):
            corr = corr_matrix[i, j]
            if abs(corr) > 0.7:  # Threshold for "high correlation"
                high_corr_pairs.append({
                    "dim1": dimension_names[i],
                    "dim2": dimension_names[j],
                    "correlation": float(corr),
                    "strength": "very high" if abs(corr) > 0.9 else "high",
                })

    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "correlation_matrix": corr_matrix.tolist(),
        "dimension_names": dimension_names,
        "high_correlation_pairs": high_corr_pairs,
    }


def perform_pca_analysis(labels: np.ndarray, dimension_names: List[str]) -> Dict:
    """Perform PCA to determine intrinsic dimensionality."""
    # Normalize data (important for PCA)
    labels_normalized = (labels - labels.mean(axis=0)) / labels.std(axis=0)

    # Fit PCA
    pca = PCA()
    pca.fit(labels_normalized)

    # Variance explained
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)

    # Find number of components needed for 90%, 95%, 99% variance
    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1

    # Component loadings (which dimensions contribute to each PC)
    loadings = pca.components_

    return {
        "variance_explained": variance_explained.tolist(),
        "cumulative_variance": cumulative_variance.tolist(),
        "n_components_90": int(n_components_90),
        "n_components_95": int(n_components_95),
        "n_components_99": int(n_components_99),
        "loadings": loadings.tolist(),
        "interpretation": {
            "total_dimensions": len(dimension_names),
            "effective_dimensions_90": int(n_components_90),
            "effective_dimensions_95": int(n_components_95),
            "effective_dimensions_99": int(n_components_99),
            "redundancy_ratio": float(1.0 - n_components_95 / len(dimension_names)),
        }
    }


def suggest_dimension_reduction(corr_results: Dict, pca_results: Dict, threshold: float = 0.85) -> Dict:
    """Suggest which dimensions to merge based on correlation and PCA."""
    suggestions = []

    # Group highly correlated dimensions
    merged_groups = []
    used_dims = set()

    for pair in corr_results["high_correlation_pairs"]:
        if abs(pair["correlation"]) >= threshold:
            dim1, dim2 = pair["dim1"], pair["dim2"]

            # Find if either dimension is already in a group
            found_group = None
            for group in merged_groups:
                if dim1 in group or dim2 in group:
                    found_group = group
                    break

            if found_group:
                found_group.update([dim1, dim2])
            else:
                merged_groups.append({dim1, dim2})

            used_dims.update([dim1, dim2])

    # Convert to list of suggestions
    for i, group in enumerate(merged_groups, 1):
        group_list = sorted(list(group))
        suggestions.append({
            "merged_group_id": i,
            "dimensions": group_list,
            "suggested_name": f"merged_{group_list[0][:15]}",
            "count": len(group_list),
            "reason": f"Correlation >= {threshold:.2f}",
        })

    # Dimensions that remain independent
    all_dims = set(corr_results["dimension_names"])
    independent_dims = sorted(list(all_dims - used_dims))

    return {
        "merge_suggestions": suggestions,
        "independent_dimensions": independent_dims,
        "original_count": len(corr_results["dimension_names"]),
        "reduced_count": len(suggestions) + len(independent_dims),
        "reduction_percentage": float(
            100 * (1 - (len(suggestions) + len(independent_dims)) / len(corr_results["dimension_names"]))
        ),
    }


def create_visualizations(
    labels: np.ndarray,
    dimension_names: List[str],
    corr_results: Dict,
    pca_results: Dict,
    output_dir: Path,
    filter_name: str,
):
    """Create visualization plots."""

    # 1. Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = np.array(corr_results["correlation_matrix"])
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        xticklabels=dimension_names,
        yticklabels=dimension_names,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Correlation Coefficient"},
    )
    plt.title(f"{filter_name}: Oracle Label Correlation Matrix\n(High values = redundant dimensions)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{filter_name}_oracle_correlations.png", dpi=150)
    plt.close()

    # 2. Hierarchical clustering dendrogram
    plt.figure(figsize=(12, 6))
    linkage_matrix = linkage(labels.T, method="ward")
    dendrogram(linkage_matrix, labels=dimension_names, leaf_rotation=90)
    plt.title(f"{filter_name}: Dimension Clustering\n(Closer branches = more similar dimensions)")
    plt.xlabel("Dimension")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output_dir / f"{filter_name}_dimension_clustering.png", dpi=150)
    plt.close()

    # 3. PCA variance explained
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scree plot
    variance_explained = pca_results["variance_explained"]
    ax1.bar(range(1, len(variance_explained) + 1), variance_explained)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained")
    ax1.set_title("Scree Plot: Variance per Component")
    ax1.grid(True, alpha=0.3)

    # Cumulative variance
    cumulative_variance = pca_results["cumulative_variance"]
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
    ax2.axhline(y=0.90, color="r", linestyle="--", label="90% variance")
    ax2.axhline(y=0.95, color="orange", linestyle="--", label="95% variance")
    ax2.axhline(y=0.99, color="purple", linestyle="--", label="99% variance")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{filter_name}_pca_variance.png", dpi=150)
    plt.close()

    # 4. PCA loading heatmap (first 4 components)
    n_components_show = min(4, len(dimension_names))
    loadings = np.array(pca_results["loadings"][:n_components_show])

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        loadings,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        xticklabels=dimension_names,
        yticklabels=[f"PC{i+1}" for i in range(n_components_show)],
        cbar_kws={"label": "Loading"},
    )
    plt.title(f"{filter_name}: PCA Component Loadings\n(Which dimensions contribute to each component)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{filter_name}_pca_loadings.png", dpi=150)
    plt.close()

    print(f"Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dimension redundancy in oracle training data"
    )
    parser.add_argument(
        "--filter",
        type=Path,
        required=True,
        help="Path to filter directory (e.g., filters/investment-risk/v4)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to training data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: {filter}/dimension_analysis)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.85,
        help="Correlation threshold for merge suggestions (default: 0.85)",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.filter / "dimension_analysis"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Oracle Dimension Redundancy Analysis")
    print("=" * 80)
    print(f"Filter: {args.filter}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print()

    # Load training data
    train_path = args.data_dir / "train.jsonl"
    print(f"Loading training data from: {train_path}")
    labels, dimension_names = load_training_data(train_path)
    print(f"Loaded {len(labels)} training examples")
    print(f"Dimensions ({len(dimension_names)}): {dimension_names}")
    print()

    # Correlation analysis
    print("Computing correlation analysis...")
    corr_results = compute_correlation_analysis(labels, dimension_names)

    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    if corr_results["high_correlation_pairs"]:
        print(f"\nFound {len(corr_results['high_correlation_pairs'])} highly correlated dimension pairs (|r| > 0.7):")
        print()
        for pair in corr_results["high_correlation_pairs"][:10]:  # Show top 10
            print(f"  {pair['dim1']:30s} <-> {pair['dim2']:30s}: r={pair['correlation']:+.3f} ({pair['strength']})")
        if len(corr_results["high_correlation_pairs"]) > 10:
            print(f"  ... and {len(corr_results['high_correlation_pairs']) - 10} more")
    else:
        print("\nNo highly correlated dimension pairs found. Dimensions are independent!")

    # PCA analysis
    print("\n" + "=" * 80)
    print("PCA ANALYSIS (Intrinsic Dimensionality)")
    print("=" * 80)
    pca_results = perform_pca_analysis(labels, dimension_names)

    print(f"\nOriginal dimensions: {len(dimension_names)}")
    print(f"Effective dimensions (90% variance): {pca_results['n_components_90']}")
    print(f"Effective dimensions (95% variance): {pca_results['n_components_95']}")
    print(f"Effective dimensions (99% variance): {pca_results['n_components_99']}")
    print()
    print(f"Redundancy ratio: {pca_results['interpretation']['redundancy_ratio']:.1%}")
    print(f"  (How much information is redundant)")
    print()

    print("Variance explained by each principal component:")
    for i, var in enumerate(pca_results["variance_explained"], 1):
        cumvar = pca_results["cumulative_variance"][i - 1]
        print(f"  PC{i}: {var:.1%} (cumulative: {cumvar:.1%})")

    # Dimension reduction suggestions
    print("\n" + "=" * 80)
    print("DIMENSION REDUCTION SUGGESTIONS")
    print("=" * 80)
    reduction_suggestions = suggest_dimension_reduction(
        corr_results, pca_results, args.correlation_threshold
    )

    if reduction_suggestions["merge_suggestions"]:
        print(f"\nSuggested dimension merges (correlation >= {args.correlation_threshold}):")
        print()
        for suggestion in reduction_suggestions["merge_suggestions"]:
            print(f"  Group {suggestion['merged_group_id']}: Merge into '{suggestion['suggested_name']}'")
            for dim in suggestion["dimensions"]:
                print(f"    - {dim}")
            print()

        print(f"Independent dimensions (keep as-is):")
        for dim in reduction_suggestions["independent_dimensions"]:
            print(f"  - {dim}")
        print()

        print(f"Dimension count reduction:")
        print(f"  Original: {reduction_suggestions['original_count']}")
        print(f"  Reduced:  {reduction_suggestions['reduced_count']}")
        print(f"  Savings:  {reduction_suggestions['reduction_percentage']:.1f}%")
    else:
        print(f"\nNo dimension merges recommended at threshold {args.correlation_threshold}")
        print("All dimensions appear to be independent!")

    # Save results
    results_file = args.output_dir / "dimension_analysis.json"
    print(f"\nSaving detailed results to: {results_file}")

    full_results = {
        "filter": str(args.filter),
        "num_training_examples": len(labels),
        "dimension_names": dimension_names,
        "correlation_analysis": corr_results,
        "pca_analysis": pca_results,
        "reduction_suggestions": reduction_suggestions,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    # Create visualizations
    print("\nGenerating visualizations...")
    filter_name = args.filter.name
    create_visualizations(
        labels, dimension_names, corr_results, pca_results, args.output_dir, filter_name
    )

    # Final recommendation
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)

    redundancy_ratio = pca_results["interpretation"]["redundancy_ratio"]
    n_high_corr = len(corr_results["high_correlation_pairs"])

    if redundancy_ratio < 0.1 and n_high_corr == 0:
        print("\n[OK] NO DIMENSION REDUCTION NEEDED")
        print("   - Dimensions are largely independent")
        print("   - Low redundancy detected")
        print("   - Proceed with all dimensions")
    elif redundancy_ratio < 0.2 and n_high_corr < 5:
        print("\n[WARNING] MINOR REDUNDANCY DETECTED")
        print("   - Some dimension correlation exists")
        print("   - Consider merging if you need faster training")
        print("   - Otherwise, current dimensions are acceptable")
    else:
        print("\n[ACTION NEEDED] DIMENSION REDUCTION RECOMMENDED")
        print(f"   - High redundancy detected ({redundancy_ratio:.1%})")
        print(f"   - {n_high_corr} highly correlated dimension pairs")
        print(f"   - Reduce from {len(dimension_names)} to {reduction_suggestions['reduced_count']} dimensions")
        print("   - Benefits:")
        print("     * Faster training")
        print("     * Lower model size")
        print("     * Reduced overfitting risk")
        print("     * More interpretable results")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
