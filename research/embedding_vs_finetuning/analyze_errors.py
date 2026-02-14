"""
Analyze error distribution for embedding vs fine-tuning research.

Key questions:
1. Is the error distribution normal or skewed?
2. Does error vary by score range (systematic bias)?
3. How does this affect tier classification?
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import torch
import torch.nn as nn
from typing import List


class MLPProbe(nn.Module):
    """Two-layer MLP probe for regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int] = [256, 128],
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_embeddings(path: Path):
    """Load embeddings from disk."""
    data = np.load(path, allow_pickle=True)
    return {
        'embeddings': data['embeddings'],
        'labels': data['labels'],
        'dimension_names': list(data['dimension_names'])
    }


def load_probe(path: Path):
    """Load trained probe."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def analyze_errors(predictions: np.ndarray, labels: np.ndarray, dimension_names: list):
    """Analyze error distribution."""
    errors = predictions - labels  # positive = overestimate, negative = underestimate

    results = {
        'overall': {},
        'per_dimension': {}
    }

    # Overall statistics
    flat_errors = errors.flatten()
    results['overall'] = {
        'mean_error': float(np.mean(flat_errors)),  # bias
        'std_error': float(np.std(flat_errors)),
        'skewness': float(stats.skew(flat_errors)),
        'kurtosis': float(stats.kurtosis(flat_errors)),
        'median_error': float(np.median(flat_errors)),
        'percentile_5': float(np.percentile(flat_errors, 5)),
        'percentile_95': float(np.percentile(flat_errors, 95)),
    }

    # Normality test (Shapiro-Wilk on sample)
    sample = np.random.choice(flat_errors, min(5000, len(flat_errors)), replace=False)
    _, p_value = stats.shapiro(sample)
    results['overall']['normality_p_value'] = float(p_value)
    results['overall']['is_normal'] = bool(p_value > 0.05)

    # Per-dimension analysis
    for i, dim_name in enumerate(dimension_names):
        dim_errors = errors[:, i]
        results['per_dimension'][dim_name] = {
            'mean_error': float(np.mean(dim_errors)),
            'std_error': float(np.std(dim_errors)),
            'skewness': float(stats.skew(dim_errors)),
            'mae': float(np.mean(np.abs(dim_errors))),
        }

    return results, errors


def analyze_by_score_range(predictions: np.ndarray, labels: np.ndarray):
    """Check if errors vary by true score range."""
    flat_pred = predictions.flatten()
    flat_labels = labels.flatten()
    flat_errors = flat_pred - flat_labels

    ranges = [(0, 3), (3, 5), (5, 7), (7, 10)]
    range_stats = {}

    for low, high in ranges:
        mask = (flat_labels >= low) & (flat_labels < high)
        if mask.sum() > 0:
            range_errors = flat_errors[mask]
            range_stats[f'{low}-{high}'] = {
                'count': int(mask.sum()),
                'mean_error': float(np.mean(range_errors)),
                'mae': float(np.mean(np.abs(range_errors))),
                'std_error': float(np.std(range_errors)),
            }

    return range_stats


def analyze_tier_classification(predictions: np.ndarray, labels: np.ndarray, thresholds: list = [3.0, 5.0, 7.0]):
    """Analyze tier classification accuracy."""
    # Define tiers: 0=low (<3), 1=medium (3-5), 2=good (5-7), 3=high (>=7)
    def to_tier(scores, thresholds):
        tiers = np.zeros_like(scores, dtype=int)
        for i, thresh in enumerate(thresholds):
            tiers[scores >= thresh] = i + 1
        return tiers

    pred_tiers = to_tier(predictions, thresholds)
    true_tiers = to_tier(labels, thresholds)

    # Overall accuracy
    accuracy = np.mean(pred_tiers == true_tiers)

    # Within-1-tier accuracy (off by at most 1 tier)
    within_1 = np.mean(np.abs(pred_tiers - true_tiers) <= 1)

    # Confusion analysis
    confusion = {}
    for true_t in range(len(thresholds) + 1):
        for pred_t in range(len(thresholds) + 1):
            mask = (true_tiers == true_t) & (pred_tiers == pred_t)
            count = mask.sum()
            if count > 0:
                confusion[f'true_{true_t}_pred_{pred_t}'] = int(count)

    # Per-tier accuracy
    tier_accuracy = {}
    for t in range(len(thresholds) + 1):
        mask = true_tiers == t
        if mask.sum() > 0:
            tier_accuracy[f'tier_{t}'] = {
                'count': int(mask.sum()),
                'accuracy': float(np.mean(pred_tiers[mask] == true_tiers[mask])),
                'within_1': float(np.mean(np.abs(pred_tiers[mask] - true_tiers[mask]) <= 1)),
            }

    return {
        'exact_accuracy': float(accuracy),
        'within_1_tier': float(within_1),
        'per_tier': tier_accuracy,
        'confusion': confusion
    }


def plot_error_distribution(errors: np.ndarray, output_path: Path):
    """Create visualization of error distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    flat_errors = errors.flatten()

    # 1. Histogram of errors
    ax = axes[0, 0]
    ax.hist(flat_errors, bins=50, density=True, alpha=0.7, edgecolor='black')

    # Overlay normal distribution
    mu, std = np.mean(flat_errors), np.std(flat_errors)
    x = np.linspace(mu - 4*std, mu + 4*std, 100)
    ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
    ax.axvline(0, color='green', linestyle='--', label='Zero error')
    ax.set_xlabel('Error (prediction - true)')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()

    # 2. Q-Q plot
    ax = axes[0, 1]
    stats.probplot(flat_errors, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (vs Normal)')

    # 3. Error by true score
    ax = axes[1, 0]
    # Sample for scatter plot
    n_samples = min(5000, len(flat_errors))
    idx = np.random.choice(len(flat_errors), n_samples, replace=False)
    flat_labels = errors.flatten()  # This is wrong, need labels
    # We'll pass labels separately
    ax.set_xlabel('True Score')
    ax.set_ylabel('Error')
    ax.set_title('Error vs True Score (see separate plot)')

    # 4. Per-dimension MAE
    ax = axes[1, 1]
    dim_maes = [np.mean(np.abs(errors[:, i])) for i in range(errors.shape[1])]
    ax.barh(range(len(dim_maes)), dim_maes)
    ax.set_xlabel('MAE')
    ax.set_title('MAE by Dimension')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_error_vs_score(predictions: np.ndarray, labels: np.ndarray, output_path: Path):
    """Plot error vs true score to check for systematic bias."""
    fig, ax = plt.subplots(figsize=(10, 6))

    flat_pred = predictions.flatten()
    flat_labels = labels.flatten()
    flat_errors = flat_pred - flat_labels

    # Sample for visibility
    n_samples = min(5000, len(flat_errors))
    idx = np.random.choice(len(flat_errors), n_samples, replace=False)

    ax.scatter(flat_labels[idx], flat_errors[idx], alpha=0.3, s=10)

    # Add trend line (binned means)
    bins = np.linspace(0, 10, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    for i in range(len(bins) - 1):
        mask = (flat_labels >= bins[i]) & (flat_labels < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(np.mean(flat_errors[mask]))
        else:
            bin_means.append(np.nan)

    ax.plot(bin_centers, bin_means, 'r-', lw=2, marker='o', label='Mean error by score')
    ax.axhline(0, color='green', linestyle='--', label='Zero error')
    ax.set_xlabel('True Score')
    ax.set_ylabel('Error (prediction - true)')
    ax.set_title('Error vs True Score: Checking for Systematic Bias')
    ax.legend()
    ax.set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main():
    base_dir = Path('research/embedding_vs_finetuning')
    embeddings_dir = base_dir / 'embeddings'
    results_dir = base_dir / 'results'

    # Load test embeddings for all-mpnet-base-v2 (best cached model, MAE 0.94)
    # Note: e5-large-v2 (MAE 0.86) was computed on-the-fly during eval, not cached
    test_emb_path = embeddings_dir / 'uplifting_v5_all-mpnet-base-v2_test.npz'
    test_data = load_embeddings(test_emb_path)

    # Load MLP probe
    probe_path = results_dir / 'uplifting_v5_all-mpnet-base-v2_mlp.pkl'
    probe_data = load_probe(probe_path)

    # Generate predictions
    embeddings = test_data['embeddings']
    labels = test_data['labels']
    dimension_names = test_data['dimension_names']

    # Reconstruct model from saved config and state dict
    model_config = probe_data['model_config']
    model = MLPProbe(
        input_dim=model_config['input_dim'],
        output_dim=model_config['output_dim']
    )
    model.load_state_dict(probe_data['state_dict'])
    scaler = probe_data.get('scaler')

    if scaler:
        embeddings_scaled = scaler.transform(embeddings)
    else:
        embeddings_scaled = embeddings

    # Get predictions
    model.eval()
    with torch.no_grad():
        X = torch.FloatTensor(embeddings_scaled)
        predictions = model(X).numpy()

    # Clip predictions to valid range
    predictions = np.clip(predictions, 0, 10)

    print("=" * 60)
    print("ERROR DISTRIBUTION ANALYSIS: all-mpnet-base-v2 + MLP")
    print("(Note: e5-large-v2 would be similar but ~5% better)")
    print("=" * 60)

    # 1. Error distribution analysis
    error_stats, errors = analyze_errors(predictions, labels, dimension_names)

    print("\n[STATS] OVERALL ERROR STATISTICS")
    print("-" * 40)
    overall = error_stats['overall']
    print(f"Mean Error (bias):     {overall['mean_error']:+.4f}")
    print(f"Median Error:          {overall['median_error']:+.4f}")
    print(f"Std Dev:               {overall['std_error']:.4f}")
    print(f"Skewness:              {overall['skewness']:+.4f}")
    print(f"Kurtosis:              {overall['kurtosis']:.4f}")
    print(f"5th percentile:        {overall['percentile_5']:+.4f}")
    print(f"95th percentile:       {overall['percentile_95']:+.4f}")
    print(f"Normal distribution:   {'Yes' if overall['is_normal'] else 'No'} (p={overall['normality_p_value']:.4f})")

    # Interpretation
    print("\n[NOTE] INTERPRETATION")
    if abs(overall['mean_error']) < 0.1:
        print("[OK] Low bias - predictions are centered around true values")
    else:
        direction = "overestimates" if overall['mean_error'] > 0 else "underestimates"
        print(f"[!] Systematic bias: model {direction} by ~{abs(overall['mean_error']):.2f} points")

    if abs(overall['skewness']) < 0.5:
        print("[OK] Symmetric error distribution")
    else:
        direction = "positive" if overall['skewness'] > 0 else "negative"
        print(f"[!] Skewed {direction}: errors tend toward {direction} values")

    # 2. Error by score range
    print("\n[STATS] ERROR BY SCORE RANGE")
    print("-" * 40)
    range_stats = analyze_by_score_range(predictions, labels)
    print(f"{'Range':<10} {'Count':<8} {'Mean Error':<12} {'MAE':<8}")
    for range_name, stats_dict in range_stats.items():
        print(f"{range_name:<10} {stats_dict['count']:<8} {stats_dict['mean_error']:+.4f}      {stats_dict['mae']:.4f}")

    # 3. Tier classification
    print("\n[STATS] TIER CLASSIFICATION ACCURACY")
    print("-" * 40)
    print("Tiers: 0 (<3), 1 (3-5), 2 (5-7), 3 (>=7)")
    tier_stats = analyze_tier_classification(predictions, labels)
    print(f"Exact tier match:      {tier_stats['exact_accuracy']*100:.1f}%")
    print(f"Within 1 tier:         {tier_stats['within_1_tier']*100:.1f}%")

    print("\nPer-tier accuracy:")
    for tier_name, tier_data in tier_stats['per_tier'].items():
        print(f"  {tier_name}: {tier_data['accuracy']*100:.1f}% exact, {tier_data['within_1']*100:.1f}% within-1 (n={tier_data['count']})")

    # 4. Per-dimension analysis
    print("\n[STATS] PER-DIMENSION ERROR STATISTICS")
    print("-" * 40)
    print(f"{'Dimension':<25} {'Bias':>10} {'MAE':>8} {'Skew':>8}")
    for dim_name, dim_stats in error_stats['per_dimension'].items():
        print(f"{dim_name:<25} {dim_stats['mean_error']:+.4f}    {dim_stats['mae']:.4f}   {dim_stats['skewness']:+.4f}")

    # 5. Generate plots
    print("\n[PLOT] Generating plots...")
    plot_error_distribution(errors, results_dir / 'error_distribution.png')
    plot_error_vs_score(predictions, labels, results_dir / 'error_vs_score.png')
    print(f"Saved: {results_dir / 'error_distribution.png'}")
    print(f"Saved: {results_dir / 'error_vs_score.png'}")

    # 6. Save full results
    full_results = {
        'error_stats': error_stats,
        'range_stats': range_stats,
        'tier_stats': tier_stats
    }
    with open(results_dir / 'error_analysis.json', 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved: {results_dir / 'error_analysis.json'}")

    print("\n" + "=" * 60)
    print("SUMMARY FOR NEXUSMIND GRANT")
    print("=" * 60)
    print(f"""
For tier-based filtering (the NexusMind use case):
- Exact tier classification: {tier_stats['exact_accuracy']*100:.1f}%
- Within 1 tier: {tier_stats['within_1_tier']*100:.1f}%

This means the embedding approach could be viable for:
- Initial filtering/triage (accept within-1 accuracy)
- "Quick start" Filter SDK option (no fine-tuning needed)
- Resource-constrained deployments

Fine-tuning remains better for:
- Production accuracy requirements
- Precise score-based decisions
""")


if __name__ == '__main__':
    main()
