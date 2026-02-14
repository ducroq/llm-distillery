"""
Analyze correlation between article length and prediction error.

Hypothesis: Embedding models with short context windows (256-512 tokens)
truncate long articles, losing information and increasing prediction error.

Usage:
    python research/embedding_vs_finetuning/analyze_length_vs_error.py \
        --dataset uplifting_v5 \
        --model e5-large-v2 \
        --probe mlp
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import stats

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_articles(data_dir: Path, split: str = 'test') -> List[Dict[str, Any]]:
    """Load articles from JSONL file."""
    file_path = data_dir / f'{split}.jsonl'
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def load_embeddings(input_path: Path) -> Dict[str, Any]:
    """Load embeddings from disk."""
    data = np.load(input_path, allow_pickle=True)
    return {
        'embeddings': data['embeddings'],
        'article_ids': list(data['article_ids']),
        'labels': data['labels'],
        'dimension_names': list(data['dimension_names'])
    }


def load_probe(probe_path: Path) -> Dict[str, Any]:
    """Load trained probe from disk."""
    with open(probe_path, 'rb') as f:
        return pickle.load(f)


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (words * 1.3)."""
    words = len(text.split())
    return int(words * 1.3)


def compute_article_lengths(articles: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Compute character and token lengths for each article."""
    lengths = {}
    for article in articles:
        text = f"{article.get('title', '')}\n\n{article.get('content', '')}"
        lengths[article['id']] = {
            'chars': len(text),
            'tokens_est': estimate_tokens(text),
            'words': len(text.split())
        }
    return lengths


def predict_with_probe(
    probe_type: str,
    probe_data: Dict,
    embeddings: np.ndarray,
    config: Dict[str, Any]
) -> np.ndarray:
    """Generate predictions using a trained probe."""

    if probe_type == 'ridge':
        scaler = probe_data['scaler']
        model = probe_data['model']
        embeddings_scaled = scaler.transform(embeddings)
        return model.predict(embeddings_scaled)

    elif probe_type == 'mlp':
        from train_probes import MLPProbe

        scaler = probe_data['scaler']
        model_config = probe_data['model_config']

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = MLPProbe(
            input_dim=model_config['input_dim'],
            output_dim=model_config['output_dim'],
            hidden_sizes=config.get('hidden_sizes', [256, 128]),
            dropout=0.0
        ).to(device)

        model.load_state_dict(probe_data['state_dict'])
        model.eval()

        embeddings_scaled = scaler.transform(embeddings)
        embeddings_tensor = torch.FloatTensor(embeddings_scaled).to(device)

        with torch.no_grad():
            return model(embeddings_tensor).cpu().numpy()

    elif probe_type == 'lightgbm':
        models = probe_data['models']
        return np.column_stack([model.predict(embeddings) for model in models])

    else:
        raise ValueError(f"Unknown probe type: {probe_type}")


def analyze_length_error_correlation(
    article_ids: List[str],
    lengths: Dict[str, Dict[str, int]],
    predictions: np.ndarray,
    labels: np.ndarray,
    dimension_names: List[str],
    model_max_tokens: int
) -> Dict[str, Any]:
    """Analyze correlation between article length and prediction error."""

    # Compute per-article MAE (averaged across dimensions)
    per_article_mae = np.mean(np.abs(predictions - labels), axis=1)

    # Get lengths in order
    tokens = np.array([lengths[aid]['tokens_est'] for aid in article_ids])
    chars = np.array([lengths[aid]['chars'] for aid in article_ids])

    # Identify truncated articles
    truncated_mask = tokens > model_max_tokens
    n_truncated = np.sum(truncated_mask)
    pct_truncated = 100 * n_truncated / len(tokens)

    results = {
        'n_articles': len(article_ids),
        'n_truncated': int(n_truncated),
        'pct_truncated': float(pct_truncated),
        'model_max_tokens': model_max_tokens,
        'length_stats': {
            'tokens_mean': float(np.mean(tokens)),
            'tokens_median': float(np.median(tokens)),
            'tokens_std': float(np.std(tokens)),
            'tokens_min': int(np.min(tokens)),
            'tokens_max': int(np.max(tokens)),
        },
        'error_stats': {
            'mae_mean': float(np.mean(per_article_mae)),
            'mae_std': float(np.std(per_article_mae)),
        }
    }

    # Correlation: tokens vs MAE
    corr_tokens, p_tokens = stats.spearmanr(tokens, per_article_mae)
    results['correlation_tokens_mae'] = {
        'spearman': float(corr_tokens),
        'p_value': float(p_tokens),
        'significant': bool(p_tokens < 0.05)
    }

    # Compare truncated vs non-truncated
    if n_truncated > 0 and n_truncated < len(tokens):
        mae_truncated = per_article_mae[truncated_mask]
        mae_not_truncated = per_article_mae[~truncated_mask]

        results['truncated_vs_not'] = {
            'mae_truncated_mean': float(np.mean(mae_truncated)),
            'mae_not_truncated_mean': float(np.mean(mae_not_truncated)),
            'mae_difference': float(np.mean(mae_truncated) - np.mean(mae_not_truncated)),
        }

        # Statistical test
        stat, p_value = stats.mannwhitneyu(mae_truncated, mae_not_truncated, alternative='greater')
        results['truncated_vs_not']['mannwhitney_p'] = float(p_value)
        results['truncated_vs_not']['significant'] = bool(p_value < 0.05)

    # Per-dimension correlations
    results['per_dimension'] = {}
    for i, dim_name in enumerate(dimension_names):
        dim_errors = np.abs(predictions[:, i] - labels[:, i])
        corr, p = stats.spearmanr(tokens, dim_errors)
        results['per_dimension'][dim_name] = {
            'spearman': float(corr),
            'p_value': float(p),
            'significant': bool(p < 0.05)
        }

    return results, tokens, per_article_mae, truncated_mask


def plot_length_vs_error(
    tokens: np.ndarray,
    errors: np.ndarray,
    truncated_mask: np.ndarray,
    model_max_tokens: int,
    model_name: str,
    output_path: Path
):
    """Create scatter plot of article length vs prediction error."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Scatter plot
    ax1 = axes[0]
    ax1.scatter(tokens[~truncated_mask], errors[~truncated_mask],
                alpha=0.4, s=20, label='Within context', color='blue')
    ax1.scatter(tokens[truncated_mask], errors[truncated_mask],
                alpha=0.4, s=20, label='Truncated', color='red')
    ax1.axvline(x=model_max_tokens, color='gray', linestyle='--',
                label=f'Max tokens ({model_max_tokens})')

    # Add trend line
    z = np.polyfit(tokens, errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(tokens.min(), tokens.max(), 100)
    ax1.plot(x_line, p(x_line), 'g--', alpha=0.8, label='Trend')

    ax1.set_xlabel('Estimated Tokens')
    ax1.set_ylabel('MAE (per article)')
    ax1.set_title(f'{model_name}: Article Length vs Prediction Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Box plot comparing truncated vs not
    ax2 = axes[1]
    data_to_plot = [errors[~truncated_mask], errors[truncated_mask]]
    labels = [f'Within context\n(n={np.sum(~truncated_mask)})',
              f'Truncated\n(n={np.sum(truncated_mask)})']

    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')

    ax2.set_ylabel('MAE (per article)')
    ax2.set_title('Error Distribution: Truncated vs Non-Truncated')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add means as points
    means = [np.mean(d) for d in data_to_plot]
    ax2.scatter([1, 2], means, color='black', s=50, zorder=5, marker='D', label='Mean')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze length vs error correlation')
    parser.add_argument('--config', type=str,
                        default='research/embedding_vs_finetuning/config.yaml')
    parser.add_argument('--dataset', type=str, default='uplifting_v5')
    parser.add_argument('--model', type=str, default='e5-large-v2',
                        help='Embedding model to analyze')
    parser.add_argument('--probe', type=str, default='mlp',
                        help='Probe type (ridge, mlp, lightgbm)')
    parser.add_argument('--output-dir', type=str,
                        default='research/embedding_vs_finetuning/results')

    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))

    # Paths
    dataset_config = config['datasets'][args.dataset]
    model_config = config['embedding_models'][args.model]

    data_dir = Path(args.config).parent.parent.parent / dataset_config['path']
    embeddings_dir = Path(args.config).parent / 'embeddings'
    results_dir = Path(args.output_dir)

    model_max_tokens = model_config.get('max_tokens', 512)

    print(f"\n{'='*60}")
    print(f"Analyzing: {args.model} + {args.probe}")
    print(f"Model max tokens: {model_max_tokens}")
    print(f"{'='*60}\n")

    # Load test articles and compute lengths
    articles = load_articles(data_dir, 'test')
    lengths = compute_article_lengths(articles)
    print(f"Loaded {len(articles)} test articles")

    # Load embeddings
    embed_path = embeddings_dir / f"{args.dataset}_{args.model.replace('/', '_')}_test.npz"
    embed_data = load_embeddings(embed_path)
    print(f"Loaded embeddings: {embed_data['embeddings'].shape}")

    # Load probe and predict
    probe_path = results_dir / f"{args.dataset}_{args.model.replace('/', '_')}_{args.probe}.pkl"
    probe_data = load_probe(probe_path)

    predictions = predict_with_probe(
        args.probe,
        probe_data,
        embed_data['embeddings'],
        config['probe_methods'].get(args.probe, {})
    )

    # Analyze correlation
    results, tokens, errors, truncated_mask = analyze_length_error_correlation(
        article_ids=embed_data['article_ids'],
        lengths=lengths,
        predictions=predictions,
        labels=embed_data['labels'],
        dimension_names=embed_data['dimension_names'],
        model_max_tokens=model_max_tokens
    )

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    print(f"\nArticle Length Statistics:")
    print(f"  Mean tokens: {results['length_stats']['tokens_mean']:.0f}")
    print(f"  Median tokens: {results['length_stats']['tokens_median']:.0f}")
    print(f"  Range: {results['length_stats']['tokens_min']} - {results['length_stats']['tokens_max']}")

    print(f"\nTruncation:")
    print(f"  Model max tokens: {model_max_tokens}")
    print(f"  Articles truncated: {results['n_truncated']} / {results['n_articles']} ({results['pct_truncated']:.1f}%)")

    print(f"\nCorrelation (tokens vs MAE):")
    corr = results['correlation_tokens_mae']
    sig = "***" if corr['p_value'] < 0.001 else "**" if corr['p_value'] < 0.01 else "*" if corr['p_value'] < 0.05 else ""
    print(f"  Spearman r = {corr['spearman']:.4f} (p = {corr['p_value']:.4f}) {sig}")

    if 'truncated_vs_not' in results:
        print(f"\nTruncated vs Non-Truncated:")
        tvn = results['truncated_vs_not']
        print(f"  MAE (truncated): {tvn['mae_truncated_mean']:.4f}")
        print(f"  MAE (not truncated): {tvn['mae_not_truncated_mean']:.4f}")
        print(f"  Difference: {tvn['mae_difference']:+.4f}")
        sig = "significant" if tvn['significant'] else "not significant"
        print(f"  Mann-Whitney p = {tvn['mannwhitney_p']:.4f} ({sig})")

    print(f"\nPer-Dimension Correlations (tokens vs error):")
    for dim, dim_results in results['per_dimension'].items():
        sig = "*" if dim_results['significant'] else ""
        print(f"  {dim}: r = {dim_results['spearman']:.4f} {sig}")

    # Save results
    output_json = results_dir / f"{args.dataset}_{args.model.replace('/', '_')}_{args.probe}_length_analysis.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_json}")

    # Create plot
    plot_path = results_dir / f"{args.dataset}_{args.model.replace('/', '_')}_{args.probe}_length_vs_error.png"
    plot_length_vs_error(tokens, errors, truncated_mask, model_max_tokens, args.model, plot_path)


if __name__ == '__main__':
    main()
