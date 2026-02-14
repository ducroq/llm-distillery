"""
Analyze error distributions across models and probe types.

This script generates comprehensive error analysis including:
1. Error distribution histograms and violin plots
2. Per-dimension error breakdown
3. Truncated vs non-truncated comparison
4. Model comparison visualizations

Usage:
    python research/embedding_vs_finetuning/analyze_error_distribution.py \
        --dataset uplifting_v5 \
        --output-dir research/embedding_vs_finetuning/results/analysis
"""

import argparse
import io
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_embeddings(input_path: Path) -> Dict[str, Any]:
    """Load embeddings from disk."""
    data = np.load(input_path, allow_pickle=True)
    return {
        'embeddings': data['embeddings'],
        'article_ids': list(data['article_ids']),
        'labels': data['labels'],
        'dimension_names': list(data['dimension_names'])
    }


def load_articles(data_dir: Path, split: str = 'test') -> List[Dict[str, Any]]:
    """Load articles from JSONL file."""
    file_path = data_dir / f'{split}.jsonl'
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def load_probe(probe_path: Path, device: str = 'cpu') -> Dict[str, Any]:
    """Load trained probe from disk.

    Handles CUDA tensors saved in probes by mapping them to the specified device.
    """
    # Custom unpickler that maps CUDA tensors to CPU when needed
    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Handle torch.storage classes that may have been saved on CUDA
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location=device, weights_only=False)
            return super().find_class(module, name)

    with open(probe_path, 'rb') as f:
        try:
            # First try standard pickle (works if no CUDA tensors or on CUDA machine)
            probe_data = pickle.load(f)
        except RuntimeError as e:
            if 'CUDA' in str(e):
                # Rewind and use CPU unpickler
                f.seek(0)
                probe_data = CPUUnpickler(f).load()
            else:
                raise

    # If MLP probe, ensure state_dict tensors are on correct device
    if 'state_dict' in probe_data:
        state_dict = probe_data['state_dict']
        probe_data['state_dict'] = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in state_dict.items()
        }

    return probe_data


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (words * 1.3)."""
    words = len(text.split())
    return int(words * 1.3)


def predict_with_probe(
    probe_type: str,
    probe_data: Dict,
    embeddings: np.ndarray,
    config: Dict[str, Any],
    device: str = 'cuda'
) -> np.ndarray:
    """Generate predictions using a trained probe."""
    sys.path.insert(0, str(Path(__file__).parent))

    if probe_type == 'ridge':
        scaler = probe_data['scaler']
        model = probe_data['model']
        embeddings_scaled = scaler.transform(embeddings)
        return model.predict(embeddings_scaled)

    elif probe_type == 'mlp':
        from train_probes import MLPProbe

        scaler = probe_data['scaler']
        model_config = probe_data['model_config']

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


def compute_errors(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute various error metrics."""
    absolute_errors = np.abs(predictions - labels)
    squared_errors = (predictions - labels) ** 2

    return {
        'per_article_mae': np.mean(absolute_errors, axis=1),  # (n_articles,)
        'per_dimension_mae': np.mean(absolute_errors, axis=0),  # (n_dims,)
        'per_article_per_dim': absolute_errors,  # (n_articles, n_dims)
        'overall_mae': np.mean(absolute_errors),
        'overall_rmse': np.sqrt(np.mean(squared_errors)),
        'predictions': predictions,
        'labels': labels
    }


def plot_error_histogram(
    errors_by_model: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Error Distribution by Model"
):
    """Plot error distribution histograms for multiple models."""
    n_models = len(errors_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)

    if n_models == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for ax, (model_name, errors), color in zip(axes, errors_by_model.items(), colors):
        ax.hist(errors, bins=50, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
        ax.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
        ax.axvline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.3f}')
        ax.set_xlabel('MAE per Article')
        ax.set_title(model_name.replace('_', ' '))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Count')
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved histogram: {output_path}")


def plot_error_violin(
    errors_by_model: Dict[str, np.ndarray],
    output_path: Path,
    baseline_mae: Optional[float] = None,
    title: str = "Error Distribution Comparison"
):
    """Plot violin plots comparing error distributions."""
    fig, ax = plt.subplots(figsize=(max(8, len(errors_by_model) * 1.5), 6))

    data = list(errors_by_model.values())
    labels = [name.replace('_', '\n') for name in errors_by_model.keys()]

    parts = ax.violinplot(data, positions=range(len(data)), showmeans=True, showmedians=True)

    # Customize violin plots
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    parts['cmeans'].set_color('red')
    parts['cmedians'].set_color('green')

    # Add baseline line if provided
    if baseline_mae:
        ax.axhline(baseline_mae, color='orange', linestyle='--', linewidth=2,
                   label=f'Fine-tuned Baseline: {baseline_mae:.3f}')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('MAE per Article')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved violin plot: {output_path}")


def plot_per_dimension_errors(
    errors_by_model: Dict[str, np.ndarray],
    dimension_names: List[str],
    output_path: Path,
    title: str = "Per-Dimension MAE Comparison"
):
    """Plot per-dimension errors as grouped bar chart."""
    n_dims = len(dimension_names)
    n_models = len(errors_by_model)

    fig, ax = plt.subplots(figsize=(max(10, n_dims * 1.5), 6))

    x = np.arange(n_dims)
    width = 0.8 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for i, (model_name, errors) in enumerate(errors_by_model.items()):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, errors, width, label=model_name, color=colors[i], alpha=0.8)

    ax.set_xlabel('Dimension')
    ax.set_ylabel('MAE')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(dimension_names, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved per-dimension plot: {output_path}")


def plot_truncation_comparison(
    errors_truncated: np.ndarray,
    errors_not_truncated: np.ndarray,
    model_name: str,
    max_tokens: int,
    output_path: Path
):
    """Compare errors for truncated vs non-truncated articles."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Violin plot
    ax1 = axes[0]
    parts = ax1.violinplot([errors_not_truncated, errors_truncated],
                           positions=[0, 1], showmeans=True, showmedians=True)

    colors = ['lightgreen', 'lightcoral']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([
        f'Not Truncated\n(n={len(errors_not_truncated)})',
        f'Truncated (>{max_tokens} tokens)\n(n={len(errors_truncated)})'
    ])
    ax1.set_ylabel('MAE per Article')
    ax1.set_title(f'{model_name}: Truncation Effect')
    ax1.grid(True, alpha=0.3, axis='y')

    # Histogram comparison
    ax2 = axes[1]
    bins = np.linspace(0, max(errors_truncated.max(), errors_not_truncated.max()), 30)
    ax2.hist(errors_not_truncated, bins=bins, alpha=0.6, label='Not Truncated', color='green')
    ax2.hist(errors_truncated, bins=bins, alpha=0.6, label='Truncated', color='red')
    ax2.axvline(np.mean(errors_not_truncated), color='darkgreen', linestyle='--')
    ax2.axvline(np.mean(errors_truncated), color='darkred', linestyle='--')
    ax2.set_xlabel('MAE per Article')
    ax2.set_ylabel('Count')
    ax2.set_title('Error Distribution Overlay')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved truncation comparison: {output_path}")


def plot_error_vs_score(
    predictions: np.ndarray,
    labels: np.ndarray,
    dimension_names: List[str],
    output_path: Path,
    model_name: str
):
    """Plot prediction vs actual for each dimension."""
    n_dims = len(dimension_names)
    n_cols = min(3, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_dims == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]

    for idx, dim_name in enumerate(dimension_names):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]

        pred = predictions[:, idx]
        actual = labels[:, idx]

        ax.scatter(actual, pred, alpha=0.3, s=10)
        ax.plot([0, 10], [0, 10], 'r--', label='Perfect')

        # Correlation
        corr, _ = stats.spearmanr(actual, pred)
        mae = np.mean(np.abs(pred - actual))

        ax.set_xlabel('Actual Score')
        ax.set_ylabel('Predicted Score')
        ax.set_title(f'{dim_name}\nMAE={mae:.2f}, r={corr:.2f}')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_dims, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].set_visible(False)

    fig.suptitle(f'{model_name}: Prediction vs Actual', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved prediction scatter: {output_path}")


def generate_error_summary(
    results: Dict[str, Dict[str, Any]],
    baseline_mae: Optional[float],
    output_path: Path
):
    """Generate JSON summary of error analysis."""
    summary = {
        'baseline_mae': baseline_mae,
        'models': {}
    }

    for model_name, model_results in results.items():
        summary['models'][model_name] = {
            'overall_mae': float(model_results['overall_mae']),
            'overall_rmse': float(model_results['overall_rmse']),
            'mae_std': float(np.std(model_results['per_article_mae'])),
            'mae_median': float(np.median(model_results['per_article_mae'])),
            'mae_q25': float(np.percentile(model_results['per_article_mae'], 25)),
            'mae_q75': float(np.percentile(model_results['per_article_mae'], 75)),
            'per_dimension_mae': {
                dim: float(mae) for dim, mae in zip(
                    model_results.get('dimension_names', [f'dim_{i}' for i in range(len(model_results['per_dimension_mae']))]),
                    model_results['per_dimension_mae']
                )
            }
        }

        if baseline_mae:
            summary['models'][model_name]['gap_to_baseline'] = float(model_results['overall_mae'] - baseline_mae)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze error distributions across models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, default='research/embedding_vs_finetuning/config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., uplifting_v5)')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Embedding models to analyze (default: all)')
    parser.add_argument('--probe', type=str, default='mlp',
                       choices=['ridge', 'mlp', 'lightgbm'],
                       help='Probe type to analyze')
    parser.add_argument('--embeddings-dir', type=str, default='research/embedding_vs_finetuning/embeddings',
                       help='Directory with embeddings')
    parser.add_argument('--results-dir', type=str, default='research/embedding_vs_finetuning/results',
                       help='Directory with trained probes')
    parser.add_argument('--output-dir', type=str, default='research/embedding_vs_finetuning/results/analysis',
                       help='Output directory for analysis')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Auto-detect device - fall back to CPU if CUDA not available
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Load config
    config_path = Path(args.config)
    config = load_config(config_path)

    dataset_config = config['datasets'][args.dataset]
    baseline_mae = dataset_config.get('baseline_mae')

    # Determine models
    if args.models:
        models = args.models
    else:
        models = list(config['embedding_models'].keys())

    embeddings_dir = Path(args.embeddings_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load article lengths for truncation analysis
    data_dir = config_path.parent.parent.parent / dataset_config['path']
    articles = load_articles(data_dir, 'test')
    article_lengths = {}
    for article in articles:
        text = f"{article.get('title', '')}\n\n{article.get('content', '')}"
        article_lengths[article['id']] = estimate_tokens(text)

    # Collect results for all models
    all_results = {}
    errors_by_model = {}
    per_dim_errors_by_model = {}

    for model_name in models:
        logger.info(f"\n{'='*60}\nAnalyzing: {model_name}\n{'='*60}")

        try:
            # Load test embeddings
            test_path = embeddings_dir / f"{args.dataset}_{model_name.replace('/', '_')}_test.npz"
            if not test_path.exists():
                logger.warning(f"Embeddings not found: {test_path}")
                continue

            test_data = load_embeddings(test_path)

            # Load probe
            probe_path = results_dir / f"{args.dataset}_{model_name.replace('/', '_')}_{args.probe}.pkl"
            if not probe_path.exists():
                logger.warning(f"Probe not found: {probe_path}")
                continue

            probe_data = load_probe(probe_path, device=args.device)

            # Predict
            probe_config = config['probe_methods'].get(args.probe, {})
            predictions = predict_with_probe(
                args.probe,
                probe_data,
                test_data['embeddings'],
                probe_config,
                args.device
            )

            # Compute errors
            errors = compute_errors(predictions, test_data['labels'])
            errors['dimension_names'] = test_data['dimension_names']
            errors['article_ids'] = test_data['article_ids']

            all_results[model_name] = errors
            errors_by_model[model_name] = errors['per_article_mae']
            per_dim_errors_by_model[model_name] = errors['per_dimension_mae']

            # Model-specific plots
            model_config = config['embedding_models'].get(model_name, {})
            max_tokens = model_config.get('max_tokens', 512)

            # Truncation analysis
            truncated_mask = np.array([
                article_lengths.get(aid, 0) > max_tokens
                for aid in test_data['article_ids']
            ])

            if truncated_mask.any() and (~truncated_mask).any():
                plot_truncation_comparison(
                    errors['per_article_mae'][truncated_mask],
                    errors['per_article_mae'][~truncated_mask],
                    model_name,
                    max_tokens,
                    output_dir / f"{args.dataset}_{model_name.replace('/', '_')}_truncation.png"
                )

            # Prediction vs actual scatter
            plot_error_vs_score(
                predictions,
                test_data['labels'],
                test_data['dimension_names'],
                output_dir / f"{args.dataset}_{model_name.replace('/', '_')}_scatter.png",
                model_name
            )

        except Exception as e:
            logger.error(f"Failed to analyze {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-model comparison plots
    if len(errors_by_model) > 1:
        plot_error_histogram(
            errors_by_model,
            output_dir / f"{args.dataset}_error_histograms.png",
            f"Error Distribution by Model ({args.probe})"
        )

        plot_error_violin(
            errors_by_model,
            output_dir / f"{args.dataset}_error_violin.png",
            baseline_mae,
            f"Error Distribution Comparison ({args.probe})"
        )

        # Use dimension names from first model
        first_model = list(all_results.keys())[0]
        dimension_names = all_results[first_model]['dimension_names']

        plot_per_dimension_errors(
            per_dim_errors_by_model,
            dimension_names,
            output_dir / f"{args.dataset}_per_dimension.png",
            f"Per-Dimension MAE Comparison ({args.probe})"
        )

    # Generate summary JSON
    generate_error_summary(
        all_results,
        baseline_mae,
        output_dir / f"{args.dataset}_error_summary.json"
    )

    logger.info(f"\n{'='*60}\nAnalysis Complete\n{'='*60}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
