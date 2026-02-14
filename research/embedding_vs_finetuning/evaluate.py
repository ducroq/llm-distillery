"""
Evaluate trained probes on test set and compare to fine-tuned baseline.

This script loads trained probes and evaluates them on the test set,
computing MAE, RMSE, and per-dimension metrics.

Usage:
    python research/embedding_vs_finetuning/evaluate.py \
        --dataset uplifting_v5 \
        --models all-MiniLM-L6-v2 bge-large-en-v1.5 \
        --probes ridge mlp lightgbm
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        'article_ids': data['article_ids'],
        'labels': data['labels'],
        'dimension_names': list(data['dimension_names'])
    }


def load_probe(probe_path: Path) -> Dict[str, Any]:
    """Load trained probe from disk."""
    with open(probe_path, 'rb') as f:
        return pickle.load(f)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    dimension_names: List[str]
) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    metrics = {}

    # Overall metrics
    mae = np.mean(np.abs(predictions - labels))
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))

    # Spearman correlation (average across dimensions)
    spearman_corrs = []
    for i in range(predictions.shape[1]):
        corr, _ = stats.spearmanr(predictions[:, i], labels[:, i])
        spearman_corrs.append(corr)
    avg_spearman = np.mean(spearman_corrs)

    metrics['overall_mae'] = float(mae)
    metrics['overall_rmse'] = float(rmse)
    metrics['overall_spearman'] = float(avg_spearman)

    # Per-dimension metrics
    per_dimension = {}
    for i, dim_name in enumerate(dimension_names):
        dim_pred = predictions[:, i]
        dim_labels = labels[:, i]

        dim_mae = np.mean(np.abs(dim_pred - dim_labels))
        dim_rmse = np.sqrt(np.mean((dim_pred - dim_labels) ** 2))
        dim_spearman, _ = stats.spearmanr(dim_pred, dim_labels)

        per_dimension[dim_name] = {
            'mae': float(dim_mae),
            'rmse': float(dim_rmse),
            'spearman': float(dim_spearman)
        }

    metrics['per_dimension'] = per_dimension

    return metrics


def predict_ridge(probe_data: Dict, embeddings: np.ndarray) -> np.ndarray:
    """Generate predictions using Ridge probe."""
    scaler = probe_data['scaler']
    model = probe_data['model']

    embeddings_scaled = scaler.transform(embeddings)
    predictions = model.predict(embeddings_scaled)

    return predictions


def predict_mlp(
    probe_data: Dict,
    embeddings: np.ndarray,
    config: Dict[str, Any],
    device: str = 'cuda'
) -> np.ndarray:
    """Generate predictions using MLP probe."""
    from train_probes import MLPProbe

    scaler = probe_data['scaler']
    model_config = probe_data['model_config']

    # Recreate model
    model = MLPProbe(
        input_dim=model_config['input_dim'],
        output_dim=model_config['output_dim'],
        hidden_sizes=config.get('hidden_sizes', [256, 128]),
        dropout=0.0  # No dropout at inference
    ).to(device)

    model.load_state_dict(probe_data['state_dict'])
    model.eval()

    # Scale embeddings
    embeddings_scaled = scaler.transform(embeddings)
    embeddings_tensor = torch.FloatTensor(embeddings_scaled).to(device)

    with torch.no_grad():
        predictions = model(embeddings_tensor).cpu().numpy()

    return predictions


def predict_lightgbm(probe_data: Dict, embeddings: np.ndarray) -> np.ndarray:
    """Generate predictions using LightGBM probe."""
    models = probe_data['models']

    predictions = np.column_stack([
        model.predict(embeddings) for model in models
    ])

    return predictions


def evaluate_probe(
    probe_type: str,
    probe_path: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    dimension_names: List[str],
    config: Dict[str, Any],
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Evaluate a single probe."""
    probe_data = load_probe(probe_path)

    start_time = time.time()

    if probe_type == 'ridge':
        predictions = predict_ridge(probe_data, embeddings)
    elif probe_type == 'mlp':
        predictions = predict_mlp(probe_data, embeddings, config['probe_methods']['mlp'], device)
    elif probe_type == 'lightgbm':
        predictions = predict_lightgbm(probe_data, embeddings)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

    inference_time = time.time() - start_time

    metrics = compute_metrics(predictions, labels, dimension_names)
    metrics['inference_time_seconds'] = inference_time
    metrics['inference_time_per_article_ms'] = (inference_time / len(labels)) * 1000

    return metrics


def evaluate_all_probes(
    config: Dict[str, Any],
    dataset_name: str,
    model_name: str,
    probe_types: List[str],
    embeddings_dir: Path,
    results_dir: Path,
    device: str = 'cuda'
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all probes for a given embedding model."""

    # Load test embeddings
    test_path = embeddings_dir / f"{dataset_name}_{model_name.replace('/', '_')}_test.npz"
    test_data = load_embeddings(test_path)

    embeddings = test_data['embeddings']
    labels = test_data['labels']
    dimension_names = test_data['dimension_names']

    logger.info(f"Test set: {embeddings.shape[0]} articles")

    results = {}

    for probe_type in probe_types:
        probe_path = results_dir / f"{dataset_name}_{model_name.replace('/', '_')}_{probe_type}.pkl"

        if not probe_path.exists():
            logger.warning(f"Probe not found: {probe_path}")
            results[probe_type] = {'error': 'Probe file not found'}
            continue

        try:
            logger.info(f"Evaluating {probe_type}...")
            metrics = evaluate_probe(
                probe_type=probe_type,
                probe_path=probe_path,
                embeddings=embeddings,
                labels=labels,
                dimension_names=dimension_names,
                config=config,
                device=device
            )
            results[probe_type] = metrics

            logger.info(f"  MAE: {metrics['overall_mae']:.4f}, RMSE: {metrics['overall_rmse']:.4f}, "
                       f"Spearman: {metrics['overall_spearman']:.4f}")

        except Exception as e:
            logger.error(f"Failed to evaluate {probe_type}: {e}")
            results[probe_type] = {'error': str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained probes on test set',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, default='research/embedding_vs_finetuning/config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., uplifting_v5)')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Embedding models to evaluate (default: all)')
    parser.add_argument('--probes', type=str, nargs='+', default=['ridge', 'mlp', 'lightgbm'],
                       help='Probe types to evaluate')
    parser.add_argument('--embeddings-dir', type=str, default='research/embedding_vs_finetuning/embeddings',
                       help='Directory with cached embeddings')
    parser.add_argument('--results-dir', type=str, default='research/embedding_vs_finetuning/results',
                       help='Directory with trained probes')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))

    # Get baseline MAE for comparison
    baseline_mae = config['datasets'][args.dataset].get('baseline_mae', None)

    # Determine models to use
    if args.models:
        models = args.models
    else:
        models = list(config['embedding_models'].keys())

    embeddings_dir = Path(args.embeddings_dir)
    results_dir = Path(args.results_dir)

    # Evaluate all models
    all_results = {
        'baseline_mae': baseline_mae,
        'dataset': args.dataset,
        'models': {}
    }

    for model_name in models:
        logger.info(f"\n{'='*60}\nEvaluating: {model_name}\n{'='*60}")

        try:
            results = evaluate_all_probes(
                config=config,
                dataset_name=args.dataset,
                model_name=model_name,
                probe_types=args.probes,
                embeddings_dir=embeddings_dir,
                results_dir=results_dir,
                device=args.device
            )

            # Add embedding metadata
            embed_config = config['embedding_models'].get(model_name, {})
            results['embedding_dim'] = embed_config.get('dimensions', 'unknown')

            all_results['models'][model_name] = results

        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            all_results['models'][model_name] = {'error': str(e)}

    # Save evaluation results
    output_path = results_dir / f"{args.dataset}_evaluation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nEvaluation results saved to {output_path}")

    # Print comparison table
    print_comparison_table(all_results, baseline_mae)


def print_comparison_table(results: Dict[str, Any], baseline_mae: Optional[float]):
    """Print a comparison table of all results."""
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION RESULTS COMPARISON")
    logger.info(f"{'='*80}")

    if baseline_mae:
        logger.info(f"\nFine-tuned Baseline MAE: {baseline_mae:.4f}")

    logger.info(f"\n{'Model':<30} {'Probe':<12} {'MAE':<8} {'RMSE':<8} {'Spearman':<10} {'vs Baseline'}")
    logger.info("-" * 80)

    for model_name, model_results in results.get('models', {}).items():
        if 'error' in model_results:
            logger.info(f"{model_name:<30} ERROR: {model_results['error']}")
            continue

        for probe_type in ['ridge', 'mlp', 'lightgbm']:
            if probe_type not in model_results:
                continue

            probe_results = model_results[probe_type]
            if 'error' in probe_results:
                continue

            mae = probe_results.get('overall_mae', 'N/A')
            rmse = probe_results.get('overall_rmse', 'N/A')
            spearman = probe_results.get('overall_spearman', 'N/A')

            if baseline_mae and isinstance(mae, float):
                delta = mae - baseline_mae
                delta_str = f"{delta:+.4f}"
            else:
                delta_str = "N/A"

            logger.info(f"{model_name:<30} {probe_type:<12} {mae:<8.4f} {rmse:<8.4f} {spearman:<10.4f} {delta_str}")


if __name__ == '__main__':
    main()
