"""
Train linear and non-linear probes on frozen embeddings.

This script loads cached embeddings and trains various probe architectures
(Ridge regression, MLP, LightGBM) to predict dimensional scores.

Usage:
    python research/embedding_vs_finetuning/train_probes.py \
        --dataset uplifting_v5 \
        --models all-MiniLM-L6-v2 bge-large-en-v1.5 \
        --probes ridge mlp lightgbm
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

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


def train_ridge_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    alphas: List[float] = [0.01, 0.1, 1.0, 10.0, 100.0]
) -> Tuple[Any, StandardScaler, Dict[str, float]]:
    """Train Ridge regression probe with cross-validation."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Standardize embeddings
    scaler = StandardScaler()
    train_embeddings_scaled = scaler.fit_transform(train_embeddings)
    val_embeddings_scaled = scaler.transform(val_embeddings)

    # Train Ridge with cross-validation for alpha selection
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(train_embeddings_scaled, train_labels)

    # Evaluate on validation set
    val_predictions = model.predict(val_embeddings_scaled)
    val_mae = mean_absolute_error(val_labels, val_predictions)
    val_rmse = np.sqrt(mean_squared_error(val_labels, val_predictions))

    metrics = {
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'best_alpha': model.alpha_
    }

    logger.info(f"Ridge - Best alpha: {model.alpha_}, Val MAE: {val_mae:.4f}")

    return model, scaler, metrics


def train_mlp_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    config: Dict[str, Any],
    device: str = 'cuda'
) -> Tuple[MLPProbe, StandardScaler, Dict[str, float]]:
    """Train MLP probe with early stopping."""
    from torch.utils.data import DataLoader, TensorDataset

    # Standardize embeddings
    scaler = StandardScaler()
    train_embeddings_scaled = scaler.fit_transform(train_embeddings)
    val_embeddings_scaled = scaler.transform(val_embeddings)

    # Convert to tensors
    train_X = torch.FloatTensor(train_embeddings_scaled)
    train_y = torch.FloatTensor(train_labels)
    val_X = torch.FloatTensor(val_embeddings_scaled).to(device)
    val_y = torch.FloatTensor(val_labels).to(device)

    # Create dataloader
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Create model
    input_dim = train_embeddings.shape[1]
    output_dim = train_labels.shape[1]
    model = MLPProbe(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=config.get('hidden_sizes', [256, 128]),
        dropout=config.get('dropout', 0.2)
    ).to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    criterion = nn.L1Loss()  # MAE loss

    epochs = config.get('epochs', 100)
    patience = config.get('patience', 10)
    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_X)
            val_mae = torch.mean(torch.abs(val_predictions - val_y)).item()
            val_rmse = torch.sqrt(torch.mean((val_predictions - val_y) ** 2)).item()

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val MAE={val_mae:.4f}")

    # Load best state
    model.load_state_dict(best_state)

    metrics = {
        'val_mae': best_val_mae,
        'val_rmse': val_rmse,
        'epochs_trained': epoch + 1
    }

    logger.info(f"MLP - Best Val MAE: {best_val_mae:.4f}")

    return model, scaler, metrics


def train_lightgbm_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[List[Any], None, Dict[str, float]]:
    """Train LightGBM probe (one model per dimension)."""
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    num_dimensions = train_labels.shape[1]
    models = []
    per_dim_mae = []

    for dim_idx in range(num_dimensions):
        train_y = train_labels[:, dim_idx]
        val_y = val_labels[:, dim_idx]

        train_data = lgb.Dataset(train_embeddings, label=train_y)
        val_data = lgb.Dataset(val_embeddings, label=val_y, reference=train_data)

        params = {
            'objective': 'regression',
            'metric': 'mae',
            'n_estimators': config.get('n_estimators', 500),
            'learning_rate': config.get('learning_rate', 0.05),
            'max_depth': config.get('max_depth', 6),
            'num_leaves': config.get('num_leaves', 31),
            'verbose': -1
        }

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(config.get('early_stopping_rounds', 50)),
                lgb.log_evaluation(period=0)
            ]
        )
        models.append(model)

        val_pred = model.predict(val_embeddings)
        dim_mae = mean_absolute_error(val_y, val_pred)
        per_dim_mae.append(dim_mae)

    # Overall metrics
    all_predictions = np.column_stack([m.predict(val_embeddings) for m in models])
    val_mae = mean_absolute_error(val_labels, all_predictions)
    val_rmse = np.sqrt(mean_squared_error(val_labels, all_predictions))

    metrics = {
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'per_dim_mae': per_dim_mae
    }

    logger.info(f"LightGBM - Val MAE: {val_mae:.4f}")

    return models, None, metrics


def save_probe(
    probe_type: str,
    model: Any,
    scaler: Optional[StandardScaler],
    metrics: Dict[str, Any],
    output_path: Path
):
    """Save trained probe to disk."""
    import pickle

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'probe_type': probe_type,
        'metrics': metrics
    }

    if probe_type == 'ridge':
        data['model'] = model
        data['scaler'] = scaler
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

    elif probe_type == 'mlp':
        data['scaler'] = scaler
        data['state_dict'] = model.state_dict()
        data['model_config'] = {
            'input_dim': model.network[0].in_features,
            'output_dim': model.network[-1].out_features
        }
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

    elif probe_type == 'lightgbm':
        data['models'] = model  # List of models, one per dimension
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

    logger.info(f"Saved {probe_type} probe to {output_path}")


def train_all_probes(
    config: Dict[str, Any],
    dataset_name: str,
    model_name: str,
    probe_types: List[str],
    embeddings_dir: Path,
    output_dir: Path,
    device: str = 'cuda'
) -> Dict[str, Dict[str, Any]]:
    """Train all specified probes for a given embedding model."""

    # Load embeddings
    train_path = embeddings_dir / f"{dataset_name}_{model_name.replace('/', '_')}_train.npz"
    val_path = embeddings_dir / f"{dataset_name}_{model_name.replace('/', '_')}_val.npz"

    train_data = load_embeddings(train_path)
    val_data = load_embeddings(val_path)

    train_embeddings = train_data['embeddings']
    train_labels = train_data['labels']
    val_embeddings = val_data['embeddings']
    val_labels = val_data['labels']

    logger.info(f"Train: {train_embeddings.shape}, Val: {val_embeddings.shape}")

    results = {}

    for probe_type in probe_types:
        logger.info(f"\nTraining {probe_type} probe...")
        start_time = time.time()

        try:
            if probe_type == 'ridge':
                probe_config = config['probe_methods']['ridge']
                model, scaler, metrics = train_ridge_probe(
                    train_embeddings, train_labels,
                    val_embeddings, val_labels,
                    alphas=probe_config.get('alpha', [0.01, 0.1, 1.0, 10.0, 100.0])
                )

            elif probe_type == 'mlp':
                probe_config = config['probe_methods']['mlp']
                model, scaler, metrics = train_mlp_probe(
                    train_embeddings, train_labels,
                    val_embeddings, val_labels,
                    config=probe_config,
                    device=device
                )

            elif probe_type == 'lightgbm':
                probe_config = config['probe_methods']['lightgbm']
                model, scaler, metrics = train_lightgbm_probe(
                    train_embeddings, train_labels,
                    val_embeddings, val_labels,
                    config=probe_config
                )

            else:
                logger.warning(f"Unknown probe type: {probe_type}")
                continue

            elapsed = time.time() - start_time
            metrics['training_time_seconds'] = elapsed

            # Save probe
            probe_path = output_dir / f"{dataset_name}_{model_name.replace('/', '_')}_{probe_type}.pkl"
            save_probe(probe_type, model, scaler, metrics, probe_path)

            results[probe_type] = {
                'metrics': metrics,
                'path': str(probe_path)
            }

        except Exception as e:
            logger.error(f"Failed to train {probe_type}: {e}")
            results[probe_type] = {'error': str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train probes on frozen embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, default='research/embedding_vs_finetuning/config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., uplifting_v5)')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Embedding models to use (default: all)')
    parser.add_argument('--probes', type=str, nargs='+', default=['ridge', 'mlp', 'lightgbm'],
                       help='Probe types to train')
    parser.add_argument('--embeddings-dir', type=str, default='research/embedding_vs_finetuning/embeddings',
                       help='Directory with cached embeddings')
    parser.add_argument('--output-dir', type=str, default='research/embedding_vs_finetuning/results',
                       help='Output directory for trained probes')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))

    # Determine models to use
    if args.models:
        models = args.models
    else:
        models = list(config['embedding_models'].keys())

    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)

    # Train probes for each embedding model
    all_results = {}
    for model_name in models:
        logger.info(f"\n{'='*60}\nTraining probes for: {model_name}\n{'='*60}")

        try:
            results = train_all_probes(
                config=config,
                dataset_name=args.dataset,
                model_name=model_name,
                probe_types=args.probes,
                embeddings_dir=embeddings_dir,
                output_dir=output_dir,
                device=args.device
            )
            all_results[model_name] = results

        except Exception as e:
            logger.error(f"Failed to process {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}

    # Save summary
    summary_path = output_dir / f"{args.dataset}_training_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nTraining summary saved to {summary_path}")

    # Print summary table
    logger.info(f"\n{'='*60}\nTraining Results Summary\n{'='*60}")
    for model_name, results in all_results.items():
        if 'error' in results:
            logger.info(f"{model_name}: ERROR - {results['error']}")
            continue

        logger.info(f"\n{model_name}:")
        for probe_type, probe_results in results.items():
            if 'error' in probe_results:
                logger.info(f"  {probe_type}: ERROR")
            else:
                mae = probe_results['metrics'].get('val_mae', 'N/A')
                logger.info(f"  {probe_type}: Val MAE = {mae:.4f}" if isinstance(mae, float) else f"  {probe_type}: Val MAE = {mae}")


if __name__ == '__main__':
    main()
