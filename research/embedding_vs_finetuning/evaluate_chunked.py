#!/usr/bin/env python3
"""
Evaluate chunked embeddings with different aggregation strategies.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLPProbe(nn.Module):
    """Simple MLP probe for regression."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_probe(X_train, y_train, X_val, y_val, device='cuda', max_epochs=100, patience=10):
    """Train MLP probe with early stopping."""
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = MLPProbe(input_dim, output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy()
            val_mae = mean_absolute_error(y_val, val_pred)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_mae


def evaluate_model(model, X_test, y_test, device='cuda', is_mlp=False):
    """Evaluate model on test set."""
    if is_mlp:
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(device)
            predictions = model(X_t).cpu().numpy()
    else:
        predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Average Spearman across dimensions
    spearman_scores = []
    for i in range(y_test.shape[1]):
        corr, _ = spearmanr(y_test[:, i], predictions[:, i])
        if not np.isnan(corr):
            spearman_scores.append(corr)
    avg_spearman = np.mean(spearman_scores)

    return mae, rmse, avg_spearman


def main():
    parser = argparse.ArgumentParser(description='Evaluate chunked embeddings')
    parser.add_argument('--dataset', type=str, default='uplifting_v5')
    parser.add_argument('--models', nargs='+',
                        default=['multilingual-MiniLM-L12-v2', 'multilingual-mpnet-base-v2'])
    parser.add_argument('--strategies', nargs='+',
                        default=['mean', 'max', 'mean_max', 'first_last'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--embeddings-dir', type=str,
                        default='research/embedding_vs_finetuning/embeddings')
    parser.add_argument('--output', type=str,
                        default='research/embedding_vs_finetuning/results/chunked_evaluation.json')

    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    results = {
        'baseline_mae': 0.68,
        'dataset': args.dataset,
        'models': {}
    }

    for model_name in args.models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*60}")

        results['models'][model_name] = {'strategies': {}}

        # Also load truncated baseline for comparison
        truncated_file = embeddings_dir / f"{args.dataset}_{model_name}_train.npz"
        if truncated_file.exists():
            data = np.load(truncated_file)
            X_train_trunc = data['embeddings']
            y_train = data['labels']

            data = np.load(embeddings_dir / f"{args.dataset}_{model_name}_val.npz")
            X_val_trunc = data['embeddings']
            y_val = data['labels']

            data = np.load(embeddings_dir / f"{args.dataset}_{model_name}_test.npz")
            X_test_trunc = data['embeddings']
            y_test = data['labels']

            # Train and evaluate truncated baseline
            logger.info("Training MLP on truncated embeddings (baseline)...")
            mlp, val_mae = train_mlp_probe(X_train_trunc, y_train, X_val_trunc, y_val, args.device)
            mae, rmse, spearman = evaluate_model(mlp, X_test_trunc, y_test, args.device, is_mlp=True)

            results['models'][model_name]['truncated'] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'spearman': float(spearman),
                'embedding_dim': int(X_train_trunc.shape[1])
            }
            logger.info(f"  Truncated: MAE={mae:.4f}, RMSE={rmse:.4f}, Spearman={spearman:.4f}")

        # Evaluate each chunking strategy
        for strategy in args.strategies:
            train_file = embeddings_dir / f"{args.dataset}_{model_name}_chunked_{strategy}_train.npz"

            if not train_file.exists():
                logger.warning(f"  Skipping {strategy} - file not found: {train_file}")
                continue

            logger.info(f"\nEvaluating strategy: {strategy}")

            # Load chunked embeddings
            data = np.load(train_file)
            X_train = data['embeddings']
            y_train = data['labels']

            data = np.load(embeddings_dir / f"{args.dataset}_{model_name}_chunked_{strategy}_val.npz")
            X_val = data['embeddings']
            y_val = data['labels']

            data = np.load(embeddings_dir / f"{args.dataset}_{model_name}_chunked_{strategy}_test.npz")
            X_test = data['embeddings']
            y_test = data['labels']

            logger.info(f"  Embedding dim: {X_train.shape[1]}")

            # Train MLP probe
            logger.info("  Training MLP probe...")
            mlp, val_mae = train_mlp_probe(X_train, y_train, X_val, y_val, args.device)
            mae, rmse, spearman = evaluate_model(mlp, X_test, y_test, args.device, is_mlp=True)

            results['models'][model_name]['strategies'][strategy] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'spearman': float(spearman),
                'embedding_dim': int(X_train.shape[1])
            }

            gap = mae - 0.68
            logger.info(f"  {strategy}: MAE={mae:.4f} (+{gap:.4f}), RMSE={rmse:.4f}, Spearman={spearman:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("CHUNKING EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"\nBaseline (fine-tuned): MAE = 0.68")
    logger.info("-"*80)

    for model_name, model_results in results['models'].items():
        logger.info(f"\n{model_name}:")

        if 'truncated' in model_results:
            r = model_results['truncated']
            gap = r['mae'] - 0.68
            logger.info(f"  Truncated (128 tok):  MAE={r['mae']:.4f} (+{gap:.4f})")

        for strategy, r in model_results.get('strategies', {}).items():
            gap = r['mae'] - 0.68
            logger.info(f"  Chunked ({strategy:10s}): MAE={r['mae']:.4f} (+{gap:.4f})")


if __name__ == '__main__':
    main()
