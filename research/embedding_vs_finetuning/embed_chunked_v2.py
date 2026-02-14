#!/usr/bin/env python3
"""
Extended chunking experiments with more aggregation strategies and overlap variations.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks by words."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += step

    return chunks


def aggregate_embeddings(embeddings: np.ndarray, strategy: str) -> np.ndarray:
    """
    Aggregate chunk embeddings with various strategies.

    New strategies:
    - weighted_position: Weight by position (U-shaped: start/end matter more)
    - percentile_25_75: Concatenate 25th and 75th percentile
    - mean_std: Concatenate mean and std deviation
    - top2_norm: Average top 2 chunks by L2 norm
    - all_concat: Concatenate mean + max + first + last (4x dims)
    """
    if len(embeddings) == 1:
        emb = embeddings[0]
        if strategy in ['mean_max', 'first_last', 'percentile_25_75', 'mean_std', 'min_max']:
            return np.concatenate([emb, emb])
        elif strategy == 'first_middle_last':
            return np.concatenate([emb, emb, emb])
        elif strategy == 'all_concat':
            return np.concatenate([emb, emb, emb, emb])
        return emb

    if strategy == 'mean':
        return np.mean(embeddings, axis=0)

    elif strategy == 'max':
        return np.max(embeddings, axis=0)

    elif strategy == 'mean_max':
        return np.concatenate([np.mean(embeddings, axis=0), np.max(embeddings, axis=0)])

    elif strategy == 'first_last':
        return np.concatenate([embeddings[0], embeddings[-1]])

    elif strategy == 'first_middle_last':
        if len(embeddings) == 2:
            # For 2 chunks, duplicate middle
            return np.concatenate([embeddings[0], embeddings[0], embeddings[-1]])
        mid_idx = len(embeddings) // 2
        return np.concatenate([embeddings[0], embeddings[mid_idx], embeddings[-1]])

    elif strategy == 'weighted_position':
        # U-shaped weights: start and end matter more
        n = len(embeddings)
        positions = np.arange(n)
        # Parabola with minimum at center
        weights = (positions - n/2) ** 2
        weights = weights / weights.sum()
        return np.average(embeddings, axis=0, weights=weights)

    elif strategy == 'percentile_25_75':
        p25 = np.percentile(embeddings, 25, axis=0)
        p75 = np.percentile(embeddings, 75, axis=0)
        return np.concatenate([p25, p75])

    elif strategy == 'mean_std':
        return np.concatenate([np.mean(embeddings, axis=0), np.std(embeddings, axis=0)])

    elif strategy == 'top2_norm':
        # Select top 2 chunks by L2 norm (most "activated")
        if len(embeddings) <= 2:
            return np.mean(embeddings, axis=0)
        norms = np.linalg.norm(embeddings, axis=1)
        top_indices = np.argsort(norms)[-2:]
        return np.mean(embeddings[top_indices], axis=0)

    elif strategy == 'min_max':
        # Concatenate min and max pooling
        return np.concatenate([np.min(embeddings, axis=0), np.max(embeddings, axis=0)])

    elif strategy == 'all_concat':
        # Concatenate all: mean + max + first + last
        return np.concatenate([
            np.mean(embeddings, axis=0),
            np.max(embeddings, axis=0),
            embeddings[0],
            embeddings[-1]
        ])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def load_articles(dataset_path: Path, split: str) -> Tuple[List[str], np.ndarray]:
    """Load articles and labels."""
    file_path = dataset_path / f"{split}.jsonl"
    texts, labels = [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            title = item.get('title', '')
            content = item.get('content', item.get('text', ''))
            texts.append(f"{title}\n\n{content}" if title else content)
            labels.append(item['labels'])

    return texts, np.array(labels)


def run_experiment(
    model: SentenceTransformer,
    texts: List[str],
    chunk_size: int,
    overlap: int,
    strategies: List[str],
    batch_size: int = 64,
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """Run chunking experiment with given parameters."""

    # Chunk all texts
    all_chunks = []
    chunk_indices = []

    for text in texts:
        chunks = chunk_text(text, chunk_size, overlap)
        start_idx = len(all_chunks)
        all_chunks.extend(chunks)
        chunk_indices.append((start_idx, len(all_chunks)))

    avg_chunks = len(all_chunks) / len(texts)
    logger.info(f"  Chunks: {len(all_chunks)} total, {avg_chunks:.1f} avg/article")

    # Embed all chunks
    chunk_embeddings = model.encode(
        all_chunks, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, device=device
    )

    # Aggregate with each strategy
    results = {}
    for strategy in strategies:
        aggregated = []
        for start_idx, end_idx in chunk_indices:
            text_chunks = chunk_embeddings[start_idx:end_idx]
            agg_emb = aggregate_embeddings(text_chunks, strategy)
            aggregated.append(agg_emb)
        results[strategy] = np.array(aggregated)

    return results


class MLPProbe(nn.Module):
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


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, device='cuda'):
    """Train MLP and return test MAE."""
    from sklearn.metrics import mean_absolute_error

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = MLPProbe(input_dim, output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)

    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy()
            val_mae = mean_absolute_error(y_val, val_pred)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        test_pred = model(X_test_t).cpu().numpy()

    return mean_absolute_error(y_test, test_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='uplifting_v5')
    parser.add_argument('--model', default='multilingual-mpnet-base-v2')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Model mapping
    model_map = {
        'multilingual-MiniLM-L12-v2': 'paraphrase-multilingual-MiniLM-L12-v2',
        'multilingual-mpnet-base-v2': 'paraphrase-multilingual-mpnet-base-v2',
    }

    dataset_path = Path(f"datasets/training/{args.dataset}")

    # Load data
    logger.info("Loading data...")
    train_texts, y_train = load_articles(dataset_path, 'train')
    val_texts, y_val = load_articles(dataset_path, 'val')
    test_texts, y_test = load_articles(dataset_path, 'test')

    # Load model
    logger.info(f"Loading model: {args.model}")
    model = SentenceTransformer(model_map.get(args.model, args.model), device=args.device)

    # Extended strategies
    strategies = [
        'mean', 'max', 'mean_max', 'first_last',
        'weighted_position', 'percentile_25_75', 'mean_std',
        'top2_norm', 'min_max', 'first_middle_last'
    ]

    # Test different chunk sizes and overlaps
    experiments = [
        # (chunk_size, overlap, description)
        (200, 100, "200w/50%"),   # Original
        (200, 150, "200w/75%"),   # Higher overlap
        (100, 50, "100w/50%"),    # Smaller chunks
        (100, 75, "100w/75%"),    # Small + high overlap
        (300, 150, "300w/50%"),   # Larger chunks
    ]

    results = {}

    for chunk_size, overlap, desc in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {desc} (chunk={chunk_size}, overlap={overlap})")
        logger.info(f"{'='*60}")

        # Generate embeddings for each split
        logger.info("Generating train embeddings...")
        train_embs = run_experiment(model, train_texts, chunk_size, overlap, strategies, device=args.device)

        logger.info("Generating val embeddings...")
        val_embs = run_experiment(model, val_texts, chunk_size, overlap, strategies, device=args.device)

        logger.info("Generating test embeddings...")
        test_embs = run_experiment(model, test_texts, chunk_size, overlap, strategies, device=args.device)

        # Evaluate each strategy
        results[desc] = {}
        logger.info("\nEvaluating strategies:")

        for strategy in strategies:
            try:
                mae = train_and_evaluate(
                    train_embs[strategy], y_train,
                    val_embs[strategy], y_val,
                    test_embs[strategy], y_test,
                    device=args.device
                )
                results[desc][strategy] = mae
                gap = mae - 0.68
                logger.info(f"  {strategy:20s}: MAE={mae:.4f} (+{gap:.4f})")
            except Exception as e:
                logger.error(f"  {strategy}: FAILED - {e}")

    # Save results
    output_path = Path(f"research/embedding_vs_finetuning/results/chunked_extended_{args.model}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY: Best MAE per chunk configuration")
    logger.info("="*80)

    for desc, strat_results in results.items():
        if strat_results:
            best_strat = min(strat_results, key=strat_results.get)
            best_mae = strat_results[best_strat]
            logger.info(f"{desc:15s}: {best_mae:.4f} ({best_strat})")


if __name__ == '__main__':
    main()
