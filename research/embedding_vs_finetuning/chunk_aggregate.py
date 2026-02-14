"""
Aggregate chunk embeddings using various methods.

This script implements Top-K chunk selection and aggregation:
1. Score chunks using various methods (variance, centroid distance, etc.)
2. Select top-K most informative chunks
3. Aggregate selected chunks (mean, attention, GRU)

Usage:
    python research/embedding_vs_finetuning/chunk_aggregate.py \
        --dataset uplifting_v5 \
        --model all-MiniLM-L6-v2 \
        --scoring variance \
        --top-k 5 \
        --aggregation mean
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
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


def load_chunk_embeddings(input_path: Path) -> Dict[str, Any]:
    """Load chunk embeddings from disk."""
    data = np.load(input_path, allow_pickle=True)

    article_ids = list(data['article_ids'])
    chunk_counts = data['chunk_counts']
    padded_embeddings = data['embeddings']

    # Unpack to dict
    embeddings = {}
    for i, article_id in enumerate(article_ids):
        n_chunks = chunk_counts[i]
        embeddings[article_id] = padded_embeddings[i, :n_chunks]

    return {
        'embeddings': embeddings,
        'article_ids': article_ids,
        'labels': data['labels'],
        'dimension_names': list(data['dimension_names']),
        'chunk_counts': chunk_counts,
        'padded_embeddings': padded_embeddings
    }


# =============================================================================
# Chunk Scoring Methods
# =============================================================================

def score_chunks_variance(embeddings: np.ndarray) -> np.ndarray:
    """
    Score chunks by embedding variance.

    Higher variance = more information content.

    Args:
        embeddings: (num_chunks, embedding_dim)

    Returns:
        scores: (num_chunks,) - higher is more important
    """
    # Variance across embedding dimensions
    return np.var(embeddings, axis=1)


def score_chunks_centroid_distance(embeddings: np.ndarray) -> np.ndarray:
    """
    Score chunks by distance from centroid.

    Chunks further from mean are more distinctive.

    Args:
        embeddings: (num_chunks, embedding_dim)

    Returns:
        scores: (num_chunks,) - higher is more important
    """
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return distances


def score_chunks_norm(embeddings: np.ndarray) -> np.ndarray:
    """
    Score chunks by embedding L2 norm.

    Higher norm may indicate more content.

    Args:
        embeddings: (num_chunks, embedding_dim)

    Returns:
        scores: (num_chunks,) - higher is more important
    """
    return np.linalg.norm(embeddings, axis=1)


def score_chunks_random(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Random chunk scores (baseline).

    Args:
        embeddings: (num_chunks, embedding_dim)
        seed: Random seed

    Returns:
        scores: (num_chunks,) - random values
    """
    rng = np.random.RandomState(seed)
    return rng.random(embeddings.shape[0])


def score_chunks_first_k(embeddings: np.ndarray) -> np.ndarray:
    """
    Prefer earlier chunks (beginning of article often most important).

    Args:
        embeddings: (num_chunks, embedding_dim)

    Returns:
        scores: (num_chunks,) - higher for earlier chunks
    """
    n = embeddings.shape[0]
    return np.arange(n, 0, -1).astype(float)


SCORING_METHODS = {
    'variance': score_chunks_variance,
    'centroid_distance': score_chunks_centroid_distance,
    'norm': score_chunks_norm,
    'random': score_chunks_random,
    'first_k': score_chunks_first_k
}


def select_top_k_chunks(
    embeddings: np.ndarray,
    k: int,
    scoring_method: str = 'variance'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top-K chunks based on scoring method.

    Args:
        embeddings: (num_chunks, embedding_dim)
        k: Number of chunks to select
        scoring_method: Method to score chunks

    Returns:
        selected_embeddings: (k, embedding_dim) or (num_chunks, embedding_dim) if num_chunks < k
        selected_indices: indices of selected chunks
    """
    n_chunks = embeddings.shape[0]

    if n_chunks <= k:
        # Return all chunks if fewer than k
        return embeddings, np.arange(n_chunks)

    # Score chunks
    scorer = SCORING_METHODS.get(scoring_method, score_chunks_variance)
    scores = scorer(embeddings)

    # Select top-k
    top_indices = np.argsort(scores)[-k:]
    top_indices = np.sort(top_indices)  # Maintain order

    return embeddings[top_indices], top_indices


# =============================================================================
# Aggregation Methods
# =============================================================================

class MeanAggregator:
    """Simple mean pooling of chunk embeddings."""

    def __init__(self):
        pass

    def aggregate(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Args:
            embeddings: (num_chunks, embedding_dim)

        Returns:
            aggregated: (embedding_dim,)
        """
        return np.mean(embeddings, axis=0)

    def aggregate_batch(self, embeddings: np.ndarray, chunk_counts: np.ndarray) -> np.ndarray:
        """
        Aggregate batch of articles.

        Args:
            embeddings: (batch, max_chunks, embedding_dim)
            chunk_counts: (batch,) - number of valid chunks per article

        Returns:
            aggregated: (batch, embedding_dim)
        """
        batch_size = embeddings.shape[0]
        embedding_dim = embeddings.shape[2]
        result = np.zeros((batch_size, embedding_dim))

        for i in range(batch_size):
            n = chunk_counts[i]
            result[i] = np.mean(embeddings[i, :n], axis=0)

        return result


class AttentionAggregator(nn.Module):
    """Learned attention-based aggregation."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, num_chunks, embedding_dim)
            mask: (batch, num_chunks) - 1 for valid, 0 for padding

        Returns:
            aggregated: (batch, embedding_dim)
        """
        # Compute attention scores
        scores = self.attention(embeddings).squeeze(-1)  # (batch, num_chunks)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        weights = torch.softmax(scores, dim=-1)  # (batch, num_chunks)

        # Weighted sum
        aggregated = torch.bmm(weights.unsqueeze(1), embeddings).squeeze(1)

        return aggregated


class GRUAggregator(nn.Module):
    """GRU-based sequential aggregation."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True
    ):
        super().__init__()

        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.projection = nn.Linear(output_dim, embedding_dim)

    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, num_chunks, embedding_dim)
            lengths: (batch,) - number of valid chunks per article

        Returns:
            aggregated: (batch, embedding_dim)
        """
        # Pack sequences if lengths provided
        if lengths is not None:
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

            # Sort by length (required for pack_padded_sequence)
            sorted_lengths, sort_indices = torch.sort(lengths, descending=True)
            sorted_embeddings = embeddings[sort_indices]

            packed = pack_padded_sequence(
                sorted_embeddings,
                sorted_lengths.cpu(),
                batch_first=True,
                enforce_sorted=True
            )

            _, hidden = self.gru(packed)

            # Unsort
            _, unsort_indices = torch.sort(sort_indices)
            if self.gru.bidirectional:
                # Concatenate forward and backward hidden states
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            else:
                hidden = hidden[-1]

            hidden = hidden[unsort_indices]
        else:
            _, hidden = self.gru(embeddings)
            if self.gru.bidirectional:
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            else:
                hidden = hidden[-1]

        # Project back to embedding dim
        return self.projection(hidden)


# =============================================================================
# Training Aggregators
# =============================================================================

def train_attention_aggregator(
    train_embeddings: np.ndarray,
    train_chunk_counts: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_chunk_counts: np.ndarray,
    val_labels: np.ndarray,
    embedding_dim: int,
    hidden_dim: int = 128,
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    device: str = 'cuda'
) -> Tuple[AttentionAggregator, nn.Module, StandardScaler, Dict[str, float]]:
    """
    Train attention aggregator with a probe head.

    Returns:
        aggregator, probe_head, scaler, metrics
    """
    from torch.utils.data import DataLoader, TensorDataset

    # Create models
    aggregator = AttentionAggregator(embedding_dim, hidden_dim).to(device)
    probe_head = nn.Linear(embedding_dim, train_labels.shape[1]).to(device)

    # Create masks from chunk counts
    max_chunks = train_embeddings.shape[1]

    def create_mask(counts, max_len):
        mask = np.zeros((len(counts), max_len))
        for i, c in enumerate(counts):
            mask[i, :c] = 1
        return mask

    train_mask = create_mask(train_chunk_counts, max_chunks)
    val_mask = create_mask(val_chunk_counts, val_embeddings.shape[1])

    # Convert to tensors
    train_X = torch.FloatTensor(train_embeddings)
    train_M = torch.FloatTensor(train_mask)
    train_y = torch.FloatTensor(train_labels)

    val_X = torch.FloatTensor(val_embeddings).to(device)
    val_M = torch.FloatTensor(val_mask).to(device)
    val_y = torch.FloatTensor(val_labels).to(device)

    # Dataloader
    train_dataset = TensorDataset(train_X, train_M, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training
    optimizer = torch.optim.Adam(
        list(aggregator.parameters()) + list(probe_head.parameters()),
        lr=learning_rate
    )
    criterion = nn.L1Loss()

    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        aggregator.train()
        probe_head.train()
        train_loss = 0.0

        for batch_X, batch_M, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_M = batch_M.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            aggregated = aggregator(batch_X, batch_M)
            predictions = probe_head(aggregated)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        aggregator.eval()
        probe_head.eval()
        with torch.no_grad():
            val_aggregated = aggregator(val_X, val_M)
            val_predictions = probe_head(val_aggregated)
            val_mae = torch.mean(torch.abs(val_predictions - val_y)).item()

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {
                'aggregator': aggregator.state_dict(),
                'probe_head': probe_head.state_dict()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val MAE={val_mae:.4f}")

    # Load best state
    aggregator.load_state_dict(best_state['aggregator'])
    probe_head.load_state_dict(best_state['probe_head'])

    metrics = {
        'val_mae': best_val_mae,
        'epochs_trained': epoch + 1
    }

    return aggregator, probe_head, None, metrics


def train_gru_aggregator(
    train_embeddings: np.ndarray,
    train_chunk_counts: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_chunk_counts: np.ndarray,
    val_labels: np.ndarray,
    embedding_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 1,
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    device: str = 'cuda'
) -> Tuple[GRUAggregator, nn.Module, StandardScaler, Dict[str, float]]:
    """
    Train GRU aggregator with a probe head.

    Returns:
        aggregator, probe_head, scaler, metrics
    """
    from torch.utils.data import DataLoader, TensorDataset

    # Create models
    aggregator = GRUAggregator(embedding_dim, hidden_dim, num_layers).to(device)
    probe_head = nn.Linear(embedding_dim, train_labels.shape[1]).to(device)

    # Convert to tensors
    train_X = torch.FloatTensor(train_embeddings)
    train_lengths = torch.LongTensor(train_chunk_counts)
    train_y = torch.FloatTensor(train_labels)

    val_X = torch.FloatTensor(val_embeddings).to(device)
    val_lengths = torch.LongTensor(val_chunk_counts).to(device)
    val_y = torch.FloatTensor(val_labels).to(device)

    # Dataloader
    train_dataset = TensorDataset(train_X, train_lengths, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training
    optimizer = torch.optim.Adam(
        list(aggregator.parameters()) + list(probe_head.parameters()),
        lr=learning_rate
    )
    criterion = nn.L1Loss()

    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        aggregator.train()
        probe_head.train()
        train_loss = 0.0

        for batch_X, batch_lengths, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_lengths = batch_lengths.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            aggregated = aggregator(batch_X, batch_lengths)
            predictions = probe_head(aggregated)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        aggregator.eval()
        probe_head.eval()
        with torch.no_grad():
            val_aggregated = aggregator(val_X, val_lengths)
            val_predictions = probe_head(val_aggregated)
            val_mae = torch.mean(torch.abs(val_predictions - val_y)).item()

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {
                'aggregator': aggregator.state_dict(),
                'probe_head': probe_head.state_dict()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val MAE={val_mae:.4f}")

    # Load best state
    aggregator.load_state_dict(best_state['aggregator'])
    probe_head.load_state_dict(best_state['probe_head'])

    metrics = {
        'val_mae': best_val_mae,
        'epochs_trained': epoch + 1
    }

    return aggregator, probe_head, None, metrics


# =============================================================================
# Aggregation Pipeline
# =============================================================================

def aggregate_with_topk(
    chunk_embeddings: np.ndarray,
    chunk_counts: np.ndarray,
    top_k: int,
    scoring_method: str = 'variance',
    aggregation: str = 'mean'
) -> np.ndarray:
    """
    Apply top-K selection and aggregation to chunk embeddings.

    Args:
        chunk_embeddings: (batch, max_chunks, embedding_dim)
        chunk_counts: (batch,) - valid chunk counts
        top_k: Number of chunks to select
        scoring_method: How to score chunks
        aggregation: Aggregation method ('mean' only for non-learned)

    Returns:
        aggregated: (batch, embedding_dim)
    """
    batch_size = chunk_embeddings.shape[0]
    embedding_dim = chunk_embeddings.shape[2]
    result = np.zeros((batch_size, embedding_dim))

    for i in range(batch_size):
        n_chunks = chunk_counts[i]
        embeddings = chunk_embeddings[i, :n_chunks]

        # Select top-K
        selected, _ = select_top_k_chunks(embeddings, top_k, scoring_method)

        # Aggregate
        if aggregation == 'mean':
            result[i] = np.mean(selected, axis=0)
        else:
            raise ValueError(f"Non-learned aggregation must be 'mean', got: {aggregation}")

    return result


def save_aggregated_embeddings(
    embeddings: np.ndarray,
    article_ids: List[str],
    labels: np.ndarray,
    dimension_names: List[str],
    output_path: Path,
    metadata: Dict[str, Any]
):
    """Save aggregated embeddings in same format as regular embeddings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        article_ids=np.array(article_ids, dtype=object),
        labels=labels,
        dimension_names=np.array(dimension_names, dtype=object),
        **{f"meta_{k}": v for k, v in metadata.items() if isinstance(v, (int, float, str))}
    )
    logger.info(f"Saved aggregated embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate chunk embeddings with Top-K selection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, default='research/embedding_vs_finetuning/config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., uplifting_v5)')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                       help='Embedding model used for chunks')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                       help='Dataset splits to process')
    parser.add_argument('--scoring', type=str, default='variance',
                       choices=['variance', 'centroid_distance', 'norm', 'random', 'first_k'],
                       help='Chunk scoring method')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of chunks to select')
    parser.add_argument('--aggregation', type=str, default='mean',
                       choices=['mean', 'attention', 'gru'],
                       help='Aggregation method')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--embeddings-dir', type=str, default='research/embedding_vs_finetuning/embeddings',
                       help='Directory with chunk embeddings')
    parser.add_argument('--output-dir', type=str, default='research/embedding_vs_finetuning/embeddings',
                       help='Output directory for aggregated embeddings')

    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))

    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)

    # Output file naming
    suffix = f"topk{args.top_k}_{args.scoring}_{args.aggregation}"

    if args.aggregation == 'mean':
        # Simple aggregation - just aggregate each split
        for split in args.splits:
            input_path = embeddings_dir / f"{args.dataset}_{args.model.replace('/', '_')}_chunked_{split}.npz"
            output_path = output_dir / f"{args.dataset}_{args.model.replace('/', '_')}_{suffix}_{split}.npz"

            if not input_path.exists():
                logger.error(f"Chunk embeddings not found: {input_path}")
                continue

            logger.info(f"\n{'='*60}\nProcessing {split}\n{'='*60}")

            # Load chunk embeddings
            data = load_chunk_embeddings(input_path)

            # Aggregate
            aggregated = aggregate_with_topk(
                data['padded_embeddings'],
                data['chunk_counts'],
                top_k=args.top_k,
                scoring_method=args.scoring,
                aggregation='mean'
            )

            logger.info(f"Aggregated: {aggregated.shape}")

            # Save
            metadata = {
                'model': args.model,
                'scoring': args.scoring,
                'top_k': args.top_k,
                'aggregation': args.aggregation
            }
            save_aggregated_embeddings(
                aggregated,
                data['article_ids'],
                data['labels'],
                data['dimension_names'],
                output_path,
                metadata
            )

    else:
        # Learned aggregation - train on train set, apply to all
        logger.info(f"\n{'='*60}\nTraining {args.aggregation} aggregator\n{'='*60}")

        # Load train and val
        train_path = embeddings_dir / f"{args.dataset}_{args.model.replace('/', '_')}_chunked_train.npz"
        val_path = embeddings_dir / f"{args.dataset}_{args.model.replace('/', '_')}_chunked_val.npz"

        train_data = load_chunk_embeddings(train_path)
        val_data = load_chunk_embeddings(val_path)

        # Apply top-K selection first
        train_selected = aggregate_with_topk(
            train_data['padded_embeddings'],
            train_data['chunk_counts'],
            top_k=args.top_k,
            scoring_method=args.scoring,
            aggregation='mean'
        )
        # For learned methods, we need the chunk-level data
        # Re-select to get the selected chunk embeddings

        embedding_dim = train_data['padded_embeddings'].shape[2]

        # Train aggregator
        chunk_config = config.get('chunking', {}).get('aggregation_methods', {}).get(args.aggregation, {})
        hidden_dim = chunk_config.get('hidden_dim', 128)

        if args.aggregation == 'attention':
            aggregator, probe_head, _, metrics = train_attention_aggregator(
                train_data['padded_embeddings'],
                train_data['chunk_counts'],
                train_data['labels'],
                val_data['padded_embeddings'],
                val_data['chunk_counts'],
                val_data['labels'],
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                device=args.device
            )
        elif args.aggregation == 'gru':
            num_layers = chunk_config.get('num_layers', 1)
            aggregator, probe_head, _, metrics = train_gru_aggregator(
                train_data['padded_embeddings'],
                train_data['chunk_counts'],
                train_data['labels'],
                val_data['padded_embeddings'],
                val_data['chunk_counts'],
                val_data['labels'],
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                device=args.device
            )

        logger.info(f"Training complete: Val MAE = {metrics['val_mae']:.4f}")

        # Save trained models
        models_dir = output_dir.parent / 'results'
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / f"{args.dataset}_{args.model.replace('/', '_')}_{suffix}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'aggregator_state': aggregator.state_dict(),
                'probe_head_state': probe_head.state_dict(),
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'aggregation': args.aggregation,
                'metrics': metrics
            }, f)
        logger.info(f"Saved models to {model_path}")


if __name__ == '__main__':
    main()
