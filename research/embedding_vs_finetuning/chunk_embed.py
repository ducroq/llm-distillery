"""
Chunk articles and generate per-chunk embeddings.

This script implements the chunking approach for long articles:
1. Split articles into overlapping chunks
2. Embed each chunk independently
3. Store chunk embeddings for aggregation

Usage:
    python research/embedding_vs_finetuning/chunk_embed.py \
        --dataset uplifting_v5 \
        --model all-MiniLM-L6-v2 \
        --chunk-size 256 \
        --overlap 128
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_dataset(data_dir: Path, split: str = 'train') -> List[Dict[str, Any]]:
    """Load articles from JSONL file."""
    file_path = data_dir / f'{split}.jsonl'
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    logger.info(f"Loaded {len(articles)} articles from {file_path}")
    return articles


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (words * 1.3)."""
    words = len(text.split())
    return int(words * 1.3)


def chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap: int = 128,
    min_chunks: int = 1,
    max_chunks: int = 32
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text to chunk
        chunk_size: Target size in estimated tokens per chunk
        overlap: Number of overlapping tokens between chunks
        min_chunks: Minimum number of chunks to return
        max_chunks: Maximum number of chunks to return

    Returns:
        List of text chunks
    """
    words = text.split()
    if not words:
        return [text] if text.strip() else [""]

    # Convert token targets to word counts (tokens / 1.3)
    words_per_chunk = int(chunk_size / 1.3)
    overlap_words = int(overlap / 1.3)
    step_size = max(1, words_per_chunk - overlap_words)

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))

        start += step_size

        # Stop if we've reached max chunks
        if len(chunks) >= max_chunks:
            break

    # Ensure minimum chunks (pad with duplicates if needed)
    while len(chunks) < min_chunks:
        chunks.append(chunks[-1] if chunks else "")

    return chunks


def chunk_article(
    article: Dict[str, Any],
    chunk_size: int = 256,
    overlap: int = 128,
    min_chunks: int = 1,
    max_chunks: int = 32,
    prefix: str = ""
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Split an article into chunks with metadata.

    Returns:
        Tuple of (list of chunk texts, metadata dict)
    """
    title = article.get('title', '')
    content = article.get('content', '')

    # Combine title and content
    if title and content:
        full_text = f"{title}\n\n{content}"
    elif title:
        full_text = title
    else:
        full_text = content

    chunks = chunk_text(
        full_text,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunks=min_chunks,
        max_chunks=max_chunks
    )

    # Add prefix to each chunk if needed
    if prefix:
        chunks = [prefix + chunk for chunk in chunks]

    metadata = {
        'article_id': article['id'],
        'num_chunks': len(chunks),
        'total_tokens_est': estimate_tokens(full_text),
        'chunk_sizes': [estimate_tokens(c) for c in chunks]
    }

    return chunks, metadata


class ChunkEmbedder:
    """Embed article chunks using any embedding model."""

    def __init__(
        self,
        model_name: str,
        device: str = 'cuda',
        trust_remote_code: bool = False
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code
        )
        self.device = device
        logger.info(f"Loaded SentenceTransformer: {model_name}")

    def embed_chunks(
        self,
        chunks: List[str],
        batch_size: int = 64,
        show_progress: bool = False
    ) -> np.ndarray:
        """Embed a list of chunks."""
        embeddings = self.model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings


def embed_articles_chunked(
    articles: List[Dict[str, Any]],
    embedder: ChunkEmbedder,
    chunk_size: int = 256,
    overlap: int = 128,
    min_chunks: int = 1,
    max_chunks: int = 32,
    prefix: str = "",
    batch_size: int = 64
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """
    Embed all articles with chunking.

    Returns:
        Tuple of:
        - Dict mapping article_id -> chunk embeddings (num_chunks, embedding_dim)
        - Dict mapping article_id -> chunk metadata
    """
    from tqdm import tqdm

    all_embeddings = {}
    all_metadata = {}

    # First pass: chunk all articles and collect all chunks
    all_chunks = []
    chunk_indices = []  # (article_id, chunk_idx)

    for article in tqdm(articles, desc="Chunking articles"):
        chunks, metadata = chunk_article(
            article,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunks=min_chunks,
            max_chunks=max_chunks,
            prefix=prefix
        )

        article_id = article['id']
        all_metadata[article_id] = metadata

        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_indices.append((article_id, idx))

    logger.info(f"Total chunks: {len(all_chunks)} from {len(articles)} articles")
    logger.info(f"Average chunks per article: {len(all_chunks) / len(articles):.1f}")

    # Second pass: embed all chunks in batches
    logger.info("Embedding chunks...")
    all_chunk_embeddings = embedder.embed_chunks(
        all_chunks,
        batch_size=batch_size,
        show_progress=True
    )

    embedding_dim = all_chunk_embeddings.shape[1]

    # Third pass: reorganize embeddings by article
    for article_id in all_metadata:
        num_chunks = all_metadata[article_id]['num_chunks']
        all_embeddings[article_id] = np.zeros((num_chunks, embedding_dim))

    for idx, (article_id, chunk_idx) in enumerate(chunk_indices):
        all_embeddings[article_id][chunk_idx] = all_chunk_embeddings[idx]

    return all_embeddings, all_metadata


def save_chunk_embeddings(
    embeddings: Dict[str, np.ndarray],
    metadata: Dict[str, Dict[str, Any]],
    labels: Dict[str, np.ndarray],
    dimension_names: List[str],
    output_path: Path,
    config_info: Dict[str, Any]
):
    """Save chunk embeddings to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to arrays for storage
    article_ids = list(embeddings.keys())
    max_chunks = max(e.shape[0] for e in embeddings.values())
    embedding_dim = next(iter(embeddings.values())).shape[1]

    # Padded array (num_articles, max_chunks, embedding_dim)
    padded_embeddings = np.zeros((len(article_ids), max_chunks, embedding_dim))
    chunk_counts = np.zeros(len(article_ids), dtype=np.int32)
    labels_array = np.zeros((len(article_ids), len(dimension_names)))

    for i, article_id in enumerate(article_ids):
        n_chunks = embeddings[article_id].shape[0]
        padded_embeddings[i, :n_chunks] = embeddings[article_id]
        chunk_counts[i] = n_chunks
        labels_array[i] = labels[article_id]

    np.savez_compressed(
        output_path,
        embeddings=padded_embeddings,
        chunk_counts=chunk_counts,
        article_ids=np.array(article_ids, dtype=object),
        labels=labels_array,
        dimension_names=np.array(dimension_names, dtype=object),
        chunk_size=config_info.get('chunk_size', 256),
        overlap=config_info.get('overlap', 128)
    )

    logger.info(f"Saved chunk embeddings to {output_path}")
    logger.info(f"  Shape: {padded_embeddings.shape}")
    logger.info(f"  Chunk counts: min={chunk_counts.min()}, max={chunk_counts.max()}, mean={chunk_counts.mean():.1f}")


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
        'chunk_size': int(data['chunk_size']),
        'overlap': int(data['overlap'])
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate chunk embeddings for articles',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, default='research/embedding_vs_finetuning/config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., uplifting_v5)')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                       help='Embedding model to use')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                       help='Dataset splits to process')
    parser.add_argument('--chunk-size', type=int, default=None,
                       help='Chunk size in tokens (default: from config)')
    parser.add_argument('--overlap', type=int, default=None,
                       help='Overlap in tokens (default: from config)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for embedding')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default='research/embedding_vs_finetuning/embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration of cached embeddings')

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    config = load_config(config_path)
    config['root_dir'] = str(config_path.parent.parent.parent)

    # Get chunking config
    chunk_config = config.get('chunking', {})
    chunk_size = args.chunk_size or chunk_config.get('chunk_size', 256)
    overlap = args.overlap or chunk_config.get('overlap', 128)
    min_chunks = chunk_config.get('min_chunks', 1)
    max_chunks = chunk_config.get('max_chunks', 32)

    logger.info(f"Chunking config: size={chunk_size}, overlap={overlap}, min={min_chunks}, max={max_chunks}")

    # Get model config
    model_config = config['embedding_models'].get(args.model, {})
    actual_model_name = model_config.get('model_name', args.model)
    trust_remote_code = model_config.get('trust_remote_code', False)
    prefix = model_config.get('prefix', '')

    # Create embedder
    embedder = ChunkEmbedder(
        actual_model_name,
        device=args.device,
        trust_remote_code=trust_remote_code
    )

    # Process each split
    dataset_config = config['datasets'][args.dataset]
    data_dir = Path(config['root_dir']) / dataset_config['path']
    output_dir = Path(args.output_dir)

    for split in args.splits:
        output_path = output_dir / f"{args.dataset}_{args.model.replace('/', '_')}_chunked_{split}.npz"

        if output_path.exists() and not args.force:
            logger.info(f"Skipping {split} (cached): {output_path}")
            continue

        logger.info(f"\n{'='*60}\nProcessing {split} split\n{'='*60}")

        # Load articles
        articles = load_dataset(data_dir, split)

        # Prepare labels dict
        labels = {a['id']: np.array(a['labels']) for a in articles}
        dimension_names = articles[0]['dimension_names']

        # Embed with chunking
        start_time = time.time()
        embeddings, metadata = embed_articles_chunked(
            articles,
            embedder,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunks=min_chunks,
            max_chunks=max_chunks,
            prefix=prefix,
            batch_size=args.batch_size
        )
        elapsed = time.time() - start_time

        logger.info(f"Embedding time: {elapsed:.2f}s ({len(articles)/elapsed:.1f} articles/sec)")

        # Save
        config_info = {
            'model_name': args.model,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'min_chunks': min_chunks,
            'max_chunks': max_chunks
        }
        save_chunk_embeddings(embeddings, metadata, labels, dimension_names, output_path, config_info)


if __name__ == '__main__':
    main()
