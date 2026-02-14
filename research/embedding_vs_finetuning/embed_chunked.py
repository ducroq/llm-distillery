#!/usr/bin/env python3
"""
Chunked embedding with multiple aggregation strategies.

Tests whether chunking + aggregation can improve short-context models
(128 tokens) to match longer-context models (512+ tokens).
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Aggregation strategies
AGGREGATION_STRATEGIES = ['mean', 'max', 'mean_max', 'first_last']


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 128) -> List[str]:
    """
    Split text into overlapping chunks by words.

    Args:
        text: Input text
        chunk_size: Target words per chunk
        overlap: Words of overlap between chunks

    Returns:
        List of text chunks
    """
    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap

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
    Aggregate multiple chunk embeddings into a single embedding.

    Args:
        embeddings: Shape (num_chunks, embedding_dim)
        strategy: One of 'mean', 'max', 'mean_max', 'first_last'

    Returns:
        Aggregated embedding of shape (embedding_dim,) or (2*embedding_dim,) for concat strategies
    """
    if len(embeddings) == 1:
        if strategy in ['mean_max', 'first_last']:
            # For concat strategies with single chunk, duplicate
            return np.concatenate([embeddings[0], embeddings[0]])
        return embeddings[0]

    if strategy == 'mean':
        return np.mean(embeddings, axis=0)
    elif strategy == 'max':
        return np.max(embeddings, axis=0)
    elif strategy == 'mean_max':
        # Concatenate mean and max pooling
        mean_emb = np.mean(embeddings, axis=0)
        max_emb = np.max(embeddings, axis=0)
        return np.concatenate([mean_emb, max_emb])
    elif strategy == 'first_last':
        # Concatenate first and last chunk embeddings
        return np.concatenate([embeddings[0], embeddings[-1]])
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")


def load_articles(dataset_path: Path, split: str) -> Tuple[List[str], np.ndarray]:
    """Load articles and labels from dataset."""
    file_path = dataset_path / f"{split}.jsonl"

    texts = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Combine title and content for embedding
            title = item.get('title', '')
            content = item.get('content', item.get('text', ''))
            texts.append(f"{title}\n\n{content}" if title else content)
            labels.append(item['labels'])

    return texts, np.array(labels)


def embed_with_chunking(
    model: SentenceTransformer,
    texts: List[str],
    chunk_size: int = 256,
    overlap: int = 128,
    strategies: List[str] = None,
    batch_size: int = 64,
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """
    Embed texts using chunking and multiple aggregation strategies.

    Returns:
        Dictionary mapping strategy name to embeddings array
    """
    if strategies is None:
        strategies = AGGREGATION_STRATEGIES

    # First, chunk all texts and track which chunks belong to which text
    all_chunks = []
    chunk_indices = []  # (start_idx, end_idx) for each text

    logger.info(f"Chunking {len(texts)} texts (chunk_size={chunk_size}, overlap={overlap})...")

    for text in texts:
        chunks = chunk_text(text, chunk_size, overlap)
        start_idx = len(all_chunks)
        all_chunks.extend(chunks)
        end_idx = len(all_chunks)
        chunk_indices.append((start_idx, end_idx))

    avg_chunks = len(all_chunks) / len(texts)
    logger.info(f"Created {len(all_chunks)} chunks (avg {avg_chunks:.1f} chunks/article)")

    # Embed all chunks
    logger.info("Embedding all chunks...")
    start_time = time.time()

    chunk_embeddings = model.encode(
        all_chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device
    )

    embed_time = time.time() - start_time
    logger.info(f"Embedded {len(all_chunks)} chunks in {embed_time:.1f}s ({len(all_chunks)/embed_time:.1f} chunks/sec)")

    # Aggregate for each strategy
    results = {}

    for strategy in strategies:
        logger.info(f"Aggregating with strategy: {strategy}")

        aggregated = []
        for start_idx, end_idx in chunk_indices:
            text_chunks = chunk_embeddings[start_idx:end_idx]
            agg_emb = aggregate_embeddings(text_chunks, strategy)
            aggregated.append(agg_emb)

        results[strategy] = np.array(aggregated)
        logger.info(f"  {strategy}: shape {results[strategy].shape}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate chunked embeddings with aggregation')
    parser.add_argument('--dataset', type=str, default='uplifting_v5',
                        help='Dataset name')
    parser.add_argument('--models', nargs='+',
                        default=['multilingual-MiniLM-L12-v2', 'multilingual-mpnet-base-v2'],
                        help='Models to test (short-context models)')
    parser.add_argument('--chunk-size', type=int, default=200,
                        help='Words per chunk (default: 200, ~256 tokens)')
    parser.add_argument('--overlap', type=int, default=100,
                        help='Overlap words between chunks (default: 100)')
    parser.add_argument('--strategies', nargs='+', default=AGGREGATION_STRATEGIES,
                        help='Aggregation strategies to test')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for embedding')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output-dir', type=str,
                        default='research/embedding_vs_finetuning/embeddings',
                        help='Output directory')

    args = parser.parse_args()

    # Load config for model mappings
    config_path = Path('research/embedding_vs_finetuning/config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_path = Path(f"datasets/training/{args.dataset}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model name mappings
    model_mappings = {
        'multilingual-MiniLM-L12-v2': 'paraphrase-multilingual-MiniLM-L12-v2',
        'multilingual-mpnet-base-v2': 'paraphrase-multilingual-mpnet-base-v2',
    }

    for model_name in args.models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'='*60}")

        # Load model
        hf_name = model_mappings.get(model_name, model_name)
        model = SentenceTransformer(hf_name, device=args.device)

        for split in ['train', 'val', 'test']:
            logger.info(f"\nProcessing {split} split...")

            # Load data
            texts, labels = load_articles(dataset_path, split)
            logger.info(f"Loaded {len(texts)} articles")

            # Embed with chunking
            embeddings_by_strategy = embed_with_chunking(
                model=model,
                texts=texts,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                strategies=args.strategies,
                batch_size=args.batch_size,
                device=args.device
            )

            # Save embeddings for each strategy
            for strategy, embeddings in embeddings_by_strategy.items():
                output_file = output_dir / f"{args.dataset}_{model_name}_chunked_{strategy}_{split}.npz"
                np.savez_compressed(
                    output_file,
                    embeddings=embeddings,
                    labels=labels
                )
                logger.info(f"Saved {output_file}")

        # Clear GPU memory
        del model
        torch.cuda.empty_cache()

    logger.info("\n" + "="*60)
    logger.info("Chunked embedding generation complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
