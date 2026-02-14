"""
Generate embeddings for articles using various embedding models.

This script loads training data and generates embeddings using different
embedding models (sentence-transformers, frozen Qwen, etc.) and caches
them to disk for efficient reuse during probe training.

Usage:
    python research/embedding_vs_finetuning/embed_articles.py \
        --dataset uplifting_v5 \
        --models all-MiniLM-L6-v2 bge-large-en-v1.5 \
        --batch-size 32
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def prepare_text(article: Dict[str, Any], prefix: str = "") -> str:
    """Prepare article text for embedding.

    Combines title and content into a single string for embedding.
    """
    title = article.get('title', '')
    content = article.get('content', '')

    # Combine title and content
    if title and content:
        text = f"{title}\n\n{content}"
    elif title:
        text = title
    else:
        text = content

    return prefix + text


class SentenceTransformerEmbedder:
    """Embedding using sentence-transformers library."""

    def __init__(
        self,
        model_name: str,
        device: str = 'cuda',
        trust_remote_code: bool = False,
        max_seq_length: Optional[int] = None
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code
        )

        # Override max sequence length if specified (for long-context models)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
            logger.info(f"Set max_seq_length to {max_seq_length}")

        self.device = device
        logger.info(f"Loaded SentenceTransformer: {model_name} (trust_remote_code={trust_remote_code})")

    def embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings


class QwenFrozenEmbedder:
    """Frozen Qwen2.5-1.5B embeddings using mean pooling."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        device: str = 'cuda',
        max_length: int = 512
    ):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.default_max_length = max_length

        # Handle padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loaded frozen Qwen model: {model_name} (max_length={max_length})")

    def embed(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings using mean pooling of last hidden states."""
        from tqdm import tqdm

        if max_length is None:
            max_length = self.default_max_length

        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding with Qwen")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)

                outputs = self.model(**inputs)

                # Mean pool over sequence length (excluding padding)
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                hidden_states = outputs.last_hidden_state
                masked_hidden = hidden_states * attention_mask
                sum_hidden = masked_hidden.sum(dim=1)
                count = attention_mask.sum(dim=1)
                embeddings = sum_hidden / count

                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


def get_embedder(model_config: Dict[str, Any], model_name: str, device: str):
    """Factory function to create appropriate embedder."""
    source = model_config.get('source', 'sentence-transformers')

    if source == 'sentence-transformers':
        actual_model_name = model_config.get('model_name', model_name)
        trust_remote_code = model_config.get('trust_remote_code', False)
        max_tokens = model_config.get('max_tokens')

        return SentenceTransformerEmbedder(
            actual_model_name,
            device=device,
            trust_remote_code=trust_remote_code,
            max_seq_length=max_tokens
        )
    elif source == 'transformers':
        actual_model_name = model_config.get('model_name', 'Qwen/Qwen2.5-1.5B')
        max_tokens = model_config.get('max_tokens', 512)
        return QwenFrozenEmbedder(actual_model_name, device, max_length=max_tokens)
    else:
        raise ValueError(f"Unknown embedding source: {source}")


def save_embeddings(
    embeddings: np.ndarray,
    article_ids: List[str],
    labels: np.ndarray,
    dimension_names: List[str],
    output_path: Path,
    metadata: Dict[str, Any]
):
    """Save embeddings and associated data to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        article_ids=np.array(article_ids, dtype=object),
        labels=labels,
        dimension_names=np.array(dimension_names, dtype=object),
        **{f"meta_{k}": v for k, v in metadata.items() if isinstance(v, (int, float, str))}
    )
    logger.info(f"Saved embeddings to {output_path}")


def load_embeddings(input_path: Path) -> Dict[str, Any]:
    """Load embeddings from disk."""
    data = np.load(input_path, allow_pickle=True)
    return {
        'embeddings': data['embeddings'],
        'article_ids': data['article_ids'],
        'labels': data['labels'],
        'dimension_names': list(data['dimension_names'])
    }


def embed_dataset(
    config: Dict[str, Any],
    dataset_name: str,
    model_name: str,
    split: str,
    output_dir: Path,
    batch_size: int = 32,
    device: str = 'cuda',
    force: bool = False
) -> Path:
    """Generate and cache embeddings for a dataset split."""

    # Check if already cached
    output_path = output_dir / f"{dataset_name}_{model_name.replace('/', '_')}_{split}.npz"
    if output_path.exists() and not force:
        logger.info(f"Using cached embeddings: {output_path}")
        return output_path

    # Load dataset
    dataset_config = config['datasets'][dataset_name]
    data_dir = Path(config.get('root_dir', '.')) / dataset_config['path']
    articles = load_dataset(data_dir, split)

    # Get model config
    model_config = config['embedding_models'][model_name]
    prefix = model_config.get('prefix', '')

    # Prepare texts
    texts = [prepare_text(article, prefix) for article in articles]
    article_ids = [article['id'] for article in articles]
    labels = np.array([article['labels'] for article in articles])
    dimension_names = articles[0]['dimension_names']

    # Create embedder and generate embeddings
    embedder = get_embedder(model_config, model_name, device)

    logger.info(f"Generating embeddings for {len(texts)} articles with {model_name}")
    start_time = time.time()

    if isinstance(embedder, QwenFrozenEmbedder):
        embeddings = embedder.embed(texts, batch_size=batch_size // 4, max_length=model_config.get('max_tokens', 512))
    else:
        embeddings = embedder.embed(texts, batch_size=batch_size)

    elapsed = time.time() - start_time

    # Save with metadata
    metadata = {
        'model_name': model_name,
        'dataset': dataset_name,
        'split': split,
        'num_articles': len(articles),
        'embedding_dim': embeddings.shape[1],
        'embedding_time_seconds': elapsed
    }

    save_embeddings(embeddings, article_ids, labels, dimension_names, output_path, metadata)

    logger.info(f"Embedding time: {elapsed:.2f}s ({len(texts)/elapsed:.1f} articles/sec)")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings for articles',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, default='research/embedding_vs_finetuning/config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., uplifting_v5)')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Embedding models to use (default: all)')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                       help='Dataset splits to process')
    parser.add_argument('--batch-size', type=int, default=32,
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

    # Determine models to use
    if args.models:
        models = args.models
    else:
        models = list(config['embedding_models'].keys())

    output_dir = Path(args.output_dir)

    # Generate embeddings for each model and split
    results = {}
    for model_name in models:
        logger.info(f"\n{'='*60}\nProcessing model: {model_name}\n{'='*60}")

        model_results = {}
        for split in args.splits:
            try:
                output_path = embed_dataset(
                    config=config,
                    dataset_name=args.dataset,
                    model_name=model_name,
                    split=split,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    device=args.device,
                    force=args.force
                )
                model_results[split] = str(output_path)
            except Exception as e:
                logger.error(f"Failed to embed {split} with {model_name}: {e}")
                model_results[split] = None

        results[model_name] = model_results

    # Print summary
    logger.info(f"\n{'='*60}\nEmbedding Summary\n{'='*60}")
    for model_name, splits in results.items():
        logger.info(f"{model_name}:")
        for split, path in splits.items():
            status = "OK" if path else "FAILED"
            logger.info(f"  {split}: {status}")


if __name__ == '__main__':
    main()
