"""
Benchmark speed and memory usage for embedding models and probes.

This script measures:
1. Embedding generation speed (articles/second)
2. Probe training speed
3. Inference speed (predictions/second)
4. Peak memory usage (GPU and CPU)

Usage:
    python research/embedding_vs_finetuning/benchmark_speed.py \
        --dataset uplifting_v5 \
        --models all-MiniLM-L6-v2 jina-embeddings-v2-base-en \
        --output-dir research/embedding_vs_finetuning/results/benchmarks
"""

import argparse
import gc
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


def load_dataset(data_dir: Path, split: str = 'test') -> List[Dict[str, Any]]:
    """Load articles from JSONL file."""
    file_path = data_dir / f'{split}.jsonl'
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def prepare_texts(articles: List[Dict[str, Any]], prefix: str = "") -> List[str]:
    """Prepare article texts for embedding."""
    texts = []
    for article in articles:
        title = article.get('title', '')
        content = article.get('content', '')
        if title and content:
            text = f"{title}\n\n{content}"
        elif title:
            text = title
        else:
            text = content
        texts.append(prefix + text)
    return texts


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def get_peak_gpu_memory_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def benchmark_embedding_model(
    model_name: str,
    model_config: Dict[str, Any],
    texts: List[str],
    batch_size: int = 32,
    device: str = 'cuda',
    warmup_batches: int = 2
) -> Dict[str, Any]:
    """
    Benchmark embedding model speed and memory.

    Returns dict with:
        - articles_per_second
        - total_time_seconds
        - peak_gpu_memory_mb
        - batch_size_used
    """
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    source = model_config.get('source', 'sentence-transformers')
    actual_model_name = model_config.get('model_name', model_name)
    trust_remote_code = model_config.get('trust_remote_code', False)
    max_tokens = model_config.get('max_tokens', 512)

    reset_gpu_memory_stats()

    # Load model
    load_start = time.time()

    if source == 'sentence-transformers':
        model = SentenceTransformer(
            actual_model_name,
            device=device,
            trust_remote_code=trust_remote_code
        )
        if max_tokens:
            model.max_seq_length = max_tokens
    else:
        tokenizer = AutoTokenizer.from_pretrained(actual_model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(actual_model_name, trust_remote_code=True)
        model.to(device)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    load_time = time.time() - load_start
    memory_after_load = get_gpu_memory_mb()

    logger.info(f"Model loaded in {load_time:.2f}s, GPU memory: {memory_after_load:.0f}MB")

    # Warmup
    warmup_texts = texts[:batch_size * warmup_batches]
    if source == 'sentence-transformers':
        _ = model.encode(warmup_texts, batch_size=batch_size, show_progress_bar=False)
    else:
        with torch.no_grad():
            for i in range(0, len(warmup_texts), batch_size):
                batch = warmup_texts[i:i + batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=max_tokens, return_tensors='pt').to(device)
                _ = model(**inputs)

    reset_gpu_memory_stats()

    # Benchmark
    start_time = time.time()

    if source == 'sentence-transformers':
        _ = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    else:
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=max_tokens, return_tensors='pt').to(device)
                outputs = model(**inputs)
                # Mean pooling
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                hidden_states = outputs.last_hidden_state
                masked_hidden = hidden_states * attention_mask
                sum_hidden = masked_hidden.sum(dim=1)
                count = attention_mask.sum(dim=1)
                embeddings = sum_hidden / count
                all_embeddings.append(embeddings.cpu().numpy())

    elapsed = time.time() - start_time
    peak_memory = get_peak_gpu_memory_mb()

    # Cleanup
    del model
    if source == 'transformers':
        del tokenizer
    reset_gpu_memory_stats()

    return {
        'model_name': model_name,
        'source': source,
        'articles_per_second': len(texts) / elapsed,
        'total_time_seconds': elapsed,
        'num_articles': len(texts),
        'batch_size': batch_size,
        'max_tokens': max_tokens,
        'model_load_time_seconds': load_time,
        'peak_gpu_memory_mb': peak_memory,
        'memory_after_load_mb': memory_after_load
    }


def benchmark_probe_training(
    embeddings: np.ndarray,
    labels: np.ndarray,
    probe_type: str,
    probe_config: Dict[str, Any],
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Benchmark probe training speed.

    Returns dict with:
        - training_time_seconds
        - peak_gpu_memory_mb (for neural probes)
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    reset_gpu_memory_stats()

    # Split for validation
    n_train = int(len(embeddings) * 0.9)
    train_X, val_X = embeddings[:n_train], embeddings[n_train:]
    train_y, val_y = labels[:n_train], labels[n_train:]

    start_time = time.time()

    if probe_type == 'ridge':
        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)

        alphas = probe_config.get('alpha', [0.01, 0.1, 1.0, 10.0, 100.0])
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(train_X_scaled, train_y)

        peak_memory = 0.0

    elif probe_type == 'mlp':
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)

        train_tensor = torch.FloatTensor(train_X_scaled)
        train_labels = torch.FloatTensor(train_y)

        # Simple MLP
        input_dim = train_X.shape[1]
        output_dim = train_y.shape[1]
        hidden_sizes = probe_config.get('hidden_sizes', [256, 128])

        layers = []
        prev_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_size
        layers.append(nn.Linear(prev_dim, output_dim))

        model = nn.Sequential(*layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.L1Loss()

        dataset = TensorDataset(train_tensor, train_labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        epochs = min(probe_config.get('epochs', 100), 20)  # Limit for benchmark
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        peak_memory = get_peak_gpu_memory_mb()
        del model

    elif probe_type == 'lightgbm':
        import lightgbm as lgb

        models = []
        for dim_idx in range(train_y.shape[1]):
            train_data = lgb.Dataset(train_X, label=train_y[:, dim_idx])
            val_data = lgb.Dataset(val_X, label=val_y[:, dim_idx], reference=train_data)

            params = {
                'objective': 'regression',
                'metric': 'mae',
                'n_estimators': min(probe_config.get('n_estimators', 500), 100),
                'learning_rate': probe_config.get('learning_rate', 0.05),
                'max_depth': probe_config.get('max_depth', 6),
                'verbose': -1
            }

            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(20),
                    lgb.log_evaluation(period=0)
                ]
            )
            models.append(model)

        peak_memory = 0.0

    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

    elapsed = time.time() - start_time
    reset_gpu_memory_stats()

    return {
        'probe_type': probe_type,
        'training_time_seconds': elapsed,
        'num_samples': len(train_X),
        'embedding_dim': train_X.shape[1],
        'output_dim': train_y.shape[1],
        'peak_gpu_memory_mb': peak_memory
    }


def benchmark_inference(
    embeddings: np.ndarray,
    probe_type: str,
    num_runs: int = 5,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Benchmark inference speed.

    Returns dict with:
        - predictions_per_second
        - avg_latency_ms
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import torch.nn as nn

    # Create dummy probe
    output_dim = 6
    n_samples = len(embeddings)

    times = []

    if probe_type == 'ridge':
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        model = Ridge()
        model.fit(embeddings_scaled[:100], np.random.randn(100, output_dim))

        for _ in range(num_runs):
            start = time.time()
            _ = model.predict(embeddings_scaled)
            times.append(time.time() - start)

    elif probe_type == 'mlp':
        input_dim = embeddings.shape[1]
        model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        ).to(device)
        model.eval()

        embeddings_tensor = torch.FloatTensor(embeddings).to(device)

        for _ in range(num_runs):
            with torch.no_grad():
                start = time.time()
                _ = model(embeddings_tensor)
                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)

        del model, embeddings_tensor

    elif probe_type == 'lightgbm':
        import lightgbm as lgb

        # Create minimal LightGBM model
        train_data = lgb.Dataset(embeddings[:100], label=np.random.randn(100))
        model = lgb.train({'objective': 'regression', 'verbose': -1}, train_data, num_boost_round=10)

        for _ in range(num_runs):
            start = time.time()
            _ = model.predict(embeddings)
            times.append(time.time() - start)

    avg_time = np.mean(times)
    std_time = np.std(times)

    return {
        'probe_type': probe_type,
        'num_samples': n_samples,
        'predictions_per_second': n_samples / avg_time,
        'avg_latency_ms': avg_time * 1000,
        'std_latency_ms': std_time * 1000,
        'num_runs': num_runs
    }


def format_results_table(results: List[Dict[str, Any]], title: str) -> str:
    """Format results as a nice table."""
    if not results:
        return f"\n{title}\n{'='*60}\nNo results\n"

    lines = [f"\n{title}", "=" * 80]

    # Embedding benchmarks
    embed_results = [r for r in results if 'articles_per_second' in r]
    if embed_results:
        lines.append("\nEmbedding Speed:")
        lines.append(f"{'Model':<35} {'Articles/s':>12} {'Time (s)':>10} {'GPU (MB)':>10}")
        lines.append("-" * 70)
        for r in embed_results:
            lines.append(
                f"{r['model_name']:<35} {r['articles_per_second']:>12.1f} "
                f"{r['total_time_seconds']:>10.2f} {r['peak_gpu_memory_mb']:>10.0f}"
            )

    # Probe training benchmarks
    train_results = [r for r in results if 'training_time_seconds' in r and 'probe_type' in r]
    if train_results:
        lines.append("\nProbe Training Speed:")
        lines.append(f"{'Probe Type':<20} {'Time (s)':>12} {'GPU (MB)':>10}")
        lines.append("-" * 45)
        for r in train_results:
            lines.append(
                f"{r['probe_type']:<20} {r['training_time_seconds']:>12.2f} "
                f"{r['peak_gpu_memory_mb']:>10.0f}"
            )

    # Inference benchmarks
    infer_results = [r for r in results if 'predictions_per_second' in r]
    if infer_results:
        lines.append("\nInference Speed:")
        lines.append(f"{'Probe Type':<20} {'Pred/s':>12} {'Latency (ms)':>12}")
        lines.append("-" * 48)
        for r in infer_results:
            lines.append(
                f"{r['probe_type']:<20} {r['predictions_per_second']:>12.0f} "
                f"{r['avg_latency_ms']:>12.2f}"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark embedding models and probes',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, default='research/embedding_vs_finetuning/config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., uplifting_v5)')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Embedding models to benchmark (default: all)')
    parser.add_argument('--probes', type=str, nargs='+', default=['ridge', 'mlp', 'lightgbm'],
                       help='Probe types to benchmark')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default='research/embedding_vs_finetuning/results/benchmarks',
                       help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to use (default: all)')

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    config = load_config(config_path)

    # Determine models
    if args.models:
        models = args.models
    else:
        models = list(config['embedding_models'].keys())

    # Load dataset
    dataset_config = config['datasets'][args.dataset]
    data_dir = config_path.parent.parent.parent / dataset_config['path']
    articles = load_dataset(data_dir, 'test')

    if args.num_samples:
        articles = articles[:args.num_samples]

    logger.info(f"Benchmarking with {len(articles)} articles")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Benchmark embedding models
    for model_name in models:
        logger.info(f"\n{'='*60}\nBenchmarking: {model_name}\n{'='*60}")

        model_config = config['embedding_models'].get(model_name, {})
        prefix = model_config.get('prefix', '')
        texts = prepare_texts(articles, prefix)

        try:
            result = benchmark_embedding_model(
                model_name,
                model_config,
                texts,
                batch_size=args.batch_size,
                device=args.device
            )
            all_results.append(result)
            logger.info(f"  Speed: {result['articles_per_second']:.1f} articles/sec")
            logger.info(f"  Peak GPU: {result['peak_gpu_memory_mb']:.0f} MB")

        except Exception as e:
            logger.error(f"Failed to benchmark {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Benchmark probe training and inference
    # Use dummy embeddings for probe benchmarks
    embedding_dim = 768
    n_samples = len(articles)
    dummy_embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    dummy_labels = np.random.randn(n_samples, 6).astype(np.float32)

    for probe_type in args.probes:
        logger.info(f"\n{'='*60}\nBenchmarking probe: {probe_type}\n{'='*60}")

        probe_config = config['probe_methods'].get(probe_type, {})

        try:
            # Training benchmark
            train_result = benchmark_probe_training(
                dummy_embeddings,
                dummy_labels,
                probe_type,
                probe_config,
                device=args.device
            )
            train_result['benchmark_type'] = 'training'
            all_results.append(train_result)
            logger.info(f"  Training time: {train_result['training_time_seconds']:.2f}s")

            # Inference benchmark
            infer_result = benchmark_inference(
                dummy_embeddings,
                probe_type,
                device=args.device
            )
            infer_result['benchmark_type'] = 'inference'
            all_results.append(infer_result)
            logger.info(f"  Inference: {infer_result['predictions_per_second']:.0f} pred/sec")

        except Exception as e:
            logger.error(f"Failed to benchmark {probe_type}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = output_dir / f"{args.dataset}_benchmarks.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved results to {output_path}")

    # Print summary table
    print(format_results_table(all_results, f"Benchmark Results: {args.dataset}"))


if __name__ == '__main__':
    main()
