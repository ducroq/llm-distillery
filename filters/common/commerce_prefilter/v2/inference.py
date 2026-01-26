"""
Commerce Prefilter v2 - Embedding + MLP Classifier

Uses frozen sentence-transformers embeddings with a trained MLP classifier.
Achieves same accuracy as v1 (97.8% F1) with simpler architecture.

Usage:
    from filters.common.commerce_prefilter.v2.inference import CommercePrefilterV2

    detector = CommercePrefilterV2(threshold=0.95)
    result = detector.is_commerce(article)
    # {"is_commerce": True, "score": 0.97, "version": "v2"}
"""

import pickle
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class CommercePrefilterV2:
    """
    Commerce detection using frozen embeddings + MLP classifier.

    Architecture:
        Article text -> [Frozen Embedder] -> 768-dim vector -> [MLP] -> score (0-1)

    Attributes:
        threshold: Score above which article is classified as commerce
        embedder: SentenceTransformer model for generating embeddings
        classifier: Trained MLP classifier
        scaler: StandardScaler for normalizing embeddings
    """

    def __init__(
        self,
        threshold: float = 0.95,
        model_dir: Optional[Path] = None,
        device: str = 'cpu'
    ):
        """
        Initialize the commerce prefilter.

        Args:
            threshold: Classification threshold (default 0.95 for high precision)
            model_dir: Path to directory containing classifier and scaler
            device: Device for embedder ('cpu' or 'cuda')
        """
        self.threshold = threshold
        self.device = device

        # Set model directory
        if model_dir is None:
            model_dir = Path(__file__).parent / 'models'
        self.model_dir = Path(model_dir)

        # Load models lazily
        self._embedder = None
        self._classifier = None
        self._scaler = None
        self._loaded = False

    def _load_models(self):
        """Load embedder, classifier, and scaler."""
        if self._loaded:
            return

        # Load embedder
        self._embedder = SentenceTransformer(
            'paraphrase-multilingual-mpnet-base-v2',
            device=self.device
        )

        # Load classifier
        classifier_path = self.model_dir / 'mlp_classifier.pkl'
        with open(classifier_path, 'rb') as f:
            self._classifier = pickle.load(f)

        # Load scaler
        scaler_path = self.model_dir / 'scaler.pkl'
        with open(scaler_path, 'rb') as f:
            self._scaler = pickle.load(f)

        self._loaded = True

    def _prepare_text(self, article: Union[dict, str]) -> str:
        """
        Prepare text from article for embedding.

        Args:
            article: Article dict with 'title' and 'content', or raw text string

        Returns:
            Combined text for embedding
        """
        if isinstance(article, str):
            return article

        title = article.get('title', '')
        content = article.get('content', '')

        # Combine title and content
        # Note: Embedder has 128-token limit, so title + first ~100 words is used
        return f"{title} {content}"

    def is_commerce(self, article: Union[dict, str]) -> dict:
        """
        Check if an article is commerce/promotional content.

        Args:
            article: Article dict with 'title' and 'content', or raw text

        Returns:
            Dict with:
                - is_commerce: bool
                - score: float (0-1)
                - inference_time_ms: float
                - version: str
        """
        self._load_models()

        start_time = time.perf_counter()

        # Prepare text
        text = self._prepare_text(article)

        # Generate embedding
        embedding = self._embedder.encode([text], show_progress_bar=False)

        # Scale
        embedding_scaled = self._scaler.transform(embedding)

        # Predict
        score = self._classifier.predict_proba(embedding_scaled)[0, 1]

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            'is_commerce': bool(score >= self.threshold),
            'score': float(score),
            'inference_time_ms': inference_time_ms,
            'version': 'v2'
        }

    def batch_predict(self, articles: list, batch_size: int = 32) -> list:
        """
        Predict commerce scores for multiple articles.

        Args:
            articles: List of article dicts or text strings
            batch_size: Batch size for embedding generation

        Returns:
            List of result dicts (same format as is_commerce)
        """
        self._load_models()

        start_time = time.perf_counter()

        # Prepare texts
        texts = [self._prepare_text(a) for a in articles]

        # Generate embeddings in batch
        embeddings = self._embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False
        )

        # Scale
        embeddings_scaled = self._scaler.transform(embeddings)

        # Predict
        scores = self._classifier.predict_proba(embeddings_scaled)[:, 1]

        total_time_ms = (time.perf_counter() - start_time) * 1000
        avg_time_ms = total_time_ms / len(articles)

        results = []
        for score in scores:
            results.append({
                'is_commerce': bool(score >= self.threshold),
                'score': float(score),
                'inference_time_ms': avg_time_ms,
                'version': 'v2'
            })

        return results

    def get_score(self, article: Union[dict, str]) -> float:
        """
        Get raw commerce score without threshold application.

        Args:
            article: Article dict or text string

        Returns:
            Commerce probability (0-1)
        """
        result = self.is_commerce(article)
        return result['score']


# Convenience function for quick checks
def is_commerce(article: Union[dict, str], threshold: float = 0.95) -> bool:
    """
    Quick check if article is commerce content.

    Note: Creates new detector instance each call. For batch processing,
    use CommercePrefilterV2 class directly.
    """
    detector = CommercePrefilterV2(threshold=threshold)
    return detector.is_commerce(article)['is_commerce']
