"""
Commerce Prefilter SLM - Inference Module

DistilBERT-based classifier for detecting commerce/promotional content.

Usage:
    from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM

    detector = CommercePrefilterSLM()
    result = detector.is_commerce(article)
    # {"is_commerce": True, "score": 0.87, "inference_time_ms": 23}

    # Batch processing
    results = detector.batch_predict(articles)
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class CommercePrefilterSLM:
    """
    ML-based classifier for detecting commerce/promotional content.

    Uses DistilBERT fine-tuned for binary classification.
    Single sigmoid output with configurable threshold.
    """

    # Default model path (relative to this file)
    DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "distilbert"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
        max_length: int = 512,
    ):
        """
        Initialize the commerce prefilter.

        Args:
            model_path: Path to trained model directory. If None, uses default path.
            device: Device to use ('cuda', 'cpu', or None for auto).
            threshold: Score threshold for commerce classification (default: 0.7).
            max_length: Maximum sequence length for tokenization.
        """
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self.threshold = threshold
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load the trained model and tokenizer."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Train the model first using: "
                f"python -m filters.common.commerce_prefilter.training.train"
            )

        logger.info(f"Loading commerce prefilter from {self.model_path}")
        logger.info(f"Device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path)
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info("Commerce prefilter loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {type(e).__name__}: {e}")

    def _prepare_text(self, article: Dict) -> str:
        """Prepare input text from article fields using [TITLE] [CONTENT] format."""
        title = article.get('title', '')
        content = article.get('content', article.get('text', ''))

        # Truncate content for efficiency (model max_length handles final truncation)
        content = content[:4000] if content else ''

        return f"[TITLE] {title} [CONTENT] {content}"

    def predict_score(self, article: Dict) -> float:
        """
        Get raw commerce score for an article.

        Args:
            article: Dict with 'title' and 'content'/'text' fields.

        Returns:
            Commerce score between 0 and 1.
        """
        text = self._prepare_text(article)

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # 2-label classification: use softmax and get probability of class 1 (commerce)
            probs = torch.softmax(outputs.logits, dim=-1)
            score = probs[0, 1].item()

        return score

    def is_commerce(self, article: Dict) -> Dict:
        """
        Check if article is commerce/promotional content.

        Args:
            article: Dict with 'title' and 'content'/'text' fields.

        Returns:
            Dict with:
                - is_commerce: bool
                - score: float (0-1)
                - inference_time_ms: float
        """
        start_time = time.perf_counter()

        score = self.predict_score(article)
        is_commerce = score >= self.threshold

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "is_commerce": is_commerce,
            "score": round(score, 4),
            "inference_time_ms": round(inference_time_ms, 2),
        }

    def batch_predict(
        self,
        articles: List[Dict],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Batch prediction for multiple articles.

        Args:
            articles: List of article dicts.
            batch_size: Batch size for inference.

        Returns:
            List of result dicts (same format as is_commerce).
        """
        results = []
        start_time = time.perf_counter()

        # Prepare all texts
        texts = [self._prepare_text(article) for article in articles]

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # 2-label classification: use softmax and get probability of class 1 (commerce)
                probs = torch.softmax(outputs.logits, dim=-1)
                scores = probs[:, 1].cpu().numpy()

            # Handle single item batch
            if scores.ndim == 0:
                scores = [scores.item()]

            for score in scores:
                results.append({
                    "is_commerce": float(score) >= self.threshold,
                    "score": round(float(score), 4),
                })

        total_time_ms = (time.perf_counter() - start_time) * 1000
        avg_time_ms = total_time_ms / len(articles) if articles else 0

        # Add timing info
        for result in results:
            result["inference_time_ms"] = round(avg_time_ms, 2)

        return results

    def set_threshold(self, threshold: float):
        """
        Update the classification threshold.

        Args:
            threshold: New threshold value (0-1).
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        self.threshold = threshold
        logger.info(f"Commerce threshold updated to {threshold}")


# Singleton instance for lazy loading
_commerce_detector_instance: Optional[CommercePrefilterSLM] = None


def get_commerce_detector(
    model_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.5,
) -> CommercePrefilterSLM:
    """
    Get singleton instance of commerce detector.

    Lazy loads the model on first call.

    Args:
        model_path: Path to model (only used on first call).
        threshold: Classification threshold (only used on first call).

    Returns:
        CommercePrefilterSLM instance.
    """
    global _commerce_detector_instance

    if _commerce_detector_instance is None:
        _commerce_detector_instance = CommercePrefilterSLM(
            model_path=model_path,
            threshold=threshold,
        )

    return _commerce_detector_instance


def main():
    """CLI for testing the commerce prefilter."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Test commerce prefilter on articles"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input JSONL file with articles"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.7,
        help="Classification threshold"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to model directory"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Commerce Prefilter SLM - Inference")
    print("=" * 60)

    # Initialize
    print(f"\nLoading model...")
    detector = CommercePrefilterSLM(
        model_path=args.model_path,
        threshold=args.threshold,
    )
    print(f"Model loaded. Threshold: {detector.threshold}")
    print(f"Device: {detector.device}")

    if args.input:
        # Process file
        print(f"\nProcessing {args.input}...")
        articles = []
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    articles.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        results = detector.batch_predict(articles)

        # Summary
        commerce_count = sum(1 for r in results if r['is_commerce'])
        print(f"\nResults: {commerce_count}/{len(results)} classified as commerce")

        avg_time = sum(r['inference_time_ms'] for r in results) / len(results) if results else 0
        print(f"Average inference time: {avg_time:.2f}ms")

        # Score distribution
        scores = [r['score'] for r in results]
        print(f"\nScore distribution:")
        print(f"  Min: {min(scores):.4f}")
        print(f"  Max: {max(scores):.4f}")
        print(f"  Mean: {sum(scores)/len(scores):.4f}")

    else:
        # Demo
        print("\n--- Demo ---")
        demo_articles = [
            {
                "title": "Green Deals: Save $500 on Jackery Solar Generator",
                "content": "Today's Green Deals are headlined by an exclusive discount on "
                          "the Jackery Explorer 1000 Plus solar generator kit. Originally "
                          "priced at $1,999, you can now get it for just $1,499 - that's "
                          "$500 in savings! This deal ends tonight at midnight."
            },
            {
                "title": "New Solar Technology Achieves Record 30% Efficiency",
                "content": "Researchers at MIT have developed a breakthrough perovskite-silicon "
                          "tandem solar cell that achieves 30% efficiency, surpassing traditional "
                          "silicon panels. The technology uses abundant materials and could be "
                          "manufactured at scale using existing equipment."
            },
        ]

        for article in demo_articles:
            print(f"\nTitle: {article['title'][:60]}...")
            result = detector.is_commerce(article)
            print(f"  Commerce: {result['is_commerce']} (score: {result['score']:.4f})")
            print(f"  Inference: {result['inference_time_ms']:.2f}ms")


if __name__ == "__main__":
    main()
