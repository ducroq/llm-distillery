"""
Cultural Discovery Filter v3 - Base Scorer Class

Shared logic for CulturalDiscoveryScorer and CulturalDiscoveryScorerHub.
Eliminates code duplication between local and Hub inference.

This module provides:
- Common constants (dimensions, weights, gatekeeper thresholds)
- Tier assignment logic (HIGH/MEDIUM/LOW)
- Score processing (clamping, weighted average, gatekeeper)
- Input validation
- Result dict structure
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class BaseCulturalDiscoveryScorer(ABC):
    """
    Abstract base class for cultural discovery scoring.

    Provides shared logic for both local model loading (CulturalDiscoveryScorer)
    and HuggingFace Hub loading (CulturalDiscoveryScorerHub).

    Subclasses must implement:
        - _load_model(): Load model from local files or Hub
    """

    # Filter metadata
    FILTER_NAME = "cultural-discovery"
    FILTER_VERSION = "3.0"

    # Dimension configuration (5 dimensions)
    DIMENSION_NAMES = [
        "discovery_novelty",
        "heritage_significance",
        "cross_cultural_connection",
        "human_resonance",
        "evidence_quality",
    ]

    DIMENSION_WEIGHTS = {
        "discovery_novelty": 0.25,
        "heritage_significance": 0.20,
        "cross_cultural_connection": 0.25,
        "human_resonance": 0.15,
        "evidence_quality": 0.15,
    }

    # Gatekeeper: Evidence < 3 caps overall at 3.0
    EVIDENCE_GATEKEEPER_MIN = 3.0
    EVIDENCE_GATEKEEPER_CAP = 3.0

    # Tier thresholds
    TIER_HIGH_THRESHOLD = 7.0
    TIER_MEDIUM_THRESHOLD = 4.0

    # Model configuration
    MAX_TOKEN_LENGTH = 512
    DEFAULT_BATCH_SIZE = 16

    def __init__(
        self,
        device: Optional[str] = None,
        use_prefilter: bool = True,
    ):
        """
        Initialize the base scorer.

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_prefilter: Whether to apply prefilter before scoring
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_prefilter = use_prefilter
        self.prefilter = None
        self.model = None
        self.tokenizer = None

        if use_prefilter:
            self._load_prefilter()

    def _load_prefilter(self):
        """Load the prefilter module."""
        import importlib.util
        import sys

        # Add project root to path for filters.common imports
        project_root = Path(__file__).parent.parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        prefilter_path = Path(__file__).parent / "prefilter.py"
        spec = importlib.util.spec_from_file_location("prefilter", prefilter_path)
        prefilter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prefilter_module)
        self.prefilter = prefilter_module.CulturalDiscoveryPreFilterV1()

    @abstractmethod
    def _load_model(self):
        """Load the model. Implemented by subclasses."""
        pass

    def _validate_article(self, article: Dict) -> None:
        """
        Validate article structure.

        Args:
            article: Article dict to validate

        Raises:
            TypeError: If article is not a dict
            ValueError: If required keys are missing or empty
        """
        if not isinstance(article, dict):
            raise TypeError(f"article must be dict, got {type(article).__name__}")

        if "title" not in article:
            raise ValueError("article must contain 'title' key")
        if "content" not in article:
            raise ValueError("article must contain 'content' key")

        if not article.get("title"):
            raise ValueError("article 'title' cannot be empty")
        if not article.get("content"):
            raise ValueError("article 'content' cannot be empty")

    def _create_empty_result(self) -> Dict:
        """Create an empty result dict structure."""
        return {
            "passed_prefilter": True,
            "prefilter_reason": None,
            "scores": None,
            "weighted_average": None,
            "tier": None,
            "tier_description": None,
            "gatekeeper_applied": False,
        }

    def _assign_tier(
        self, scores: Dict[str, float], weighted_avg: float
    ) -> Tuple[str, str]:
        """
        Assign tier based on weighted average score.

        Tiers:
            - HIGH: Significant discovery or deep cross-cultural insight (>= 7.0)
            - MEDIUM: Meaningful cultural content with some value (>= 4.0)
            - LOW: Superficial or speculative content (< 4.0)

        Args:
            scores: Dict of dimension scores
            weighted_avg: Weighted average score

        Returns:
            Tuple of (tier_name, tier_description)
        """
        if weighted_avg >= self.TIER_HIGH_THRESHOLD:
            return ("HIGH", "Significant discovery or deep cross-cultural insight")
        elif weighted_avg >= self.TIER_MEDIUM_THRESHOLD:
            return ("MEDIUM", "Meaningful cultural content with some discovery value")
        else:
            return ("LOW", "Superficial, speculative, or single-culture content")

    def _process_raw_scores(
        self, raw_scores, result: Dict
    ) -> Dict:
        """
        Process raw model output into final scores with gatekeeper logic.

        Args:
            raw_scores: Raw numpy array from model output
            result: Result dict to update

        Returns:
            Updated result dict
        """
        # Clamp scores to 0-10 range
        scores = {
            dim: float(max(0.0, min(10.0, raw_scores[i])))
            for i, dim in enumerate(self.DIMENSION_NAMES)
        }
        result["scores"] = scores

        # Compute weighted average
        weighted_avg = sum(
            scores[dim] * self.DIMENSION_WEIGHTS[dim]
            for dim in self.DIMENSION_NAMES
        )

        # Apply gatekeeper: Evidence < 3 caps overall at 3.0
        if scores["evidence_quality"] < self.EVIDENCE_GATEKEEPER_MIN:
            if weighted_avg > self.EVIDENCE_GATEKEEPER_CAP:
                weighted_avg = self.EVIDENCE_GATEKEEPER_CAP
                result["gatekeeper_applied"] = True

        result["weighted_average"] = weighted_avg

        # Assign tier
        tier, tier_desc = self._assign_tier(scores, weighted_avg)
        result["tier"] = tier
        result["tier_description"] = tier_desc

        return result

    def score_article(
        self,
        article: Dict,
        skip_prefilter: bool = False,
    ) -> Dict:
        """
        Score a single article.

        Args:
            article: Dict with 'title' and 'content' keys
            skip_prefilter: Force skip prefilter even if enabled

        Returns:
            Dict with:
                - passed_prefilter: bool
                - prefilter_reason: str (if blocked)
                - scores: Dict[dimension_name, float] (if passed)
                - weighted_average: float (if passed)
                - tier: str (if passed)
                - tier_description: str (if passed)
                - gatekeeper_applied: bool

        Raises:
            TypeError: If article is not a dict
            ValueError: If article is missing required keys
        """
        # Validate input
        self._validate_article(article)

        result = self._create_empty_result()

        # Apply prefilter
        if self.use_prefilter and not skip_prefilter:
            passed, reason = self.prefilter.apply_filter(article)
            if not passed:
                result["passed_prefilter"] = False
                result["prefilter_reason"] = reason
                return result

        # Prepare input
        text = f"{article['title']}\n\n{article['content']}"
        inputs = self.tokenizer(
            text,
            max_length=self.MAX_TOKEN_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_scores = outputs.logits[0].cpu().float().numpy()

        # Process scores
        return self._process_raw_scores(raw_scores, result)

    def score_batch(
        self,
        articles: List[Dict],
        batch_size: int = None,
        skip_prefilter: bool = False,
    ) -> List[Dict]:
        """
        Score a batch of articles efficiently.

        Uses batched inference for better GPU utilization.

        Args:
            articles: List of article dicts
            batch_size: Batch size for inference (default: DEFAULT_BATCH_SIZE)
            skip_prefilter: Skip prefilter for all articles

        Returns:
            List of result dicts (same structure as score_article)

        Raises:
            ValueError: If batch_size is not positive
        """
        if not articles:
            return []

        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        results = []

        # First pass: prefilter and validate
        articles_to_score = []
        article_indices = []

        for i, article in enumerate(articles):
            result = self._create_empty_result()

            # Validate (but don't raise - just mark as failed)
            try:
                self._validate_article(article)
            except (TypeError, ValueError) as e:
                result["passed_prefilter"] = False
                result["prefilter_reason"] = f"Invalid article: {e}"
                results.append(result)
                continue

            # Apply prefilter
            if self.use_prefilter and not skip_prefilter:
                passed, reason = self.prefilter.apply_filter(article)
                if not passed:
                    result["passed_prefilter"] = False
                    result["prefilter_reason"] = reason
                    results.append(result)
                    continue

            articles_to_score.append(article)
            article_indices.append(i)
            results.append(result)

        # Second pass: batch inference
        if articles_to_score:
            for batch_start in range(0, len(articles_to_score), batch_size):
                batch_end = min(batch_start + batch_size, len(articles_to_score))
                batch = articles_to_score[batch_start:batch_end]
                batch_indices = article_indices[batch_start:batch_end]

                # Tokenize batch
                texts = [f"{a['title']}\n\n{a['content']}" for a in batch]
                inputs = self.tokenizer(
                    texts,
                    max_length=self.MAX_TOKEN_LENGTH,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_scores = outputs.logits.cpu().float().numpy()

                # Process each result
                for j, idx in enumerate(batch_indices):
                    raw_scores = batch_scores[j]
                    self._process_raw_scores(raw_scores, results[idx])

        return results
