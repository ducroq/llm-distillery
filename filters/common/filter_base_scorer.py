"""
FilterBaseScorer - Abstract base class for all filter scorers.

Extracted from the 4 production base_scorer.py files (uplifting v6,
sustainability_technology v3, investment-risk v6, cultural-discovery v4)
which were ~400 lines each with ~350 lines identical.

Subclasses define constants and _load_prefilter(); everything else lives here.

See GitHub issue #10.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

logger = logging.getLogger(__name__)


class FilterBaseScorer(ABC):
    """
    Abstract base class for filter scoring.

    Provides shared logic for both local model loading and HuggingFace Hub
    loading. Subclasses must define class constants and implement:
        - _load_model(): Load model from local files or Hub
        - _load_prefilter(): Load the filter-specific prefilter
    """

    # --- Subclasses MUST define these ---
    FILTER_NAME: str
    FILTER_VERSION: str
    DIMENSION_NAMES: List[str]
    DIMENSION_WEIGHTS: Dict[str, float]
    TIER_THRESHOLDS: List[Tuple[str, float, str]]

    # --- Gatekeeper (subclasses MAY override; None = no gatekeeper) ---
    GATEKEEPER_DIMENSION: Optional[str] = None
    GATEKEEPER_MIN: float = 0.0
    GATEKEEPER_CAP: float = 0.0

    # --- Fixed constants ---
    MAX_TOKEN_LENGTH = 512
    DEFAULT_BATCH_SIZE = 16

    def __init__(
        self,
        device: Optional[str] = None,
        use_prefilter: bool = True,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_prefilter = use_prefilter
        self.prefilter = None
        self.model = None
        self.tokenizer = None

        self._load_preprocessing_config()
        self._load_calibration()

        if use_prefilter:
            self._load_prefilter()

    # --- Directory resolution ---

    def _get_filter_dir(self) -> Path:
        """Return the concrete subclass's directory (not the common/ dir)."""
        return Path(inspect.getfile(type(self))).parent

    # --- Property for HybridScorer compatibility ---

    @property
    def device_str(self) -> str:
        """Return string representation of device."""
        return str(self.device)

    # --- Calibration ---

    def _load_calibration(self):
        """Load score calibration if calibration.json exists in the filter directory."""
        self.calibration = None
        cal_path = self._get_filter_dir() / "calibration.json"
        if cal_path.exists():
            from filters.common.score_calibration import load_calibration
            self.calibration = load_calibration(str(cal_path))
            if self.calibration:
                dims_in_file = set(self.calibration.get("dimensions", {}).keys())
                expected_dims = set(self.DIMENSION_NAMES)
                missing = expected_dims - dims_in_file
                if missing:
                    logger.warning(
                        f"Calibration file missing dimensions: {missing}. "
                        f"Those dimensions will not be calibrated."
                    )
                logger.info(
                    f"Score calibration loaded ({self.calibration.get('n_samples', '?')} samples)"
                )

    # --- Preprocessing config ---

    def _load_preprocessing_config(self):
        """Load preprocessing config from config.yaml."""
        config_path = self._get_filter_dir() / "config.yaml"

        self.use_head_tail = False
        self.head_tokens = 256
        self.tail_tokens = 256
        self.head_tail_separator = " [...] "

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            preprocessing = config.get("preprocessing", {})
            head_tail = preprocessing.get("head_tail", {})

            self.use_head_tail = head_tail.get("enabled", False)
            self.head_tokens = head_tail.get("head_tokens", 256)
            self.tail_tokens = head_tail.get("tail_tokens", 256)
            self.head_tail_separator = head_tail.get("separator", " [...] ")

            if self.use_head_tail:
                logger.info(
                    f"Head+tail preprocessing enabled: {self.head_tokens} + {self.tail_tokens} tokens"
                )

    # --- Abstract methods ---

    @abstractmethod
    def _load_model(self):
        """Load the model. Implemented by subclasses."""
        pass

    @abstractmethod
    def _load_prefilter(self):
        """Load the filter-specific prefilter. Implemented by subclasses."""
        pass

    # --- Validation ---

    def _validate_article(self, article: Dict) -> None:
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

    # --- Tier assignment ---

    def _assign_tier(self, weighted_avg: float) -> Tuple[str, str]:
        for tier_name, threshold, description in self.TIER_THRESHOLDS:
            if weighted_avg >= threshold:
                return (tier_name, description)
        return ("low", "No tier matched")

    # --- Score processing ---

    def _process_raw_scores(self, raw_scores, result: Dict) -> Dict:
        """Process raw model output into final scores with gatekeeper logic."""
        if self.calibration is not None:
            from filters.common.score_calibration import apply_calibration
            raw_scores = apply_calibration(raw_scores, self.calibration, self.DIMENSION_NAMES)

        scores = {
            dim: float(max(0.0, min(10.0, raw_scores[i])))
            for i, dim in enumerate(self.DIMENSION_NAMES)
        }
        result["scores"] = scores

        weighted_avg = sum(
            scores[dim] * self.DIMENSION_WEIGHTS[dim]
            for dim in self.DIMENSION_NAMES
        )

        if self.GATEKEEPER_DIMENSION is not None:
            if scores[self.GATEKEEPER_DIMENSION] < self.GATEKEEPER_MIN:
                if weighted_avg > self.GATEKEEPER_CAP:
                    weighted_avg = self.GATEKEEPER_CAP
                    result["gatekeeper_applied"] = True

        result["weighted_average"] = weighted_avg

        tier, tier_desc = self._assign_tier(weighted_avg)
        result["tier"] = tier
        result["tier_description"] = tier_desc

        return result

    # --- Single article scoring ---

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
            Dict with scores, tier, gatekeeper info
        """
        self._validate_article(article)

        result = self._create_empty_result()

        if self.use_prefilter and not skip_prefilter:
            passed, reason = self.prefilter.apply_filter(article)
            if not passed:
                result["passed_prefilter"] = False
                result["prefilter_reason"] = reason
                return result

        text = f"{article['title']}\n\n{article['content']}"

        if self.use_head_tail:
            from filters.common.text_preprocessing import extract_head_tail
            text = extract_head_tail(
                text,
                self.tokenizer,
                self.head_tokens,
                self.tail_tokens,
                self.head_tail_separator,
            )

        inputs = self.tokenizer(
            text,
            max_length=self.MAX_TOKEN_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_scores = outputs.logits[0].cpu().numpy()

        return self._process_raw_scores(raw_scores, result)

    # --- Batch scoring ---

    def score_batch(
        self,
        articles: List[Dict],
        batch_size: int = None,
        skip_prefilter: bool = False,
    ) -> List[Dict]:
        """
        Score a batch of articles efficiently.

        Args:
            articles: List of article dicts
            batch_size: Batch size for inference (default: DEFAULT_BATCH_SIZE)
            skip_prefilter: Skip prefilter for all articles

        Returns:
            List of result dicts (same structure as score_article)
        """
        if not articles:
            return []

        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        results = []
        articles_to_score = []
        article_indices = []

        for i, article in enumerate(articles):
            result = self._create_empty_result()

            try:
                self._validate_article(article)
            except (TypeError, ValueError) as e:
                result["passed_prefilter"] = False
                result["prefilter_reason"] = f"Invalid article: {e}"
                results.append(result)
                continue

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

        if articles_to_score:
            for batch_start in range(0, len(articles_to_score), batch_size):
                batch_end = min(batch_start + batch_size, len(articles_to_score))
                batch = articles_to_score[batch_start:batch_end]
                batch_indices = article_indices[batch_start:batch_end]

                texts = [f"{a['title']}\n\n{a['content']}" for a in batch]

                if self.use_head_tail:
                    from filters.common.text_preprocessing import extract_head_tail
                    texts = [
                        extract_head_tail(
                            t,
                            self.tokenizer,
                            self.head_tokens,
                            self.tail_tokens,
                            self.head_tail_separator,
                        )
                        for t in texts
                    ]

                inputs = self.tokenizer(
                    texts,
                    max_length=self.MAX_TOKEN_LENGTH,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_scores = outputs.logits.cpu().numpy()

                for j, idx in enumerate(batch_indices):
                    raw_scores = batch_scores[j]
                    self._process_raw_scores(raw_scores, results[idx])

        return results
