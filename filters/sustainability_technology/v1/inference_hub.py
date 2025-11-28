"""
Sustainability Technology v1 - HuggingFace Hub Inference

Loads the model directly from HuggingFace Hub for inference.
Use this for testing or when you don't have local model files.

Usage:
    from filters.sustainability_technology.v1.inference_hub import SustainabilityTechnologyScorerHub

    scorer = SustainabilityTechnologyScorerHub(
        repo_id="jeergrvgreg/sustainability-technology-v1",
        token="hf_..."
    )
    result = scorer.score_article(article)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Suppress the "should TRAIN this model" warning - we load trained weights from LoRA adapter
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
from peft import PeftModel, PeftConfig
from huggingface_hub import hf_hub_download

# Import prefilter
from filters.sustainability_technology.v1.prefilter import SustainabilityTechnologyPreFilterV1


class SustainabilityTechnologyScorerHub:
    """
    Scorer that loads model from HuggingFace Hub.
    """

    FILTER_NAME = "sustainability_technology"
    FILTER_VERSION = "1.0"

    DIMENSION_NAMES = [
        "technology_readiness_level",
        "technical_performance",
        "economic_competitiveness",
        "life_cycle_environmental_impact",
        "social_equity_impact",
        "governance_systemic_impact",
    ]

    DIMENSION_WEIGHTS = {
        "technology_readiness_level": 0.15,
        "technical_performance": 0.15,
        "economic_competitiveness": 0.20,
        "life_cycle_environmental_impact": 0.30,
        "social_equity_impact": 0.10,
        "governance_systemic_impact": 0.10,
    }

    TIER_THRESHOLDS = [
        ("high_sustainability", 7.0, "Mass deployed, proven sustainable, competitive"),
        ("medium_high", 5.0, "Commercial deployment, good sustainability"),
        ("medium", 3.0, "Pilot/early commercial, mixed profile"),
        ("low", 0.0, "Lab stage or poor sustainability performance"),
    ]

    TRL_GATEKEEPER_MIN = 3.0
    TRL_GATEKEEPER_CAP = 2.9

    def __init__(
        self,
        repo_id: str = "jeergrvgreg/sustainability-technology-v1",
        token: Optional[str] = None,
        device: Optional[str] = None,
        use_prefilter: bool = True,
    ):
        """
        Initialize scorer from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID
            token: HuggingFace token (for private repos)
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_prefilter: Whether to apply prefilter
        """
        self.repo_id = repo_id
        self.token = token

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_prefilter = use_prefilter
        if use_prefilter:
            self.prefilter = SustainabilityTechnologyPreFilterV1()

        self._load_model_from_hub()

    def _load_model_from_hub(self):
        """Load model from HuggingFace Hub."""
        print(f"Loading model from HuggingFace Hub: {self.repo_id}")
        print(f"Device: {self.device}")

        # Download and load adapter config
        config_path = hf_hub_download(
            repo_id=self.repo_id,
            filename="adapter_config.json",
            token=self.token,
        )

        with open(config_path, "r") as f:
            adapter_config = json.load(f)

        base_model_name = adapter_config["base_model_name_or_path"]
        print(f"Base model: {base_model_name}")

        # Load tokenizer from base model (not from hub repo which may not have all files)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            token=self.token,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        print("Loading base model...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(self.DIMENSION_NAMES),
            problem_type="regression",
        )

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Load PEFT model from hub
        print("Loading LoRA adapter from Hub...")
        self.model = PeftModel.from_pretrained(
            base_model,
            self.repo_id,
            token=self.token,
        )

        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully")

    def score_article(
        self,
        article: Dict,
        skip_prefilter: bool = False,
    ) -> Dict:
        """
        Score a single article.

        Args:
            article: Dict with 'title' and 'content' keys
            skip_prefilter: Force skip prefilter

        Returns:
            Dict with scores, tier, etc.
        """
        result = {
            "passed_prefilter": True,
            "prefilter_reason": None,
            "scores": None,
            "weighted_average": None,
            "tier": None,
            "tier_description": None,
            "gatekeeper_applied": False,
        }

        # Prefilter
        if self.use_prefilter and not skip_prefilter:
            passed, reason = self.prefilter.apply_filter(article)
            if not passed:
                result["passed_prefilter"] = False
                result["prefilter_reason"] = reason
                return result

        # Tokenize
        text = f"{article['title']}\n\n{article['content']}"
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_scores = outputs.logits[0].cpu().numpy()

        # Process scores
        scores = {
            dim: float(max(0.0, min(10.0, raw_scores[i])))
            for i, dim in enumerate(self.DIMENSION_NAMES)
        }
        result["scores"] = scores

        # Weighted average
        weighted_avg = sum(
            scores[dim] * self.DIMENSION_WEIGHTS[dim]
            for dim in self.DIMENSION_NAMES
        )

        # Gatekeeper
        if scores["technology_readiness_level"] < self.TRL_GATEKEEPER_MIN:
            if weighted_avg > self.TRL_GATEKEEPER_CAP:
                weighted_avg = self.TRL_GATEKEEPER_CAP
                result["gatekeeper_applied"] = True

        result["weighted_average"] = weighted_avg

        # Tier
        for tier_name, threshold, description in self.TIER_THRESHOLDS:
            if weighted_avg >= threshold:
                result["tier"] = tier_name
                result["tier_description"] = description
                break

        return result

    def score_batch(
        self,
        articles: List[Dict],
        batch_size: int = 16,
        skip_prefilter: bool = False,
    ) -> List[Dict]:
        """Score multiple articles."""
        results = []
        for article in articles:
            results.append(self.score_article(article, skip_prefilter))
        return results
