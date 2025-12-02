"""
Investment Risk Filter v5 - HuggingFace Hub Inference

Loads the model directly from HuggingFace Hub for inference.
Use this for testing or when you don't have local model files.

Usage:
    from filters.investment_risk.v5.inference_hub import InvestmentRiskScorerHub

    scorer = InvestmentRiskScorerHub(
        repo_id="your-username/investment-risk-filter-v5",
        token="hf_..."  # Only needed for private repos
    )
    result = scorer.score_article(article)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Suppress the "should TRAIN this model" warning
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
from peft import PeftModel
from huggingface_hub import hf_hub_download

# Import prefilter (handle hyphen in path via importlib)
import importlib.util
_prefilter_path = Path(__file__).parent / "prefilter.py"
_spec = importlib.util.spec_from_file_location("prefilter", _prefilter_path)
_prefilter_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prefilter_module)
InvestmentRiskPreFilterV5 = _prefilter_module.InvestmentRiskPreFilterV5


class InvestmentRiskScorerHub:
    """
    Scorer that loads model from HuggingFace Hub.
    """

    FILTER_NAME = "investment-risk"
    FILTER_VERSION = "5.0"

    DIMENSION_NAMES = [
        "risk_domain_type",
        "severity_magnitude",
        "materialization_timeline",
        "evidence_quality",
        "impact_breadth",
        "retail_actionability",
    ]

    DIMENSION_WEIGHTS = {
        "risk_domain_type": 0.20,
        "severity_magnitude": 0.25,
        "materialization_timeline": 0.15,
        "evidence_quality": 0.15,
        "impact_breadth": 0.15,
        "retail_actionability": 0.10,
    }

    # Gatekeeper: Evidence < 4 caps overall at 3.0
    EVIDENCE_GATEKEEPER_MIN = 4.0
    EVIDENCE_GATEKEEPER_CAP = 3.0

    def __init__(
        self,
        repo_id: str = "your-username/investment-risk-filter-v5",
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
            self.prefilter = InvestmentRiskPreFilterV5()

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

        # Load tokenizer from base model
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

    def _assign_signal_tier(self, scores: Dict[str, float], weighted_avg: float) -> tuple:
        """
        Assign signal tier based on dimension scores.

        Returns:
            (tier_name, tier_description)
        """
        risk_domain = scores["risk_domain_type"]
        severity = scores["severity_magnitude"]
        timeline = scores["materialization_timeline"]
        evidence = scores["evidence_quality"]
        breadth = scores["impact_breadth"]
        actionability = scores["retail_actionability"]

        # RED: Act now - reduce risk immediately
        if (risk_domain >= 7 and severity >= 7 and timeline >= 7 and evidence >= 5):
            return ("RED", "Act now - reduce risk immediately")

        # YELLOW: Monitor closely - prepare for defense
        if ((severity >= 5 or risk_domain >= 6) and evidence >= 4 and timeline >= 5):
            return ("YELLOW", "Monitor closely - prepare for defense")

        # GREEN: Counter-cyclical opportunity
        if (severity >= 6 and timeline <= 4 and evidence >= 6 and actionability >= 5):
            return ("GREEN", "Consider accumulating - opportunity emerging")

        # BLUE: Educational - understand but no action
        if (evidence >= 7 and actionability <= 3):
            return ("BLUE", "Educational - understand but no action")

        # NOISE: Ignore - not investment-relevant
        if (risk_domain <= 3 or evidence < 3):
            return ("NOISE", "Ignore - not investment-relevant")

        # Default to NOISE if no tier matched
        return ("NOISE", "No clear signal")

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

        # Gatekeeper: Evidence < 4 caps at 3.0
        if scores["evidence_quality"] < self.EVIDENCE_GATEKEEPER_MIN:
            if weighted_avg > self.EVIDENCE_GATEKEEPER_CAP:
                weighted_avg = self.EVIDENCE_GATEKEEPER_CAP
                result["gatekeeper_applied"] = True

        result["weighted_average"] = weighted_avg

        # Tier
        tier, tier_desc = self._assign_signal_tier(scores, weighted_avg)
        result["tier"] = tier
        result["tier_description"] = tier_desc

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


def main():
    """Demo loading from HuggingFace Hub."""
    import os

    # Get token from environment or secrets
    token = os.environ.get("HF_TOKEN")
    if not token:
        # Try loading from secrets.ini
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read("config/credentials/secrets.ini")
            token = config.get("api_keys", "huggingface_token", fallback=None)
        except Exception:
            pass

    print("Loading investment-risk scorer from HuggingFace Hub...")
    scorer = InvestmentRiskScorerHub(
        repo_id="your-username/investment-risk-filter-v5",
        token=token,
    )

    # Demo article
    demo_article = {
        "title": "Fed Signals Rate Cuts May Come Later Than Expected Amid Sticky Inflation",
        "content": """
        Federal Reserve officials indicated Wednesday that interest rate cuts
        could be delayed until later in 2025 as inflation remains stubbornly
        above the central bank's 2% target. The latest FOMC minutes revealed
        concerns about persistent price pressures in services and housing.

        Markets had been pricing in cuts as early as March, but traders are
        now adjusting expectations. The S&P 500 fell 1.2% following the release.
        Bond yields rose as investors recalibrated their outlook.

        For retail investors, this suggests maintaining a defensive posture
        and avoiding long-duration bonds until the path becomes clearer.
        """
    }

    print(f"\nScoring demo article: {demo_article['title']}")
    result = scorer.score_article(demo_article)

    print(f"\nResults:")
    print(f"  Passed prefilter: {result['passed_prefilter']}")
    if result['scores']:
        print(f"  Scores:")
        for dim, score in result['scores'].items():
            print(f"    {dim}: {score:.2f}")
        print(f"  Weighted average: {result['weighted_average']:.2f}")
        print(f"  Signal tier: {result['tier']} ({result['tier_description']})")


if __name__ == "__main__":
    main()
