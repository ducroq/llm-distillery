"""
Sustainability Technology v3 - HuggingFace Hub Inference

Loads the model directly from HuggingFace Hub for inference.
Use this when you don't have local model files or want to use a shared model.

Usage:
    from filters.sustainability_technology.v3.inference_hub import SustainabilityTechnologyScorerHub

    scorer = SustainabilityTechnologyScorerHub(
        repo_id="jeergrvgreg/sustainability-technology-v3",
        token="hf_..."  # Only needed for private repos
    )
    result = scorer.score_article(article)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch

from filters.common.model_loading import load_lora_hub
from filters.sustainability_technology.v3.base_scorer import BaseSustainabilityTechnologyScorer

logger = logging.getLogger(__name__)


class SustainabilityTechnologyScorerHub(BaseSustainabilityTechnologyScorer):
    """
    Scorer that loads model from HuggingFace Hub.

    Inherits all scoring logic from BaseSustainabilityTechnologyScorer.
    Only implements Hub-specific model loading.

    For loading from local files, use SustainabilityTechnologyScorer instead.
    """

    def __init__(
        self,
        repo_id: str = "jeergrvgreg/sustainability-technology-v3",
        token: Optional[str] = None,
        device: Optional[str] = None,
        use_prefilter: bool = True,
        torch_dtype=None,
    ):
        self.repo_id = repo_id
        self.token = token
        self.torch_dtype = torch_dtype

        # Initialize base class (sets device, loads prefilter, loads calibration)
        super().__init__(device=device, use_prefilter=use_prefilter)

        # Load the model from Hub
        self._load_model()

    def _load_model(self):
        """Load model from HuggingFace Hub."""
        self.model, self.tokenizer = load_lora_hub(
            self.repo_id, len(self.DIMENSION_NAMES), self.device,
            token=self.token, torch_dtype=self.torch_dtype,
        )


def main():
    """Demo loading from HuggingFace Hub."""
    import os

    # Get token from environment or secrets
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read("config/credentials/secrets.ini")
            token = config.get("api_keys", "huggingface_token", fallback=None)
        except Exception:
            pass

    print("Loading sustainability_technology scorer from HuggingFace Hub...")
    scorer = SustainabilityTechnologyScorerHub(token=token)

    # Demo article
    demo_article = {
        "title": "New Solar Panel Technology Achieves 30% Efficiency",
        "content": """
        Researchers at MIT have developed a new perovskite-silicon tandem solar cell
        that achieves 30% efficiency, surpassing traditional silicon panels. The technology
        uses abundant materials and can be manufactured at scale using existing equipment.
        Early pilot deployments show promising results with 25-year durability projections.
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
        print(f"  Tier: {result['tier']}")


if __name__ == "__main__":
    main()
