"""
Foresight Filter v1 - HuggingFace Hub Inference

Loads the model directly from HuggingFace Hub for inference.
Use this when you don't have local model files or want to use a shared model.

Usage:
    from filters.foresight.v1.inference_hub import ForesightScorerHub

    scorer = ForesightScorerHub(
        repo_id="jeergrvgreg/foresight-filter-v1",
        token="hf_..."  # Only needed for private repos
    )
    result = scorer.score_article(article)
"""

import logging
from typing import Optional

from filters.common.model_loading import load_lora_hub
from filters.foresight.v1.base_scorer import BaseForesightScorer

logger = logging.getLogger(__name__)


class ForesightScorerHub(BaseForesightScorer):
    """
    Scorer that loads model from HuggingFace Hub.

    Inherits all scoring logic from BaseForesightScorer.
    Only implements Hub-specific model loading.

    For loading from local files, use ForesightScorer instead.
    """

    def __init__(
        self,
        repo_id: str = "jeergrvgreg/foresight-filter-v1",
        token: Optional[str] = None,
        device: Optional[str] = None,
        use_prefilter: bool = True,
        torch_dtype=None,
    ):
        self.repo_id = repo_id
        self.token = token
        self.torch_dtype = torch_dtype

        super().__init__(device=device, use_prefilter=use_prefilter)
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

    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read("config/credentials/secrets.ini")
            token = config.get("api_keys", "huggingface_token", fallback=None)
        except Exception:
            pass

    print("Loading foresight scorer from HuggingFace Hub...")
    scorer = ForesightScorerHub(token=token)

    demo_article = {
        "title": "New Zealand replaces GDP with wellbeing metrics in historic budget reform",
        "content": """
        New Zealand became the first country to require all government spending to be
        evaluated against wellbeing metrics rather than GDP growth. Finance Minister
        Grant Robertson acknowledged that 'GDP never captured what matters' and committed
        to measuring success across 12 domains including mental health, child poverty,
        and environmental sustainability over a 30-year horizon.
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
