"""
Cultural Discovery Filter v4 - HuggingFace Hub Inference

Loads the model directly from HuggingFace Hub for inference.
Use this when you don't have local model files or want to use a shared model.

Usage:
    from importlib import import_module
    mod = import_module("filters.cultural-discovery.v4.inference_hub")
    scorer = mod.CulturalDiscoveryScorerHub(
        repo_id="jeergrvgreg/cultural-discovery-v4",
        token="hf_..."  # Only needed for private repos
    )
    result = scorer.score_article(article)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

# Suppress the "should TRAIN this model" warning
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
from peft import PeftModel
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError, GatedRepoError

from filters.common.model_loading import load_base_model_for_seq_cls

# Import base class (handle hyphen in path via importlib)
import importlib.util
_base_path = Path(__file__).parent / "base_scorer.py"
_spec = importlib.util.spec_from_file_location("base_scorer", _base_path)
_base_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base_module)
BaseCulturalDiscoveryScorer = _base_module.BaseCulturalDiscoveryScorer

logger = logging.getLogger(__name__)


class CulturalDiscoveryScorerHub(BaseCulturalDiscoveryScorer):
    """
    Scorer that loads model from HuggingFace Hub.

    Inherits all scoring logic from BaseCulturalDiscoveryScorer.
    Only implements Hub-specific model loading.

    For loading from local files, use CulturalDiscoveryScorer instead.
    """

    def __init__(
        self,
        repo_id: str = "jeergrvgreg/cultural-discovery-v4",
        token: Optional[str] = None,
        device: Optional[str] = None,
        use_prefilter: bool = True,
        torch_dtype=None,
    ):
        """
        Initialize scorer from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID (e.g., "username/model-name")
            token: HuggingFace token (required for private repos)
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_prefilter: Whether to apply prefilter
            torch_dtype: Model dtype (e.g., torch.float16). If None, uses model
                default. Use torch.float16 as workaround on hardware without
                bfloat16 support.
        """
        self.repo_id = repo_id
        self.token = token
        self.torch_dtype = torch_dtype

        # Initialize base class (sets device, loads prefilter)
        super().__init__(device=device, use_prefilter=use_prefilter)

        # Load the model from Hub
        self._load_model()

    def _load_model(self):
        """
        Load model from HuggingFace Hub.

        Raises:
            RuntimeError: If model loading fails
            RepositoryNotFoundError: If repo doesn't exist
        """
        try:
            logger.info(f"Loading model from HuggingFace Hub: {self.repo_id}")
            logger.info(f"Device: {self.device}")

            # Download and load adapter config
            config_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="adapter_config.json",
                token=self.token,
            )

            with open(config_path, "r") as f:
                adapter_config = json.load(f)

            base_model_name = adapter_config["base_model_name_or_path"]
            logger.info(f"Base model: {base_model_name}")

            # Load tokenizer from base model
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                token=self.token,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model (uses Gemma-3 compatible loader)
            logger.info("Loading base model...")
            base_model = load_base_model_for_seq_cls(
                base_model_name,
                num_labels=len(self.DIMENSION_NAMES),
                problem_type="regression",
                torch_dtype=self.torch_dtype,
            )

            if base_model.config.pad_token_id is None:
                base_model.config.pad_token_id = self.tokenizer.pad_token_id

            # Load PEFT model from hub
            logger.info("Loading LoRA adapter from Hub...")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.repo_id,
                token=self.token,
            )

            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info("Model loaded successfully")

        except (RepositoryNotFoundError, GatedRepoError):
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from Hub ({self.repo_id}): "
                f"{type(e).__name__}: {e}"
            )


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

    print("Loading cultural discovery scorer from HuggingFace Hub...")
    scorer = CulturalDiscoveryScorerHub(token=token)

    # Demo article
    demo_article = {
        "title": "Ancient Silk Road Temple Reveals Unexpected Buddhist-Zoroastrian Syncretism",
        "content": """
        Excavations at a 4th-century temple in Uzbekistan have uncovered evidence
        of an unprecedented religious synthesis. The site contains Buddhist statues
        with distinctly Zoroastrian fire altar iconography, suggesting practitioners
        of both faiths worshipped together during the height of Silk Road trade.
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
