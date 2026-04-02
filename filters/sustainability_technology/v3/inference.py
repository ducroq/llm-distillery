"""
Sustainability Technology v3 - Production Inference Pipeline

This module provides the complete inference pipeline for scoring articles
using the trained sustainability_technology filter with local model files.

Pipeline: Article -> Prefilter -> Model -> Calibration -> Gatekeeper -> Tier

Usage:
    # Python API
    from filters.sustainability_technology.v3.inference import SustainabilityTechnologyScorer

    scorer = SustainabilityTechnologyScorer()
    result = scorer.score_article(article)

    # CLI
    python filters/sustainability_technology/v3/inference.py --input articles.jsonl --output results.jsonl
"""

import logging
from pathlib import Path
from typing import Optional

from filters.common.model_loading import load_lora_local
from filters.sustainability_technology.v3.base_scorer import BaseSustainabilityTechnologyScorer

logger = logging.getLogger(__name__)


class SustainabilityTechnologyScorer(BaseSustainabilityTechnologyScorer):
    """
    Production scorer for sustainability_technology filter v3.

    Loads the trained LoRA model from local files and provides scoring with:
    - Optional prefiltering for efficiency
    - Per-dimension scores (6 LCSA dimensions)
    - Score calibration (isotonic regression)
    - TRL gatekeeper logic
    - Tier assignment (high/medium/low)

    For loading from HuggingFace Hub, use SustainabilityTechnologyScorerHub instead.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        use_prefilter: bool = True,
    ):
        # Set model path before calling super().__init__
        if model_path is None:
            model_path = Path(__file__).parent / "model"
        self.model_path = Path(model_path)

        # Initialize base class (sets device, loads prefilter, loads calibration)
        super().__init__(device=device, use_prefilter=use_prefilter)

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the trained LoRA model from local files."""
        self.model, self.tokenizer = load_lora_local(
            self.model_path, len(self.DIMENSION_NAMES), self.device
        )


def main():
    from filters.common.cli import run_scorer_cli
    run_scorer_cli(
        SustainabilityTechnologyScorer,
        "sustainability_technology filter v3",
        {
            "title": "New Solar Panel Technology Achieves 30% Efficiency",
            "content": """
            Researchers at MIT have developed a new perovskite-silicon tandem solar cell
            that achieves 30% efficiency, surpassing traditional silicon panels. The technology
            uses abundant materials and can be manufactured at scale using existing equipment.
            Early pilot deployments show promising results with 25-year durability projections.
            The cells are expected to reach cost parity with conventional panels by 2026.
            """
        },
    )


if __name__ == "__main__":
    main()
