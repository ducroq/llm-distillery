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

import json
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
    - Tier assignment (high_sustainability/medium_high/medium/low)

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
    """CLI interface for batch scoring."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Score articles with sustainability_technology filter v3"
    )
    parser.add_argument(
        "--input", "-i", type=Path, help="Input JSONL file with articles"
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Output JSONL file for results"
    )
    parser.add_argument(
        "--no-prefilter", action="store_true", help="Skip prefilter"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for inference"
    )

    args = parser.parse_args()

    # Initialize scorer
    logging.basicConfig(level=logging.INFO)
    scorer = SustainabilityTechnologyScorer(use_prefilter=not args.no_prefilter)

    if args.input:
        # Score from file
        logger.info(f"Loading articles from {args.input}")
        articles = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))

        logger.info(f"Scoring {len(articles)} articles...")
        results = scorer.score_batch(articles, batch_size=args.batch_size)

        # Output
        if args.output:
            logger.info(f"Writing results to {args.output}")
            with open(args.output, "w", encoding="utf-8") as f:
                for article, result in zip(articles, results):
                    article_id = article.get("id") or article.get("article_id", "")
                    output = {"article_id": article_id, **result}
                    f.write(json.dumps(output) + "\n")
        else:
            # Print summary
            passed = sum(1 for r in results if r["passed_prefilter"])
            print(f"\nResults: {passed}/{len(results)} passed prefilter")

            if passed > 0:
                tiers = {}
                for r in results:
                    if r["tier"]:
                        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
                print("Tier distribution:")
                for tier, count in sorted(tiers.items()):
                    print(f"  {tier}: {count}")
    else:
        # Interactive demo
        print("\n--- Interactive Demo ---")
        demo_article = {
            "title": "New Solar Panel Technology Achieves 30% Efficiency",
            "content": """
            Researchers at MIT have developed a new perovskite-silicon tandem solar cell
            that achieves 30% efficiency, surpassing traditional silicon panels. The technology
            uses abundant materials and can be manufactured at scale using existing equipment.
            Early pilot deployments show promising results with 25-year durability projections.
            The cells are expected to reach cost parity with conventional panels by 2026.
            """
        }

        print(f"\nDemo article: {demo_article['title']}")
        result = scorer.score_article(demo_article)

        print(f"\nResults:")
        print(f"  Passed prefilter: {result['passed_prefilter']}")
        if result["scores"]:
            print(f"  Scores:")
            for dim, score in result["scores"].items():
                print(f"    {dim}: {score:.2f}")
            print(f"  Weighted average: {result['weighted_average']:.2f}")
            print(f"  Tier: {result['tier']} ({result['tier_description']})")
            if result["gatekeeper_applied"]:
                print(f"  Note: TRL gatekeeper applied")


if __name__ == "__main__":
    main()
