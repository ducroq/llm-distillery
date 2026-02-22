"""
Investment Risk Filter v6 - Production Inference Pipeline

This module provides the complete inference pipeline for scoring articles
using the trained investment risk filter with local model files.

Pipeline: Article -> Prefilter -> Model -> Calibration -> Gatekeeper -> Tier

Usage:
    # Python API
    from importlib import import_module
    mod = import_module("filters.investment-risk.v6.inference")
    scorer = mod.InvestmentRiskScorer()
    result = scorer.score_article(article)

    # CLI
    python filters/investment-risk/v6/inference.py --input articles.jsonl --output results.jsonl
"""

import json
import logging
from pathlib import Path
from typing import Optional

from filters.common.model_loading import load_lora_local

# Import base class (handle hyphen in path via importlib)
import importlib.util
_base_path = Path(__file__).parent / "base_scorer.py"
_spec = importlib.util.spec_from_file_location("base_scorer", _base_path)
_base_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base_module)
BaseInvestmentRiskScorer = _base_module.BaseInvestmentRiskScorer

logger = logging.getLogger(__name__)


class InvestmentRiskScorer(BaseInvestmentRiskScorer):
    """
    Production scorer for investment risk filter v6.

    Loads the trained LoRA model from local files and provides scoring with:
    - Optional prefiltering for efficiency
    - Per-dimension scores (6 orthogonal dimensions)
    - Score calibration (isotonic regression)
    - Evidence gatekeeper logic
    - Tier assignment (high/medium_high/medium/low)

    For loading from HuggingFace Hub, use InvestmentRiskScorerHub instead.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        use_prefilter: bool = True,
    ):
        """
        Initialize the scorer with local model files.

        Args:
            model_path: Path to model directory (default: ./model)
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_prefilter: Whether to apply prefilter before scoring
        """
        # Set model path before calling super().__init__
        if model_path is None:
            model_path = Path(__file__).parent / "model"
        self.model_path = Path(model_path)

        # Initialize base class (sets device, loads prefilter)
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
        description="Score articles with investment risk filter v6"
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
    scorer = InvestmentRiskScorer(use_prefilter=not args.no_prefilter)

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
                print(f"  Note: Evidence gatekeeper applied (speculation capped)")


if __name__ == "__main__":
    main()
