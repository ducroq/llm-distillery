"""
Foresight Filter v1 - Production Inference Pipeline

Scores articles for evidence of foresighted decision-making: long-term thinking,
systems awareness, course correction, intergenerational investment.

Pipeline: Article -> Prefilter -> Model -> Calibration -> Gatekeeper -> Tier

Usage:
    # Python API
    from filters.foresight.v1.inference import ForesightScorer
    scorer = ForesightScorer()
    result = scorer.score_article(article)

    # CLI
    python filters/foresight/v1/inference.py --input articles.jsonl --output results.jsonl
"""

import json
import logging
from pathlib import Path
from typing import Optional

from filters.common.model_loading import load_lora_local
from filters.foresight.v1.base_scorer import BaseForesightScorer

logger = logging.getLogger(__name__)


class ForesightScorer(BaseForesightScorer):
    """
    Production scorer for foresight filter v1.

    Loads the trained LoRA model from local files and provides scoring with:
    - Optional prefiltering for efficiency
    - Per-dimension scores (6 dimensions)
    - Score calibration (isotonic regression)
    - Evidence foundation gatekeeper logic
    - Tier assignment (high/medium/low)

    For loading from HuggingFace Hub, use ForesightScorerHub instead.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        use_prefilter: bool = True,
    ):
        if model_path is None:
            model_path = Path(__file__).parent / "model" / "model"
        self.model_path = Path(model_path)

        super().__init__(device=device, use_prefilter=use_prefilter)
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
        description="Score articles with foresight filter v1"
    )
    parser.add_argument("--input", "-i", type=Path, help="Input JSONL file with articles")
    parser.add_argument("--output", "-o", type=Path, help="Output JSONL file for results")
    parser.add_argument("--no-prefilter", action="store_true", help="Skip prefilter")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    scorer = ForesightScorer(use_prefilter=not args.no_prefilter)

    if args.input:
        articles = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))

        logger.info(f"Scoring {len(articles)} articles...")
        results = scorer.score_batch(articles, batch_size=args.batch_size)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for article, result in zip(articles, results):
                    article_id = article.get("id") or article.get("article_id", "")
                    output = {"article_id": article_id, **result}
                    f.write(json.dumps(output) + "\n")
        else:
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
        print("\n--- Interactive Demo ---")
        demo_article = {
            "title": "New Zealand replaces GDP with wellbeing metrics in historic budget reform",
            "content": """
            New Zealand became the first country to require all government spending to be
            evaluated against wellbeing metrics rather than GDP growth. Finance Minister
            Grant Robertson acknowledged that 'GDP never captured what matters' and committed
            to measuring success across 12 domains including mental health, child poverty,
            and environmental sustainability over a 30-year horizon. The reform embeds
            wellbeing measurement into the Treasury's core functions, ensuring it outlasts
            any single government. Critics note the transition will be difficult and that
            measurement methodologies are still evolving, but cross-party support suggests
            the reform will survive changes in government.
            """
        }

        print(f"\nDemo article: {demo_article['title']}")
        result = scorer.score_article(demo_article)
        print(f"\nResults:")
        print(f"  Passed prefilter: {result['passed_prefilter']}")
        if result["scores"]:
            for dim, score in result["scores"].items():
                print(f"    {dim}: {score:.2f}")
            print(f"  Weighted average: {result['weighted_average']:.2f}")
            print(f"  Tier: {result['tier']}")
            if result.get("gatekeeper_applied"):
                print(f"  Note: Evidence foundation gatekeeper applied")


if __name__ == "__main__":
    main()
