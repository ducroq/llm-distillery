"""
Investment Risk Filter v5 - Production Inference Pipeline

This module provides the complete inference pipeline for scoring articles
using the trained investment risk filter.

Pipeline: Article -> Prefilter -> Model -> Gatekeeper -> Signal Tier

Usage:
    from filters.investment_risk.v5.inference import InvestmentRiskScorer

    scorer = InvestmentRiskScorer()
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
from peft import PeftConfig, get_peft_model
from safetensors.torch import load_file

# Import prefilter (handle hyphen in path via importlib)
import importlib.util
_prefilter_path = Path(__file__).parent / "prefilter.py"
_spec = importlib.util.spec_from_file_location("prefilter", _prefilter_path)
_prefilter_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prefilter_module)
InvestmentRiskPreFilterV5 = _prefilter_module.InvestmentRiskPreFilterV5


class InvestmentRiskScorer:
    """
    Production scorer for investment risk filter v5.

    Loads the trained LoRA model and provides scoring with:
    - Optional prefiltering for efficiency
    - Per-dimension scores (6 orthogonal dimensions)
    - Evidence gatekeeper logic
    - Signal tier assignment (RED/YELLOW/GREEN/BLUE/NOISE)
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
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        use_prefilter: bool = True,
    ):
        """
        Initialize the scorer.

        Args:
            model_path: Path to model directory (default: auto-detect)
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_prefilter: Whether to apply prefilter before scoring
        """
        if model_path is None:
            model_path = Path(__file__).parent / "model"
        self.model_path = Path(model_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_prefilter = use_prefilter
        if use_prefilter:
            self.prefilter = InvestmentRiskPreFilterV5()

        self._load_model()

    def _load_model(self):
        """Load the trained LoRA model."""
        print(f"Loading model from {self.model_path}")
        print(f"Device: {self.device}")

        # Load PEFT config
        peft_config = PeftConfig.from_pretrained(str(self.model_path))
        base_model_name = peft_config.base_model_name_or_path

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(self.DIMENSION_NAMES),
            problem_type="regression",
        )

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Create PEFT model
        self.model = get_peft_model(base_model, peft_config)

        # Load adapter weights
        adapter_path = self.model_path / "adapter_model.safetensors"
        adapter_state_dict = load_file(str(adapter_path))

        # Remap keys for PEFT compatibility
        remapped = {}
        for key, value in adapter_state_dict.items():
            if ".lora_A.weight" in key or ".lora_B.weight" in key:
                new_key = key.replace(".lora_A.weight", ".lora_A.default.weight")
                new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
                remapped[new_key] = value
            elif key == "base_model.model.score.weight":
                remapped["base_model.model.score.modules_to_save.default.weight"] = value
            elif key == "base_model.model.score.bias":
                remapped["base_model.model.score.modules_to_save.default.bias"] = value
            else:
                remapped[key] = value

        self.model.load_state_dict(remapped, strict=False)
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
            skip_prefilter: Force skip prefilter even if enabled

        Returns:
            Dict with:
                - passed_prefilter: bool
                - prefilter_reason: str (if blocked)
                - scores: Dict[dimension_name, float] (if passed)
                - weighted_average: float (if passed)
                - tier: str (if passed)
                - tier_description: str (if passed)
                - gatekeeper_applied: bool
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

        # Apply prefilter
        if self.use_prefilter and not skip_prefilter:
            passed, reason = self.prefilter.apply_filter(article)
            if not passed:
                result["passed_prefilter"] = False
                result["prefilter_reason"] = reason
                return result

        # Prepare input
        text = f"{article['title']}\n\n{article['content']}"
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_scores = outputs.logits[0].cpu().numpy()

        # Clamp scores to 0-10 range
        scores = {
            dim: float(max(0.0, min(10.0, raw_scores[i])))
            for i, dim in enumerate(self.DIMENSION_NAMES)
        }
        result["scores"] = scores

        # Compute weighted average
        weighted_avg = sum(
            scores[dim] * self.DIMENSION_WEIGHTS[dim]
            for dim in self.DIMENSION_NAMES
        )

        # Apply gatekeeper: Evidence < 4 caps overall at 3.0
        if scores["evidence_quality"] < self.EVIDENCE_GATEKEEPER_MIN:
            if weighted_avg > self.EVIDENCE_GATEKEEPER_CAP:
                weighted_avg = self.EVIDENCE_GATEKEEPER_CAP
                result["gatekeeper_applied"] = True

        result["weighted_average"] = weighted_avg

        # Assign signal tier
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
        """
        Score a batch of articles efficiently.

        Args:
            articles: List of article dicts
            batch_size: Batch size for inference
            skip_prefilter: Skip prefilter for all articles

        Returns:
            List of result dicts (same as score_article)
        """
        results = []

        # First pass: prefilter
        articles_to_score = []
        article_indices = []

        for i, article in enumerate(articles):
            result = {
                "passed_prefilter": True,
                "prefilter_reason": None,
                "scores": None,
                "weighted_average": None,
                "tier": None,
                "tier_description": None,
                "gatekeeper_applied": False,
            }

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

        # Second pass: batch inference
        if articles_to_score:
            for batch_start in range(0, len(articles_to_score), batch_size):
                batch_end = min(batch_start + batch_size, len(articles_to_score))
                batch = articles_to_score[batch_start:batch_end]
                batch_indices = article_indices[batch_start:batch_end]

                # Tokenize batch
                texts = [f"{a['title']}\n\n{a['content']}" for a in batch]
                inputs = self.tokenizer(
                    texts,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_scores = outputs.logits.cpu().numpy()

                # Process each result
                for j, idx in enumerate(batch_indices):
                    raw_scores = batch_scores[j]

                    scores = {
                        dim: float(max(0.0, min(10.0, raw_scores[k])))
                        for k, dim in enumerate(self.DIMENSION_NAMES)
                    }
                    results[idx]["scores"] = scores

                    # Weighted average
                    weighted_avg = sum(
                        scores[dim] * self.DIMENSION_WEIGHTS[dim]
                        for dim in self.DIMENSION_NAMES
                    )

                    # Gatekeeper
                    if scores["evidence_quality"] < self.EVIDENCE_GATEKEEPER_MIN:
                        if weighted_avg > self.EVIDENCE_GATEKEEPER_CAP:
                            weighted_avg = self.EVIDENCE_GATEKEEPER_CAP
                            results[idx]["gatekeeper_applied"] = True

                    results[idx]["weighted_average"] = weighted_avg

                    # Tier
                    tier, tier_desc = self._assign_signal_tier(scores, weighted_avg)
                    results[idx]["tier"] = tier
                    results[idx]["tier_description"] = tier_desc

        return results


def main():
    """Example usage / CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Score articles with investment risk filter v5")
    parser.add_argument("--input", "-i", type=Path, help="Input JSONL file with articles")
    parser.add_argument("--output", "-o", type=Path, help="Output JSONL file for results")
    parser.add_argument("--no-prefilter", action="store_true", help="Skip prefilter")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")

    args = parser.parse_args()

    # Initialize scorer
    print("Initializing scorer...")
    scorer = InvestmentRiskScorer(use_prefilter=not args.no_prefilter)

    if args.input:
        # Score from file
        print(f"Loading articles from {args.input}")
        articles = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))

        print(f"Scoring {len(articles)} articles...")
        results = scorer.score_batch(articles, batch_size=args.batch_size)

        # Output
        if args.output:
            print(f"Writing results to {args.output}")
            with open(args.output, "w", encoding="utf-8") as f:
                for article, result in zip(articles, results):
                    output = {"article_id": article.get("id", ""), **result}
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
                print("Signal tier distribution:")
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
        if result['scores']:
            print(f"  Scores:")
            for dim, score in result['scores'].items():
                print(f"    {dim}: {score:.2f}")
            print(f"  Weighted average: {result['weighted_average']:.2f}")
            print(f"  Signal tier: {result['tier']} ({result['tier_description']})")
            if result['gatekeeper_applied']:
                print(f"  Note: Evidence gatekeeper applied (speculation capped)")


if __name__ == "__main__":
    main()
