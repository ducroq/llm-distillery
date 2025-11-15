"""
Post-filter: Convert dimensional scores to tier classification.

This module takes dimensional scores from student models and applies
filter-specific logic to assign tier classifications.

Architecture:
  1. Student model outputs: [dim1_score, dim2_score, ..., dim8_score]
  2. Post-filter calculates weighted average
  3. Post-filter applies gatekeeper rules / content caps
  4. Post-filter assigns tier based on thresholds
  5. Post-filter flags top articles for reasoning (optional)

Usage:
    from scripts.postfilter import PostFilter

    pf = PostFilter("filters/sustainability_tech_deployment/v1")
    result = pf.classify({
        "deployment_maturity": 7.2,
        "technology_performance": 6.8,
        ...
    })

    print(result)
    # {
    #   "tier": "commercial_proven",
    #   "overall_score": 7.1,
    #   "needs_reasoning": False,
    #   "applied_rules": ["deployment_maturity gatekeeper passed"]
    # }
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml


class PostFilter:
    """Generic post-filter for tier classification from dimensional scores."""

    def __init__(self, filter_path: str):
        """
        Initialize post-filter with filter configuration.

        Args:
            filter_path: Path to filter directory (e.g., "filters/uplifting/v1")
        """
        self.filter_path = Path(filter_path)
        self.config = self._load_config()
        self.dimensions = self.config["scoring"]["dimensions"]
        self.tiers = self.config["scoring"]["tiers"]
        self.gatekeeper_rules = self.config["scoring"].get("gatekeeper_rules", {})
        self.content_type_caps = self.config["scoring"].get("content_type_caps", {})

        # Extract dimension names and weights
        self.dimension_names = list(self.dimensions.keys())
        self.weights = {name: dim["weight"] for name, dim in self.dimensions.items()}

        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Dimension weights sum to {total_weight}, expected 1.0")

    def _load_config(self) -> Dict[str, Any]:
        """Load filter configuration from config.yaml."""
        config_path = self.filter_path / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted average of dimensional scores.

        Args:
            scores: Dictionary of {dimension_name: score}

        Returns:
            Weighted average score (0-10 scale)
        """
        # Validate all dimensions present
        missing = set(self.dimension_names) - set(scores.keys())
        if missing:
            raise ValueError(f"Missing dimensions: {missing}")

        # Calculate weighted average
        overall = sum(scores[name] * self.weights[name] for name in self.dimension_names)
        return overall

    def apply_gatekeeper_rules(
        self,
        scores: Dict[str, float],
        overall_score: float
    ) -> tuple[float, List[str]]:
        """
        Apply gatekeeper rules (hard limits based on critical dimensions).

        Example from sustainability_tech_deployment:
          - If deployment_maturity < 5.0, overall score capped at 4.9
          - If proof_of_impact < 4.0, overall score capped at 3.9

        Args:
            scores: Dimensional scores
            overall_score: Calculated overall score

        Returns:
            (adjusted_score, list of applied rules)
        """
        adjusted_score = overall_score
        applied_rules = []

        for dimension_name, rule in self.gatekeeper_rules.items():
            threshold = rule["threshold"]
            max_overall = rule["max_overall_if_below"]

            if scores[dimension_name] < threshold:
                if adjusted_score > max_overall:
                    applied_rules.append(
                        f"Gatekeeper: {dimension_name} < {threshold} → "
                        f"overall capped at {max_overall}"
                    )
                    adjusted_score = max_overall

        if not applied_rules:
            applied_rules.append("No gatekeeper rules triggered")

        return adjusted_score, applied_rules

    def apply_content_type_caps(
        self,
        scores: Dict[str, float],
        overall_score: float,
        article_text: Optional[str] = None
    ) -> tuple[float, List[str]]:
        """
        Apply content type caps (mathematical conditions only).

        **NOTE (2025-11-13):** Content type caps are now enforced in oracle prompts,
        not in post-filter. See: docs/decisions/2025-11-13-content-caps-in-oracle-not-postfilter.md

        This method only applies condition-based caps (mathematical, e.g. "collective_benefit < 6").
        Semantic content caps (corporate_finance, military_security) are handled in oracle.

        Rationale:
        - Oracle understands semantic context better than keyword/tag matching
        - Avoids false positives (e.g., worker cooperative tagged as "business")
        - Keeps post-filter simple (pure arithmetic)

        Args:
            scores: Dimensional scores
            overall_score: Calculated overall score
            article_text: Article text (not currently used)

        Returns:
            (adjusted_score, list of applied caps)
        """
        adjusted_score = overall_score
        applied_caps = []

        # Only apply condition-based caps (mathematical conditions on dimension scores)
        # Semantic content detection (triggers/exceptions) handled in oracle prompt

        for cap_name, cap_config in self.content_type_caps.items():
            max_score = cap_config["max_score"]

            # Check if condition-based cap applies
            condition = cap_config.get("condition")
            if condition:
                # Parse condition like "collective_benefit < 6"
                # Simple parsing: "dimension_name < threshold"
                parts = condition.split("<")
                if len(parts) == 2:
                    dim_name = parts[0].strip()
                    threshold = float(parts[1].strip())

                    if dim_name in scores and scores[dim_name] < threshold:
                        if adjusted_score > max_score:
                            applied_caps.append(
                                f"Content cap ({cap_name}): {dim_name} < {threshold} → "
                                f"overall capped at {max_score}"
                            )
                            adjusted_score = max_score

        if not applied_caps:
            applied_caps.append("No content type caps triggered")

        return adjusted_score, applied_caps

    def assign_tier(self, overall_score: float) -> str:
        """
        Assign tier based on overall score and tier thresholds.

        Tiers are sorted by threshold (descending) and the first matching tier is returned.

        Args:
            overall_score: Final overall score after gatekeeper rules

        Returns:
            Tier name (e.g., "commercial_proven", "impact", etc.)
        """
        # Sort tiers by threshold descending
        sorted_tiers = sorted(
            self.tiers.items(),
            key=lambda x: x[1]["threshold"],
            reverse=True
        )

        # Find first tier where score >= threshold
        for tier_name, tier_config in sorted_tiers:
            if overall_score >= tier_config["threshold"]:
                return tier_name

        # Should never reach here if lowest tier has threshold 0.0
        # Return lowest tier as fallback
        return sorted_tiers[-1][0]

    def classify(
        self,
        scores: Dict[str, float],
        article_text: Optional[str] = None,
        flag_reasoning_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Full classification pipeline: scores → tier.

        Args:
            scores: Dimensional scores from student model
            article_text: Article text for content caps (optional)
            flag_reasoning_threshold: Score threshold for flagging articles
                                     that need reasoning (e.g., 7.0 for top tier)

        Returns:
            Classification result with tier, score, and metadata
        """
        # Step 1: Calculate weighted average
        overall_score = self.calculate_overall_score(scores)

        # Step 2: Apply gatekeeper rules or content caps
        applied_rules = []
        if self.gatekeeper_rules:
            overall_score, rules = self.apply_gatekeeper_rules(scores, overall_score)
            applied_rules.extend(rules)

        if self.content_type_caps:
            overall_score, caps = self.apply_content_type_caps(scores, overall_score, article_text)
            applied_rules.extend(caps)

        # Step 3: Assign tier
        tier = self.assign_tier(overall_score)

        # Step 4: Flag for reasoning if in top tier(s)
        needs_reasoning = False
        if flag_reasoning_threshold is not None:
            needs_reasoning = overall_score >= flag_reasoning_threshold

        return {
            "tier": tier,
            "overall_score": round(overall_score, 2),
            "dimensional_scores": scores,
            "needs_reasoning": needs_reasoning,
            "applied_rules": applied_rules,
            "tier_description": self.tiers[tier]["description"]
        }


def main():
    """Demo usage of PostFilter."""
    import argparse

    parser = argparse.ArgumentParser(description="Post-filter: dimensional scores → tier")
    parser.add_argument("--filter", required=True, help="Filter directory path")
    parser.add_argument("--scores", required=True, help="Dimensional scores as JSON string")
    parser.add_argument("--flag-reasoning-threshold", type=float, default=7.0,
                       help="Score threshold for flagging articles needing reasoning")

    args = parser.parse_args()

    import json
    scores = json.loads(args.scores)

    pf = PostFilter(args.filter)
    result = pf.classify(scores, flag_reasoning_threshold=args.flag_reasoning_threshold)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
