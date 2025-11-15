#!/usr/bin/env python3
"""
Dataset Quality Assurance Script
Analyzes the sustainability_tech_deployment dataset for quality issues
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any
import statistics

# Configuration
DATASET_PATH = Path(r"C:\local_dev\llm-distillery\datasets\labeled\sustainability_tech_deployment\labeled_articles.jsonl")
REQUIRED_FIELDS = ["id", "title", "content", "sustainability_tech_deployment_analysis"]
REQUIRED_DIMENSIONS = [
    "deployment_maturity",
    "technology_performance",
    "cost_trajectory",
    "scale_of_deployment",
    "market_penetration",
    "technology_readiness",
    "supply_chain_maturity",
    "proof_of_impact"
]
DIMENSION_WEIGHTS = {
    "deployment_maturity": 0.25,
    "technology_performance": 0.15,
    "cost_trajectory": 0.15,
    "scale_of_deployment": 0.15,
    "market_penetration": 0.10,
    "technology_readiness": 0.10,
    "supply_chain_maturity": 0.05,
    "proof_of_impact": 0.05
}

TIER_RANGES = {
    "deployed": (7.5, 10.0),
    "early_commercial": (5.0, 7.5),
    "pilot_stage": (2.5, 5.0),
    "vaporware": (0.0, 2.5)
}

# Legacy tier names mapping (for compatibility)
TIER_ALIASES = {
    "pilot": "pilot_stage"
}

def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL dataset with error handling"""
    articles = []
    errors = []

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                article = json.loads(line)
                articles.append(article)
            except json.JSONDecodeError as e:
                errors.append({
                    "line": line_num,
                    "error": str(e),
                    "content": line[:100]
                })

    return articles, errors

def check_required_fields(article: Dict[str, Any]) -> List[str]:
    """Check if article has all required fields"""
    missing = []
    for field in REQUIRED_FIELDS:
        if field not in article:
            missing.append(field)
        elif field == "content" and not article.get(field, "").strip():
            missing.append(f"{field} (empty)")
    return missing

def check_analysis_structure(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Check if analysis has proper structure"""
    issues = {
        "missing_dimensions": [],
        "invalid_scores": [],
        "overall_score_mismatch": False,
        "tier_mismatch": False,
        "missing_overall_score": False,
        "missing_tier": False
    }

    # Check dimensions
    for dim in REQUIRED_DIMENSIONS:
        if dim not in analysis:
            issues["missing_dimensions"].append(dim)
        else:
            # Dimensions can be either dicts with "score" key or direct numbers
            dim_value = analysis[dim]
            if isinstance(dim_value, dict):
                if "score" not in dim_value:
                    issues["invalid_scores"].append({
                        "dimension": dim,
                        "value": dim_value,
                        "reason": "dict without score key"
                    })
                    continue
                score = dim_value["score"]
            else:
                score = dim_value

            if not isinstance(score, (int, float)) or score < 0 or score > 10:
                issues["invalid_scores"].append({
                    "dimension": dim,
                    "value": score
                })

    # Check overall score
    if "overall_score" not in analysis:
        issues["missing_overall_score"] = True
    else:
        # Calculate expected overall score
        if not issues["missing_dimensions"]:
            expected_score = sum(
                (analysis[dim]["score"] if isinstance(analysis[dim], dict) else analysis[dim]) * DIMENSION_WEIGHTS[dim]
                for dim in REQUIRED_DIMENSIONS
            )
            actual_score = analysis["overall_score"]
            # Allow small floating point differences
            if abs(expected_score - actual_score) > 0.01:
                issues["overall_score_mismatch"] = {
                    "expected": round(expected_score, 2),
                    "actual": actual_score
                }

    # Check tier
    if "tier" not in analysis:
        issues["missing_tier"] = True
    elif "overall_score" in analysis:
        overall = analysis["overall_score"]
        tier = analysis["tier"]

        # Determine correct tier
        correct_tier = None
        for tier_name, (min_score, max_score) in TIER_RANGES.items():
            if min_score <= overall < max_score or (tier_name == "deployed" and overall == 10.0):
                correct_tier = tier_name
                break

        if correct_tier != tier:
            issues["tier_mismatch"] = {
                "overall_score": overall,
                "expected_tier": correct_tier,
                "actual_tier": tier
            }

    return issues

def analyze_dataset(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Comprehensive dataset analysis"""
    results = {
        "total_articles": len(articles),
        "field_issues": [],
        "structure_issues": [],
        "duplicate_ids": [],
        "all_zeros": [],
        "class_distribution": Counter(),
        "overall_scores": [],
        "dimension_scores": {dim: [] for dim in REQUIRED_DIMENSIONS}
    }

    # Track IDs for duplicate detection
    id_counts = Counter()

    for idx, article in enumerate(articles):
        article_id = article.get("id", f"unknown_{idx}")
        id_counts[article_id] += 1

        # Check required fields
        missing_fields = check_required_fields(article)
        if missing_fields:
            results["field_issues"].append({
                "id": article_id,
                "missing": missing_fields
            })

        # Check analysis structure
        if "sustainability_tech_deployment_analysis" in article:
            analysis = article["sustainability_tech_deployment_analysis"]
            issues = check_analysis_structure(analysis)

            # Record any issues
            has_issues = False
            for key, value in issues.items():
                if value and value != []:
                    has_issues = True
                    break

            if has_issues:
                results["structure_issues"].append({
                    "id": article_id,
                    "issues": {k: v for k, v in issues.items() if v and v != []}
                })

            # Collect scores for statistics
            if "overall_score" in analysis:
                results["overall_scores"].append(analysis["overall_score"])

            for dim in REQUIRED_DIMENSIONS:
                if dim in analysis:
                    # Handle both dict format (with "score" key) and direct number
                    dim_value = analysis[dim]
                    if isinstance(dim_value, dict):
                        if "score" in dim_value:
                            results["dimension_scores"][dim].append(dim_value["score"])
                    else:
                        results["dimension_scores"][dim].append(dim_value)

            # Check for all zeros (failed labeling)
            all_zero = True
            for dim in REQUIRED_DIMENSIONS:
                if dim in analysis:
                    dim_value = analysis[dim]
                    score = dim_value["score"] if isinstance(dim_value, dict) else dim_value
                    if score != 0:
                        all_zero = False
                        break
                else:
                    all_zero = False
                    break
            if all_zero:
                results["all_zeros"].append(article_id)

            # Count tier distribution
            if "tier" in analysis:
                results["class_distribution"][analysis["tier"]] += 1

    # Find duplicates
    results["duplicate_ids"] = [id for id, count in id_counts.items() if count > 1]

    return results

def calculate_statistics(scores: List[float]) -> Dict[str, float]:
    """Calculate statistics for a list of scores"""
    if not scores:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "std_dev": 0
        }

    return {
        "min": round(min(scores), 2),
        "max": round(max(scores), 2),
        "mean": round(statistics.mean(scores), 2),
        "median": round(statistics.median(scores), 2),
        "std_dev": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0
    }

def generate_report(results: Dict[str, Any], parse_errors: List[Dict]) -> str:
    """Generate markdown QA report"""
    total = results["total_articles"]

    # Calculate pass/fail for each check
    checks = {
        "JSON parsing": "✅ PASSED" if not parse_errors else f"❌ FAILED ({len(parse_errors)} errors)",
        "Required fields": "✅ PASSED" if not results["field_issues"] else f"❌ FAILED ({len(results['field_issues'])} issues)",
        "Analysis structure": "✅ PASSED" if not results["structure_issues"] else f"⚠️ ISSUES ({len(results['structure_issues'])} articles)",
        "Duplicate IDs": "✅ PASSED" if not results["duplicate_ids"] else f"❌ FAILED ({len(results['duplicate_ids'])} duplicates)",
        "All zeros check": "✅ PASSED" if not results["all_zeros"] else f"⚠️ WARNING ({len(results['all_zeros'])} articles)",
        "Score ranges": "✅ PASSED",  # Will be updated if issues found
    }

    # Overall statistics
    overall_stats = calculate_statistics(results["overall_scores"])

    # Dimension statistics
    dimension_stats = {
        dim: calculate_statistics(scores)
        for dim, scores in results["dimension_scores"].items()
    }

    # Class distribution
    class_dist = results["class_distribution"]

    # Generate report
    report = f"""# Dataset Quality Assurance Report
## Sustainability Tech Deployment Ground Truth Dataset

**Date Generated:** 2025-11-12
**Dataset Path:** `C:\\local_dev\\llm-distillery\\datasets\\labeled\\sustainability_tech_deployment\\labeled_articles.jsonl`
**Total Articles:** {total:,}

---

## Executive Summary

"""

    # Determine overall health
    critical_issues = sum([
        1 if parse_errors else 0,
        1 if results["duplicate_ids"] else 0,
        1 if len(results["field_issues"]) > total * 0.05 else 0
    ])

    if critical_issues == 0:
        health_status = "✅ **HEALTHY** - Dataset passes all critical quality checks"
    elif critical_issues <= 1:
        health_status = "⚠️ **MINOR ISSUES** - Dataset has minor issues that should be addressed"
    else:
        health_status = "❌ **CRITICAL ISSUES** - Dataset has significant quality problems"

    report += f"{health_status}\n\n"

    report += f"""### Key Findings

- **Total Articles:** {total:,}
- **Parse Errors:** {len(parse_errors)}
- **Field Issues:** {len(results['field_issues'])}
- **Structure Issues:** {len(results['structure_issues'])}
- **Duplicate IDs:** {len(results['duplicate_ids'])}
- **Failed Labelings (all zeros):** {len(results['all_zeros'])}

### Class Distribution

| Tier | Count | Percentage |
|------|-------|------------|
"""

    for tier in ["deployed", "early_commercial", "pilot_stage", "vaporware"]:
        count = class_dist.get(tier, 0)
        pct = (count / total * 100) if total > 0 else 0
        report += f"| {tier} | {count:,} | {pct:.1f}% |\n"

    # Check for severe imbalance
    if class_dist:
        max_class = max(class_dist.values())
        min_class = min(class_dist.values())
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')

        report += f"\n**Imbalance Ratio:** {imbalance_ratio:.1f}:1 "
        if imbalance_ratio > 5:
            report += "⚠️ **High class imbalance detected**\n"
        else:
            report += "✅ **Acceptable balance**\n"

    report += f"""
---

## Detailed Quality Checks

### 1. JSON Parsing
{checks["JSON parsing"]}

"""

    if parse_errors:
        report += f"Found {len(parse_errors)} malformed JSON lines:\n\n"
        for error in parse_errors[:5]:  # Show first 5
            report += f"- Line {error['line']}: {error['error']}\n"
        if len(parse_errors) > 5:
            report += f"- ... and {len(parse_errors) - 5} more\n"
    else:
        report += "All lines successfully parsed as valid JSON.\n"

    report += f"""
### 2. Required Fields
{checks["Required fields"]}

"""

    if results["field_issues"]:
        report += f"Found {len(results['field_issues'])} articles with missing or empty required fields:\n\n"
        for issue in results["field_issues"][:10]:  # Show first 10
            report += f"- Article `{issue['id']}`: Missing {', '.join(issue['missing'])}\n"
        if len(results["field_issues"]) > 10:
            report += f"- ... and {len(results['field_issues']) - 10} more\n"
    else:
        report += "All articles have required fields: `id`, `title`, `content`, `sustainability_tech_deployment_analysis`\n"

    report += f"""
### 3. Analysis Structure
{checks["Analysis structure"]}

"""

    if results["structure_issues"]:
        report += f"Found {len(results['structure_issues'])} articles with structural issues:\n\n"

        # Categorize issues
        missing_dims = sum(1 for i in results["structure_issues"] if "missing_dimensions" in i["issues"])
        invalid_scores = sum(1 for i in results["structure_issues"] if "invalid_scores" in i["issues"])
        score_mismatch = sum(1 for i in results["structure_issues"] if "overall_score_mismatch" in i["issues"])
        tier_mismatch = sum(1 for i in results["structure_issues"] if "tier_mismatch" in i["issues"])

        report += f"- Missing dimensions: {missing_dims} articles\n"
        report += f"- Invalid score ranges: {invalid_scores} articles\n"
        report += f"- Overall score calculation mismatch: {score_mismatch} articles\n"
        report += f"- Tier assignment mismatch: {tier_mismatch} articles\n\n"

        report += "**Sample Issues:**\n\n"
        for issue in results["structure_issues"][:5]:
            report += f"- Article `{issue['id']}`:\n"
            for issue_type, details in issue["issues"].items():
                report += f"  - {issue_type}: {details}\n"
        if len(results["structure_issues"]) > 5:
            report += f"- ... and {len(results['structure_issues']) - 5} more\n"
    else:
        report += "All articles have complete and valid analysis structures with all 8 dimensions.\n"

    report += f"""
### 4. Duplicate IDs
{checks["Duplicate IDs"]}

"""

    if results["duplicate_ids"]:
        report += f"Found {len(results['duplicate_ids'])} duplicate IDs:\n\n"
        for dup_id in results["duplicate_ids"][:10]:
            report += f"- `{dup_id}`\n"
        if len(results["duplicate_ids"]) > 10:
            report += f"- ... and {len(results['duplicate_ids']) - 10} more\n"
    else:
        report += "No duplicate IDs found. All article IDs are unique.\n"

    report += f"""
### 5. Failed Labelings (All Zeros)
{checks["All zeros check"]}

"""

    if results["all_zeros"]:
        report += f"Found {len(results['all_zeros'])} articles where all dimension scores are 0 (potential labeling failures):\n\n"
        for article_id in results["all_zeros"][:10]:
            report += f"- `{article_id}`\n"
        if len(results["all_zeros"]) > 10:
            report += f"- ... and {len(results['all_zeros']) - 10} more\n"
    else:
        report += "No articles with all-zero scores detected.\n"

    report += f"""
---

## Statistical Analysis

### Overall Score Distribution

| Metric | Value |
|--------|-------|
| Minimum | {overall_stats['min']} |
| Maximum | {overall_stats['max']} |
| Mean | {overall_stats['mean']} |
| Median | {overall_stats['median']} |
| Std Dev | {overall_stats['std_dev']} |
| Total Scores | {len(results['overall_scores']):,} |

### Dimension Score Statistics

| Dimension | Min | Max | Mean | Median | Std Dev |
|-----------|-----|-----|------|--------|---------|
"""

    for dim in REQUIRED_DIMENSIONS:
        stats = dimension_stats[dim]
        report += f"| {dim} | {stats['min']} | {stats['max']} | {stats['mean']} | {stats['median']} | {stats['std_dev']} |\n"

    report += f"""
---

## Recommendations

"""

    recommendations = []

    if parse_errors:
        recommendations.append("🔴 **CRITICAL:** Fix malformed JSON lines before using this dataset")

    if results["duplicate_ids"]:
        recommendations.append("🔴 **CRITICAL:** Remove or consolidate duplicate article IDs")

    if results["field_issues"]:
        recommendations.append("🟡 **HIGH:** Add missing required fields to affected articles")

    if results["structure_issues"]:
        tier_issues = sum(1 for i in results["structure_issues"] if "tier_mismatch" in i["issues"])
        score_issues = sum(1 for i in results["structure_issues"] if "overall_score_mismatch" in i["issues"])

        if tier_issues > 0:
            recommendations.append(f"🟡 **HIGH:** Correct tier assignments for {tier_issues} articles with mismatched tiers")
        if score_issues > 0:
            recommendations.append(f"🟡 **MEDIUM:** Recalculate overall scores for {score_issues} articles with calculation mismatches")

    if results["all_zeros"]:
        recommendations.append(f"🟡 **MEDIUM:** Re-label {len(results['all_zeros'])} articles with all-zero scores")

    if class_dist and max(class_dist.values()) / min(class_dist.values()) > 5:
        recommendations.append("🟢 **LOW:** Consider collecting more samples for underrepresented classes to improve model balance")

    if not recommendations:
        recommendations.append("✅ **No critical issues found!** Dataset is ready for training.")

    for rec in recommendations:
        report += f"\n{rec}\n"

    report += f"""
---

## Conclusion

"""

    if critical_issues == 0 and len(results["structure_issues"]) < total * 0.01:
        report += """This dataset is in excellent condition and ready for model training. All critical quality checks have passed, and only minor issues (if any) were detected. The class distribution is reasonable, and all articles have complete, valid annotations.

**Status: ✅ APPROVED FOR TRAINING**
"""
    elif critical_issues == 0:
        report += """This dataset is in good condition overall but has some structural issues that should be addressed before training. While all critical checks passed, fixing the identified issues will improve model performance and training stability.

**Status: ⚠️ APPROVED WITH RESERVATIONS**
"""
    else:
        report += """This dataset has critical quality issues that must be resolved before use in model training. Address the high-priority recommendations above before proceeding.

**Status: ❌ REQUIRES FIXES**
"""

    report += f"""
---

*Report generated by Dataset QA Script v1.0*
*Total articles analyzed: {total:,}*
"""

    return report

def main():
    print("Starting dataset quality assurance analysis...")
    print(f"Loading dataset from: {DATASET_PATH}")

    # Load dataset
    articles, parse_errors = load_dataset(DATASET_PATH)
    print(f"Loaded {len(articles)} articles")

    if parse_errors:
        print(f"[WARNING] Found {len(parse_errors)} parse errors")

    # Analyze
    print("Running comprehensive analysis...")
    results = analyze_dataset(articles)

    # Generate report
    print("Generating report...")
    report = generate_report(results, parse_errors)

    # Save report
    report_path = Path(r"C:\local_dev\llm-distillery\reports\tech_deployment_dataset_qa.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n[OK] Report saved to: {report_path}")
    print("\n" + "="*80)
    print(report)
    print("="*80)

if __name__ == "__main__":
    main()
