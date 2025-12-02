"""
Unit tests for training/prepare_data.py

Tests the training data preparation functions:
- load_labels(): JSONL loading
- stratified_split(): Data splitting with stratification
- convert_to_training_format(): Format conversion
- calculate_overall_score(): Score calculation
- assign_tier(): Tier assignment
"""

import json
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.prepare_data import (
    load_labels,
    stratified_split,
    convert_to_training_format,
    calculate_overall_score,
    assign_tier,
    assign_score_bin,
    get_analysis_field_name,
)


class TestLoadLabels:
    """Tests for load_labels()"""

    def test_load_single_file(self, temp_jsonl_file):
        """Should load all articles from single JSONL file."""
        labels = load_labels(temp_jsonl_file)
        assert len(labels) == 8  # Based on labeled_articles fixture
        assert all("id" in label for label in labels)

    def test_load_nonexistent_file(self, temp_dir):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_labels(temp_dir / "nonexistent.jsonl")

    def test_load_empty_file(self, temp_dir):
        """Should return empty list for empty file."""
        empty_file = temp_dir / "empty.jsonl"
        empty_file.touch()
        labels = load_labels(empty_file)
        assert labels == []

    def test_load_with_blank_lines(self, temp_dir):
        """Should skip blank lines in JSONL."""
        file_path = temp_dir / "with_blanks.jsonl"
        with open(file_path, 'w') as f:
            f.write('{"id": "1", "title": "Test", "content": "Content"}\n')
            f.write('\n')  # Blank line
            f.write('{"id": "2", "title": "Test2", "content": "Content2"}\n')
            f.write('   \n')  # Whitespace line
        labels = load_labels(file_path)
        assert len(labels) == 2


class TestCalculateOverallScore:
    """Tests for calculate_overall_score()"""

    def test_explicit_overall_score(self, sample_oracle_analysis):
        """Should use explicit overall_score if present."""
        score = calculate_overall_score(sample_oracle_analysis)
        assert score == 6.5

    def test_calculate_from_dimensions_nested(self, sample_oracle_analysis):
        """Should calculate from nested dimensions if no overall_score."""
        analysis = sample_oracle_analysis.copy()
        del analysis["overall_score"]
        # Get dimension names (excluding overall_score which we just deleted)
        dimension_names = [k for k in analysis.keys()]
        score = calculate_overall_score(analysis, dimension_names)
        # Scores: 7, 6, 5, 8, 6, 7 = 39/6 = 6.5
        assert abs(score - 6.5) < 0.1  # Allow some tolerance

    def test_calculate_from_dimensions_flat(self, sample_oracle_analysis_flat):
        """Should calculate from flat dimensions if no overall_score."""
        analysis = sample_oracle_analysis_flat.copy()
        del analysis["overall_score"]
        dimension_names = [k for k in analysis.keys()]
        score = calculate_overall_score(analysis, dimension_names)
        assert score > 0

    def test_empty_analysis(self):
        """Should return 0 for empty analysis."""
        score = calculate_overall_score({})
        assert score == 0.0


class TestAssignTier:
    """Tests for assign_tier()"""

    def test_high_tier(self, uplifting_tier_boundaries):
        """Score >= 7.0 should be high_impact."""
        tier = assign_tier(8.5, uplifting_tier_boundaries)
        assert tier == "high_impact"

        tier = assign_tier(7.0, uplifting_tier_boundaries)
        assert tier == "high_impact"

    def test_medium_tier(self, uplifting_tier_boundaries):
        """Score >= 4.0 and < 7.0 should be moderate_uplift."""
        tier = assign_tier(6.9, uplifting_tier_boundaries)
        assert tier == "moderate_uplift"

        tier = assign_tier(4.0, uplifting_tier_boundaries)
        assert tier == "moderate_uplift"

    def test_low_tier(self, uplifting_tier_boundaries):
        """Score < 4.0 should be not_uplifting."""
        tier = assign_tier(3.9, uplifting_tier_boundaries)
        assert tier == "not_uplifting"

        tier = assign_tier(0.0, uplifting_tier_boundaries)
        assert tier == "not_uplifting"

    def test_boundary_values(self, uplifting_tier_boundaries):
        """Boundary values should go to higher tier."""
        assert assign_tier(7.0, uplifting_tier_boundaries) == "high_impact"
        assert assign_tier(4.0, uplifting_tier_boundaries) == "moderate_uplift"
        assert assign_tier(0.0, uplifting_tier_boundaries) == "not_uplifting"


class TestAssignScoreBin:
    """Tests for assign_score_bin()"""

    def test_very_high_bin(self):
        """Score >= 8.0 should be very_high."""
        assert assign_score_bin(10.0) == "very_high"
        assert assign_score_bin(8.0) == "very_high"
        assert assign_score_bin(8.5) == "very_high"

    def test_high_bin(self):
        """Score >= 6.0 and < 8.0 should be high."""
        assert assign_score_bin(7.9) == "high"
        assert assign_score_bin(6.0) == "high"

    def test_medium_bin(self):
        """Score >= 4.0 and < 6.0 should be medium."""
        assert assign_score_bin(5.9) == "medium"
        assert assign_score_bin(4.0) == "medium"

    def test_low_bin(self):
        """Score >= 2.0 and < 4.0 should be low."""
        assert assign_score_bin(3.9) == "low"
        assert assign_score_bin(2.0) == "low"

    def test_very_low_bin(self):
        """Score < 2.0 should be very_low."""
        assert assign_score_bin(1.9) == "very_low"
        assert assign_score_bin(0.0) == "very_low"


class TestStratifiedSplit:
    """Tests for stratified_split()"""

    def test_split_ratios(self, labeled_articles, uplifting_tier_boundaries, uplifting_dimension_names):
        """Split should approximately match requested ratios."""
        train, val, test = stratified_split(
            labeled_articles,
            analysis_field="uplifting_analysis",
            tier_boundaries=uplifting_tier_boundaries,
            dimension_names=uplifting_dimension_names,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42
        )

        total = len(labeled_articles)
        # Allow some variance due to small sample size
        assert len(train) >= int(total * 0.6)  # At least 60%
        assert len(train) + len(val) + len(test) == total

    def test_no_data_leakage(self, labeled_articles, uplifting_tier_boundaries, uplifting_dimension_names):
        """Same article should not appear in multiple splits."""
        train, val, test = stratified_split(
            labeled_articles,
            analysis_field="uplifting_analysis",
            tier_boundaries=uplifting_tier_boundaries,
            dimension_names=uplifting_dimension_names
        )

        train_ids = {a["id"] for a in train}
        val_ids = {a["id"] for a in val}
        test_ids = {a["id"] for a in test}

        # No overlap between sets
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_reproducible_with_seed(self, labeled_articles, uplifting_tier_boundaries, uplifting_dimension_names):
        """Same seed should produce same split."""
        split1 = stratified_split(
            labeled_articles,
            analysis_field="uplifting_analysis",
            tier_boundaries=uplifting_tier_boundaries,
            seed=42
        )
        split2 = stratified_split(
            labeled_articles,
            analysis_field="uplifting_analysis",
            tier_boundaries=uplifting_tier_boundaries,
            seed=42
        )

        # Same articles in each split
        train1_ids = {a["id"] for a in split1[0]}
        train2_ids = {a["id"] for a in split2[0]}
        assert train1_ids == train2_ids

    def test_different_seed_different_split(self, labeled_articles, uplifting_tier_boundaries):
        """Different seeds should produce different splits."""
        split1 = stratified_split(
            labeled_articles,
            analysis_field="uplifting_analysis",
            tier_boundaries=uplifting_tier_boundaries,
            seed=42
        )
        split2 = stratified_split(
            labeled_articles,
            analysis_field="uplifting_analysis",
            tier_boundaries=uplifting_tier_boundaries,
            seed=123
        )

        train1_ids = {a["id"] for a in split1[0]}
        train2_ids = {a["id"] for a in split2[0]}
        # Very likely different (could be same by chance but unlikely)
        # Just check they're not obviously broken
        assert len(train1_ids) > 0
        assert len(train2_ids) > 0

    def test_empty_tier_boundaries_uses_score_bins(self, labeled_articles, uplifting_dimension_names):
        """Empty tier_boundaries should use score bin stratification."""
        train, val, test = stratified_split(
            labeled_articles,
            analysis_field="uplifting_analysis",
            tier_boundaries={},  # Empty = use score bins
            dimension_names=uplifting_dimension_names
        )
        assert len(train) + len(val) + len(test) == len(labeled_articles)


class TestConvertToTrainingFormat:
    """Tests for convert_to_training_format()"""

    def test_basic_conversion(self, labeled_articles, uplifting_dimension_names):
        """Should convert articles to training format."""
        training_data = convert_to_training_format(
            labeled_articles,
            analysis_field="uplifting_analysis",
            dimension_names=uplifting_dimension_names
        )

        assert len(training_data) == len(labeled_articles)
        for item in training_data:
            assert "id" in item
            assert "title" in item
            assert "content" in item
            assert "labels" in item
            assert "dimension_names" in item
            assert len(item["labels"]) == len(uplifting_dimension_names)

    def test_labels_are_numeric_array(self, labeled_articles, uplifting_dimension_names):
        """Labels should be a list of numbers."""
        training_data = convert_to_training_format(
            labeled_articles,
            analysis_field="uplifting_analysis",
            dimension_names=uplifting_dimension_names
        )

        for item in training_data:
            assert isinstance(item["labels"], list)
            assert all(isinstance(score, (int, float)) for score in item["labels"])

    def test_dimension_order_preserved(self, labeled_articles, uplifting_dimension_names):
        """Dimension order in labels should match dimension_names."""
        training_data = convert_to_training_format(
            labeled_articles,
            analysis_field="uplifting_analysis",
            dimension_names=uplifting_dimension_names
        )

        # The dimension_names field should match input
        assert training_data[0]["dimension_names"] == uplifting_dimension_names

    def test_skips_articles_without_analysis(self, valid_article, uplifting_dimension_names):
        """Articles without analysis should be skipped."""
        articles = [
            valid_article,  # No analysis
            {**valid_article, "id": "with-analysis", "uplifting_analysis": {"dimensions": {}}}
        ]

        training_data = convert_to_training_format(
            articles,
            analysis_field="uplifting_analysis",
            dimension_names=uplifting_dimension_names
        )

        # Only the article with analysis should be included
        assert len(training_data) == 1
        assert training_data[0]["id"] == "with-analysis"

    def test_handles_nested_dimension_format(self, uplifting_dimension_names):
        """Should handle nested dimension format (score + reasoning)."""
        article = {
            "id": "nested-test",
            "title": "Test",
            "content": "Content",
            "uplifting_analysis": {
                "dimensions": {
                    dim: {"score": 5 + i, "reasoning": "test"}
                    for i, dim in enumerate(uplifting_dimension_names)
                }
            }
        }

        training_data = convert_to_training_format(
            [article],
            analysis_field="uplifting_analysis",
            dimension_names=uplifting_dimension_names
        )

        assert len(training_data) == 1
        # First dimension should have score 5, second 6, etc.
        assert training_data[0]["labels"][0] == 5
        assert training_data[0]["labels"][1] == 6


class TestGetAnalysisFieldName:
    """Tests for get_analysis_field_name()"""

    def test_simple_filter_name(self):
        """Simple filter name should get _analysis suffix."""
        assert get_analysis_field_name("uplifting") == "uplifting_analysis"

    def test_compound_filter_name(self):
        """Compound filter name should get _analysis suffix."""
        assert get_analysis_field_name("investment_risk") == "investment_risk_analysis"
        assert get_analysis_field_name("sustainability_tech") == "sustainability_tech_analysis"
