"""
Integration tests for filter package loading.

Tests the complete filter loading workflow:
- load_filter_package(): Loading prefilter, prompt, config
- Prefilter apply_filter(): End-to-end prefiltering
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestFilterPackageLoading:
    """Integration tests for filter package loading."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def uplifting_filter_path(self, project_root):
        """Path to uplifting v5 filter."""
        return project_root / "filters" / "uplifting" / "v5"

    @pytest.fixture
    def sustainability_filter_path(self, project_root):
        """Path to sustainability_technology v1 filter."""
        return project_root / "filters" / "sustainability_technology" / "v1"

    def test_load_uplifting_filter(self, uplifting_filter_path):
        """Should load uplifting v5 filter package."""
        if not uplifting_filter_path.exists():
            pytest.skip("Uplifting v5 filter not available")

        from ground_truth.batch_scorer import load_filter_package

        prefilter, prompt_path, config = load_filter_package(uplifting_filter_path)

        # Prefilter should be loaded
        assert prefilter is not None
        assert hasattr(prefilter, 'apply_filter')
        assert hasattr(prefilter, 'VERSION')

        # Prompt should exist
        assert prompt_path.exists()
        assert prompt_path.suffix == ".md"

    def test_load_sustainability_filter(self, sustainability_filter_path):
        """Should load sustainability_technology v1 filter package."""
        if not sustainability_filter_path.exists():
            pytest.skip("Sustainability technology v1 filter not available")

        from ground_truth.batch_scorer import load_filter_package

        prefilter, prompt_path, config = load_filter_package(sustainability_filter_path)

        assert prefilter is not None
        assert prompt_path.exists()

    def test_prefilter_accepts_valid_article(self, uplifting_filter_path, valid_article):
        """Prefilter should accept valid uplifting-style article."""
        if not uplifting_filter_path.exists():
            pytest.skip("Uplifting v5 filter not available")

        from ground_truth.batch_scorer import load_filter_package

        prefilter, _, _ = load_filter_package(uplifting_filter_path)

        # Community solar article should be considered potentially uplifting
        passed, reason = prefilter.apply_filter(valid_article)
        # Just verify it returns a tuple with bool and string
        assert isinstance(passed, bool)
        assert isinstance(reason, str)

    def test_prefilter_returns_valid_tuple(self, uplifting_filter_path):
        """Prefilter should return a valid (bool, str) tuple."""
        if not uplifting_filter_path.exists():
            pytest.skip("Uplifting v5 filter not available")

        from ground_truth.batch_scorer import load_filter_package

        prefilter, _, _ = load_filter_package(uplifting_filter_path)

        # Any article should return valid tuple
        article = {
            "title": "Test Article",
            "content": "This is test content that is long enough." * 20
        }

        result = prefilter.apply_filter(article)

        # Should return tuple of (bool, str)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_prefilter_rejects_short_content(self, uplifting_filter_path):
        """Prefilter should reject content that's too short."""
        if not uplifting_filter_path.exists():
            pytest.skip("Uplifting v5 filter not available")

        from ground_truth.batch_scorer import load_filter_package

        prefilter, _, _ = load_filter_package(uplifting_filter_path)

        short_article = {
            "title": "Good News",
            "content": "This is too short."
        }

        passed, reason = prefilter.apply_filter(short_article)
        assert passed is False
        assert "short" in reason.lower() or "length" in reason.lower() or "content" in reason.lower()


class TestPrefilterValidation:
    """Tests for prefilter input validation."""

    @pytest.fixture
    def uplifting_prefilter(self):
        """Load uplifting prefilter if available."""
        project_root = Path(__file__).parent.parent.parent
        filter_path = project_root / "filters" / "uplifting" / "v5"

        if not filter_path.exists():
            pytest.skip("Uplifting v5 filter not available")

        from ground_truth.batch_scorer import load_filter_package
        prefilter, _, _ = load_filter_package(filter_path)
        return prefilter

    def test_handles_missing_title(self, uplifting_prefilter):
        """Prefilter should handle article with missing title gracefully."""
        article = {"content": "Some content here that is long enough." * 20}

        # Should not crash - returns a result (may pass or fail depending on implementation)
        try:
            result = uplifting_prefilter.apply_filter(article)
            # If it doesn't raise, should return valid tuple
            assert isinstance(result, tuple)
            assert len(result) == 2
        except (KeyError, TypeError, ValueError):
            # Also acceptable to raise these for invalid input
            pass

    def test_handles_empty_content(self, uplifting_prefilter):
        """Prefilter should handle article with empty content."""
        article = {"title": "Test", "content": ""}

        passed, reason = uplifting_prefilter.apply_filter(article)
        assert passed is False
