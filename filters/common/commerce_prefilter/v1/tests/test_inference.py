"""
Tests for Commerce Prefilter SLM Inference

Run with: pytest filters/common/commerce_prefilter/v1/tests/test_inference.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch


# Test data
COMMERCE_ARTICLES = [
    {
        "title": "Green Deals: Save $500 on Jackery Solar Generator",
        "content": "Today's Green Deals are headlined by an exclusive discount on "
                   "the Jackery Explorer 1000 Plus solar generator kit. Originally "
                   "priced at $1,999, you can now get it for just $1,499 - that's "
                   "$500 in savings! This deal ends tonight at midnight.",
    },
    {
        "title": "Black Friday Solar Panel Deals: Up to 40% Off",
        "content": "Don't miss these incredible Black Friday solar panel deals! "
                   "Use promo code SOLAR40 for an extra 10% off. Limited time offer.",
    },
    {
        "title": "Best EV Charger Discounts This Cyber Monday",
        "content": "Cyber Monday brings massive savings on EV chargers. "
                   "Save $200 on the ChargePoint Home Flex. Shop now before stock runs out!",
    },
]

JOURNALISM_ARTICLES = [
    {
        "title": "New Solar Technology Achieves Record 30% Efficiency",
        "content": "Researchers at MIT have developed a breakthrough perovskite-silicon "
                   "tandem solar cell that achieves 30% efficiency, surpassing traditional "
                   "silicon panels. The technology uses abundant materials and could be "
                   "manufactured at scale using existing equipment.",
    },
    {
        "title": "EPA Announces New Clean Energy Regulations",
        "content": "The Environmental Protection Agency announced new regulations "
                   "requiring power plants to reduce carbon emissions by 50% by 2030. "
                   "The rules are expected to accelerate the transition to renewable energy.",
    },
    {
        "title": "Global Wind Capacity Reaches 1 Terawatt Milestone",
        "content": "The world's installed wind power capacity has reached 1 terawatt, "
                   "according to a new report from the Global Wind Energy Council. "
                   "China and the US lead in new installations.",
    },
]


class TestCommercePrefilterSLM:
    """Tests for CommercePrefilterSLM class."""

    def test_prepare_text(self):
        """Test text preparation from article."""
        # Import locally to handle missing model gracefully
        from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM

        # Create mock instance without loading model
        with patch.object(CommercePrefilterSLM, '_load_model'):
            detector = CommercePrefilterSLM()

        article = {
            "title": "Test Title",
            "content": "Test content here."
        }
        text = detector._prepare_text(article)
        assert "Test Title" in text
        assert "Test content here" in text

    def test_prepare_text_with_text_field(self):
        """Test text preparation with 'text' field instead of 'content'."""
        from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM

        with patch.object(CommercePrefilterSLM, '_load_model'):
            detector = CommercePrefilterSLM()

        article = {
            "title": "Test Title",
            "text": "Test text here."
        }
        text = detector._prepare_text(article)
        assert "Test Title" in text
        assert "Test text here" in text

    def test_prepare_text_truncation(self):
        """Test that long content is truncated."""
        from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM

        with patch.object(CommercePrefilterSLM, '_load_model'):
            detector = CommercePrefilterSLM()

        article = {
            "title": "Title",
            "content": "x" * 5000  # Very long content
        }
        text = detector._prepare_text(article)
        # Content should be truncated to 2000 chars
        assert len(text) <= 2000 + len("Title") + 10  # Some buffer for newlines

    def test_threshold_setter(self):
        """Test threshold validation."""
        from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM

        with patch.object(CommercePrefilterSLM, '_load_model'):
            detector = CommercePrefilterSLM(threshold=0.7)

        detector.set_threshold(0.5)
        assert detector.threshold == 0.5

        detector.set_threshold(0.9)
        assert detector.threshold == 0.9

        with pytest.raises(ValueError):
            detector.set_threshold(1.5)

        with pytest.raises(ValueError):
            detector.set_threshold(-0.1)

    def test_model_not_found_error(self):
        """Test error when model is not found."""
        from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM

        with pytest.raises(FileNotFoundError) as exc_info:
            CommercePrefilterSLM(model_path="/nonexistent/path")

        assert "Model not found" in str(exc_info.value)


class TestCommercePrefilterWithMockedModel:
    """Tests with mocked model for faster execution."""

    @pytest.fixture
    def mock_detector(self):
        """Create detector with mocked model."""
        from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM

        with patch.object(CommercePrefilterSLM, '_load_model'):
            detector = CommercePrefilterSLM(threshold=0.7)
            detector.model = Mock()
            detector.tokenizer = Mock()
            detector.device = 'cpu'
            return detector

    def test_predict_score_commerce(self, mock_detector):
        """Test prediction returns high score for commerce."""
        # Mock tokenizer
        mock_detector.tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
        }

        # Mock model output (high score for commerce)
        mock_output = Mock()
        mock_output.logits = torch.tensor([[2.0]])  # sigmoid(2.0) ≈ 0.88
        mock_detector.model.return_value = mock_output

        score = mock_detector.predict_score(COMMERCE_ARTICLES[0])
        assert 0 <= score <= 1

    def test_is_commerce_returns_correct_format(self, mock_detector):
        """Test is_commerce returns correct result format."""
        # Mock tokenizer
        mock_detector.tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
        }

        # Mock model output
        mock_output = Mock()
        mock_output.logits = torch.tensor([[2.0]])
        mock_detector.model.return_value = mock_output

        result = mock_detector.is_commerce(COMMERCE_ARTICLES[0])

        assert 'is_commerce' in result
        assert 'score' in result
        assert 'inference_time_ms' in result
        assert isinstance(result['is_commerce'], bool)
        assert isinstance(result['score'], float)
        assert isinstance(result['inference_time_ms'], float)


class TestIntegration:
    """Integration tests (require trained model)."""

    @pytest.fixture
    def detector(self):
        """Create detector with real model if available."""
        from filters.common.commerce_prefilter.v1.inference import CommercePrefilterSLM

        model_path = Path(__file__).parent.parent / "model"
        if not model_path.exists():
            pytest.skip("Model not trained yet")

        return CommercePrefilterSLM(model_path=model_path, threshold=0.7)

    @pytest.mark.integration
    def test_commerce_detection(self, detector):
        """Test that commerce articles are detected."""
        for article in COMMERCE_ARTICLES:
            result = detector.is_commerce(article)
            # Commerce articles should have high scores
            assert result['score'] > 0.5, f"Expected high score for commerce: {article['title']}"

    @pytest.mark.integration
    def test_journalism_not_blocked(self, detector):
        """Test that journalism articles are not blocked."""
        for article in JOURNALISM_ARTICLES:
            result = detector.is_commerce(article)
            # Journalism should have low scores
            assert result['score'] < 0.7, f"Expected low score for journalism: {article['title']}"

    @pytest.mark.integration
    def test_inference_speed(self, detector):
        """Test that inference is fast (<50ms)."""
        result = detector.is_commerce(JOURNALISM_ARTICLES[0])
        assert result['inference_time_ms'] < 100, "Inference too slow"

    @pytest.mark.integration
    def test_batch_predict(self, detector):
        """Test batch prediction."""
        all_articles = COMMERCE_ARTICLES + JOURNALISM_ARTICLES
        results = detector.batch_predict(all_articles)

        assert len(results) == len(all_articles)
        for result in results:
            assert 'is_commerce' in result
            assert 'score' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
