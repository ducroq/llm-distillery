"""
Oracle for Sustainability Technology V1

This module provides the oracle (Gemini-based scorer) for the sustainability_technology v1 filter.
"""

import os
import json
import configparser
from pathlib import Path
from typing import Dict, Tuple
import google.generativeai as genai


class SustainabilityTechnologyOracleV1:
    """
    Oracle for scoring articles on sustainability technology using Gemini API.

    Scores articles across 6 LCSA dimensions:
    - Technology Readiness Level (TRL)
    - Technical Performance
    - Economic Competitiveness
    - Life Cycle Environmental Impact
    - Social & Equity Impact
    - Governance & Systemic Impact
    """

    def __init__(self, model_name: str = "models/gemini-2.5-flash"):
        """Initialize oracle with Gemini model."""
        # Get API key from environment or config file
        api_key = self._load_api_key()

        genai.configure(api_key=api_key)

        # Use Gemini 2.5 Flash (stable, fast, cost-effective)
        self.model = genai.GenerativeModel(model_name)
        print(f"Initialized oracle with {model_name}")

        # Load prompt
        prompt_path = Path(__file__).parent / "prompt-compressed.md"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

        # Dimensions
        self.dimensions = [
            'technology_readiness_level',
            'technical_performance',
            'economic_competitiveness',
            'life_cycle_environmental_impact',
            'social_equity_impact',
            'governance_systemic_impact'
        ]

    def score_article(self, article: Dict) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Score an article using the oracle.

        Args:
            article: Article dictionary with 'title', 'content', etc.

        Returns:
            Tuple of (scores_dict, reasoning_dict)
            - scores_dict: {dimension: score (0.0-10.0)}
            - reasoning_dict: {dimension: reasoning text}
        """
        # Prepare article summary for prompt
        article_summary = self._prepare_article_summary(article)

        # Build full prompt
        full_prompt = self._build_prompt(article_summary)

        # Call Gemini API
        response = self.model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Low temperature for consistency
                response_mime_type="application/json",
            )
        )

        # Parse response
        scores, reasoning = self._parse_response(response.text)

        return scores, reasoning

    def _prepare_article_summary(self, article: Dict) -> str:
        """Prepare article summary for the prompt."""
        title = article.get('title', 'No title')
        content = article.get('content', article.get('description', ''))

        # Truncate content if too long (Gemini context limit)
        max_content_length = 5000  # characters
        if len(content) > max_content_length:
            content = content[:max_content_length] + "... [truncated]"

        summary = f"**Title:** {title}\n\n**Content:**\n{content}"
        return summary

    def _build_prompt(self, article_summary: str) -> str:
        """Build the full prompt with article summary."""
        # Replace the INPUT DATA placeholder with actual article
        prompt = self.prompt_template.replace(
            "**INPUT DATA:** [Paste the summary of the article here]",
            f"**INPUT DATA:**\n\n{article_summary}"
        )

        # Add JSON response format instruction
        prompt += "\n\n---\n\n"
        prompt += "## Response Format\n\n"
        prompt += "Respond with a JSON object in this exact format:\n\n"
        prompt += "```json\n"
        prompt += "{\n"
        prompt += '  "technology_readiness_level": {"score": 7.5, "reasoning": "..."},\n'
        prompt += '  "technical_performance": {"score": 6.0, "reasoning": "..."},\n'
        prompt += '  "economic_competitiveness": {"score": 5.5, "reasoning": "..."},\n'
        prompt += '  "life_cycle_environmental_impact": {"score": 7.0, "reasoning": "..."},\n'
        prompt += '  "social_equity_impact": {"score": 4.0, "reasoning": "..."},\n'
        prompt += '  "governance_systemic_impact": {"score": 6.5, "reasoning": "..."}\n'
        prompt += "}\n"
        prompt += "```\n\n"
        prompt += "**CRITICAL:** Provide scores as floats (0.0-10.0) and brief reasoning (1-2 sentences) for each dimension."

        return prompt

    def _parse_response(self, response_text: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Parse Gemini response into scores and reasoning."""
        try:
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            # Parse JSON
            data = json.loads(response_text.strip())

            scores = {}
            reasoning = {}

            for dim in self.dimensions:
                if dim in data:
                    if isinstance(data[dim], dict):
                        scores[dim] = float(data[dim].get('score', 0.0))
                        reasoning[dim] = data[dim].get('reasoning', '')
                    else:
                        # Fallback if format is different
                        scores[dim] = float(data[dim])
                        reasoning[dim] = ''
                else:
                    scores[dim] = 0.0
                    reasoning[dim] = 'ERROR: Dimension not found in response'

            return scores, reasoning

        except Exception as e:
            # Fallback: return zeros if parsing fails
            scores = {dim: 0.0 for dim in self.dimensions}
            reasoning = {dim: f'ERROR: Failed to parse response - {str(e)}' for dim in self.dimensions}
            return scores, reasoning

    def _load_api_key(self) -> str:
        """Load API key from environment variable or config file."""
        # Try environment variable first
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            return api_key

        # Try config/credentials/secrets.ini
        secrets_path = Path(__file__).parent.parent.parent.parent / "config" / "credentials" / "secrets.ini"
        if secrets_path.exists():
            config = configparser.ConfigParser()
            config.read(secrets_path)
            if 'api_keys' in config:
                # Try billing key first (no rate limits)
                if 'gemini_billing_api_key' in config['api_keys']:
                    api_key = config['api_keys']['gemini_billing_api_key']
                    if api_key and not api_key.startswith('AIza-your-'):
                        print("Using gemini_billing_api_key (no rate limits)")
                        return api_key
                # Fallback to free tier key
                if 'gemini_api_key' in config['api_keys']:
                    api_key = config['api_keys']['gemini_api_key']
                    if api_key and not api_key.startswith('AIza-your-'):
                        print("WARNING: Using gemini_api_key (free tier, 250 requests/day limit)")
                        return api_key

        raise ValueError(
            "Gemini API key not found. Please set either:\n"
            "  1. GOOGLE_API_KEY environment variable, or\n"
            "  2. gemini_billing_api_key in config/credentials/secrets.ini"
        )
