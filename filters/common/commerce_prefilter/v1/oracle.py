"""
Oracle for Commerce Content Detection

Uses Gemini to classify articles as commerce/promotional vs journalism/editorial.
Outputs a single score (0-10) where higher = more commercial.
"""

import configparser
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import google.generativeai as genai


class CommerceOracle:
    """
    Oracle for classifying commerce vs journalism content using Gemini API.

    Outputs:
    - commerce_score: 0-10 (0=journalism, 10=commerce)
    - reasoning: explanation for the score
    - key_signals: list of detected signals
    """

    def __init__(self, model_name: str = "models/gemini-2.5-pro"):
        """
        Initialize oracle with Gemini model.

        Args:
            model_name: Gemini model to use.
                       Use "models/gemini-2.5-pro" for calibration (accurate).
                       Use "models/gemini-2.5-flash" for production (fast/cheap).
        """
        api_key = self._load_api_key()
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        print(f"Initialized CommerceOracle with {model_name}")

        # Load prompt
        prompt_path = Path(__file__).parent / "prompt.md"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

    def score_article(self, article: Dict) -> Dict:
        """
        Score an article for commerce content.

        Args:
            article: Dict with 'title' and 'content' fields

        Returns:
            Dict with:
            - commerce_score: float (0-10)
            - reasoning: str
            - key_signals: list[str]
            - error: str (if any)
        """
        try:
            # Prepare article text
            article_text = self._prepare_article(article)

            # Build prompt
            full_prompt = self._build_prompt(article_text)

            # Call API
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                )
            )

            # Parse response
            result = self._parse_response(response.text)
            return result

        except Exception as e:
            return {
                "commerce_score": -1,
                "reasoning": f"ERROR: {str(e)}",
                "key_signals": [],
                "error": str(e),
            }

    def score_batch(
        self,
        articles: list,
        delay: float = 0.5,
        progress_every: int = 10,
    ) -> list:
        """
        Score multiple articles with rate limiting.

        Args:
            articles: List of article dicts
            delay: Seconds between API calls
            progress_every: Print progress every N articles

        Returns:
            List of result dicts
        """
        results = []

        for i, article in enumerate(articles):
            if i > 0 and i % progress_every == 0:
                print(f"  Progress: {i}/{len(articles)}")

            result = self.score_article(article)
            result['article_id'] = article.get('id', i)
            result['title'] = article.get('title', '')[:100]
            results.append(result)

            if delay > 0 and i < len(articles) - 1:
                time.sleep(delay)

        return results

    def _prepare_article(self, article: Dict) -> str:
        """Prepare article text for the prompt."""
        title = article.get('title', 'No title')
        content = article.get('content', article.get('text', ''))
        source = article.get('source', 'unknown')
        url = article.get('url', '')

        # Truncate content
        max_length = 3000
        if len(content) > max_length:
            content = content[:max_length] + "... [truncated]"

        return f"**Title:** {title}\n**Source:** {source}\n**URL:** {url}\n\n**Content:**\n{content}"

    def _build_prompt(self, article_text: str) -> str:
        """Build full prompt with article."""
        prompt = self.prompt_template.replace(
            "**INPUT DATA:** [Paste the article here]",
            f"**INPUT DATA:**\n\n{article_text}"
        )
        return prompt

    def _parse_response(self, response_text: str) -> Dict:
        """Parse JSON response from Gemini."""
        try:
            # Remove markdown code blocks if present
            text = response_text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())

            return {
                "commerce_score": float(data.get("commerce_score", 0)),
                "reasoning": data.get("reasoning", ""),
                "key_signals": data.get("key_signals", []),
                "error": None,
            }

        except Exception as e:
            return {
                "commerce_score": -1,
                "reasoning": f"Parse error: {str(e)}",
                "key_signals": [],
                "error": str(e),
            }

    def _load_api_key(self) -> str:
        """Load Gemini API key."""
        # Try environment variable
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            return api_key

        # Try config file
        secrets_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "credentials" / "secrets.ini"
        if secrets_path.exists():
            config = configparser.ConfigParser()
            config.read(secrets_path)
            if 'api_keys' in config:
                if 'gemini_billing_api_key' in config['api_keys']:
                    key = config['api_keys']['gemini_billing_api_key']
                    if key and not key.startswith('AIza-your-'):
                        print("Using gemini_billing_api_key")
                        return key
                if 'gemini_api_key' in config['api_keys']:
                    key = config['api_keys']['gemini_api_key']
                    if key and not key.startswith('AIza-your-'):
                        print("WARNING: Using free tier API key (rate limited)")
                        return key

        raise ValueError("Gemini API key not found")


def main():
    """Test the oracle on a few examples."""
    oracle = CommerceOracle()

    test_articles = [
        {
            "title": "Green Deals: Save $500 on Jackery Solar Generator",
            "content": "Today's Green Deals are headlined by an exclusive discount on the Jackery Explorer 1000 Plus. Originally $1,999, now just $1,499!",
            "source": "electrek",
        },
        {
            "title": "EPA Announces New Clean Energy Regulations",
            "content": "The Environmental Protection Agency announced new regulations requiring power plants to reduce carbon emissions by 50% by 2030.",
            "source": "reuters",
        },
    ]

    for article in test_articles:
        print(f"\n--- {article['title'][:50]}... ---")
        result = oracle.score_article(article)
        print(f"Score: {result['commerce_score']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Signals: {result['key_signals']}")


if __name__ == "__main__":
    main()
