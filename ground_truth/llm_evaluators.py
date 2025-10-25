"""LLM evaluators for generating ground truth ratings."""

import json
from pathlib import Path
from typing import Dict, Optional

import anthropic
import google.generativeai as genai
import openai


class BaseLLMEvaluator:
    """Base class for LLM evaluators."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def load_prompt_template(self, prompt_path: Path) -> str:
        """Load and parse prompt template from markdown file."""
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract prompt from markdown code blocks
        start_marker = "```\nAnalyze this article"
        end_marker = "DO NOT include any text outside the JSON object.\n```"

        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker, start_idx)

        if start_idx == -1 or end_idx == -1:
            raise ValueError("Could not find prompt template in markdown file")

        prompt = content[start_idx + 4:end_idx + len(end_marker) - 4]
        return prompt.strip()

    def evaluate(self, article: Dict, prompt_template: str) -> Optional[Dict]:
        """Evaluate a single article. Must be implemented by subclasses."""
        raise NotImplementedError


class ClaudeEvaluator(BaseLLMEvaluator):
    """Evaluator using Anthropic Claude models."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(api_key)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def evaluate(self, article: Dict, prompt_template: str) -> Optional[Dict]:
        """Rate article using Claude."""
        # Format prompt with article data
        prompt = prompt_template.format(
            title=article.get('title', 'N/A'),
            source=article.get('source', 'N/A'),
            published_date=article.get('published_date', 'N/A'),
            text=article.get('content', '')[:4000]  # Truncate to fit context
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.3,  # Lower for more consistent ratings
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON response
            response_text = response.content[0].text.strip()

            # Remove markdown formatting if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            ratings = json.loads(response_text.strip())
            return ratings

        except Exception as e:
            print(f"Error rating article {article.get('id')}: {e}")
            return None


class GeminiEvaluator(BaseLLMEvaluator):
    """Evaluator using Google Gemini models."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        super().__init__(api_key)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def evaluate(self, article: Dict, prompt_template: str) -> Optional[Dict]:
        """Rate article using Gemini."""
        # Format prompt
        prompt = prompt_template.format(
            title=article.get('title', 'N/A'),
            source=article.get('source', 'N/A'),
            published_date=article.get('published_date', 'N/A'),
            text=article.get('content', '')[:4000]
        )

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                )
            )

            # Parse JSON
            response_text = response.text.strip()

            # Remove markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            ratings = json.loads(response_text.strip())
            return ratings

        except Exception as e:
            print(f"Error rating article {article.get('id')}: {e}")
            return None


class GPT4Evaluator(BaseLLMEvaluator):
    """Evaluator using OpenAI GPT-4 models."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        super().__init__(api_key)
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def evaluate(self, article: Dict, prompt_template: str) -> Optional[Dict]:
        """Rate article using GPT-4."""
        prompt = prompt_template.format(
            title=article.get('title', 'N/A'),
            source=article.get('source', 'N/A'),
            published_date=article.get('published_date', 'N/A'),
            text=article.get('content', '')[:4000]
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048,
            )

            response_text = response.choices[0].message.content.strip()

            # Remove markdown
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            ratings = json.loads(response_text.strip())
            return ratings

        except Exception as e:
            print(f"Error rating article {article.get('id')}: {e}")
            return None
